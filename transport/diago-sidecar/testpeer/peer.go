// Package testpeer is an offline stand-in for miniSIPServer plus the far-end
// phone: a digest-authenticating registrar that answers the bridge's INVITE,
// plays scripted caller speech over real RTP, records what the bridge sends,
// and audits the received RTP headers. It never touches a real network peer.
package testpeer

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/emiago/diago"
	"github.com/emiago/diago/audio"
	"github.com/emiago/diago/media"
	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
	"github.com/icholy/digest"
	"github.com/pion/rtp"
)

const (
	frameMS      = 20
	frameSamples = 160
	frameBytes   = frameSamples * 2
)

// Utterance modes.
const (
	// AfterBot waits until the bot has spoken and then stayed silent.
	AfterBot = "after_bot"
	// BargeIn starts while the bot is speaking.
	BargeIn = "barge_in"
	// AfterDelay starts a fixed delay after the previous caller utterance.
	AfterDelay = "after_delay"
)

// Final actions once all utterances were played.
const (
	FinalHangupAfterReply = "hangup_after_reply"
	FinalHangupAfterDelay = "hangup_after_delay"
	FinalWaitForBye       = "wait_for_bye"
	FinalStopRTP          = "stop_rtp"
)

type Utterance struct {
	Name string `json:"name"`
	PCM  []byte `json:"-"`
	// Digits, when set, are sent as RFC 2833 keypad events instead of PCM.
	Digits  string `json:"digits"`
	Mode    string `json:"mode"`
	DelayMS int    `json:"delay_ms"`
}

type Script struct {
	Utterances     []Utterance
	BotSilenceMS   int
	ReplyTimeout   time.Duration
	FinalAction    string
	FinalDelayMS   int
	SpeechRMS      float64
	MaxCallTimeout time.Duration
}

type Config struct {
	BindHost string
	BindPort int
	Username string
	Password string
	Realm    string
	Codec    media.Codec
	Script   Script
}

type Segment struct {
	StartMS int64 `json:"start_ms"`
	EndMS   int64 `json:"end_ms"`
}

type UtteranceLog struct {
	Name               string `json:"name"`
	Mode               string `json:"mode"`
	StartMS            int64  `json:"start_ms"`
	EndMS              int64  `json:"end_ms"`
	BotSpeakingAtStart bool   `json:"bot_speaking_at_start"`
	// BotStopMS is how long the bot kept talking after a barge-in began.
	BotStopMS *int64 `json:"bot_stop_ms,omitempty"`
	// ReplyLatencyMS is caller speech end to the first following bot speech.
	ReplyLatencyMS *int64 `json:"reply_latency_ms,omitempty"`
}

type RTPStats struct {
	Packets           int            `json:"packets"`
	PayloadTypes      map[uint8]int  `json:"payload_types"`
	SSRCs             int            `json:"ssrcs"`
	SeqGaps           int            `json:"seq_gaps"`
	SeqReorders       int            `json:"seq_reorders"`
	TimestampErrors   int            `json:"timestamp_errors"`
	MarkerPackets     int            `json:"marker_packets"`
	PayloadSizes      map[int]int    `json:"payload_sizes"`
	MaxInterArrivalMS float64        `json:"max_interarrival_ms"`
	JitterMS          float64        `json:"jitter_ms"`
	ClockDriftMS      float64        `json:"clock_drift_ms"`
	firstArrival      time.Time      `json:"-"`
	firstTS           uint32         `json:"-"`
	lastArrival       time.Time      `json:"-"`
	lastSeq           uint16         `json:"-"`
	lastTS            uint32         `json:"-"`
	ssrcSeen          map[uint32]int `json:"-"`
	transit           float64        `json:"-"`
}

type Report struct {
	RegisterChallenged bool           `json:"register_challenged"`
	RegisterAuthorized bool           `json:"register_authorized"`
	InviteChallenged   bool           `json:"invite_challenged"`
	InviteAuthorized   bool           `json:"invite_authorized"`
	DialedUser         string         `json:"dialed_user"`
	Codec              string         `json:"codec"`
	Result             string         `json:"result"`
	HangupBy           string         `json:"hangup_by"`
	DurationMS         int64          `json:"duration_ms"`
	FirstBotSpeechMS   *int64         `json:"first_bot_speech_ms,omitempty"`
	BotSegments        []Segment      `json:"bot_segments"`
	Utterances         []UtteranceLog `json:"utterances"`
	RTP                *RTPStats      `json:"rtp"`
	ReceivedPCM        []byte         `json:"-"`
	SentPCM            []byte         `json:"-"`
}

type Peer struct {
	cfg    Config
	log    *slog.Logger
	nonce  string
	mu     sync.Mutex
	report Report
	done   chan Report
	once   sync.Once
}

func New(cfg Config, log *slog.Logger) *Peer {
	if cfg.Script.BotSilenceMS == 0 {
		cfg.Script.BotSilenceMS = 1200
	}
	if cfg.Script.ReplyTimeout == 0 {
		cfg.Script.ReplyTimeout = 60 * time.Second
	}
	if cfg.Script.SpeechRMS == 0 {
		cfg.Script.SpeechRMS = 300
	}
	if cfg.Script.MaxCallTimeout == 0 {
		cfg.Script.MaxCallTimeout = 10 * time.Minute
	}
	if cfg.Script.FinalAction == "" {
		cfg.Script.FinalAction = FinalHangupAfterReply
	}
	nonce := make([]byte, 16)
	_, _ = rand.Read(nonce)
	return &Peer{cfg: cfg, log: log, nonce: hex.EncodeToString(nonce), done: make(chan Report, 1)}
}

// Done delivers the report of the first (and only) answered call.
func (p *Peer) Done() <-chan Report { return p.done }

func (p *Peer) Start(ctx context.Context) error {
	ua, err := sipgo.NewUA(sipgo.WithUserAgent("nrsp-test-pbx"))
	if err != nil {
		return err
	}
	go func() {
		<-ctx.Done()
		_ = ua.Close()
	}()
	server, err := sipgo.NewServer(ua)
	if err != nil {
		return err
	}
	server.OnRegister(p.handleRegister)
	dg := diago.NewDiago(
		ua,
		diago.WithServer(server),
		diago.WithTransport(diago.Transport{Transport: "udp", BindHost: p.cfg.BindHost, BindPort: p.cfg.BindPort}),
		diago.WithMediaConfig(diago.MediaConfig{Codecs: []media.Codec{p.cfg.Codec}}),
		diago.WithServerRequestMiddleware(p.inviteAuth),
	)
	return dg.ServeBackground(ctx, p.serveCall)
}

func (p *Peer) challenge() *digest.Challenge {
	return &digest.Challenge{Realm: p.cfg.Realm, Nonce: p.nonce, Algorithm: "MD5"}
}

func (p *Peer) authorized(req *sip.Request, headerName string) bool {
	header := req.GetHeader(headerName)
	if header == nil {
		return false
	}
	cred, err := digest.ParseCredentials(header.Value())
	if err != nil || cred.Username != p.cfg.Username {
		return false
	}
	expected, err := digest.Digest(p.challenge(), digest.Options{
		Method: req.Method.String(), URI: cred.URI, Username: p.cfg.Username, Password: p.cfg.Password,
	})
	return err == nil && expected.Response == cred.Response
}

func (p *Peer) handleRegister(req *sip.Request, tx sip.ServerTransaction) {
	if !p.authorized(req, "Authorization") {
		p.mu.Lock()
		p.report.RegisterChallenged = true
		p.mu.Unlock()
		res := sip.NewResponseFromRequest(req, sip.StatusUnauthorized, "Unauthorized", nil)
		res.AppendHeader(sip.NewHeader("WWW-Authenticate", p.challenge().String()))
		_ = tx.Respond(res)
		return
	}
	p.mu.Lock()
	p.report.RegisterAuthorized = true
	p.mu.Unlock()
	res := sip.NewResponseFromRequest(req, sip.StatusOK, "OK", nil)
	if contact := req.Contact(); contact != nil {
		res.AppendHeader(sip.HeaderClone(contact))
	}
	res.AppendHeader(sip.NewHeader("Expires", "300"))
	_ = tx.Respond(res)
}

// inviteAuth challenges the initial INVITE with 407 like a PBX would.
func (p *Peer) inviteAuth(next sipgo.RequestHandler) sipgo.RequestHandler {
	return func(req *sip.Request, tx sip.ServerTransaction) {
		if req.Method == sip.INVITE && !req.To().Params.Has("tag") {
			if !p.authorized(req, "Proxy-Authorization") {
				p.mu.Lock()
				p.report.InviteChallenged = true
				p.mu.Unlock()
				res := sip.NewResponseFromRequest(req, sip.StatusProxyAuthRequired, "Proxy Authentication Required", nil)
				res.AppendHeader(sip.NewHeader("Proxy-Authenticate", p.challenge().String()))
				_ = tx.Respond(res)
				return
			}
			p.mu.Lock()
			p.report.InviteAuthorized = true
			p.mu.Unlock()
		}
		next(req, tx)
	}
}

type headerTap struct {
	inner  io.Reader
	reader *media.RTPPacketReader
	onPkt  func(rtp.Header)
}

func (h *headerTap) Read(b []byte) (int, error) {
	n, err := h.inner.Read(b)
	if err == nil && n > 0 {
		h.onPkt(h.reader.PacketHeader)
	}
	return n, err
}

// callState is shared between the RTP reader and the scripted writer.
type callState struct {
	mu              sync.Mutex
	start           time.Time
	botSpeaking     bool
	botSpeechStart  int64
	lastBotSpeechMS int64
	botSpokeSince   bool
	segments        []Segment
	received        []byte
	rtp             RTPStats
}

func (s *callState) nowMS() int64 { return time.Since(s.start).Milliseconds() }

func (p *Peer) serveCall(dialog *diago.DialogServerSession) {
	report := p.runCall(dialog)
	p.once.Do(func() { p.done <- report })
}

func (p *Peer) runCall(dialog *diago.DialogServerSession) Report {
	p.mu.Lock()
	report := p.report
	p.mu.Unlock()
	report.DialedUser = dialog.InviteRequest.Recipient.User
	if err := dialog.Ringing(); err != nil {
		report.Result = "ringing_failed"
		return report
	}
	med, err := dialog.Answer(diago.AnswerOptions{
		Codecs: []media.Codec{p.cfg.Codec, media.CodecTelephoneEvent8000},
	})
	if err != nil {
		report.Result = "answer_failed"
		return report
	}
	defer med.Close()
	state := &callState{start: time.Now(), rtp: RTPStats{
		PayloadTypes: map[uint8]int{}, PayloadSizes: map[int]int{}, ssrcSeen: map[uint32]int{},
	}}
	report.Codec = p.cfg.Codec.Name

	readerProps := diago.MediaProps{}
	encodedReader, err := med.AudioReader(diago.WithAudioReaderMediaProps(&readerProps))
	if err != nil {
		report.Result = "media_failed"
		return report
	}
	tap := &headerTap{inner: encodedReader, reader: med.RTPPacketReader}
	tap.onPkt = func(h rtp.Header) { state.recordRTP(h, time.Now()) }
	decoder := &audio.PCMDecoderReader{}
	if err := decoder.Init(readerProps.Codec, tap); err != nil {
		report.Result = "media_failed"
		return report
	}
	writerProps := diago.MediaProps{}
	dtmf := &diago.DTMFWriter{}
	encodedWriter, err := med.AudioWriter(
		diago.WithAudioWriterMediaProps(&writerProps), diago.WithAudioWriterDTMF(dtmf),
	)
	if err != nil {
		report.Result = "media_failed"
		return report
	}
	encoder := &audio.PCMEncoderWriter{}
	if err := encoder.Init(writerProps.Codec, encodedWriter); err != nil {
		report.Result = "media_failed"
		return report
	}

	go p.readLoop(decoder, state)
	ctx, cancel := context.WithTimeout(dialog.Context(), p.cfg.Script.MaxCallTimeout)
	defer cancel()
	result, sent, logs := p.writeLoop(ctx, encoder, dtmf, state)

	hangupBy := "bridge"
	if dialog.Context().Err() == nil {
		hangupBy = "peer"
		hctx, hcancel := context.WithTimeout(context.Background(), 3*time.Second)
		_ = dialog.Hangup(hctx)
		hcancel()
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	state.closeSegment()
	report.Result = result
	report.HangupBy = hangupBy
	report.DurationMS = state.nowMS()
	report.BotSegments = append([]Segment(nil), state.segments...)
	report.Utterances = annotate(logs, state.segments)
	if len(state.segments) > 0 {
		first := state.segments[0].StartMS
		report.FirstBotSpeechMS = &first
	}
	stats := state.rtp
	report.RTP = &stats
	report.RTP.SSRCs = len(state.rtp.ssrcSeen)
	report.ReceivedPCM = append([]byte(nil), state.received...)
	report.SentPCM = sent
	return report
}

func (p *Peer) readLoop(reader io.Reader, state *callState) {
	buf := make([]byte, 2*media.RTPBufSize)
	for {
		n, err := reader.Read(buf)
		if err != nil {
			return
		}
		pcm := buf[:n]
		speech := rms(pcm) >= p.cfg.Script.SpeechRMS
		state.mu.Lock()
		state.received = append(state.received, pcm...)
		now := state.nowMS()
		if speech {
			if !state.botSpeaking {
				state.botSpeaking = true
				state.botSpeechStart = now
				state.segments = append(state.segments, Segment{StartMS: now, EndMS: now})
			}
			state.lastBotSpeechMS = now
			state.botSpokeSince = true
			state.segments[len(state.segments)-1].EndMS = now
		} else if state.botSpeaking && now-state.lastBotSpeechMS >= 300 {
			state.botSpeaking = false
		}
		state.mu.Unlock()
	}
}

func (s *callState) closeSegment() {
	s.botSpeaking = false
}

// botIdle reports whether the bot spoke since the marker and then stayed
// silent for at least silenceMS.
func (s *callState) botIdle(silenceMS int) bool {
	return s.botSpokeSince && !s.botSpeaking && s.nowMS()-s.lastBotSpeechMS >= int64(silenceMS)
}

func (p *Peer) writeLoop(ctx context.Context, writer io.Writer, dtmf *diago.DTMFWriter, state *callState) (string, []byte, []UtteranceLog) {
	script := p.cfg.Script
	silence := make([]byte, frameBytes)
	var sent []byte
	var logs []UtteranceLog
	index := 0
	offset := 0
	playing := false
	waitStart := time.Now()
	lastUtteranceEnd := time.Now()
	finalStart := time.Time{}
	rtpStopped := false

	for {
		if ctx.Err() != nil {
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return "max_call_timeout", sent, logs
			}
			if index >= len(script.Utterances) && !playing {
				return "completed", sent, logs
			}
			return "bridge_hung_up_early", sent, logs
		}
		frame := silence
		state.mu.Lock()
		now := state.nowMS()
		if !playing && index < len(script.Utterances) {
			u := script.Utterances[index]
			ready := false
			switch u.Mode {
			case BargeIn:
				ready = state.botSpeaking && now-state.botSpeechStart >= int64(u.DelayMS)
			case AfterDelay:
				ready = time.Since(lastUtteranceEnd) >= time.Duration(u.DelayMS)*time.Millisecond
			default:
				ready = state.botIdle(script.BotSilenceMS)
			}
			if ready && u.Digits != "" {
				// Keypad entry: events go out on their own RTP payload type;
				// the lock is released while each key's events are paced.
				logs = append(logs, UtteranceLog{Name: u.Name, Mode: u.Mode, StartMS: now, BotSpeakingAtStart: state.botSpeaking})
				state.mu.Unlock()
				for _, key := range u.Digits {
					if err := dtmf.WriteDTMF(key); err != nil {
						return "dtmf_failed", sent, logs
					}
				}
				state.mu.Lock()
				logs[len(logs)-1].EndMS = state.nowMS()
				index++
				state.botSpokeSince = false
				waitStart = time.Now()
				lastUtteranceEnd = time.Now()
				state.mu.Unlock()
				continue
			}
			if ready {
				playing = true
				offset = 0
				logs = append(logs, UtteranceLog{Name: u.Name, Mode: u.Mode, StartMS: now, BotSpeakingAtStart: state.botSpeaking})
			} else if time.Since(waitStart) > script.ReplyTimeout {
				state.mu.Unlock()
				return "timeout_waiting_for_bot", sent, logs
			}
		}
		if playing {
			pcm := script.Utterances[index].PCM
			end := min(offset+frameBytes, len(pcm))
			chunk := make([]byte, frameBytes)
			copy(chunk, pcm[offset:end])
			frame = chunk
			offset = end
			if offset >= len(pcm) {
				playing = false
				logs[len(logs)-1].EndMS = now + frameMS
				index++
				state.botSpokeSince = false
				waitStart = time.Now()
				lastUtteranceEnd = time.Now()
			}
		}
		if index >= len(script.Utterances) && !playing {
			if finalStart.IsZero() {
				finalStart = time.Now()
			}
			switch script.FinalAction {
			case FinalHangupAfterReply:
				if state.botIdle(script.BotSilenceMS) {
					state.mu.Unlock()
					return "completed", sent, logs
				}
				if time.Since(finalStart) > script.ReplyTimeout {
					state.mu.Unlock()
					return "timeout_waiting_for_bot", sent, logs
				}
			case FinalHangupAfterDelay:
				if time.Since(finalStart) >= time.Duration(script.FinalDelayMS)*time.Millisecond {
					state.mu.Unlock()
					return "completed", sent, logs
				}
			case FinalStopRTP:
				rtpStopped = true
			}
		}
		state.mu.Unlock()

		if rtpStopped {
			select {
			case <-ctx.Done():
			case <-time.After(frameMS * time.Millisecond):
			}
			continue
		}
		sent = append(sent, frame...)
		if _, err := writer.Write(frame); err != nil {
			if index >= len(script.Utterances) && !playing {
				return "completed", sent, logs
			}
			return "bridge_hung_up_early", sent, logs
		}
	}
}

// annotate derives barge-in stop time and reply latency from bot segments.
func annotate(logs []UtteranceLog, segments []Segment) []UtteranceLog {
	out := append([]UtteranceLog(nil), logs...)
	for i := range out {
		u := &out[i]
		if u.BotSpeakingAtStart {
			for _, seg := range segments {
				if seg.StartMS <= u.StartMS && seg.EndMS >= u.StartMS {
					stop := seg.EndMS - u.StartMS
					u.BotStopMS = &stop
				}
			}
		}
		if u.EndMS == 0 {
			continue
		}
		for _, seg := range segments {
			if seg.StartMS >= u.EndMS {
				latency := seg.StartMS - u.EndMS
				u.ReplyLatencyMS = &latency
				break
			}
		}
	}
	return out
}

func (s *callState) recordRTP(h rtp.Header, arrival time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := &s.rtp
	st.PayloadTypes[h.PayloadType]++
	st.ssrcSeen[h.SSRC]++
	if h.Marker {
		st.MarkerPackets++
	}
	if st.Packets == 0 {
		st.firstArrival, st.firstTS = arrival, h.Timestamp
	} else {
		seqDelta := h.SequenceNumber - st.lastSeq
		switch {
		case seqDelta == 0 || seqDelta > 0x8000:
			st.SeqReorders++
		case seqDelta > 1:
			st.SeqGaps++
		}
		if seqDelta >= 1 && seqDelta < 0x8000 && h.Timestamp-st.lastTS != uint32(seqDelta)*frameSamples {
			st.TimestampErrors++
		}
		gap := arrival.Sub(st.lastArrival).Seconds() * 1000
		st.MaxInterArrivalMS = math.Max(st.MaxInterArrivalMS, gap)
		// RFC 3550 interarrival jitter estimate, in milliseconds.
		transit := arrival.Sub(st.firstArrival).Seconds()*1000 - float64(h.Timestamp-st.firstTS)/8.0
		if st.Packets > 1 {
			d := math.Abs(transit - st.transit)
			st.JitterMS += (d - st.JitterMS) / 16
		}
		st.transit = transit
		st.ClockDriftMS = transit
	}
	st.Packets++
	st.lastSeq, st.lastTS, st.lastArrival = h.SequenceNumber, h.Timestamp, arrival
}

func rms(pcm []byte) float64 {
	if len(pcm) < 2 {
		return 0
	}
	var sum float64
	for i := 0; i+1 < len(pcm); i += 2 {
		v := float64(int16(uint16(pcm[i]) | uint16(pcm[i+1])<<8))
		sum += v * v
	}
	return math.Sqrt(sum / float64(len(pcm)/2))
}

// String is used in logs only.
func (r Report) String() string {
	return fmt.Sprintf("result=%s hangup_by=%s utterances=%d bot_segments=%d", r.Result, r.HangupBy, len(r.Utterances), len(r.BotSegments))
}
