package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"regexp"
	"sync"
	"sync/atomic"
	"time"

	"github.com/emiago/diago"
	"github.com/emiago/diago/audio"
	"github.com/emiago/diago/media"
	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
)

var safeID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$`)

var errRTPTimeout = errors.New("rtp_timeout")

const (
	rtpFrameDuration = 20 * time.Millisecond
	// 10 s of 8 kHz PCM16; the Python sender paces to a small lead, so this
	// bound only protects the bridge from a misbehaving controller.
	maxPlayoutBytes = 160_000
	maxDrainWait    = 10 * time.Second
)

type bridge struct {
	cfg        config
	diago      *diago.Diago
	ua         *sipgo.UserAgent
	startedAt  time.Time
	registered atomic.Bool
	callsMu    sync.Mutex
	calls      map[string]*callSession
}

type callSession struct {
	bridge     *bridge
	owner      *clientConn
	callID     string
	ctx        context.Context
	cancel     context.CancelFunc
	playout    *playoutBuffer
	lastRx     atomic.Int64
	clockSkips atomic.Int64

	mu          sync.Mutex
	dialog      *diago.DialogClientSession
	media       *diago.DialogMedia
	pcmWriter   io.Writer
	localReason string
	answered    bool
	ringing     bool
	finished    bool
}

func newBridge(cfg config) (*bridge, error) {
	ua, err := sipgo.NewUA(
		sipgo.WithUserAgent(cfg.SIPUser),
		sipgo.WithUserAgentHostname(cfg.PBXHost),
	)
	if err != nil {
		return nil, fmt.Errorf("create SIP user agent: %w", err)
	}
	dg := diago.NewDiago(
		ua,
		diago.WithTransport(diago.Transport{
			Transport: "udp",
			BindHost:  cfg.SIPBindHost,
			BindPort:  cfg.SIPBindPort,
		}),
		diago.WithMediaConfig(diago.MediaConfig{
			Codecs: []media.Codec{
				media.CodecAudioAlaw,
				media.CodecAudioUlaw,
				media.CodecTelephoneEvent8000,
			},
		}),
	)
	return &bridge{
		cfg:       cfg,
		diago:     dg,
		ua:        ua,
		startedAt: time.Now(),
		calls:     make(map[string]*callSession),
	}, nil
}

func (b *bridge) run(ctx context.Context) error {
	defer b.ua.Close()
	if err := b.diago.ServeBackground(ctx, func(dialog *diago.DialogServerSession) {
		_ = dialog.Hangup(dialog.Context())
	}); err != nil {
		return fmt.Errorf("start SIP listener: %w", err)
	}
	go b.registrationLoop(ctx)

	listener, err := net.Listen("tcp", b.cfg.ControlAddress)
	if err != nil {
		return fmt.Errorf("listen on control socket: %w", err)
	}
	defer listener.Close()
	go func() {
		<-ctx.Done()
		_ = listener.Close()
	}()

	slog.Info("media bridge ready", "control_address", b.cfg.ControlAddress)
	for {
		conn, acceptErr := listener.Accept()
		if acceptErr != nil {
			if ctx.Err() != nil {
				return nil
			}
			return fmt.Errorf("accept control connection: %w", acceptErr)
		}
		go b.serveClient(ctx, newClientConn(conn))
	}
}

func (b *bridge) registrationLoop(ctx context.Context) {
	params := sip.NewParams()
	params.Add("transport", "udp")
	recipient := sip.Uri{
		Scheme:    "sip",
		User:      b.cfg.SIPUser,
		Host:      b.cfg.PBXHost,
		Port:      b.cfg.PBXPort,
		UriParams: params,
	}
	for ctx.Err() == nil {
		err := b.diago.Register(ctx, recipient, diago.RegisterOptions{
			Username: b.cfg.SIPUser,
			Password: b.cfg.SIPPassword,
			Expiry:   5 * time.Minute,
			OnRegistered: func() {
				b.registered.Store(true)
				slog.Info("SIP account registered")
			},
		})
		b.registered.Store(false)
		if ctx.Err() != nil {
			return
		}
		slog.Warn("SIP registration stopped; retrying", "error_kind", classifyError(err))
		select {
		case <-ctx.Done():
			return
		case <-time.After(5 * time.Second):
		}
	}
}

func (b *bridge) serveClient(ctx context.Context, client *clientConn) {
	defer client.conn.Close()
	defer b.cancelOwnedCalls(client)
	scanner := bufio.NewScanner(client.conn)
	scanner.Buffer(make([]byte, 64*1024), maxControlMessageBytes)
	for scanner.Scan() {
		request, decoderErr := decodeCommand(scanner.Bytes())
		if decoderErr != nil {
			_ = client.send(response{Type: "response", OK: false, Error: "invalid_json"})
			continue
		}
		if request.Type != "command" || request.RequestID == "" {
			_ = client.send(response{Type: "response", RequestID: request.RequestID, OK: false, Error: "invalid_request"})
			continue
		}
		if !tokenMatches(request.AuthToken, b.cfg.AuthToken) {
			_ = client.send(response{Type: "response", RequestID: request.RequestID, OK: false, Error: "unauthorized"})
			continue
		}
		if err := b.handleCommand(ctx, client, request); err != nil {
			_ = client.send(response{Type: "response", RequestID: request.RequestID, OK: false, Error: err.Error()})
			continue
		}
		if request.Command != "health" {
			_ = client.send(response{Type: "response", RequestID: request.RequestID, OK: true})
		}
	}
}

func (b *bridge) handleCommand(ctx context.Context, client *clientConn, request command) error {
	switch request.Command {
	case "health":
		registered := b.registered.Load()
		return client.send(response{Type: "response", RequestID: request.RequestID, OK: true, Registered: &registered})
	case "start_call":
		return b.startCall(ctx, client, request)
	case "hangup":
		return b.hangup(request.CallID, request.Reason, request.Drain)
	case "send_audio":
		return b.sendAudio(request)
	case "clear_audio":
		return b.clearAudio(request.CallID)
	default:
		return errors.New("unknown_command")
	}
}

func (b *bridge) startCall(parent context.Context, client *clientConn, request command) error {
	if !safeID.MatchString(request.CallID) || !safeID.MatchString(request.GatewayID) {
		return errors.New("invalid_call_or_gateway_id")
	}
	if request.GatewayID != b.cfg.GatewayID {
		return errors.New("gateway_not_configured")
	}
	if !b.cfg.LiveCalls {
		return errors.New("live_calls_disabled")
	}
	if _, allowed := b.cfg.AllowedNumbers[request.Number]; !allowed {
		return errors.New("number_not_allowlisted")
	}
	if !b.registered.Load() {
		return errors.New("sip_not_registered")
	}

	b.callsMu.Lock()
	defer b.callsMu.Unlock()
	if _, exists := b.calls[request.CallID]; exists {
		return errors.New("call_id_already_active")
	}
	for _, active := range b.calls {
		if active != nil {
			return errors.New("gateway_channel_busy")
		}
	}
	callCtx, cancel := context.WithCancel(parent)
	session := &callSession{
		bridge:  b,
		owner:   client,
		callID:  request.CallID,
		ctx:     callCtx,
		cancel:  cancel,
		playout: newPlayoutBuffer(maxPlayoutBytes),
	}
	b.calls[request.CallID] = session
	go session.run(request.Number)
	return nil
}

func (b *bridge) hangup(callID string, reason string, drain bool) error {
	session := b.getCall(callID)
	if session == nil {
		return errors.New("call_not_found")
	}
	if reason == "" {
		reason = "requested"
	}
	session.mu.Lock()
	session.localReason = reason
	session.mu.Unlock()
	if drain {
		go session.cancelAfterDrain(maxDrainWait)
		return nil
	}
	session.cancel()
	return nil
}

// cancelAfterDrain lets queued closing speech reach the caller before BYE.
func (s *callSession) cancelAfterDrain(maxWait time.Duration) {
	deadline := time.NewTimer(maxWait)
	defer deadline.Stop()
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for s.playout.len() > 0 {
		select {
		case <-s.ctx.Done():
			return
		case <-deadline.C:
			s.cancel()
			return
		case <-ticker.C:
		}
	}
	// The final frame has been taken but may still be on the media clock.
	select {
	case <-s.ctx.Done():
	case <-time.After(2 * rtpFrameDuration):
	}
	s.cancel()
}

func (b *bridge) clearAudio(callID string) error {
	session := b.getCall(callID)
	if session == nil {
		return errors.New("call_not_found")
	}
	session.playout.clear()
	return nil
}

func (b *bridge) sendAudio(request command) error {
	session := b.getCall(request.CallID)
	if session == nil {
		return errors.New("call_not_found")
	}
	if request.Encoding != "pcm_s16le" || request.Channels != 1 || request.SampleRate != 8000 {
		return errors.New("audio_must_be_pcm_s16le_mono_8000hz")
	}
	pcm, err := base64.StdEncoding.Strict().DecodeString(request.PCMBase64)
	if err != nil || len(pcm) == 0 || len(pcm)%2 != 0 || len(pcm) > 1_920_000 {
		return errors.New("invalid_pcm_payload")
	}
	session.mu.Lock()
	ready := session.pcmWriter != nil && !session.finished
	session.mu.Unlock()
	if !ready {
		return errors.New("rtp_not_ready")
	}
	return session.playout.push(pcm)
}

func (b *bridge) getCall(callID string) *callSession {
	b.callsMu.Lock()
	defer b.callsMu.Unlock()
	return b.calls[callID]
}

func (b *bridge) removeCall(callID string) {
	b.callsMu.Lock()
	delete(b.calls, callID)
	b.callsMu.Unlock()
}

func (b *bridge) cancelOwnedCalls(owner *clientConn) {
	b.callsMu.Lock()
	owned := make([]*callSession, 0)
	for _, session := range b.calls {
		if session.owner == owner {
			owned = append(owned, session)
		}
	}
	b.callsMu.Unlock()
	for _, session := range owned {
		session.mu.Lock()
		session.localReason = "controller_disconnected"
		session.mu.Unlock()
		session.cancel()
	}
}

func (s *callSession) run(number string) {
	defer s.bridge.removeCall(s.callID)
	s.emit("DIALING", "", 0, "")
	inviteCtx, inviteCancel := context.WithTimeout(s.ctx, s.bridge.cfg.InviteTimeout)
	defer inviteCancel()
	recipient := sip.Uri{Scheme: "sip", User: number, Host: s.bridge.cfg.PBXHost, Port: s.bridge.cfg.PBXPort}
	dialog, med, err := s.bridge.diago.Invite(inviteCtx, recipient, diago.InviteOptions{
		Transport: "udp",
		Username:  s.bridge.cfg.SIPUser,
		Password:  s.bridge.cfg.SIPPassword,
		OnResponse: func(res *sip.Response) error {
			if res.StatusCode == sip.StatusRinging || res.StatusCode == 183 {
				s.mu.Lock()
				first := !s.ringing
				s.ringing = true
				s.mu.Unlock()
				if first {
					s.emit("RINGING", "", res.StatusCode, "")
				}
			}
			return nil
		},
	})
	if err != nil {
		if s.ctx.Err() != nil {
			s.finish("HANGUP", s.reasonOr("cancelled"))
			return
		}
		s.finish("FAILED", classifyError(err))
		return
	}
	s.mu.Lock()
	s.dialog = dialog
	s.media = med
	s.answered = true
	s.mu.Unlock()
	defer dialog.Close()
	defer med.Close()
	s.emit("ANSWERED", "", sip.StatusOK, "")

	pcmReader, pcmWriter, codec, err := setupPCM(med)
	if err != nil {
		s.emit("RTP_LOST", "media_setup_failed", 0, "")
		s.hangupDialog()
		s.finish("HANGUP", "media_setup_failed")
		return
	}
	s.mu.Lock()
	s.pcmWriter = pcmWriter
	s.mu.Unlock()
	s.lastRx.Store(time.Now().UnixNano())
	s.emit("RTP_READY", "", 0, codec.Name)

	mediaErr := make(chan error, 3)
	frameBytes := int(codec.SamplesPCM(16))
	go s.readPCM(pcmReader, int(codec.SampleRate), mediaErr)
	go s.writePCM(pcmWriter, med.RTPPacketWriter, frameBytes, mediaErr)
	go s.watchRTP(s.bridge.cfg.RTPTimeout, mediaErr)

	terminalReason := "remote_hangup"
	select {
	case <-s.ctx.Done():
		terminalReason = s.reasonOr("cancelled")
		s.hangupDialog()
	case <-dialog.Context().Done():
		terminalReason = "remote_hangup"
	case err := <-mediaErr:
		select {
		case <-dialog.Context().Done():
			terminalReason = "remote_hangup"
		default:
			terminalReason = "media_io_failed"
			if errors.Is(err, errRTPTimeout) {
				terminalReason = "rtp_timeout"
			}
			s.emit("RTP_LOST", terminalReason, 0, "")
			s.hangupDialog()
		}
	}
	if skipped := s.clockSkips.Load(); skipped > 0 {
		slog.Warn("media clock re-aligned after scheduling stalls", "skipped_frames", skipped)
	}
	s.finish("HANGUP", terminalReason)
}

// watchRTP fails the call when the far end stops sending media without a BYE.
func (s *callSession) watchRTP(timeout time.Duration, result chan<- error) {
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case now := <-ticker.C:
			if now.Sub(time.Unix(0, s.lastRx.Load())) > timeout {
				reportMediaError(result, errRTPTimeout)
				return
			}
		}
	}
}

func setupPCM(med *diago.DialogMedia) (io.Reader, io.Writer, media.Codec, error) {
	readerProps := diago.MediaProps{}
	encodedReader, err := med.AudioReader(diago.WithAudioReaderMediaProps(&readerProps))
	if err != nil {
		return nil, nil, media.Codec{}, err
	}
	writerProps := diago.MediaProps{}
	encodedWriter, err := med.AudioWriter(diago.WithAudioWriterMediaProps(&writerProps))
	if err != nil {
		return nil, nil, media.Codec{}, err
	}
	if readerProps.Codec.PayloadType != writerProps.Codec.PayloadType || readerProps.Codec.SampleRate != 8000 {
		return nil, nil, media.Codec{}, errors.New("unsupported negotiated codec")
	}
	decoder := &audio.PCMDecoderReader{}
	if err := decoder.Init(readerProps.Codec, encodedReader); err != nil {
		return nil, nil, media.Codec{}, err
	}
	encoder := &audio.PCMEncoderWriter{}
	if err := encoder.Init(writerProps.Codec, encodedWriter); err != nil {
		return nil, nil, media.Codec{}, err
	}
	return decoder, encoder, readerProps.Codec, nil
}

// readPCM forwards each decoded RTP packet. The decoder returns one packet per
// Read, so the buffer is sized for any remote ptime rather than 20 ms only.
func (s *callSession) readPCM(reader io.Reader, sampleRate int, result chan<- error) {
	buffer := make([]byte, 2*media.RTPBufSize)
	for {
		n, err := reader.Read(buffer)
		if err != nil {
			reportMediaError(result, err)
			return
		}
		if n == 0 || n%2 != 0 {
			continue
		}
		s.lastRx.Store(time.Now().UnixNano())
		message := audioMessage{
			Type:        "audio",
			CallID:      s.callID,
			PCMBase64:   base64.StdEncoding.EncodeToString(buffer[:n]),
			SampleRate:  sampleRate,
			Channels:    1,
			Encoding:    "pcm_s16le",
			MonotonicMS: s.bridge.monotonicMS(),
		}
		if err := s.owner.send(message); err != nil {
			reportMediaError(result, err)
			return
		}
	}
}

// writePCM sends one frame per media-clock tick for the whole call: queued
// speech when available, otherwise silence. Diago's RTP writer blocks on its
// 20 ms ticker and advances the RTP timestamp per packet, so a continuous
// stream keeps timestamps aligned with wall clock across pauses.
//
// A starved process misses ticks (time.Ticker drops them), which would leave
// RTP time permanently behind wall clock; measured 4.9 s over 218 s on the
// paging demo laptop. When that happens the timestamp skips the missed
// interval, which receivers treat as a short loss, and the clock re-aligns.
func (s *callSession) writePCM(writer io.Writer, rtpWriter *media.RTPPacketWriter, frameBytes int, result chan<- error) {
	frame := make([]byte, frameBytes)
	samplesPerFrame := uint32(frameBytes / 2)
	start := time.Now()
	var sent int64
	for s.ctx.Err() == nil {
		s.playout.take(frame)
		if _, err := writer.Write(frame); err != nil {
			if s.ctx.Err() == nil {
				reportMediaError(result, err)
			}
			return
		}
		sent++
		if behind := int64(time.Since(start)/rtpFrameDuration) - sent; behind >= 3 && rtpWriter != nil {
			rtpWriter.DelayTimestamp(uint32(behind) * samplesPerFrame)
			sent += behind
			s.clockSkips.Add(behind)
		}
	}
}

func reportMediaError(result chan<- error, err error) {
	select {
	case result <- err:
	default:
	}
}

func (s *callSession) hangupDialog() {
	s.mu.Lock()
	dialog := s.dialog
	s.mu.Unlock()
	if dialog == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = dialog.Hangup(ctx)
}

func (s *callSession) finish(event string, reason string) {
	s.mu.Lock()
	if s.finished {
		s.mu.Unlock()
		return
	}
	s.finished = true
	s.mu.Unlock()
	s.emit(event, reason, 0, "")
	s.cancel()
}

func (s *callSession) reasonOr(fallback string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.localReason != "" {
		return s.localReason
	}
	return fallback
}

func (s *callSession) emit(event string, reason string, status int, codec string) {
	_ = s.owner.send(callEvent{
		Type:        "event",
		CallID:      s.callID,
		Event:       event,
		MonotonicMS: s.bridge.monotonicMS(),
		Reason:      reason,
		SIPStatus:   status,
		Codec:       codec,
	})
}

func (b *bridge) monotonicMS() int64 {
	return time.Since(b.startedAt).Milliseconds()
}

func classifyError(err error) string {
	if err == nil {
		return "unknown"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "timeout"
	}
	if errors.Is(err, context.Canceled) {
		return "cancelled"
	}
	var responseErr sipgo.ErrDialogResponse
	if errors.As(err, &responseErr) {
		return fmt.Sprintf("sip_%d", responseErr.Res.StatusCode)
	}
	return "transport_error"
}
