package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"testing"
	"time"

	"github.com/emiago/diago/media"

	"polarityiq.com/nrsp/diago-sidecar/testpeer"
)

const testToken = "integration-test-token"

type controlClient struct {
	t      *testing.T
	conn   net.Conn
	seq    int
	resp   chan map[string]any
	events chan map[string]any
	audio  chan []byte
}

func dialControl(t *testing.T, address string) *controlClient {
	t.Helper()
	conn, err := net.Dial("tcp", address)
	if err != nil {
		t.Fatalf("dial control: %v", err)
	}
	c := &controlClient{
		t: t, conn: conn,
		resp: make(chan map[string]any, 16), events: make(chan map[string]any, 64),
		audio: make(chan []byte, 4096),
	}
	go func() {
		scanner := bufio.NewScanner(conn)
		scanner.Buffer(make([]byte, 64*1024), maxControlMessageBytes)
		for scanner.Scan() {
			var msg map[string]any
			if json.Unmarshal(scanner.Bytes(), &msg) != nil {
				continue
			}
			switch msg["type"] {
			case "response":
				c.resp <- msg
			case "event":
				c.events <- msg
			case "audio":
				pcm, _ := base64.StdEncoding.DecodeString(msg["pcm_b64"].(string))
				select {
				case c.audio <- pcm:
				default:
				}
			}
		}
	}()
	t.Cleanup(func() { _ = conn.Close() })
	return c
}

func (c *controlClient) command(name string, fields map[string]any) map[string]any {
	c.t.Helper()
	c.seq++
	msg := map[string]any{"type": "command", "request_id": fmt.Sprintf("r%d", c.seq), "auth_token": testToken, "command": name}
	for k, v := range fields {
		msg[k] = v
	}
	data, _ := json.Marshal(msg)
	if _, err := c.conn.Write(append(data, '\n')); err != nil {
		c.t.Fatalf("write command: %v", err)
	}
	select {
	case r := <-c.resp:
		return r
	case <-time.After(5 * time.Second):
		c.t.Fatalf("no response to %s", name)
		return nil
	}
}

func (c *controlClient) expectEvent(want string, timeout time.Duration) map[string]any {
	c.t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case ev := <-c.events:
			if ev["event"] == want {
				return ev
			}
			if want != "HANGUP" && (ev["event"] == "HANGUP" || ev["event"] == "FAILED") {
				c.t.Fatalf("got %v while waiting for %s", ev, want)
			}
		case <-deadline:
			c.t.Fatalf("timed out waiting for %s", want)
			return nil
		}
	}
}

func (c *controlClient) sendPCM(callID string, pcm []byte) {
	c.t.Helper()
	r := c.command("send_audio", map[string]any{
		"call_id": callID, "sample_rate": 8000, "channels": 1, "encoding": "pcm_s16le",
		"pcm_b64": base64.StdEncoding.EncodeToString(pcm),
	})
	if r["ok"] != true {
		c.t.Fatalf("send_audio rejected: %v", r)
	}
}

func tone(freq float64, duration time.Duration, amplitude float64) []byte {
	samples := int(duration.Seconds() * 8000)
	pcm := make([]byte, samples*2)
	for i := range samples {
		v := int16(amplitude * math.Sin(2*math.Pi*freq*float64(i)/8000))
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(v))
	}
	return pcm
}

func reserveTCPAddress(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := l.Addr().String()
	_ = l.Close()
	return address
}

type harness struct {
	peer   *testpeer.Peer
	client *controlClient
}

func startHarness(t *testing.T, ctx context.Context, script testpeer.Script, rtpTimeout time.Duration) harness {
	t.Helper()
	peerPort := reserveUDPPort(t)
	peer := testpeer.New(testpeer.Config{
		BindHost: "127.0.0.1", BindPort: peerPort, Username: "100", Password: "test-secret",
		Realm: "nrsp-test", Codec: media.CodecAudioAlaw, Script: script,
	}, nil)
	if err := peer.Start(ctx); err != nil {
		t.Fatalf("start peer: %v", err)
	}
	cfg := config{
		ControlAddress: reserveTCPAddress(t), AuthToken: testToken,
		SIPBindHost: "127.0.0.1", SIPBindPort: reserveUDPPort(t),
		PBXHost: "127.0.0.1", PBXPort: peerPort, SIPUser: "100", SIPPassword: "test-secret",
		GatewayID: "sim-1", LiveCalls: true, AllowedNumbers: map[string]struct{}{"9000": {}},
		InviteTimeout: 10 * time.Second, RTPTimeout: rtpTimeout,
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	b, err := newBridge(cfg)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = b.run(ctx) }()

	var client *controlClient
	for deadline := time.Now().Add(5 * time.Second); ; {
		if conn, err := net.Dial("tcp", cfg.ControlAddress); err == nil {
			_ = conn.Close()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("bridge control socket never opened")
		}
		time.Sleep(20 * time.Millisecond)
	}
	client = dialControl(t, cfg.ControlAddress)
	for deadline := time.Now().Add(5 * time.Second); ; {
		if r := client.command("health", nil); r["registered"] == true {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("bridge never registered with the test PBX")
		}
		time.Sleep(50 * time.Millisecond)
	}
	return harness{peer: peer, client: client}
}

func waitReport(t *testing.T, peer *testpeer.Peer, timeout time.Duration) testpeer.Report {
	t.Helper()
	select {
	case r := <-peer.Done():
		return r
	case <-time.After(timeout):
		t.Fatal("peer report timed out")
		return testpeer.Report{}
	}
}

func assertCleanRTP(t *testing.T, report testpeer.Report) {
	t.Helper()
	rtp := report.RTP
	t.Logf("rtp=%+v segments=%+v", *rtp, report.BotSegments)
	if rtp.Packets < 50 || rtp.SSRCs != 1 || rtp.SeqGaps != 0 || rtp.SeqReorders != 0 || rtp.TimestampErrors != 0 {
		t.Fatalf("RTP integrity failure: %+v", *rtp)
	}
	if len(rtp.PayloadTypes) != 1 || rtp.PayloadTypes[media.CodecAudioAlaw.PayloadType] != rtp.Packets {
		t.Fatalf("unexpected payload types: %v", rtp.PayloadTypes)
	}
	// The continuous media clock keeps RTP time aligned with arrival time.
	if math.Abs(rtp.ClockDriftMS) > 120 || rtp.JitterMS > 15 {
		t.Fatalf("RTP clock drift %.1f ms / jitter %.1f ms", rtp.ClockDriftMS, rtp.JitterMS)
	}
}

func TestBridgeConversationOverLoopbackPBX(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	h := startHarness(t, ctx, testpeer.Script{
		BotSilenceMS: 400,
		Utterances: []testpeer.Utterance{
			{Name: "caller-tone", PCM: tone(440, time.Second, 8000), Mode: testpeer.AfterBot},
		},
		FinalAction: testpeer.FinalHangupAfterReply,
	}, 15*time.Second)
	c := h.client

	if r := c.command("start_call", map[string]any{"call_id": "call-1", "gateway_id": "sim-1", "number": "9000"}); r["ok"] != true {
		t.Fatalf("start_call rejected: %v", r)
	}
	if r := c.command("start_call", map[string]any{"call_id": "call-2", "gateway_id": "sim-1", "number": "9000"}); r["error"] != "gateway_channel_busy" {
		t.Fatalf("second concurrent call on one SIM must be refused, got %v", r)
	}
	c.expectEvent("DIALING", 5*time.Second)
	c.expectEvent("ANSWERED", 5*time.Second)
	if ev := c.expectEvent("RTP_READY", 5*time.Second); ev["codec"] != "PCMA" {
		t.Fatalf("codec %v", ev["codec"])
	}

	// Greeting with an idle gap first, so RTP timestamp alignment across
	// silence is exercised.
	time.Sleep(600 * time.Millisecond)
	c.sendPCM("call-1", tone(600, 700*time.Millisecond, 8000))

	// Caller speech must arrive through the bridge.
	heard := 0
	for deadline := time.After(10 * time.Second); heard < 25; {
		select {
		case pcm := <-c.audio:
			if pcmRMS(pcm) > 2000 {
				heard++
			}
		case <-deadline:
			t.Fatalf("caller tone not received through the bridge (frames=%d)", heard)
		}
	}
	time.Sleep(300 * time.Millisecond)
	c.sendPCM("call-1", tone(500, 500*time.Millisecond, 8000))

	ev := c.expectEvent("HANGUP", 15*time.Second)
	if ev["reason"] != "remote_hangup" {
		t.Fatalf("hangup reason %v", ev["reason"])
	}
	report := waitReport(t, h.peer, 5*time.Second)
	if !report.RegisterChallenged || !report.RegisterAuthorized || !report.InviteChallenged || !report.InviteAuthorized {
		t.Fatalf("digest auth not exercised: %+v", report)
	}
	if report.DialedUser != "9000" || report.Result != "completed" || report.HangupBy != "peer" {
		t.Fatalf("unexpected call outcome: %s dialed=%s", report, report.DialedUser)
	}
	if len(report.BotSegments) != 2 {
		t.Fatalf("expected greeting and reply segments, got %+v", report.BotSegments)
	}
	if got := report.BotSegments[0].EndMS - report.BotSegments[0].StartMS; got < 600 || got > 800 {
		t.Fatalf("greeting duration %d ms, expected ~700", got)
	}
	assertCleanRTP(t, report)
}

func TestBridgeClearAudioAndDrainHangup(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	h := startHarness(t, ctx, testpeer.Script{
		BotSilenceMS: 400, FinalAction: testpeer.FinalWaitForBye, ReplyTimeout: 20 * time.Second,
	}, 15*time.Second)
	c := h.client
	c.command("start_call", map[string]any{"call_id": "call-1", "gateway_id": "sim-1", "number": "9000"})
	c.expectEvent("RTP_READY", 5*time.Second)

	// Barge-in: three seconds queued, cleared after half a second.
	c.sendPCM("call-1", tone(600, 3*time.Second, 8000))
	time.Sleep(500 * time.Millisecond)
	if r := c.command("clear_audio", map[string]any{"call_id": "call-1"}); r["ok"] != true {
		t.Fatalf("clear_audio rejected: %v", r)
	}
	time.Sleep(400 * time.Millisecond)

	// Closing speech must play completely before BYE.
	c.sendPCM("call-1", tone(500, 800*time.Millisecond, 8000))
	if r := c.command("hangup", map[string]any{"call_id": "call-1", "reason": "conversation_complete", "drain": true}); r["ok"] != true {
		t.Fatalf("hangup rejected: %v", r)
	}
	if ev := c.expectEvent("HANGUP", 10*time.Second); ev["reason"] != "conversation_complete" {
		t.Fatalf("hangup reason %v", ev["reason"])
	}
	report := waitReport(t, h.peer, 5*time.Second)
	if report.HangupBy != "bridge" || len(report.BotSegments) != 2 {
		t.Fatalf("unexpected outcome %s segments=%+v", report, report.BotSegments)
	}
	if cut := report.BotSegments[0].EndMS - report.BotSegments[0].StartMS; cut > 650 {
		t.Fatalf("clear_audio left %d ms of stale speech playing", cut)
	}
	if closing := report.BotSegments[1].EndMS - report.BotSegments[1].StartMS; closing < 700 {
		t.Fatalf("drain hangup truncated closing speech to %d ms", closing)
	}
	assertCleanRTP(t, report)
}

func TestBridgeDetectsRTPLoss(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	h := startHarness(t, ctx, testpeer.Script{
		FinalAction: testpeer.FinalStopRTP, ReplyTimeout: 20 * time.Second,
	}, 2*time.Second)
	c := h.client
	c.command("start_call", map[string]any{"call_id": "call-1", "gateway_id": "sim-1", "number": "9000"})
	c.expectEvent("RTP_READY", 5*time.Second)
	if ev := c.expectEvent("RTP_LOST", 6*time.Second); ev["reason"] != "rtp_timeout" {
		t.Fatalf("RTP_LOST reason %v", ev["reason"])
	}
	if ev := c.expectEvent("HANGUP", 5*time.Second); ev["reason"] != "rtp_timeout" {
		t.Fatalf("HANGUP reason %v", ev["reason"])
	}
	if report := waitReport(t, h.peer, 5*time.Second); report.HangupBy != "bridge" {
		t.Fatalf("bridge must send BYE after RTP loss: %s", report)
	}
}

func pcmRMS(pcm []byte) float64 {
	var sum float64
	n := len(pcm) / 2
	for i := range n {
		v := float64(int16(binary.LittleEndian.Uint16(pcm[i*2:])))
		sum += v * v
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(sum / float64(n))
}
