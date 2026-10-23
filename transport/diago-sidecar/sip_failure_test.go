package main

import (
	"context"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
)

// A gateway that rejects the INVITE must surface as sip_<code> with the
// status, not as transport_error (first Jazz call, 2026-10-23).
func TestBridgeReportsSIPRejectionStatus(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pbxPort := reserveUDPPort(t)
	ua, err := sipgo.NewUA()
	if err != nil {
		t.Fatal(err)
	}
	defer ua.Close()
	srv, err := sipgo.NewServer(ua)
	if err != nil {
		t.Fatal(err)
	}
	srv.OnRegister(func(req *sip.Request, tx sip.ServerTransaction) {
		_ = tx.Respond(sip.NewResponseFromRequest(req, sip.StatusOK, "OK", nil))
	})
	srv.OnInvite(func(req *sip.Request, tx sip.ServerTransaction) {
		_ = tx.Respond(sip.NewResponseFromRequest(req, sip.StatusTrying, "Trying", nil))
		res := sip.NewResponseFromRequest(req, sip.StatusTemporarilyUnavailable, "Temporarily Unavailable", nil)
		res.AppendHeader(sip.NewHeader("Reason", "Q.850;cause=18"))
		_ = tx.Respond(res)
	})
	srv.OnAck(func(*sip.Request, sip.ServerTransaction) {})
	go func() {
		_ = srv.ListenAndServe(ctx, "udp", net.JoinHostPort("127.0.0.1", strconv.Itoa(pbxPort)))
	}()

	cfg := config{
		ControlAddress: reserveTCPAddress(t), AuthToken: testToken,
		SIPBindHost: "127.0.0.1", SIPBindPort: reserveUDPPort(t),
		PBXHost: "127.0.0.1", PBXPort: pbxPort, SIPUser: "100", SIPPassword: "test-secret",
		GatewayID: "sim-1", LiveCalls: true, AllowedNumbers: map[string]struct{}{"9000": {}},
		InviteTimeout: 10 * time.Second, RTPTimeout: 5 * time.Second,
	}
	b, err := newBridge(cfg)
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = b.run(ctx) }()
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
	client := dialControl(t, cfg.ControlAddress)
	for deadline := time.Now().Add(5 * time.Second); ; {
		if r := client.command("health", nil); r["registered"] == true {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("bridge never registered")
		}
		time.Sleep(50 * time.Millisecond)
	}

	r := client.command("start_call", map[string]any{"call_id": "c1", "gateway_id": "sim-1", "number": "9000"})
	if r["ok"] != true {
		t.Fatalf("start_call: %v", r)
	}
	failed := client.expectEvent("FAILED", 5*time.Second)
	if failed["reason"] != "sip_480" || failed["sip_status"] != float64(480) {
		t.Fatalf("FAILED event = %v, want reason sip_480 and sip_status 480", failed)
	}
}
