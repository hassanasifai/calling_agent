package main

import (
	"strings"
	"testing"
	"time"
)

func validTestConfig() config {
	return config{
		ControlAddress: "127.0.0.1:8765",
		AuthToken:      "0123456789abcdef",
		SIPBindHost:    "127.0.0.1",
		PBXHost:        "127.0.0.1",
		PBXPort:        5060,
		SIPUser:        "101",
		SIPPassword:    "test-only",
		GatewayID:      "sim-1",
		AllowedNumbers: map[string]struct{}{},
		InviteTimeout:  30 * time.Second,
		RTPTimeout:     15 * time.Second,
	}
}

func TestConfigRequiresLoopbackControl(t *testing.T) {
	cfg := validTestConfig()
	cfg.ControlAddress = "0.0.0.0:8765"
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "loopback") {
		t.Fatalf("expected loopback validation error, got %v", err)
	}
}

func TestConfigRequiresAllowlistWhenLiveCallsEnabled(t *testing.T) {
	cfg := validTestConfig()
	cfg.LiveCalls = true
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "ALLOWLIST") {
		t.Fatalf("expected allowlist validation error, got %v", err)
	}
}

func TestDisabledLiveCallsFailBeforeDialing(t *testing.T) {
	cfg := validTestConfig()
	b := &bridge{cfg: cfg, calls: make(map[string]*callSession)}
	err := b.startCall(t.Context(), nil, command{
		CallID: "call-1", GatewayID: "sim-1", Number: "90000000000",
	})
	if err == nil || err.Error() != "live_calls_disabled" {
		t.Fatalf("expected live_calls_disabled, got %v", err)
	}
}
