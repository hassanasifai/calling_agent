package main

import "testing"

func TestDecodeCommandRejectsUnknownFields(t *testing.T) {
	_, err := decodeCommand([]byte(`{"type":"command","request_id":"r1","auth_token":"token","command":"health","unexpected":true}`))
	if err == nil {
		t.Fatal("expected unknown field to be rejected")
	}
}

func TestDecodeCommandAcceptsOneStrictObject(t *testing.T) {
	request, err := decodeCommand([]byte(`{"type":"command","request_id":"r1","auth_token":"token","command":"health"}`))
	if err != nil {
		t.Fatalf("decode command: %v", err)
	}
	if request.Command != "health" || request.RequestID != "r1" {
		t.Fatalf("unexpected request: %#v", request)
	}
}

func TestTokenMatches(t *testing.T) {
	if !tokenMatches("correct-token", "correct-token") {
		t.Fatal("equal tokens should match")
	}
	if tokenMatches("wrong-token", "correct-token") {
		t.Fatal("different tokens must not match")
	}
}
