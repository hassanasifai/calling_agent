package main

import (
	"errors"
	"fmt"
	"testing"

	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
)

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

func TestClassifyErrorReportsSIPStatusForBothErrorForms(t *testing.T) {
	res := sip.NewResponse(486, "Busy Here")
	for _, err := range []error{
		&sipgo.ErrDialogResponse{Res: res},
		errors.Join(sipgo.ErrDialogResponse{Res: res}, nil),
		fmt.Errorf("invite: %w", &sipgo.ErrDialogResponse{Res: res}),
	} {
		if got := classifyError(err); got != "sip_486" {
			t.Fatalf("classifyError(%v) = %q, want sip_486", err, got)
		}
	}
	if got := classifyError(errors.New("transaction terminated")); got != "transport_error" {
		t.Fatalf("got %q", got)
	}
	if got := redact("auth failed for s3cret-pass", "s3cret-pass"); got != "auth failed for ***" {
		t.Fatalf("redact = %q", got)
	}
	if got := maskNumber("03001234567"); got != "*******4567" {
		t.Fatalf("maskNumber = %q", got)
	}
}
