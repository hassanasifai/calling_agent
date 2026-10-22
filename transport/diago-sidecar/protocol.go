package main

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"net"
	"sync"
)

const maxControlMessageBytes = 2_600_000

type command struct {
	Type       string `json:"type"`
	RequestID  string `json:"request_id"`
	AuthToken  string `json:"auth_token"`
	Command    string `json:"command"`
	CallID     string `json:"call_id,omitempty"`
	GatewayID  string `json:"gateway_id,omitempty"`
	Number     string `json:"number,omitempty"`
	Reason     string `json:"reason,omitempty"`
	SampleRate int    `json:"sample_rate,omitempty"`
	Channels   int    `json:"channels,omitempty"`
	Encoding   string `json:"encoding,omitempty"`
	PCMBase64  string `json:"pcm_b64,omitempty"`
}

type response struct {
	Type       string `json:"type"`
	RequestID  string `json:"request_id"`
	OK         bool   `json:"ok"`
	Error      string `json:"error,omitempty"`
	Registered *bool  `json:"registered,omitempty"`
}

type callEvent struct {
	Type        string `json:"type"`
	CallID      string `json:"call_id"`
	Event       string `json:"event"`
	MonotonicMS int64  `json:"monotonic_ms"`
	Reason      string `json:"reason,omitempty"`
	SIPStatus   int    `json:"sip_status,omitempty"`
	Codec       string `json:"codec,omitempty"`
}

type audioMessage struct {
	Type        string `json:"type"`
	CallID      string `json:"call_id"`
	PCMBase64   string `json:"pcm_b64"`
	SampleRate  int    `json:"sample_rate"`
	Channels    int    `json:"channels"`
	Encoding    string `json:"encoding"`
	MonotonicMS int64  `json:"monotonic_ms"`
}

type clientConn struct {
	conn    net.Conn
	encoder *json.Encoder
	mu      sync.Mutex
}

func newClientConn(conn net.Conn) *clientConn {
	return &clientConn{conn: conn, encoder: json.NewEncoder(conn)}
}

func (c *clientConn) send(value any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.encoder.Encode(value)
}

func tokenMatches(received string, expected string) bool {
	receivedHash := sha256.Sum256([]byte(received))
	expectedHash := sha256.Sum256([]byte(expected))
	return subtle.ConstantTimeCompare(receivedHash[:], expectedHash[:]) == 1
}

func decodeCommand(data []byte) (command, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var request command
	if err := decoder.Decode(&request); err != nil {
		return command{}, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return command{}, errors.New("multiple JSON values")
	}
	return request, nil
}
