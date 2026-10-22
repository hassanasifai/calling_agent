package main

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

type config struct {
	ControlAddress string
	AuthToken      string
	SIPBindHost    string
	SIPBindPort    int
	PBXHost        string
	PBXPort        int
	SIPUser        string
	SIPPassword    string
	GatewayID      string
	LiveCalls      bool
	AllowedNumbers map[string]struct{}
	InviteTimeout  time.Duration
	RTPTimeout     time.Duration
}

func loadConfig() (config, error) {
	cfg := config{
		ControlAddress: envOr("NRSP_BRIDGE_CONTROL_ADDRESS", "127.0.0.1:8765"),
		AuthToken:      os.Getenv("NRSP_BRIDGE_AUTH_TOKEN"),
		SIPBindHost:    envOr("NRSP_SIP_BIND_HOST", "0.0.0.0"),
		PBXHost:        os.Getenv("NRSP_SIP_PBX_HOST"),
		SIPUser:        os.Getenv("NRSP_SIP_USERNAME"),
		SIPPassword:    os.Getenv("NRSP_SIP_PASSWORD"),
		GatewayID:      envOr("NRSP_GATEWAY_ID", "sim-1"),
		LiveCalls:      strings.EqualFold(os.Getenv("NRSP_LIVE_CALLS_ENABLED"), "true"),
		AllowedNumbers: splitSet(os.Getenv("NRSP_LIVE_CALL_ALLOWLIST")),
		InviteTimeout:  90 * time.Second,
	}
	var err error
	if cfg.SIPBindPort, err = envInt("NRSP_SIP_BIND_PORT", 0); err != nil {
		return config{}, err
	}
	if cfg.PBXPort, err = envInt("NRSP_SIP_PBX_PORT", 5060); err != nil {
		return config{}, err
	}
	if seconds, parseErr := envInt("NRSP_SIP_INVITE_TIMEOUT_SECONDS", 90); parseErr != nil {
		return config{}, parseErr
	} else {
		cfg.InviteTimeout = time.Duration(seconds) * time.Second
	}
	if seconds, parseErr := envInt("NRSP_RTP_TIMEOUT_SECONDS", 15); parseErr != nil {
		return config{}, parseErr
	} else {
		cfg.RTPTimeout = time.Duration(seconds) * time.Second
	}
	if err := cfg.validate(); err != nil {
		return config{}, err
	}
	return cfg, nil
}

func (c config) validate() error {
	host, _, err := net.SplitHostPort(c.ControlAddress)
	if err != nil {
		return fmt.Errorf("NRSP_BRIDGE_CONTROL_ADDRESS must be host:port: %w", err)
	}
	ip := net.ParseIP(host)
	if !strings.EqualFold(host, "localhost") && (ip == nil || !ip.IsLoopback()) {
		return fmt.Errorf("NRSP_BRIDGE_CONTROL_ADDRESS must use a loopback host")
	}
	if len(c.AuthToken) < 16 {
		return fmt.Errorf("NRSP_BRIDGE_AUTH_TOKEN must contain at least 16 characters")
	}
	if c.PBXHost == "" || c.SIPUser == "" || c.SIPPassword == "" {
		return fmt.Errorf("NRSP_SIP_PBX_HOST, NRSP_SIP_USERNAME, and NRSP_SIP_PASSWORD are required")
	}
	if c.SIPBindPort < 0 || c.SIPBindPort > 65535 || c.PBXPort < 1 || c.PBXPort > 65535 {
		return fmt.Errorf("SIP ports must be valid TCP/UDP port numbers")
	}
	if c.InviteTimeout < time.Second || c.InviteTimeout > 10*time.Minute {
		return fmt.Errorf("NRSP_SIP_INVITE_TIMEOUT_SECONDS must be between 1 and 600")
	}
	if c.RTPTimeout < 2*time.Second || c.RTPTimeout > 5*time.Minute {
		return fmt.Errorf("NRSP_RTP_TIMEOUT_SECONDS must be between 2 and 300")
	}
	if c.LiveCalls && len(c.AllowedNumbers) == 0 {
		return fmt.Errorf("NRSP_LIVE_CALL_ALLOWLIST is required when live calls are enabled")
	}
	return nil
}

func envOr(name string, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envInt(name string, fallback int) (int, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer: %w", name, err)
	}
	return parsed, nil
}

func splitSet(value string) map[string]struct{} {
	result := make(map[string]struct{})
	for _, item := range strings.FieldsFunc(value, func(r rune) bool { return r == ',' || r == ';' }) {
		if trimmed := strings.TrimSpace(item); trimmed != "" {
			result[trimmed] = struct{}{}
		}
	}
	return result
}
