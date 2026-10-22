// Command sip-test-peer runs the offline loopback PBX/callee for end-to-end
// conversation tests. It answers exactly one call, then writes report.json,
// received.wav (bridge → caller) and sent.wav (caller → bridge) and exits.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/emiago/diago/media"

	"polarityiq.com/nrsp/diago-sidecar/testpeer"
)

type scriptFile struct {
	BotSilenceMS  int     `json:"bot_silence_ms"`
	ReplyTimeoutS int     `json:"reply_timeout_s"`
	MaxCallS      int     `json:"max_call_s"`
	FinalAction   string  `json:"final_action"`
	FinalDelayMS  int     `json:"final_delay_ms"`
	SpeechRMS     float64 `json:"speech_rms"`
	Utterances    []struct {
		Name    string `json:"name"`
		WAV     string `json:"wav"`
		Mode    string `json:"mode"`
		DelayMS int    `json:"delay_ms"`
	} `json:"utterances"`
}

func main() {
	bind := flag.String("bind", "127.0.0.1", "SIP bind host")
	port := flag.Int("port", 5070, "SIP bind port")
	user := flag.String("user", "", "expected SIP username")
	password := flag.String("password", "", "expected SIP password")
	codecName := flag.String("codec", "PCMA", "PCMA or PCMU")
	scriptPath := flag.String("script", "", "conversation script JSON")
	outDir := flag.String("out", "", "output directory for report and recordings")
	flag.Parse()
	if err := run(*bind, *port, *user, *password, *codecName, *scriptPath, *outDir); err != nil {
		slog.Error("sip test peer failed", "error", err)
		os.Exit(1)
	}
}

func run(bind string, port int, user, password, codecName, scriptPath, outDir string) error {
	if user == "" || password == "" || scriptPath == "" || outDir == "" {
		return errors.New("-user, -password, -script and -out are required")
	}
	codec := media.CodecAudioAlaw
	if codecName == "PCMU" {
		codec = media.CodecAudioUlaw
	} else if codecName != "PCMA" {
		return fmt.Errorf("unsupported codec %q", codecName)
	}
	script, err := loadScript(scriptPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	peer := testpeer.New(testpeer.Config{
		BindHost: bind, BindPort: port, Username: user, Password: password,
		Realm: "nrsp-test", Codec: codec, Script: script,
	}, slog.Default())
	if err := peer.Start(ctx); err != nil {
		return err
	}
	fmt.Printf("PEER_READY %s:%d\n", bind, port)
	select {
	case <-ctx.Done():
		return errors.New("interrupted before a call completed")
	case report := <-peer.Done():
		if err := writeWAV(filepath.Join(outDir, "received.wav"), report.ReceivedPCM); err != nil {
			return err
		}
		if err := writeWAV(filepath.Join(outDir, "sent.wav"), report.SentPCM); err != nil {
			return err
		}
		data, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(outDir, "report.json"), data, 0o644); err != nil {
			return err
		}
		fmt.Printf("PEER_DONE %s\n", report)
		// Give the BYE transaction time to complete before the UA closes.
		time.Sleep(500 * time.Millisecond)
		return nil
	}
}

func loadScript(path string) (testpeer.Script, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return testpeer.Script{}, err
	}
	var file scriptFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return testpeer.Script{}, err
	}
	script := testpeer.Script{
		BotSilenceMS:   file.BotSilenceMS,
		ReplyTimeout:   time.Duration(file.ReplyTimeoutS) * time.Second,
		MaxCallTimeout: time.Duration(file.MaxCallS) * time.Second,
		FinalAction:    file.FinalAction,
		FinalDelayMS:   file.FinalDelayMS,
		SpeechRMS:      file.SpeechRMS,
	}
	base := filepath.Dir(path)
	for _, u := range file.Utterances {
		wavPath := u.WAV
		if !filepath.IsAbs(wavPath) {
			wavPath = filepath.Join(base, wavPath)
		}
		pcm, err := readWAV8k(wavPath)
		if err != nil {
			return testpeer.Script{}, fmt.Errorf("%s: %w", u.Name, err)
		}
		script.Utterances = append(script.Utterances, testpeer.Utterance{
			Name: u.Name, PCM: pcm, Mode: u.Mode, DelayMS: u.DelayMS,
		})
	}
	return script, nil
}

// readWAV8k accepts only canonical PCM16 mono 8 kHz WAV files.
func readWAV8k(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, errors.New("not a RIFF/WAVE file")
	}
	var fmtOK bool
	for pos := 12; pos+8 <= len(raw); {
		id := string(raw[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(raw[pos+4 : pos+8]))
		body := pos + 8
		if body+size > len(raw) {
			return nil, errors.New("truncated WAV chunk")
		}
		switch id {
		case "fmt ":
			format := binary.LittleEndian.Uint16(raw[body:])
			channels := binary.LittleEndian.Uint16(raw[body+2:])
			rate := binary.LittleEndian.Uint32(raw[body+4:])
			bits := binary.LittleEndian.Uint16(raw[body+14:])
			if format != 1 || channels != 1 || rate != 8000 || bits != 16 {
				return nil, fmt.Errorf("need PCM16 mono 8000 Hz, got format=%d channels=%d rate=%d bits=%d", format, channels, rate, bits)
			}
			fmtOK = true
		case "data":
			if !fmtOK {
				return nil, errors.New("data chunk before fmt chunk")
			}
			return raw[body : body+size], nil
		}
		pos = body + size + size%2
	}
	return nil, errors.New("no data chunk")
}

func writeWAV(path string, pcm []byte) error {
	header := make([]byte, 44)
	copy(header[0:], "RIFF")
	binary.LittleEndian.PutUint32(header[4:], uint32(36+len(pcm)))
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], 8000)
	binary.LittleEndian.PutUint32(header[28:], 16000)
	binary.LittleEndian.PutUint16(header[32:], 2)
	binary.LittleEndian.PutUint16(header[34:], 16)
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], uint32(len(pcm)))
	return os.WriteFile(path, append(header, pcm...), 0o644)
}
