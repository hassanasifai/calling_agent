package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"testing"
	"time"

	"github.com/emiago/diago"
	"github.com/emiago/diago/media"
	"github.com/emiago/sipgo"
	"github.com/emiago/sipgo/sip"
)

func TestDiagoLoopbackTwoWayPCM(t *testing.T) {
	for _, codec := range []media.Codec{media.CodecAudioAlaw, media.CodecAudioUlaw} {
		t.Run(codec.Name, func(t *testing.T) {
			testDiagoLoopbackTwoWayPCM(t, codec)
		})
	}
}

func testDiagoLoopbackTwoWayPCM(t *testing.T, codec media.Codec) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	serverPort := reserveUDPPort(t)
	clientPort := reserveUDPPort(t)

	serverUA, err := sipgo.NewUA(sipgo.WithUserAgent("loopback-callee"))
	if err != nil {
		t.Fatalf("create server UA: %v", err)
	}
	defer serverUA.Close()
	server := diago.NewDiago(
		serverUA,
		diago.WithTransport(diago.Transport{
			Transport: "udp", BindHost: "127.0.0.1", BindPort: serverPort,
		}),
		diago.WithMediaConfig(diago.MediaConfig{Codecs: []media.Codec{codec}}),
	)

	remotePCM := bytes.Repeat([]byte{0x34, 0x12}, 160)
	replyPCM := bytes.Repeat([]byte{0x78, 0x06}, 160)
	serverReceived := make(chan []byte, 1)
	serverErr := make(chan error, 1)
	if err := server.ServeBackground(ctx, func(dialog *diago.DialogServerSession) {
		if err := dialog.Ringing(); err != nil {
			serverErr <- fmt.Errorf("ringing: %w", err)
			return
		}
		med, err := dialog.Answer(diago.AnswerOptions{Codecs: []media.Codec{codec}})
		if err != nil {
			serverErr <- fmt.Errorf("answer: %w", err)
			return
		}
		defer med.Close()
		reader, writer, _, err := setupPCM(med, func(rune) {})
		if err != nil {
			serverErr <- fmt.Errorf("setup server PCM: %w", err)
			return
		}
		frame := make([]byte, len(remotePCM))
		if _, err := io.ReadFull(reader, frame); err != nil {
			serverErr <- fmt.Errorf("read caller PCM: %w", err)
			return
		}
		serverReceived <- frame
		if _, err := writer.Write(replyPCM); err != nil {
			serverErr <- fmt.Errorf("write reply PCM: %w", err)
			return
		}
		<-dialog.Context().Done()
	}); err != nil {
		t.Fatalf("start server: %v", err)
	}

	clientUA, err := sipgo.NewUA(sipgo.WithUserAgent("loopback-caller"))
	if err != nil {
		t.Fatalf("create client UA: %v", err)
	}
	defer clientUA.Close()
	client := diago.NewDiago(
		clientUA,
		diago.WithTransport(diago.Transport{
			Transport: "udp", BindHost: "127.0.0.1", BindPort: clientPort,
		}),
		diago.WithMediaConfig(diago.MediaConfig{Codecs: []media.Codec{codec}}),
	)
	if err := client.ServeBackground(ctx, func(dialog *diago.DialogServerSession) {
		_ = dialog.Hangup(dialog.Context())
	}); err != nil {
		t.Fatalf("start client: %v", err)
	}

	dialog, med, err := client.Invite(ctx, sip.Uri{
		Scheme: "sip", User: "echo", Host: "127.0.0.1", Port: serverPort,
	}, diago.InviteOptions{})
	if err != nil {
		t.Fatalf("invite loopback peer: %v", err)
	}
	defer dialog.Close()
	defer med.Close()
	reader, writer, negotiated, err := setupPCM(med, func(rune) {})
	if err != nil {
		t.Fatalf("setup client PCM: %v", err)
	}
	if negotiated.PayloadType != codec.PayloadType {
		t.Fatalf("negotiated %s, expected %s", negotiated.Name, codec.Name)
	}
	if _, err := writer.Write(remotePCM); err != nil {
		t.Fatalf("write caller PCM: %v", err)
	}
	receivedReply := make([]byte, len(replyPCM))
	if _, err := io.ReadFull(reader, receivedReply); err != nil {
		t.Fatalf("read reply PCM: %v", err)
	}

	select {
	case err := <-serverErr:
		t.Fatal(err)
	case got := <-serverReceived:
		assertPCMNear(t, got, remotePCM)
	case <-ctx.Done():
		t.Fatal("timed out waiting for server PCM")
	}
	assertPCMNear(t, receivedReply, replyPCM)

	hangupCtx, hangupCancel := context.WithTimeout(context.Background(), time.Second)
	defer hangupCancel()
	if err := dialog.Hangup(hangupCtx); err != nil {
		t.Fatalf("hangup loopback call: %v", err)
	}
}

func reserveUDPPort(t *testing.T) int {
	t.Helper()
	conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve UDP port: %v", err)
	}
	port := conn.LocalAddr().(*net.UDPAddr).Port
	if err := conn.Close(); err != nil {
		t.Fatalf("release UDP port: %v", err)
	}
	return port
}

func assertPCMNear(t *testing.T, got []byte, want []byte) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("PCM length %d, expected %d", len(got), len(want))
	}
	for index := 0; index < len(got); index += 2 {
		gotSample := int16(uint16(got[index]) | uint16(got[index+1])<<8)
		wantSample := int16(uint16(want[index]) | uint16(want[index+1])<<8)
		delta := int(gotSample) - int(wantSample)
		if delta < 0 {
			delta = -delta
		}
		if delta > 512 {
			t.Fatalf("PCM sample %d delta %d exceeds G.711 tolerance", index/2, delta)
		}
	}
}
