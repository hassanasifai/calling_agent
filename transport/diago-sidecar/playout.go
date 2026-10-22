package main

import (
	"errors"
	"sync"
)

var errPlayoutFull = errors.New("audio_queue_full")

// playoutBuffer is a bounded PCM FIFO drained by the RTP media clock. It is
// byte oriented so callers may send chunks that are not RTP frame aligned.
type playoutBuffer struct {
	mu       sync.Mutex
	data     []byte
	maxBytes int
}

func newPlayoutBuffer(maxBytes int) *playoutBuffer {
	return &playoutBuffer{maxBytes: maxBytes}
}

func (p *playoutBuffer) push(pcm []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.data)+len(pcm) > p.maxBytes {
		return errPlayoutFull
	}
	p.data = append(p.data, pcm...)
	return nil
}

// take fills frame with the next queued PCM, zero-padding any shortfall, and
// returns the number of queued bytes consumed (0 means a pure silence frame).
func (p *playoutBuffer) take(frame []byte) int {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := copy(frame, p.data)
	clear(frame[n:])
	if n == len(p.data) {
		p.data = p.data[:0]
	} else {
		p.data = p.data[n:]
	}
	return n
}

// clear discards queued speech, e.g. on caller barge-in, and returns the
// number of bytes dropped.
func (p *playoutBuffer) clear() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	dropped := len(p.data)
	p.data = nil
	return dropped
}

func (p *playoutBuffer) len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.data)
}
