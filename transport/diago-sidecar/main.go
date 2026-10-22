package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
)

func main() {
	cfg, err := loadConfig()
	if err != nil {
		slog.Error("invalid configuration", "error", err)
		os.Exit(2)
	}
	mediaBridge, err := newBridge(cfg)
	if err != nil {
		slog.Error("failed to initialize media bridge", "error", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := mediaBridge.run(ctx); err != nil {
		slog.Error("media bridge stopped", "error", err)
		os.Exit(1)
	}
}
