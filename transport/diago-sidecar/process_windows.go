//go:build windows

package main

import (
	"log/slog"
	"syscall"
	"unsafe"
)

const (
	processPowerThrottlingClass      = 4 // PROCESS_INFORMATION_CLASS ProcessPowerThrottling
	powerThrottlingCurrentVersion    = 1
	powerThrottlingExecutionSpeed    = 0x1
	powerThrottlingIgnoreTimerResVal = 0x4
	aboveNormalPriorityClass         = 0x00008000
)

type processPowerThrottlingState struct {
	Version     uint32
	ControlMask uint32
	StateMask   uint32
}

// configureProcessForMedia opts the bridge out of Windows 11 power throttling
// (EcoQoS and timer-resolution coalescing for background processes) and raises
// its priority. The bridge is light on CPU but owns the 20 ms RTP clock.
func configureProcessForMedia() {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	process, err := syscall.GetCurrentProcess()
	if err != nil {
		slog.Warn("cannot get process handle for media tuning", "error", err)
		return
	}
	state := processPowerThrottlingState{
		Version:     powerThrottlingCurrentVersion,
		ControlMask: powerThrottlingExecutionSpeed | powerThrottlingIgnoreTimerResVal,
		StateMask:   0, // controlled bits cleared = throttling disabled
	}
	if ok, _, callErr := kernel32.NewProc("SetProcessInformation").Call(
		uintptr(process), processPowerThrottlingClass,
		uintptr(unsafe.Pointer(&state)), unsafe.Sizeof(state),
	); ok == 0 {
		slog.Warn("could not disable power throttling", "error", callErr)
	}
	if ok, _, callErr := kernel32.NewProc("SetPriorityClass").Call(
		uintptr(process), aboveNormalPriorityClass,
	); ok == 0 {
		slog.Warn("could not raise process priority", "error", callErr)
	}
}
