package main

import (
	"fmt"
	"strconv"
	"strings"
)

type ComputeConfig struct {
	Device string `json:"device"`
	GPUs   []int  `json:"gpus"`
}

type RuntimeDeviceConfig struct {
	Device string
	GPUs   []int
}

func normalizeRuntimeDevice(device string) string {
	d := strings.ToLower(strings.TrimSpace(device))
	switch d {
	case "", "gpu", "cuda":
		return "cuda"
	case "cpu":
		return "cpu"
	default:
		return d
	}
}

func normalizeGPUList(ids []int) []int {
	if len(ids) == 0 {
		return []int{0}
	}
	out := make([]int, 0, len(ids))
	seen := make(map[int]struct{}, len(ids))
	for _, id := range ids {
		if id < 0 {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	if len(out) == 0 {
		return []int{0}
	}
	return out
}

func parseGPUList(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid gpu id %q: %w", part, err)
		}
		out = append(out, id)
	}
	return normalizeGPUList(out), nil
}

func resolveRuntimeDevice(compute ComputeConfig, overrideDevice string, overrideGPUList []int, legacyDevice string, legacyDeviceID int) RuntimeDeviceConfig {
	device := normalizeRuntimeDevice(compute.Device)
	if device == "" {
		device = normalizeRuntimeDevice(legacyDevice)
	}
	if device == "" {
		device = "cuda"
	}
	if od := normalizeRuntimeDevice(overrideDevice); od != "" {
		device = od
	}

	gpus := normalizeGPUList(compute.GPUs)
	if len(compute.GPUs) == 0 {
		if legacyDeviceID >= 0 {
			gpus = normalizeGPUList([]int{legacyDeviceID})
		}
	}
	if len(overrideGPUList) > 0 {
		gpus = normalizeGPUList(overrideGPUList)
	}
	if device == "cpu" {
		gpus = nil
	}
	return RuntimeDeviceConfig{Device: device, GPUs: gpus}
}
