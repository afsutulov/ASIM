package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type PolygonSpec struct {
	Name   string `json:"name"`
	Public int    `json:"public"`
}

type AppConfig struct {
	ModelsPath     string
	Logs           string
	Tmp            string
	PoliPath       string
	PoligonsFile   string
	ResultsPath    string
	Ortophoto      string
	ORTLibraryPath string
	ORTDevice      string
	ORTDeviceID    int
	ORTDeviceIDs   []int
	Models         map[string]ModelSpec
	Poligons       map[string]PolygonSpec
}

type rawPaths struct {
	Logs           string `json:"logs"`
	Tmp            string `json:"tmp"`
	PoliPath       string `json:"poli_path"`
	PoligonsFile   string `json:"poligons"`
	ResultsPath    string `json:"results_path"`
	ModelsPath     string `json:"models_path"`
	Ortophoto      string `json:"ortophoto"`
	ORTLibraryPath string `json:"onnxruntime_library"`
	ORTDevice      string `json:"ort_device"`
	ORTDeviceID    int    `json:"ort_device_id"`
	ORTDeviceIDs   []int  `json:"ort_device_ids"`
	Compute        struct {
		Device string `json:"device"`
		GPUs   []int  `json:"gpus"`
	} `json:"compute"`
}

func DefaultConfigPath() string { return filepath.Join("data", "config.json") }

func LoadAppConfig(path string) (*AppConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rp rawPaths
	if err := json.Unmarshal(data, &rp); err != nil {
		return nil, err
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		return nil, err
	}
	var models map[string]ModelSpec
	if blob, ok := top["models"]; ok {
		if err := json.Unmarshal(blob, &models); err != nil {
			return nil, fmt.Errorf("parse models: %w", err)
		}
	}
	cfg := &AppConfig{
		ModelsPath:     rp.ModelsPath,
		Logs:           rp.Logs,
		Tmp:            rp.Tmp,
		PoliPath:       rp.PoliPath,
		PoligonsFile:   rp.PoligonsFile,
		ResultsPath:    rp.ResultsPath,
		Ortophoto:      rp.Ortophoto,
		ORTLibraryPath: rp.ORTLibraryPath,
		ORTDevice:      rp.ORTDevice,
		ORTDeviceID:    rp.ORTDeviceID,
		ORTDeviceIDs:   rp.ORTDeviceIDs,
		Models:         make(map[string]ModelSpec),
		Poligons:       make(map[string]PolygonSpec),
	}
	if strings.TrimSpace(cfg.ModelsPath) == "" {
		return nil, fmt.Errorf("models_path is empty")
	}
	if strings.TrimSpace(cfg.ResultsPath) == "" {
		return nil, fmt.Errorf("results_path is empty")
	}
	if strings.TrimSpace(cfg.PoligonsFile) == "" {
		return nil, fmt.Errorf("poligons is empty")
	}
	if strings.TrimSpace(cfg.Ortophoto) == "" {
		return nil, fmt.Errorf("ortophoto is empty")
	}
	if strings.TrimSpace(rp.Compute.Device) != "" {
		d := strings.ToLower(strings.TrimSpace(rp.Compute.Device))
		if d == "gpu" {
			d = "cuda"
		}
		cfg.ORTDevice = d
	}
	if len(rp.Compute.GPUs) > 0 {
		cfg.ORTDeviceIDs = append([]int(nil), rp.Compute.GPUs...)
	}
	if strings.TrimSpace(cfg.ORTDevice) == "" {
		cfg.ORTDevice = "cuda"
	}
	if strings.ToLower(strings.TrimSpace(cfg.ORTDevice)) == "gpu" {
		cfg.ORTDevice = "cuda"
	}
	if len(cfg.ORTDeviceIDs) == 0 {
		if cfg.ORTDeviceID != 0 {
			cfg.ORTDeviceIDs = []int{cfg.ORTDeviceID}
		} else {
			cfg.ORTDeviceIDs = []int{0}
		}
	}
	if len(cfg.ORTDeviceIDs) > 0 {
		cfg.ORTDeviceID = cfg.ORTDeviceIDs[0]
	}
	polyData, err := os.ReadFile(cfg.PoligonsFile)
	if err != nil {
		return nil, fmt.Errorf("read poligons json: %w", err)
	}
	if err := json.Unmarshal(polyData, &cfg.Poligons); err != nil {
		return nil, fmt.Errorf("parse poligons json: %w", err)
	}
	for key, spec := range models {
		if spec.Name == "" {
			spec.Name = key
		}
		spec = FinalizeSpec(spec)
		if strings.ToLower(strings.TrimSpace(spec.Preprocess)) != "ortophoto" {
			continue
		}
		cfg.Models[key] = spec
	}
	return cfg, nil
}

func (cfg *AppConfig) PolygonPath(name string) (string, error) {
	if _, ok := cfg.Poligons[name]; !ok {
		return "", fmt.Errorf("polygon %q not found in poligons.json", name)
	}
	if strings.TrimSpace(cfg.PoliPath) == "" {
		return "", fmt.Errorf("poli_path is empty")
	}
	return filepath.Join(cfg.PoliPath, safeName(name)+".geojson"), nil
}

func (cfg *AppConfig) ModelPath(spec ModelSpec) string {
	if filepath.IsAbs(spec.ONNXFile) || strings.TrimSpace(cfg.ModelsPath) == "" {
		return spec.ONNXFile
	}
	return filepath.Join(cfg.ModelsPath, safeName(spec.ONNXFile))
}
