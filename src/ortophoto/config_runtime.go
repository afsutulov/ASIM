package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type PolygonSpec struct {
	Name   string `json:"name"`
	Public int    `json:"public"`
}

type ComputeConfig struct {
	Device string `json:"device"`
	GPUs   []int  `json:"gpus"`
}

type RuntimeDeviceConfig struct {
	Device   string
	GPUs     []int
	DeviceID int
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
	Compute        RuntimeDeviceConfig
	Models         map[string]ModelSpec
	Poligons       map[string]PolygonSpec
}

type rawPaths struct {
	Logs           string        `json:"logs"`
	Tmp            string        `json:"tmp"`
	PoliPath       string        `json:"poli_path"`
	PoligonsFile   string        `json:"poligons"`
	ResultsPath    string        `json:"results_path"`
	ModelsPath     string        `json:"models_path"`
	Ortophoto      string        `json:"ortophoto"`
	ORTLibraryPath string        `json:"onnxruntime_library"`
	ORTDevice      string        `json:"ort_device"`
	ORTDeviceID    int           `json:"ort_device_id"`
	ORTDeviceIDs   []int         `json:"ort_device_ids"`
	Compute        ComputeConfig `json:"compute"`
}

func DefaultConfigPath() string { return filepath.Join("data", "config.json") }

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

func resolveRuntimeDevice(compute ComputeConfig, legacyDevice string, legacyDeviceIDs []int, legacyDeviceID int) RuntimeDeviceConfig {
	device := normalizeRuntimeDevice(compute.Device)
	if device == "" {
		device = normalizeRuntimeDevice(legacyDevice)
	}
	if device == "" {
		device = "cuda"
	}

	gpus := normalizeGPUList(compute.GPUs)
	if len(compute.GPUs) == 0 {
		if len(legacyDeviceIDs) > 0 {
			gpus = normalizeGPUList(legacyDeviceIDs)
		} else if legacyDeviceID >= 0 {
			gpus = normalizeGPUList([]int{legacyDeviceID})
		}
	}
	if device == "cpu" {
		gpus = nil
	}
	deviceID := 0
	if len(gpus) > 0 {
		deviceID = gpus[0]
	}
	return RuntimeDeviceConfig{Device: device, GPUs: gpus, DeviceID: deviceID}
}

func validateModelSpec(spec ModelSpec) error {
	for _, c := range spec.DetectionClasses {
		if c < 0 || c >= spec.NumClasses {
			return fmt.Errorf("model %q: class %d is outside 0..%d", spec.Name, c, spec.NumClasses-1)
		}
	}
	for raw, threshold := range spec.ClassThresholds {
		classID, err := strconv.Atoi(raw)
		if err != nil || classID < 0 || classID >= spec.NumClasses {
			return fmt.Errorf("model %q: invalid class_thresholds key %q", spec.Name, raw)
		}
		if threshold <= 0 || threshold > 1 {
			return fmt.Errorf("model %q: class_thresholds[%q]=%v must be in (0,1]", spec.Name, raw, threshold)
		}
	}
	if spec.ExclusiveClasses && spec.NumClasses < 2 {
		// Допустимо, но бессмысленно; не считаем ошибкой.
	}
	for i, rule := range spec.ClassMergeRules {
		if rule.TargetClass < 0 || rule.TargetClass >= spec.NumClasses {
			return fmt.Errorf("model %q: class_merge_rules[%d].target_class=%d is outside 0..%d", spec.Name, i, rule.TargetClass, spec.NumClasses-1)
		}
		if len(rule.SourceClasses) == 0 {
			return fmt.Errorf("model %q: class_merge_rules[%d].source_classes is empty", spec.Name, i)
		}
		for _, classID := range rule.SourceClasses {
			if classID < 0 || classID >= spec.NumClasses || classID == rule.TargetClass {
				return fmt.Errorf("model %q: invalid source class %d in class_merge_rules[%d]", spec.Name, classID, i)
			}
		}
		if rule.DistanceM < 0 {
			return fmt.Errorf("model %q: class_merge_rules[%d].distance_m must be >= 0", spec.Name, i)
		}
	}
	return nil
}

// validateSharedONNXSettings предотвращает ситуацию, когда один ONNX-файл
// запускается как несколько логических задач с различным препроцессингом или
// порогами. В таком случае независимые запуски могли бы построить разные карты
// классов и снова дать пересечения. Проверка полностью универсальна и не знает
// имён моделей или классов.
func validateSharedONNXSettings(models map[string]ModelSpec) error {
	groups := make(map[string][]ModelSpec)
	for _, spec := range models {
		if spec.ExclusiveClasses {
			groups[spec.ONNXFile] = append(groups[spec.ONNXFile], spec)
		}
	}
	for onnx, specs := range groups {
		if len(specs) < 2 {
			continue
		}
		sort.Slice(specs, func(i, j int) bool { return specs[i].Name < specs[j].Name })
		base := specs[0]
		baseThresholds, _ := json.Marshal(base.ClassThresholds)
		baseOutputs, _ := json.Marshal(base.OutputNames)
		baseMergeRules, _ := json.Marshal(base.ClassMergeRules)
		for _, cur := range specs[1:] {
			curThresholds, _ := json.Marshal(cur.ClassThresholds)
			curOutputs, _ := json.Marshal(cur.OutputNames)
			curMergeRules, _ := json.Marshal(cur.ClassMergeRules)
			if cur.NumClasses != base.NumClasses || cur.Tile != base.Tile || cur.Bound != base.Bound ||
				cur.Normalization != base.Normalization || cur.Threshold != base.Threshold ||
				cur.NMSThreshold != base.NMSThreshold || cur.AgnosticNMS != base.AgnosticNMS ||
				cur.MaskThreshold != base.MaskThreshold || cur.YoloMaskDim != base.YoloMaskDim ||
				cur.YoloProtoStride != base.YoloProtoStride || cur.InputName != base.InputName ||
				string(curOutputs) != string(baseOutputs) || string(curThresholds) != string(baseThresholds) ||
				cur.RejectBoxTileRatio != base.RejectBoxTileRatio ||
				cur.RejectEdgeOversize != base.RejectEdgeOversize || cur.RejectEdgePinPx != base.RejectEdgePinPx ||
				cur.MaskMode != base.MaskMode || cur.MaskBBoxMargin != base.MaskBBoxMargin ||
				cur.MaxMasksPerTile != base.MaxMasksPerTile ||
				cur.SkipEmptyTiles != base.SkipEmptyTiles || cur.MinContentPixels != base.MinContentPixels ||
				string(curMergeRules) != string(baseMergeRules) {
				return fmt.Errorf("exclusive models %q and %q using %q have different inference settings", base.Name, cur.Name, onnx)
			}
		}
	}
	return nil
}

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
		Compute:        resolveRuntimeDevice(rp.Compute, rp.ORTDevice, rp.ORTDeviceIDs, rp.ORTDeviceID),
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
		if err := validateModelSpec(spec); err != nil {
			return nil, err
		}
		cfg.Models[key] = spec
	}
	if err := validateSharedONNXSettings(cfg.Models); err != nil {
		return nil, err
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
