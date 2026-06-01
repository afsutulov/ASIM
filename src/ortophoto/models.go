package main

type ModelSpec struct {
	Name                string   `json:"name"`
	Description         string   `json:"description"`
	ONNXFile            string   `json:"onnx_file"`
	Channels            []string `json:"channels"`
	Tile                int      `json:"tile"`
	Bound               int      `json:"bound"`
	Overlap             int      `json:"overlap,omitempty"`
	Threshold           float32  `json:"threshold"`
	BoundaryThreshold   float32  `json:"boundary_threshold,omitempty"`
	BatchSize           int      `json:"batch_size,omitempty"`
	Preprocess          string   `json:"preprocess"`
	Inputs              int      `json:"inputs,omitempty"`
	Simplify            float64  `json:"simplify,omitempty"`
	MinArea             float64  `json:"min_area,omitempty"`
	Merge               bool     `json:"merge,omitempty"`
	InputName           string   `json:"input_name,omitempty"`
	OutputName          string   `json:"output_name,omitempty"`
	EmptyTileValue      float64  `json:"empty_tile_value,omitempty"`
	EmptyTileWhiteRatio float64  `json:"empty_tile_white_ratio,omitempty"`
	EmptyTileValidRatio float64  `json:"empty_tile_valid_ratio,omitempty"`
	BoundaryAware       bool     `json:"boundary_aware,omitempty"`
	MorphOpen           int      `json:"morph_open,omitempty"`
	MorphClose          int      `json:"morph_close,omitempty"`
	ResultMode          string   `json:"result_mode,omitempty"`
}

func FinalizeSpec(m ModelSpec) ModelSpec {
	if m.ONNXFile == "" && m.Name != "" {
		m.ONNXFile = m.Name + ".onnx"
	}
	if m.Tile <= 0 {
		m.Tile = 512
	}
	if m.Bound < 0 {
		m.Bound = 0
	}
	if m.Overlap <= 0 {
		m.Overlap = m.Bound * 2
	}
	if m.Threshold == 0 {
		m.Threshold = 0.5
	}
	if m.BoundaryThreshold == 0 {
		m.BoundaryThreshold = 0.35
	}
	if m.BatchSize <= 0 {
		m.BatchSize = 4
	}
	if m.Preprocess == "" {
		m.Preprocess = "ortophoto"
	}
	if m.Inputs <= 0 {
		m.Inputs = 1
	}
	if m.InputName == "" {
		m.InputName = "input"
	}
	if m.OutputName == "" {
		m.OutputName = "logits"
	}
	if m.EmptyTileValue == 0 {
		m.EmptyTileValue = 0.98
	}
	if m.EmptyTileWhiteRatio == 0 {
		m.EmptyTileWhiteRatio = 0.90
	}
	if m.EmptyTileValidRatio == 0 {
		m.EmptyTileValidRatio = 0.05
	}
	if !m.BoundaryAware {
		m.BoundaryAware = true
	}
	if m.MorphOpen < 0 {
		m.MorphOpen = 0
	}
	if m.MorphClose < 0 {
		m.MorphClose = 0
	}
	if m.MinArea <= 0 {
		m.MinArea = 25
	}
	if m.ResultMode == "" {
		m.ResultMode = "polygon"
	}
	return m
}
