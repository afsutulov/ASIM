package main

import (
	"encoding/json"
	"testing"
)

func specFromJSON(t *testing.T, blob string) ModelSpec {
	t.Helper()
	var m ModelSpec
	if err := json.Unmarshal([]byte(blob), &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return FinalizeSpec(m)
}

func TestFinalizeSpecBound(t *testing.T) {
	// bound не задан -> значение по умолчанию 128
	m := specFromJSON(t, `{"name":"a","num_classes":1}`)
	if m.Bound != 128 {
		t.Fatalf("default bound = %d, want 128", m.Bound)
	}
	// явный bound: 0 теперь уважается (раньше молча заменялся на 128)
	m = specFromJSON(t, `{"name":"a","num_classes":1,"bound":0}`)
	if m.Bound != 0 {
		t.Fatalf("explicit bound=0 became %d, want 0", m.Bound)
	}
	m = specFromJSON(t, `{"name":"a","num_classes":1,"bound":64}`)
	if m.Bound != 64 {
		t.Fatalf("bound=64 became %d", m.Bound)
	}
}

func TestFinalizeSpecMinAreaM2Fallback(t *testing.T) {
	// min_area_m2 не задан -> обратная совместимость: берётся min_area
	m := specFromJSON(t, `{"name":"a","num_classes":1,"min_area":10}`)
	if m.MinAreaM2 != 10 {
		t.Fatalf("MinAreaM2 fallback = %v, want 10", m.MinAreaM2)
	}
	// оба заданы -> независимые значения
	m = specFromJSON(t, `{"name":"a","num_classes":1,"min_area":10,"min_area_m2":25}`)
	if m.MinArea != 10 || m.MinAreaM2 != 25 {
		t.Fatalf("MinArea=%v MinAreaM2=%v, want 10/25", m.MinArea, m.MinAreaM2)
	}
}

func TestFinalizeSpecNormalization(t *testing.T) {
	cases := map[string]string{
		``:         "minmax",
		`"minmax"`: "minmax",
		`"FIXED"`:  "fixed",
		`"none"`:   "none",
		`"weird"`:  "minmax",
	}
	for in, want := range cases {
		blob := `{"name":"a","num_classes":1`
		if in != `` {
			blob += `,"normalization":` + in
		}
		blob += `}`
		m := specFromJSON(t, blob)
		if m.Normalization != want {
			t.Fatalf("normalization %s -> %q, want %q", in, m.Normalization, want)
		}
	}
}
