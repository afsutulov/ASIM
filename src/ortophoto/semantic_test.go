package main

import "testing"

func TestSemanticModelIsOptIn(t *testing.T) {
	a := FinalizeSpec(ModelSpec{Name: "buildings"})
	if a.ModelKind != "yolo_seg" || a.IsSemanticModel() {
		t.Fatalf("default model kind = %q, want yolo_seg", a.ModelKind)
	}
	b := FinalizeSpec(ModelSpec{Name: "hogweed", ModelKind: "semantic"})
	if !b.IsSemanticModel() {
		t.Fatalf("semantic model_kind was not enabled")
	}
	if len(b.OutputNames) != 1 || b.OutputNames[0] != "output" {
		t.Fatalf("semantic default outputs = %v, want [output]", b.OutputNames)
	}
}

func TestSemanticMergeWindowCoversRasterAndOverlaps(t *testing.T) {
	for _, tc := range []struct{ w, h, tile, bound int }{
		{20000, 20000, 512, 64},
		{12187, 11693, 512, 64},
		{10000, 10000, 512, 64},
		{400, 350, 512, 64},
	} {
		img := &OrthoImage{W: tc.w, H: tc.h}
		step := tc.tile - 2*tc.bound
		patches := collectPatches(img, tc.tile, step)
		edge := semanticAutoEdgeIgnore(tc.bound)
		coverX := make([]int, tc.w)
		coverY := make([]int, tc.h)
		for _, p := range patches {
			x0, y0, x1, y1 := semanticMergeWindow(img, p, tc.tile, edge)
			for x := x0; x < x1; x++ {
				gx := p.x + x
				if gx >= 0 && gx < tc.w {
					coverX[gx]++
				}
			}
			for y := y0; y < y1; y++ {
				gy := p.y + y
				if gy >= 0 && gy < tc.h {
					coverY[gy]++
				}
			}
		}
		for x, n := range coverX {
			if n == 0 {
				t.Fatalf("w=%d: x=%d uncovered", tc.w, x)
			}
		}
		for y, n := range coverY {
			if n == 0 {
				t.Fatalf("h=%d: y=%d uncovered", tc.h, y)
			}
		}
		if tc.w > tc.tile {
			over := false
			for _, n := range coverX {
				if n > 1 {
					over = true
					break
				}
			}
			if !over {
				t.Fatalf("w=%d: expected semantic overlap", tc.w)
			}
		}
	}
}
