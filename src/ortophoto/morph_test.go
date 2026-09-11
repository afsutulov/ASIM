package main

import (
	"math/rand"
	"testing"
)

// naiveDilate / naiveErode — прежняя (до разделимой реализации) версия
// морфологии: полный перебор квадратной окрестности с обрезкой по границам
// растра. Используется как эталон.
func naiveDilate(src []uint8, w, h, radius int) []uint8 {
	if radius <= 0 {
		out := make([]uint8, len(src))
		copy(out, src)
		return out
	}
	dst := make([]uint8, len(src))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			val := uint8(0)
			for yy := max(0, y-radius); yy <= min(h-1, y+radius) && val == 0; yy++ {
				for xx := max(0, x-radius); xx <= min(w-1, x+radius); xx++ {
					if src[yy*w+xx] != 0 {
						val = 1
						break
					}
				}
			}
			dst[y*w+x] = val
		}
	}
	return dst
}

func naiveErode(src []uint8, w, h, radius int) []uint8 {
	if radius <= 0 {
		out := make([]uint8, len(src))
		copy(out, src)
		return out
	}
	dst := make([]uint8, len(src))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			val := uint8(1)
			for yy := max(0, y-radius); yy <= min(h-1, y+radius) && val == 1; yy++ {
				for xx := max(0, x-radius); xx <= min(w-1, x+radius); xx++ {
					if src[yy*w+xx] == 0 {
						val = 0
						break
					}
				}
			}
			dst[y*w+x] = val
		}
	}
	return dst
}

// TestSeparableMorphMatchesNaive — разделимая морфология обязана давать
// побитово тот же результат, что прежняя наивная: квадратный структурный
// элемент разделим точно, включая обрезку окна на границах растра.
func TestSeparableMorphMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(20260909))
	for _, dims := range [][2]int{{1, 1}, {7, 3}, {31, 29}, {64, 17}, {97, 101}} {
		w, h := dims[0], dims[1]
		for _, density := range []float64{0.02, 0.3, 0.9} {
			src := make([]uint8, w*h)
			for i := range src {
				if rng.Float64() < density {
					src[i] = 1
				}
			}
			for _, r := range []int{0, 1, 2, 5, 13, 40} {
				gotD := dilateMask(src, w, h, r)
				wantD := naiveDilate(src, w, h, r)
				for i := range wantD {
					if gotD[i] != wantD[i] {
						t.Fatalf("dilate %dx%d r=%d d=%.2f: pixel %d got %d want %d", w, h, r, density, i, gotD[i], wantD[i])
					}
				}
				gotE := erodeMask(src, w, h, r)
				wantE := naiveErode(src, w, h, r)
				for i := range wantE {
					if gotE[i] != wantE[i] {
						t.Fatalf("erode %dx%d r=%d d=%.2f: pixel %d got %d want %d", w, h, r, density, i, gotE[i], wantE[i])
					}
				}
			}
		}
	}
}

// TestRejectBoxTileRatioIsOptIn — безусловный предел размера бокса выключен
// по умолчанию и не затрагивает модели зданий и сооружений.
func TestRejectBoxTileRatioIsOptIn(t *testing.T) {
	const (
		tile    = 1536
		bound   = 256
		stride  = 4
		maskDim = 1
	)
	protoH, protoW := tile/stride, tile/stride
	W, H := tile*2, tile*2
	img := &OrthoImage{W: W, H: H, C: 3, ValidMask: make([]uint8, W*H)}
	for i := range img.ValidMask {
		img.ValidMask[i] = 1
	}
	proto := make([]float32, protoH*protoW)
	for i := range proto {
		proto[i] = 10
	}
	patch := patchInfo{x: 0, y: 0}
	// Крупная детекция в ядре тайла: краевые фильтры её не трогают.
	huge := yoloCandidate{x1: 400, y1: 400, x2: 1000, y2: 1000,
		rawW: 600, rawH: 600, score: 0.9, coeff: []float32{1}}
	small := yoloCandidate{x1: 400, y1: 400, x2: 500, y2: 500,
		rawW: 100, rawH: 100, score: 0.9, coeff: []float32{1}}

	off := specFromJSON(t, `{"name":"a","num_classes":1,"tile":1536,"bound":256}`)
	if off.RejectBoxTileRatio != 0 {
		t.Fatalf("reject_box_tile_ratio default = %v, want 0", off.RejectBoxTileRatio)
	}
	classMap := make([]uint16, W*H)
	scoreMap := make([]float32, W*H)
	if n, rec := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, huge, proto, protoH, protoW, maskDim, 0.5, off); n == 0 {
		t.Fatalf("filter must be off by default, decision=%s", rec.decision)
	}

	on := specFromJSON(t, `{"name":"a","num_classes":1,"tile":1536,"bound":256,"reject_box_tile_ratio":0.2}`)
	classMap = make([]uint16, W*H)
	scoreMap = make([]float32, W*H)
	if n, rec := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, huge, proto, protoH, protoW, maskDim, 0.5, on); n != 0 || rec.decision != "reject_box_size" {
		t.Fatalf("oversized box must be rejected, n=%d decision=%s", n, rec.decision)
	}
	classMap = make([]uint16, W*H)
	scoreMap = make([]float32, W*H)
	if n, rec := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, small, proto, protoH, protoW, maskDim, 0.5, on); n == 0 {
		t.Fatalf("small box must be kept, decision=%s", rec.decision)
	}
}
