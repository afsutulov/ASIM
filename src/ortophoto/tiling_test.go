package main

import "testing"

// TestAxisOffsetsFullCoverage проверяет, что при новой раскладке тайлов
// (последний тайл прижат к краю растра) записываемые окна rasterizeYOLOMask
// покрывают растр целиком, без дыр, и ни один тайл не выходит за границы
// растра (когда растр не меньше тайла).
func TestAxisOffsetsFullCoverage(t *testing.T) {
	tile := 1536
	bound := 128
	step := tile - 2*bound

	sizesW := []int{1537, 1600, 2000, 2816, 2817, 3000, 5000, 9000, 10007, 12800}
	sizesH := []int{1537, 2000, 4321, 7000}

	for _, W := range sizesW {
		for _, H := range sizesH {
			cov := make([]uint8, W*H)
			for _, py := range axisOffsets(H, tile, step) {
				for _, px := range axisOffsets(W, tile, step) {
					if px+tile > W || py+tile > H {
						t.Fatalf("tile out of raster: W=%d H=%d patch=(%d,%d)", W, H, px, py)
					}
					// Повторяем логику клиппинга из rasterizeYOLOMask
					x0, y0, x1, y1 := 0, 0, tile, tile
					if px > 0 && x0 < bound {
						x0 = bound
					}
					if py > 0 && y0 < bound {
						y0 = bound
					}
					if px+tile < W && x1 > tile-bound {
						x1 = tile - bound
					}
					if py+tile < H && y1 > tile-bound {
						y1 = tile - bound
					}
					for yy := y0; yy < y1; yy++ {
						gy := py + yy
						for xx := x0; xx < x1; xx++ {
							cov[gy*W+px+xx] = 1
						}
					}
				}
			}
			for i, v := range cov {
				if v == 0 {
					t.Fatalf("uncovered pixel: W=%d H=%d x=%d y=%d", W, H, i%W, i/W)
				}
			}
		}
	}
}

// TestAxisOffsetsSmallRaster: растр меньше тайла — один тайл с нулевым смещением.
func TestAxisOffsetsSmallRaster(t *testing.T) {
	offs := axisOffsets(800, 1536, 1280)
	if len(offs) != 1 || offs[0] != 0 {
		t.Fatalf("expected [0], got %v", offs)
	}
	offs = axisOffsets(1536, 1536, 1280)
	if len(offs) != 1 || offs[0] != 0 {
		t.Fatalf("expected [0], got %v", offs)
	}
}

// TestIsMostlyEmptyTileKeepsContent: тайл, почти целиком белый/невалидный,
// но содержащий фрагмент здания, НЕ должен пропускаться.
func TestIsMostlyEmptyTileKeepsContent(t *testing.T) {
	w, h, c := 64, 64, 3
	data := make([]float32, c*w*h)
	valid := make([]uint8, w*h)
	for i := range data {
		data[i] = 1.0 // всё белое (nodata-кайма)
	}
	for i := range valid {
		valid[i] = 1
	}
	// "здание" 20x20 = 400 пикселей (~10% тайла) в углу
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			for ch := 0; ch < c; ch++ {
				data[ch*w*h+y*w+x] = 0.4
			}
		}
	}
	if isMostlyEmptyTile(data, valid, c, w, h, 0.98, 256) {
		t.Fatal("tile with a 400-px building fragment must NOT be skipped")
	}
	// А полностью белый тайл — должен пропускаться.
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			for ch := 0; ch < c; ch++ {
				data[ch*w*h+y*w+x] = 1.0
			}
		}
	}
	if !isMostlyEmptyTile(data, valid, c, w, h, 0.98, 256) {
		t.Fatal("fully white tile must be skipped")
	}
	// И полностью невалидный тайл — тоже.
	for i := range valid {
		valid[i] = 0
	}
	if !isMostlyEmptyTile(data, valid, c, w, h, 0.98, 256) {
		t.Fatal("fully invalid tile must be skipped")
	}
}

// TestMaskWriteTrim: отступ мал, не превышает bound и обнуляется при bound=0.
func TestMaskWriteTrim(t *testing.T) {
	cases := map[int]int{0: 0, 1: 1, 3: 3, 4: 4, 128: 4, 256: 4}
	for bound, want := range cases {
		if got := maskWriteTrim(bound); got != want {
			t.Fatalf("maskWriteTrim(%d) = %d, want %d", bound, got, want)
		}
	}
}

func TestHasAnyValidPixel(t *testing.T) {
	if hasAnyValidPixel([]uint8{0, 0, 0}) {
		t.Fatal("fully invalid patch must be skippable")
	}
	if !hasAnyValidPixel([]uint8{0, 1, 0}) {
		t.Fatal("a patch with one valid pixel must be processed")
	}
	if !hasAnyValidPixel(nil) {
		t.Fatal("missing validity mask must be treated as valid")
	}
}
