package main

import (
	"math"
	"math/rand"
	"testing"
)

// referenceInstanceMask повторяет ПРЕЖНИЙ (до введения mask_mode) алгоритм
// вычисления окна и вероятностей маски: свёртка по maskDim выполнялась
// отдельно для КАЖДОГО пикселя полного разрешения.
//
// Тест ниже сравнивает с ним новый путь (protoMaskWindow + probAt) при
// настройках моделей buildings/pavilion/grunt/gazone/pokritie/uchastki,
// то есть mask_mode отсутствует (-> "instance") и mask_bbox_margin = 0.
// Совпадение должно быть побитовым: порядок сложений в свёртке и выражение
// int(float32(xx)*sx) не изменились, изменилось только то, СКОЛЬКО РАЗ
// результат считается.
func referenceInstanceMask(
	img *OrthoImage, patch patchInfo, tile, bound int,
	cand yoloCandidate, proto []float32, protoH, protoW, maskDim int,
	maskThreshold float32,
) map[int]float32 {
	trim := maskWriteTrim(bound)
	sx := float32(protoW) / float32(tile)
	sy := float32(protoH) / float32(tile)

	x0 := int(math.Floor(float64(cand.x1)))
	y0 := int(math.Floor(float64(cand.y1)))
	x1 := int(math.Ceil(float64(cand.x2)))
	y1 := int(math.Ceil(float64(cand.y2)))
	if patch.x > 0 && x0 < trim {
		x0 = trim
	}
	if patch.y > 0 && y0 < trim {
		y0 = trim
	}
	if patch.x+tile < img.W && x1 > tile-trim {
		x1 = tile - trim
	}
	if patch.y+tile < img.H && y1 > tile-trim {
		y1 = tile - trim
	}
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > tile {
		x1 = tile
	}
	if y1 > tile {
		y1 = tile
	}
	out := make(map[int]float32)
	if x1 <= x0 || y1 <= y0 {
		return out
	}

	for yy := y0; yy < y1; yy++ {
		gy := patch.y + yy
		if gy < 0 || gy >= img.H {
			continue
		}
		py := int(float32(yy) * sy)
		if py < 0 {
			py = 0
		}
		if py >= protoH {
			py = protoH - 1
		}
		for xx := x0; xx < x1; xx++ {
			gx := patch.x + xx
			if gx < 0 || gx >= img.W {
				continue
			}
			idx := gy*img.W + gx
			if img.ValidMask[idx] == 0 {
				continue
			}
			px := int(float32(xx) * sx)
			if px < 0 {
				px = 0
			}
			if px >= protoW {
				px = protoW - 1
			}
			v := float32(0)
			base := py*protoW + px
			for m := 0; m < maskDim; m++ {
				v += cand.coeff[m] * proto[m*protoH*protoW+base]
			}
			maskProb := sigmoid(v)
			if maskProb >= maskThreshold {
				out[idx] = cand.score * maskProb
			}
		}
	}
	return out
}

// buildingsLikeSpec повторяет секцию config.json моделей зданий/сооружений:
// все reject_* явно нулевые, mask_mode отсутствует.
func buildingsLikeSpec(t *testing.T) ModelSpec {
	t.Helper()
	return specFromJSON(t, `{
		"name":"buildings","num_classes":5,"classes":[0],
		"tile":1536,"bound":128,"normalization":"fixed",
		"threshold":0.25,"nms_threshold":0.45,"mask_threshold":0.5,
		"yolo_mask_dim":32,"yolo_proto_stride":4,
		"exclusive_classes":true,
		"min_area":10,"min_area_m2":10,
		"reject_tile_mask_ratio":0,"reject_tile_edge_touches":0,
		"reject_rectangularity":0,"reject_min_box_tile_ratio":0,
		"reject_edge_boxfill":0,"reject_boxfill":0,"reject_boxfill_score":0,
		"reject_dfl_size_tol":0,"reject_repeat_count":0,
		"skip_empty_tiles":true,"min_content_pixels":1
	}`)
}

// TestInstanceModeUnchanged — защита поиска зданий и сооружений:
// при отсутствии mask_mode в конфиге новый код обязан давать РОВНО ту же
// карту классов и ту же карту score, что и прежний алгоритм.
func TestInstanceModeUnchanged(t *testing.T) {
	spec := buildingsLikeSpec(t)
	if spec.MaskMode != "instance" {
		t.Fatalf("mask_mode without config value = %q, want \"instance\"", spec.MaskMode)
	}
	if spec.MaskBBoxMargin != 0 {
		t.Fatalf("mask_bbox_margin default = %v, want 0", spec.MaskBBoxMargin)
	}
	if spec.MaxMasksPerTile != 0 {
		t.Fatalf("max_masks_per_tile default = %d, want 0 (unlimited)", spec.MaxMasksPerTile)
	}

	const (
		tile     = 256
		bound    = 32
		stride   = 4
		maskDim  = 8
		numTests = 200
	)
	protoH, protoW := tile/stride, tile/stride

	rng := rand.New(rand.NewSource(20260909))
	W, H := tile*2, tile*2
	img := &OrthoImage{W: W, H: H, C: 3, ValidMask: make([]uint8, W*H)}
	for i := range img.ValidMask {
		// Часть растра помечена как nodata — путь ValidMask тоже проверяется.
		if rng.Intn(20) == 0 {
			img.ValidMask[i] = 0
		} else {
			img.ValidMask[i] = 1
		}
	}

	proto := make([]float32, maskDim*protoH*protoW)
	for i := range proto {
		proto[i] = float32(rng.NormFloat64()) * 2
	}

	// Патчи: угловой, внутренний (обе внутренние границы) и прижатый к краю.
	patches := []patchInfo{{x: 0, y: 0}, {x: tile, y: 0}, {x: 0, y: tile}, {x: tile, y: tile}}

	for n := 0; n < numTests; n++ {
		patch := patches[rng.Intn(len(patches))]
		cx := rng.Float32() * float32(tile)
		cy := rng.Float32() * float32(tile)
		w := 4 + rng.Float32()*float32(tile)
		h := 4 + rng.Float32()*float32(tile)
		x1 := maxf(0, cx-w/2)
		y1 := maxf(0, cy-h/2)
		x2 := minf(float32(tile-1), cx+w/2)
		y2 := minf(float32(tile-1), cy+h/2)
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		coeff := make([]float32, maskDim)
		for m := range coeff {
			coeff[m] = float32(rng.NormFloat64())
		}
		cand := yoloCandidate{
			x1: x1, y1: y1, x2: x2, y2: y2,
			rawW: w, rawH: h,
			score:   0.3 + rng.Float32()*0.7,
			classID: rng.Intn(spec.NumClasses),
			coeff:   coeff,
		}

		classMap := make([]uint16, W*H)
		scoreMap := make([]float32, W*H)
		got, rec := rasterizeYOLOClassMap(
			classMap, scoreMap, img, patch, tile, bound, cand,
			proto, protoH, protoW, maskDim, spec.MaskThreshold, spec,
		)

		want := referenceInstanceMask(img, patch, tile, bound, cand, proto, protoH, protoW, maskDim, spec.MaskThreshold)

		if got != len(want) {
			t.Fatalf("case %d: written=%d, reference=%d (decision=%s)", n, got, len(want), rec.decision)
		}
		for idx, score := range want {
			if classMap[idx] != uint16(cand.classID+1) {
				t.Fatalf("case %d: pixel %d class=%d, want %d", n, idx, classMap[idx], cand.classID+1)
			}
			if scoreMap[idx] != score {
				t.Fatalf("case %d: pixel %d score=%v, want %v (must be bit-identical)", n, idx, scoreMap[idx], score)
			}
		}
		written := 0
		for _, v := range classMap {
			if v != 0 {
				written++
			}
		}
		if written != len(want) {
			t.Fatalf("case %d: extra pixels written: %d vs %d", n, written, len(want))
		}
	}
}

// TestSemanticModeIsOptIn — режим semantic не должен включаться сам собой
// и должен читать маску за пределами bbox только когда явно запрошен.
func TestSemanticModeIsOptIn(t *testing.T) {
	const (
		tile    = 256
		bound   = 32
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
		proto[i] = 10 // sigmoid(10) ~ 1: маска "хочет" залить весь тайл
	}
	cand := yoloCandidate{x1: 64, y1: 64, x2: 128, y2: 128, rawW: 64, rawH: 64, score: 0.9, coeff: []float32{1}}
	patch := patchInfo{x: 0, y: 0}

	instance := specFromJSON(t, `{"name":"a","num_classes":1,"bound":32}`)
	classMap := make([]uint16, W*H)
	scoreMap := make([]float32, W*H)
	nInstance, _ := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, cand, proto, protoH, protoW, maskDim, 0.5, instance)
	if nInstance > 65*65 {
		t.Fatalf("instance mode wrote %d px, must stay inside the bbox (<= %d)", nInstance, 65*65)
	}

	semantic := specFromJSON(t, `{"name":"a","num_classes":1,"bound":32,"mask_mode":"semantic"}`)
	classMap = make([]uint16, W*H)
	scoreMap = make([]float32, W*H)
	nSemantic, _ := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, cand, proto, protoH, protoW, maskDim, 0.5, semantic)
	if nSemantic <= nInstance {
		t.Fatalf("semantic mode wrote %d px, expected more than instance mode (%d)", nSemantic, nInstance)
	}
}

// TestMaskBBoxMarginExpandsWindow — mask_bbox_margin расширяет окно чтения,
// но остаётся ограниченным (в отличие от semantic).
func TestMaskBBoxMarginExpandsWindow(t *testing.T) {
	const (
		tile    = 256
		bound   = 32
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
	cand := yoloCandidate{x1: 64, y1: 64, x2: 128, y2: 128, rawW: 64, rawH: 64, score: 0.9, coeff: []float32{1}}
	patch := patchInfo{x: 0, y: 0}

	base := specFromJSON(t, `{"name":"a","num_classes":1,"bound":32}`)
	classMap := make([]uint16, W*H)
	scoreMap := make([]float32, W*H)
	n0, _ := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, cand, proto, protoH, protoW, maskDim, 0.5, base)

	widened := specFromJSON(t, `{"name":"a","num_classes":1,"bound":32,"mask_bbox_margin":0.25}`)
	classMap = make([]uint16, W*H)
	scoreMap = make([]float32, W*H)
	n1, _ := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, cand, proto, protoH, protoW, maskDim, 0.5, widened)

	if n1 <= n0 {
		t.Fatalf("mask_bbox_margin=0.25 wrote %d px, want more than %d", n1, n0)
	}
	if n1 >= tile*tile {
		t.Fatalf("mask_bbox_margin must stay bounded, wrote %d px of %d", n1, tile*tile)
	}
}

// TestRejectEdgeOversizeIsOptIn — новый фильтр краевых гигантских детекций
// по умолчанию выключен (модели зданий и сооружений не затрагиваются),
// а при включении режет только детекции, одновременно крупные, прижатые к
// границе тайла и с центром в краевой полосе.
func TestRejectEdgeOversizeIsOptIn(t *testing.T) {
	const (
		tile    = 1536
		bound   = 512
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
	patch := patchInfo{x: 0, y: 0} // сосед есть справа и снизу

	// Краевой артефакт: 500x720, правая сторона прижата к границе тайла,
	// центр (1283, 400) — правый край в краевой полосе [1024, 1536].
	artifact := yoloCandidate{x1: 1033, y1: 40, x2: 1533, y2: 760,
		rawW: 500, rawH: 720, score: 0.5, coeff: []float32{1}}
	// Настоящий объект того же размера в ядре тайла.
	real := yoloCandidate{x1: 300, y1: 300, x2: 800, y2: 1020,
		rawW: 500, rawH: 720, score: 0.5, coeff: []float32{1}}

	off := specFromJSON(t, `{"name":"a","num_classes":1,"tile":1536,"bound":512}`)
	if off.RejectEdgeOversize != 0 {
		t.Fatalf("reject_edge_oversize default = %v, want 0", off.RejectEdgeOversize)
	}
	classMap := make([]uint16, W*H)
	scoreMap := make([]float32, W*H)
	if n, rec := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, artifact, proto, protoH, protoW, maskDim, 0.5, off); n == 0 {
		t.Fatalf("filter must be off by default, got decision=%s", rec.decision)
	}

	on := specFromJSON(t, `{"name":"a","num_classes":1,"tile":1536,"bound":512,"reject_edge_oversize":0.25}`)
	if on.RejectEdgePinPx != 24 {
		t.Fatalf("reject_edge_pin_px default = %d, want 24", on.RejectEdgePinPx)
	}
	classMap = make([]uint16, W*H)
	scoreMap = make([]float32, W*H)
	if n, rec := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, artifact, proto, protoH, protoW, maskDim, 0.5, on); n != 0 || rec.decision != "reject_edge_oversize" {
		t.Fatalf("edge-pinned oversize candidate must be rejected, n=%d decision=%s", n, rec.decision)
	}
	classMap = make([]uint16, W*H)
	scoreMap = make([]float32, W*H)
	if n, rec := rasterizeYOLOClassMap(classMap, scoreMap, img, patch, tile, bound, real, proto, protoH, protoW, maskDim, 0.5, on); n == 0 {
		t.Fatalf("core-of-tile candidate of the same size must be kept, decision=%s", rec.decision)
	}
}
