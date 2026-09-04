package main

import "testing"

// makeRasterEnv готовит синтетическое окружение: изображение 2x2 тайла,
// патч в левом верхнем углу (обе внутренние границы — справа и снизу),
// proto с управляемым полем маски.
func makeRasterEnv(tile, bound, protoStride int) (*OrthoImage, patchInfo, []float32, int, int, ModelSpec) {
	W, H := tile*2, tile*2
	img := &OrthoImage{W: W, H: H, C: 3, ValidMask: make([]uint8, W*H)}
	for i := range img.ValidMask {
		img.ValidMask[i] = 1
	}
	ph, pw := tile/protoStride, tile/protoStride
	proto := make([]float32, ph*pw) // maskDim = 1
	spec := FinalizeSpec(ModelSpec{Name: "t", NumClasses: 1})
	return img, patchInfo{x: 0, y: 0}, proto, ph, pw, spec
}

func fillProto(proto []float32, v float32) {
	for i := range proto {
		proto[i] = v
	}
}

func TestRejectDegenerateEdgeSquare(t *testing.T) {
	tile, bound := 256, 32
	img, patch, proto, ph, pw, spec := makeRasterEnv(tile, bound, 4)
	fillProto(proto, 10) // sigmoid(10*1) ~ 1: маска заливает весь bbox

	// Вырожденный квадрат: центр в краевой полосе (x > tile-bound), score низкий.
	cand := yoloCandidate{x1: 200, y1: 100, x2: 250, y2: 150, score: 0.3, coeff: []float32{1}}
	dst := make([]uint8, img.W*img.H)
	if n, _ := rasterizeYOLOMask(dst, img, patch, tile, bound, cand, proto, ph, pw, 1, 0.5, spec, nil); n != 0 {
		t.Fatalf("degenerate edge square must be rejected, wrote %d px", n)
	}

	// Тот же квадрат с высоким score — реальный объект, должен записаться.
	cand.score = 0.8
	if n, _ := rasterizeYOLOMask(dst, img, patch, tile, bound, cand, proto, ph, pw, 1, 0.5, spec, nil); n == 0 {
		t.Fatal("confident edge detection must be written")
	}
}

func TestCoreDetectionsUntouchedByDefault(t *testing.T) {
	tile, bound := 256, 32
	img, patch, proto, ph, pw, spec := makeRasterEnv(tile, bound, 4)
	fillProto(proto, 10)

	// Сплошная низко-score детекция в ЯДРЕ тайла по умолчанию не фильтруется
	// (reject_boxfill по умолчанию выключен) — реальные здания с невысокой
	// уверенностью не должны пропадать.
	cand := yoloCandidate{x1: 80, y1: 80, x2: 160, y2: 160, score: 0.3, coeff: []float32{1}}
	dst := make([]uint8, img.W*img.H)
	if n, _ := rasterizeYOLOMask(dst, img, patch, tile, bound, cand, proto, ph, pw, 1, 0.5, spec, nil); n == 0 {
		t.Fatal("solid low-score core detection must be written by default")
	}
}

func TestRejectDegenerateCoreSquareOptIn(t *testing.T) {
	tile, bound := 256, 32
	img, patch, proto, ph, pw, spec := makeRasterEnv(tile, bound, 4)
	v := float32(0.995)
	spec.RejectBoxfillJSON = &v
	spec = FinalizeSpec(spec)
	fillProto(proto, 10)

	// С включённым reject_boxfill вырожденный квадрат в ядре режется...
	cand := yoloCandidate{x1: 80, y1: 80, x2: 160, y2: 160, score: 0.3, coeff: []float32{1}}
	dst := make([]uint8, img.W*img.H)
	if n, _ := rasterizeYOLOMask(dst, img, patch, tile, bound, cand, proto, ph, pw, 1, 0.5, spec, nil); n != 0 {
		t.Fatalf("degenerate core square must be rejected when opted in, wrote %d px", n)
	}
	// ...а маленький bbox (< 48px) не трогается даже при заполнении 1.0.
	small := yoloCandidate{x1: 80, y1: 80, x2: 110, y2: 110, score: 0.3, coeff: []float32{1}}
	if n, _ := rasterizeYOLOMask(dst, img, patch, tile, bound, small, proto, ph, pw, 1, 0.5, spec, nil); n == 0 {
		t.Fatal("small core detection must not be rejected by boxfill filter")
	}
}

func TestEdgeRejectionCoveredByOwnerTile(t *testing.T) {
	// Свойство-гарантия краевого правила: реальный объект, чья маска
	// отброшена в краевой полосе одного тайла, записывается соседним тайлом,
	// в ядре которого лежит его центр.
	tile, bound := 256, 32
	step := tile - 2*bound
	img, patchA, proto, ph, pw, spec := makeRasterEnv(tile, bound, 4)
	fillProto(proto, 10)

	// Объект с центром на глобальном x=230: в тайле A (offset 0) центр в
	// краевой полосе (230 > 224) -> отброшен; в тайле B (offset step=192)
	// локальный центр 38 лежит в ядре [32,224] -> записан целиком.
	dst := make([]uint8, img.W*img.H)
	candA := yoloCandidate{x1: 205, y1: 80, x2: 255, y2: 160, score: 0.3, coeff: []float32{1}}
	if n, _ := rasterizeYOLOMask(dst, img, patchA, tile, bound, candA, proto, ph, pw, 1, 0.5, spec, nil); n != 0 {
		t.Fatalf("edge-margin solid detection must be rejected in tile A, wrote %d", n)
	}
	patchB := patchInfo{x: step, y: 0}
	candB := yoloCandidate{x1: 205 - float32(step), y1: 80, x2: 255 - float32(step), y2: 160, score: 0.3, coeff: []float32{1}}
	if n, _ := rasterizeYOLOMask(dst, img, patchB, tile, bound, candB, proto, ph, pw, 1, 0.5, spec, nil); n == 0 {
		t.Fatal("owner tile B must write the object in full")
	}
}

func TestBoxfillFilterOptOut(t *testing.T) {
	zero := float32(0)
	m := FinalizeSpec(ModelSpec{Name: "t", NumClasses: 1,
		RejectEdgeBoxfillJSON: &zero, RejectBoxfillJSON: &zero})
	if m.RejectEdgeBoxfill != 0 || m.RejectBoxfill != 0 {
		t.Fatalf("explicit 0 must disable boxfill filters, got %v/%v", m.RejectEdgeBoxfill, m.RejectBoxfill)
	}
	d := FinalizeSpec(ModelSpec{Name: "t", NumClasses: 1})
	if d.RejectEdgeBoxfill != 0.95 || d.RejectBoxfill != 0 || d.RejectBoxfillScore != 0.5 {
		t.Fatalf("unexpected defaults: %v %v %v", d.RejectEdgeBoxfill, d.RejectBoxfill, d.RejectBoxfillScore)
	}
}

func TestRejectDFLCollapseAnyScore(t *testing.T) {
	tile, bound := 512, 64
	img, patch, proto, ph, pw, spec := makeRasterEnv(tile, bound, 4)
	fillProto(proto, 10)

	// Вырожденный DFL-квадрат 240x240 (15*stride16) с ВЫСОКИМ score
	// в ЯДРЕ тайла: режется сигнатурой размера.
	c := yoloCandidate{x1: 130, y1: 130, x2: 370, y2: 370, rawW: 240, rawH: 240,
		score: 0.83, coeff: []float32{1}}
	dst := make([]uint8, img.W*img.H)
	if n, rec := rasterizeYOLOMask(dst, img, patch, tile, bound, c, proto, ph, pw, 1, 0.5, spec, nil); n != 0 {
		t.Fatalf("DFL-collapse square must be rejected at any score, wrote %d (%s)", n, rec.decision)
	}

	// Бокс похожего масштаба, но вне сигнатуры (300x220) — пишется.
	r := yoloCandidate{x1: 100, y1: 140, x2: 400, y2: 360, rawW: 300, rawH: 220,
		score: 0.83, coeff: []float32{1}}
	if n, _ := rasterizeYOLOMask(dst, img, patch, tile, bound, r, proto, ph, pw, 1, 0.5, spec, nil); n == 0 {
		t.Fatal("non-DFL-sized confident detection must be written")
	}
}

func TestIsDFLCollapseSize(t *testing.T) {
	if !isDFLCollapseSize(240, 240, 0.06) || !isDFLCollapseSize(120, 124, 0.06) || !isDFLCollapseSize(480, 470, 0.06) {
		t.Fatal("canonical DFL sizes must match")
	}
	if isDFLCollapseSize(240, 300, 0.06) || isDFLCollapseSize(200, 200, 0.06) || isDFLCollapseSize(240, 240, 0) {
		t.Fatal("non-DFL sizes (or disabled tol) must not match")
	}
}

func TestRepeatedSizeFilter(t *testing.T) {
	tile, bound := 512, 64
	img, patch, proto, ph, pw, spec := makeRasterEnv(tile, bound, 4)
	fillProto(proto, 10)
	store := &deferredStore{}
	dst := make([]uint8, img.W*img.H)

	// 6 «уверенных» залитых квадратов ОДНОГО размера 280x280 (модельно-
	// специфичная константа вырождения, не совпадающая с 15*stride) в разных
	// местах ядра тайла + 1 квадрат уникального размера 200x200.
	for i := 0; i < 6; i++ {
		x := float32(70 + i*30)
		c := yoloCandidate{x1: x, y1: 100, x2: x + 280, y2: 380,
			rawW: 280.2, rawH: 279.8, score: 0.8, coeff: []float32{1}}
		if n, rec := rasterizeYOLOMask(dst, img, patch, tile, bound, c, proto, ph, pw, 1, 0.5, spec, store); n != 0 || rec.decision != "deferred" {
			t.Fatalf("suspicious candidate must be deferred, wrote %d decision=%s", n, rec.decision)
		}
	}
	u := yoloCandidate{x1: 80, y1: 60, x2: 280, y2: 260,
		rawW: 200, rawH: 201, score: 0.8, coeff: []float32{1}}
	if _, rec := rasterizeYOLOMask(dst, img, patch, tile, bound, u, proto, ph, pw, 1, 0.5, spec, store); rec.decision != "deferred" {
		t.Fatalf("unique-size candidate must also be deferred first, got %s", rec.decision)
	}

	written, recs := resolveDeferred(dst, img.W, store, spec.RejectRepeatCount)
	if written == 0 {
		t.Fatal("unique-size candidate must be written by resolveDeferred")
	}
	nRej, nWr := 0, 0
	for _, r := range recs {
		switch r.decision {
		case "reject_repeated_size":
			nRej++
		case "written":
			nWr++
		}
	}
	if nRej != 6 || nWr != 1 {
		t.Fatalf("expected 6 rejected / 1 written, got %d/%d", nRej, nWr)
	}
	// Пиксели уникального кандидата в маске, массового — нет.
	if dst[130*img.W+150] == 0 { // внутри 200x200 (y ~60..260, x ~80..280)
		t.Fatal("unique candidate pixels missing from mask")
	}
	if dst[350*img.W+100] != 0 { // внутри только массовых квадратов (y>260)
		t.Fatal("repeated-size candidate pixels must not be in mask")
	}
}

func TestNonSquareNotDeferred(t *testing.T) {
	tile, bound := 512, 64
	img, patch, proto, ph, pw, spec := makeRasterEnv(tile, bound, 4)
	fillProto(proto, 10)
	store := &deferredStore{}
	dst := make([]uint8, img.W*img.H)
	// Прямоугольное здание (300x180) с залитой маской — НЕ подозрительно,
	// пишется сразу даже при включённом частотном фильтре.
	c := yoloCandidate{x1: 100, y1: 100, x2: 400, y2: 280,
		rawW: 300, rawH: 180, score: 0.4, coeff: []float32{1}}
	if n, rec := rasterizeYOLOMask(dst, img, patch, tile, bound, c, proto, ph, pw, 1, 0.5, spec, store); n == 0 || rec.decision != "written" {
		t.Fatalf("rectangular building must be written immediately, n=%d decision=%s", n, rec.decision)
	}
	if len(store.cands) != 0 {
		t.Fatal("rectangular building must not be deferred")
	}
}
