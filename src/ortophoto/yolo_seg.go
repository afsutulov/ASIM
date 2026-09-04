package main

import (
	"fmt"
	"log"
	"math"
	"sort"
	"strconv"
	"sync"
)

type yoloCandidate struct {
	x1, y1, x2, y2 float32
	// rawW, rawH — размеры бокса, предсказанные моделью, ДО клиппинга по
	// границам тайла. Нужны для распознавания вырожденных DFL-детекций:
	// у них размер равен ровно 15*stride и клиппинг у края тайла его маскирует.
	rawW, rawH float32
	score      float32
	classID    int
	coeff      []float32
}

func classAllowed(classID int, allowed []int) bool {
	if len(allowed) == 0 {
		return true
	}
	for _, c := range allowed {
		if c == classID {
			return true
		}
	}
	return false
}

func valueAsProb(v float32) float32 {
	if v >= 0 && v <= 1 {
		return v
	}
	return sigmoid(v)
}

func boxIoU(a, b yoloCandidate) float32 {
	x1 := maxf(a.x1, b.x1)
	y1 := maxf(a.y1, b.y1)
	x2 := minf(a.x2, b.x2)
	y2 := minf(a.y2, b.y2)
	w := x2 - x1
	h := y2 - y1
	if w <= 0 || h <= 0 {
		return 0
	}
	inter := w * h
	aArea := maxf(0, a.x2-a.x1) * maxf(0, a.y2-a.y1)
	bArea := maxf(0, b.x2-b.x1) * maxf(0, b.y2-b.y1)
	den := aArea + bArea - inter
	if den <= 0 {
		return 0
	}
	return inter / den
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func classThreshold(spec ModelSpec, classID int) float32 {
	if spec.ClassThresholds != nil {
		if v, ok := spec.ClassThresholds[strconv.Itoa(classID)]; ok && v > 0 {
			return v
		}
	}
	return spec.Threshold
}

func minClassThreshold(spec ModelSpec) float32 {
	thr := spec.Threshold
	if thr <= 0 {
		thr = 0.25
	}
	for _, v := range spec.ClassThresholds {
		if v > 0 && v < thr {
			thr = v
		}
	}
	return thr
}

func decodeYOLOSeg(det []float32, detCount, numClasses, maskDim int, threshold float32, classThresholds map[string]float32, nmsThreshold float32, agnosticNMS bool, tile int) []yoloCandidate {
	cands := make([]yoloCandidate, 0)
	for i := 0; i < detCount; i++ {
		bestClass := -1
		bestScore := float32(0)
		for c := 0; c < numClasses; c++ {
			s := valueAsProb(det[(4+c)*detCount+i])
			if s > bestScore {
				bestScore = s
				bestClass = c
			}
		}
		classThr := threshold
		if classThresholds != nil {
			if v, ok := classThresholds[strconv.Itoa(bestClass)]; ok && v > 0 {
				classThr = v
			}
		}
		if bestClass < 0 || bestScore < classThr {
			continue
		}
		xc := det[0*detCount+i]
		yc := det[1*detCount+i]
		w := det[2*detCount+i]
		h := det[3*detCount+i]
		if w <= 1 || h <= 1 {
			continue
		}
		x1 := maxf(0, xc-w/2)
		y1 := maxf(0, yc-h/2)
		x2 := minf(float32(tile-1), xc+w/2)
		y2 := minf(float32(tile-1), yc+h/2)
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		coeff := make([]float32, maskDim)
		for m := 0; m < maskDim; m++ {
			coeff[m] = det[(4+numClasses+m)*detCount+i]
		}
		cands = append(cands, yoloCandidate{x1: x1, y1: y1, x2: x2, y2: y2, rawW: w, rawH: h, score: bestScore, classID: bestClass, coeff: coeff})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })
	kept := make([]yoloCandidate, 0, len(cands))
	for _, c := range cands {
		suppress := false
		for _, k := range kept {
			if (agnosticNMS || c.classID == k.classID) && boxIoU(c, k) > nmsThreshold {
				suppress = true
				break
			}
		}
		if !suppress {
			kept = append(kept, c)
		}
	}
	return kept
}

// isDFLCollapseSize сообщает, совпадают ли размеры бокса (до клиппинга)
// с сигнатурой вырожденной DFL-регрессии YOLO. При неинформативных признаках
// (однородная текстура: пашня, вода) распределение DFL по всем 4 сторонам
// вырождается в равномерное, и его матожидание даёт бокс размером ровно
// 15*stride (120/240/480 px для strides 8/16/32) НЕЗАВИСИМО от содержимого.
// Поэтому такие ложные боксы имеют константный размер с точностью до долей
// пикселя — это и наблюдается на результатах (десятки одинаковых квадратов).
// tol — относительный допуск (например 0.06 = ±6%).
func isDFLCollapseSize(w, h, tol float32) bool {
	if tol <= 0 {
		return false
	}
	for _, s := range [3]float32{8, 16, 32} {
		t := 15 * s
		if w >= t*(1-tol) && w <= t*(1+tol) && h >= t*(1-tol) && h <= t*(1+tol) {
			return true
		}
	}
	return false
}

// candidateRecord — диагностическая запись об одном кандидате детекции
// (пишется в CSV при spec.DumpCandidates; см. writeCandidateCSV).
type candidateRecord struct {
	patchX, patchY int
	cxPx, cyPx     float64 // центр бокса в глобальных пикселях листа
	rawW, rawH     float32
	score          float32
	classID        int
	fill           float32 // заполнение бокса маской (rectangularity)
	decision       string
}

// deferredCand — отложенный подозрительный кандидат частотного фильтра
// повторяющихся размеров (см. ModelSpec.RejectRepeatCount): его пиксели
// хранятся компактным битсетом над bbox в глобальных координатах листа и
// записываются в маску только если размер бокса не оказался массовым.
type deferredCand struct {
	rawW, rawH float32
	x0, y0     int // левый верхний угол bbox маски (глобальные пиксели)
	w, h       int
	bits       []uint64
	rec        candidateRecord
}

type deferredStore struct {
	cands []deferredCand
}

func sizeBinKey(rawW, rawH float32) [2]int {
	// Бины по 2 px: вырожденные размеры совпадают до долей пикселя,
	// у реальных зданий регрессия бокса шумит сильнее.
	return [2]int{int(rawW/2 + 0.5), int(rawH/2 + 0.5)}
}

// resolveDeferred применяет частотное правило: размеры, встретившиеся
// >= minRepeat раз на листе, отбрасываются; остальные кандидаты
// записываются в маску. Возвращает число новых пикселей маски и
// диагностические записи с итоговыми решениями.
func resolveDeferred(mask []uint8, imgW int, store *deferredStore, minRepeat int) (int, []candidateRecord) {
	written := 0
	recs := make([]candidateRecord, 0, len(store.cands))
	counts := make(map[[2]int]int)
	for _, c := range store.cands {
		counts[sizeBinKey(c.rawW, c.rawH)]++
	}
	rejectedSizes := make(map[[2]int]bool)
	for k, n := range counts {
		if minRepeat > 0 && n >= minRepeat {
			rejectedSizes[k] = true
		}
	}
	if len(rejectedSizes) > 0 {
		for k := range rejectedSizes {
			log.Printf("ortophoto repeated-size reject: box ~%dx%d px, %d candidates on sheet", k[0]*2, k[1]*2, counts[k])
		}
	}
	for _, c := range store.cands {
		if rejectedSizes[sizeBinKey(c.rawW, c.rawH)] {
			c.rec.decision = "reject_repeated_size"
			recs = append(recs, c.rec)
			continue
		}
		for yy := 0; yy < c.h; yy++ {
			rowBase := yy * c.w
			gy := c.y0 + yy
			for xx := 0; xx < c.w; xx++ {
				b := rowBase + xx
				if c.bits[b>>6]&(1<<uint(b&63)) == 0 {
					continue
				}
				idx := gy*imgW + c.x0 + xx
				if mask[idx] == 0 {
					written++
				}
				mask[idx] = 1
			}
		}
		c.rec.decision = "written"
		recs = append(recs, c.rec)
	}
	return written, recs
}

// isRepeatSuspicious: кандидат откладывается в частотный фильтр, если маска
// заливает бокс, бокс почти квадратный и размер в диапазоне вырождений.
func isRepeatSuspicious(spec ModelSpec, rawW, rawH, fill float32) bool {
	if spec.RejectRepeatCount <= 0 || spec.RejectEdgeBoxfill <= 0 || fill < spec.RejectEdgeBoxfill {
		return false
	}
	if rawW < 60 || rawW > 600 || rawH < 60 || rawH > 600 {
		return false
	}
	d := rawW - rawH
	if d < 0 {
		d = -d
	}
	m := rawW
	if rawH > m {
		m = rawH
	}
	tol := m * 0.02
	if tol < 2 {
		tol = 2
	}
	return d <= tol
}

// maskWriteTrim возвращает величину отступа от ВНУТРЕННИХ границ тайла,
// в пределах которого маски не записываются (отсечение артефактов самой
// крайней кромки). Отступ мал (4 px) и никогда не превышает bound, поэтому
// перекрытие соседних тайлов (2*bound) гарантированно закрывает срезанные
// полосы; при bound=0 (тайлы без перекрытия) отступ равен нулю.
func maskWriteTrim(bound int) int {
	t := 4
	if bound < t {
		t = bound
	}
	if t < 0 {
		t = 0
	}
	return t
}

func assignClassPixel(classMap []uint16, scoreMap []float32, idx, classID int, score float32) bool {
	if idx < 0 || idx >= len(classMap) || idx >= len(scoreMap) || classID < 0 || score <= scoreMap[idx] {
		return false
	}
	wasBackground := classMap[idx] == 0
	scoreMap[idx] = score
	classMap[idx] = uint16(classID + 1)
	return wasBackground
}

func rasterizeYOLOClassMap(classMap []uint16, scoreMap []float32, img *OrthoImage, patch patchInfo, tile, bound int, cand yoloCandidate, proto []float32, protoH, protoW, maskDim int, maskThreshold float32, spec ModelSpec) (int, candidateRecord) {
	rec := candidateRecord{
		patchX: patch.x, patchY: patch.y,
		cxPx: float64(patch.x) + float64(cand.x1+cand.x2)/2,
		cyPx: float64(patch.y) + float64(cand.y1+cand.y2)/2,
		rawW: cand.rawW, rawH: cand.rawH,
		score: cand.score, classID: cand.classID,
	}
	// First pass: compute the candidate mask geometry inside the usable part of the tile.
	// This allows us to reject tile-sized rectangular artefacts before they pollute the
	// accumulated raster mask.
	//
	// ВАЖНО (исправление «ровных обрезов» на стыках зон тайлов): раньше каждый
	// тайл записывал маски ТОЛЬКО внутри окна [bound, tile-bound] — растр был
	// жёстко поделён на зоны ответственности, и каждый пиксель шва записывался
	// РОВНО ОДНИМ тайлом. Любой сбой в этом единственном тайле — пропущенная
	// детекция усечённого краем тайла фрагмента здания или предсказанный YOLO
	// bbox короче реального объекта (маска обрезается по bbox) — оставлял
	// прямоугольную дыру с идеально ровными границами: по линии сетки тайлов
	// или по границе bbox.
	//
	// Теперь каждый тайл записывает ВСЮ свою маску (за вычетом крошечного
	// отступа maskWriteTrim от внутренних краёв тайла). Маски соседних тайлов
	// объединяются по OR, а перекрытие 2*bound даёт избыточность: пиксель у шва
	// пишут два тайла по каждой оси (до четырёх в углах), и здание целиком
	// восстанавливается, если его видит ХОТЯ БЫ ОДИН из перекрывающихся тайлов.
	// Усечённые краем тайла детекции безопасны: их маска — подмножество
	// настоящего здания, и объединение с полной маской соседнего тайла лишь
	// дополняет результат. Параметр bound теперь управляет только величиной
	// перекрытия тайлов (step = tile - 2*bound), а не зоной записи.
	written := 0
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
	if x1 <= x0 || y1 <= y0 {
		rec.decision = "outside_window"
		return 0, rec
	}

	type scoredPixel struct {
		x, y  int
		score float32
	}
	pixels := make([]scoredPixel, 0, 1024)
	minX, minY := tile, tile
	maxX, maxY := -1, -1
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
				pixels = append(pixels, scoredPixel{x: gx, y: gy, score: cand.score * maskProb})
				if xx < minX {
					minX = xx
				}
				if yy < minY {
					minY = yy
				}
				if xx > maxX {
					maxX = xx
				}
				if yy > maxY {
					maxY = yy
				}
			}
		}
	}
	if len(pixels) == 0 {
		rec.decision = "empty_mask"
		return 0, rec
	}

	maskArea := float32(len(pixels))
	usableArea := float32((x1 - x0) * (y1 - y0))
	if usableArea <= 0 {
		rec.decision = "outside_window"
		return 0, rec
	}
	boxW := maxX - minX + 1
	boxH := maxY - minY + 1
	boxArea := float32(boxW * boxH)
	if boxArea <= 0 {
		rec.decision = "empty_mask"
		return 0, rec
	}
	tileMaskRatio := maskArea / usableArea
	boxTileRatio := boxArea / usableArea
	rectangularity := maskArea / boxArea
	rec.fill = rectangularity

	// Фильтр вырожденных «квадратных» детекций (см. ModelSpec: reject_edge_boxfill,
	// reject_boxfill, reject_boxfill_score). Их сигнатура: маска заливает свой
	// bbox практически целиком при невысоком score. Такие детекции возникают
	// на однородных текстурах (пашня) в основном у краёв тайла; раньше их
	// молча срезало окно записи [bound, tile-bound], теперь фильтруем явно.
	// Вырожденная детекция определяется сочетанием: маска заливает бокс
	// (rectangularity ~1.0) И либо score ниже reject_boxfill_score, либо
	// размеры бокса совпадают с сигнатурой DFL-коллапса (константа 15*stride).
	// Сигнатура размера применяется при ЛЮБОМ score: на однородной текстуре
	// вырожденные детекции бывают «уверенными».
	suspicious := cand.score < spec.RejectBoxfillScore ||
		isDFLCollapseSize(cand.rawW, cand.rawH, spec.RejectDFLSizeTol)
	if suspicious {
		cxc := (cand.x1 + cand.x2) / 2
		cyc := (cand.y1 + cand.y2) / 2
		inMargin := (patch.x > 0 && cxc < float32(bound)) ||
			(patch.x+tile < img.W && cxc > float32(tile-bound)) ||
			(patch.y > 0 && cyc < float32(bound)) ||
			(patch.y+tile < img.H && cyc > float32(tile-bound))
		if inMargin && spec.RejectEdgeBoxfill > 0 && rectangularity >= spec.RejectEdgeBoxfill {
			// Центр детекции лежит в краевой полосе: по гарантии покрытия
			// сетки он находится в «ядре» соседнего тайла — реальный объект
			// будет записан оттуда, отбрасывание здесь ничего не теряет.
			rec.decision = "reject_edge_boxfill"
			return 0, rec
		}
		// DFL-сигнатура режется и в ядре тайла: вырожденные квадраты возникают
		// на любом «пустом» участке, не только у краёв. Реальное здание почти
		// никогда не совпадает с сигнатурой: нужен размер ровно 15*stride по
		// ОБЕИМ осям (до клиппинга) и маска, заливающая бокс на >= 95%.
		if spec.RejectEdgeBoxfill > 0 && rectangularity >= spec.RejectEdgeBoxfill &&
			isDFLCollapseSize(cand.rawW, cand.rawH, spec.RejectDFLSizeTol) {
			rec.decision = "reject_dfl_size"
			return 0, rec
		}
		if spec.RejectBoxfill > 0 && rectangularity >= spec.RejectBoxfill && boxW >= 48 && boxH >= 48 {
			rec.decision = "reject_boxfill"
			return 0, rec
		}
	}

	touchCount := 0
	edgeMargin := max(2, bound/8)
	if minX <= x0+edgeMargin {
		touchCount++
	}
	if minY <= y0+edgeMargin {
		touchCount++
	}
	if maxX >= x1-1-edgeMargin {
		touchCount++
	}
	if maxY >= y1-1-edgeMargin {
		touchCount++
	}

	// Reject the characteristic false positive: a very large, almost rectangular
	// mask that touches most edges of the processed tile. This is safer than a
	// pure area filter because real large roofs often have lower rectangularity or
	// do not touch 3-4 tile edges at once.
	if spec.RejectTileMaskRatio > 0 &&
		tileMaskRatio >= spec.RejectTileMaskRatio &&
		touchCount >= spec.RejectTileEdgeTouches &&
		rectangularity >= spec.RejectRectangularity &&
		boxTileRatio >= spec.RejectMinBoxTileRatio {
		rec.decision = "reject_tile_artifact"
		return 0, rec
	}

	for _, p := range pixels {
		idx := p.y*img.W + p.x
		// Один пиксель может принадлежать только одному классу. Побеждает
		// кандидат с максимальным class_confidence * mask_probability.
		if assignClassPixel(classMap, scoreMap, idx, cand.classID, p.score) {
			written++
		}
	}
	rec.decision = "written"
	return written, rec
}
func runYOLOSegOnImage(img *OrthoImage, sessions []*ORTSession, spec ModelSpec) ([]uint8, int, MaskDebugStats, error) {
	tile := spec.Tile
	if tile <= 0 {
		tile = 1536
	}
	bound := spec.Bound
	if bound < 0 {
		bound = 0
	}
	step := tile - 2*bound
	if step <= 0 {
		step = tile
	}
	patches := collectPatches(img, tile, step)
	classMap := make([]uint16, img.W*img.H)
	scoreMap := make([]float32, img.W*img.H)
	debug := MaskDebugStats{}
	if len(sessions) == 0 {
		return nil, 0, debug, fmt.Errorf("no ONNX Runtime sessions opened")
	}

	type job struct {
		idx int
		p   patchInfo
	}
	type workerResult struct {
		runs  int
		debug MaskDebugStats
		recs  []candidateRecord
		err   error
	}

	jobs := make(chan job)
	results := make(chan workerResult, len(sessions))
	var mapMu sync.Mutex

	for workerID, sess := range sessions {
		go func(workerID int, sess *ORTSession) {
			res := workerResult{}
			for j := range jobs {
				patch, valid := extractPatch(img, j.p.x, j.p.y, tile)
				// Safe acceleration: skip only patches that contain no valid raster
				// pixels at all. Any patch with even one valid pixel still reaches
				// ONNX, so edge buildings cannot be lost by this optimisation.
				if spec.SkipEmptyTiles && !hasAnyValidPixel(valid) {
					continue
				}
				det, proto, detCount, protoH, protoW, err := sess.PredictYOLOSeg(patch, img.C, tile, tile, spec.NumClasses, spec.YoloMaskDim, spec.YoloProtoStride)
				if err != nil {
					res.err = fmt.Errorf("gpu worker %d patch %d (%d,%d): %w", workerID, j.idx, j.p.x, j.p.y, err)
					break
				}
				res.runs++
				cands := decodeYOLOSeg(det, detCount, spec.NumClasses, spec.YoloMaskDim, spec.Threshold, spec.ClassThresholds, spec.NMSThreshold, spec.AgnosticNMS, tile)
				if len(cands) == 0 {
					continue
				}
				mapMu.Lock()
				for _, c := range cands {
					n, rec := rasterizeYOLOClassMap(classMap, scoreMap, img, j.p, tile, bound, c, proto, protoH, protoW, spec.YoloMaskDim, spec.MaskThreshold, spec)
					res.debug.DetectedPixels += n
					if spec.DumpCandidates {
						res.recs = append(res.recs, rec)
					}
				}
				mapMu.Unlock()
			}
			results <- res
		}(workerID, sess)
	}

	collected := make([]workerResult, 0, len(sessions))
sendLoop:
	for i, p := range patches {
		select {
		case jobs <- job{idx: i, p: p}:
		case res := <-results:
			collected = append(collected, res)
			break sendLoop
		}
	}
	close(jobs)
	for len(collected) < len(sessions) {
		collected = append(collected, <-results)
	}

	runs := 0
	var firstErr error
	for _, res := range collected {
		runs += res.runs
		debug.DetectedPixels += res.debug.DetectedPixels
		debug.Candidates = append(debug.Candidates, res.recs...)
		if res.err != nil && firstErr == nil {
			firstErr = res.err
		}
	}
	if firstErr != nil {
		return nil, runs, debug, firstErr
	}

	// Межклассовые правила применяются к общей карте до извлечения результата.
	// Например, соприкасающийся pavilion целиком переводится в building;
	// поэтому он появляется в поиске зданий и одновременно исчезает из поиска
	// сооружений при одинаковом class_merge_rules в обеих логических моделях.
	if classMergeRulesRelevant(spec.ClassMergeRules, spec.DetectionClasses) {
		applyClassMergeRules(classMap, img.W, img.H, img.Geo, spec.ClassMergeRules)
	}

	// Из единой карты классов извлекаются только классы, запрошенные текущей
	// задачей. Поскольку карта общая, пересечения между классами невозможны.
	mask := make([]uint8, len(classMap))
	for i, encoded := range classMap {
		if encoded == 0 || img.ValidMask[i] == 0 {
			continue
		}
		classID := int(encoded) - 1
		if classAllowed(classID, spec.DetectionClasses) {
			mask[i] = 1
		}
	}

	// Пространственная морфология по отдельному выходному классу способна снова
	// создать пересечения. В exclusive_classes она намеренно отключена.
	if !spec.ExclusiveClasses {
		if spec.MorphOpen > 0 {
			mask = openMask(mask, img.W, img.H, spec.MorphOpen)
		}
		if spec.MorphClose > 0 {
			mask = closeMask(mask, img.W, img.H, spec.MorphClose)
		}
		if spec.FillHoles || spec.MaxHoleArea > 0 {
			mask = fillSmallHoles(mask, img.W, img.H, spec.MaxHoleArea)
		}
	}
	beforeArea := countMaskPixels(mask)
	candidate := removeSmallComponents(mask, img.W, img.H, int(spec.MinArea))
	if countMaskPixels(candidate) > 0 || beforeArea == 0 {
		mask = candidate
	}
	applyValidMask(mask, img.ValidMask)
	debug.FinalPixels = countMaskPixels(mask)
	return mask, runs, debug, nil
}
