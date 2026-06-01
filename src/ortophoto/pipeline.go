package main

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type ProcessConfig struct {
	App           *AppConfig
	Spec          ModelSpec
	SearchPolygon string
	StartYear     int
	EndYear       int
	OutputName    string
}

type ImageStat struct {
	File                  string        `json:"file"`
	Polygons              int           `json:"polygons"`
	Duration              time.Duration `json:"duration"`
	AvgPerImage           time.Duration `json:"avg_per_image"`
	Skipped               bool          `json:"skipped"`
	SkipReason            string        `json:"skip_reason,omitempty"`
	InteriorMax           float32       `json:"interior_max"`
	InteriorMean          float32       `json:"interior_mean"`
	BoundaryMax           float32       `json:"boundary_max"`
	BoundaryMean          float32       `json:"boundary_mean"`
	InteriorPixels        int           `json:"interior_pixels"`
	BoundaryPixels        int           `json:"boundary_pixels"`
	PixelsAfterSubtract   int           `json:"pixels_after_subtract"`
	PixelsAfterMorphology int           `json:"pixels_after_morphology"`
	PixelsAfterAreaFilter int           `json:"pixels_after_area_filter"`
	BoundaryDisabled      bool          `json:"boundary_disabled"`
	AdaptiveThreshold     float32       `json:"adaptive_threshold"`
}

type MaskDebugStats struct {
	InteriorMax           float32
	InteriorMean          float32
	BoundaryMax           float32
	BoundaryMean          float32
	InteriorPixels        int
	BoundaryPixels        int
	PixelsAfterSubtract   int
	PixelsAfterMorphology int
	PixelsAfterAreaFilter int
	BoundaryDisabled      bool
	AdaptiveThreshold     float32
}

func countMaskPixels(mask []uint8) int {
	n := 0
	for _, v := range mask {
		if v != 0 {
			n++
		}
	}
	return n
}

func probStats(prob []float32) (float32, float32) {
	if len(prob) == 0 {
		return 0, 0
	}
	maxv := prob[0]
	sum := float64(0)
	for _, v := range prob {
		if v > maxv {
			maxv = v
		}
		sum += float64(v)
	}
	return maxv, float32(sum / float64(len(prob)))
}

type ProcessStats struct {
	FilesSeen      int           `json:"files_seen"`
	FilesProcessed int           `json:"files_processed"`
	ModelRuns      int           `json:"model_runs"`
	ResultPolygons int           `json:"result_polygons"`
	TotalDuration  time.Duration `json:"total_duration"`
	AvgPerImage    time.Duration `json:"avg_per_image"`
	Images         []ImageStat   `json:"images"`
}

func listOrthophotoFiles(root string, startYear, endYear int) ([]string, error) {
	files := make([]string, 0)
	for year := startYear; year <= endYear; year++ {
		yearDir := filepath.Join(root, fmt.Sprintf("%04d", year))
		if _, err := os.Stat(yearDir); err != nil {
			log.Printf("ortophoto year directory skipped: year=%d path=%s reason=%v", year, yearDir, err)
			continue
		}
		log.Printf("ortophoto scanning year directory: year=%d path=%s", year, yearDir)
		err := filepath.WalkDir(yearDir, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			lower := strings.ToLower(d.Name())
			if strings.HasSuffix(lower, ".tif") || strings.HasSuffix(lower, ".tiff") {
				files = append(files, path)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Strings(files)
	log.Printf("ortophoto files collected: count=%d years=%d..%d root=%s", len(files), startYear, endYear, root)
	return files, nil
}

type patchInfo struct{ x, y int }

func collectPatches(img *OrthoImage, tile, step int) []patchInfo {
	patches := make([]patchInfo, 0)
	for y := 0; y < img.H; y += step {
		for x := 0; x < img.W; x += step {
			patches = append(patches, patchInfo{x: x, y: y})
			if x+tile >= img.W {
				break
			}
		}
		if y+tile >= img.H {
			break
		}
	}
	return patches
}

func extractPatch(img *OrthoImage, x, y, tile int) ([]float32, []uint8) {
	data := make([]float32, img.C*tile*tile)
	valid := make([]uint8, tile*tile)
	for c := 0; c < img.C; c++ {
		srcOff := c * img.W * img.H
		dstOff := c * tile * tile
		for yy := 0; yy < tile; yy++ {
			sy := y + yy
			if sy >= img.H {
				continue
			}
			for xx := 0; xx < tile; xx++ {
				sx := x + xx
				if sx >= img.W {
					continue
				}
				data[dstOff+yy*tile+xx] = img.Data[srcOff+sy*img.W+sx]
				valid[yy*tile+xx] = img.ValidMask[sy*img.W+sx]
			}
		}
	}
	return data, valid
}

func runModelOnImage(img *OrthoImage, sessions []*ORTSession, spec ModelSpec) ([]uint8, int, MaskDebugStats, error) {
	tile := spec.Tile
	if tile <= 0 {
		tile = 512
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

	interiorSum := make([]float32, img.W*img.H)
	boundarySum := make([]float32, img.W*img.H)
	counts := make([]float32, img.W*img.H)

	batchSize := spec.BatchSize
	if batchSize <= 0 {
		batchSize = 4
	}
	if len(sessions) == 0 {
		return nil, 0, MaskDebugStats{}, fmt.Errorf("no ONNX sessions configured")
	}

	type batchTask struct {
		id      int
		patches []patchInfo
		input   []float32
	}
	type batchResult struct {
		task     batchTask
		pred     []float32
		err      error
		workerID int
	}

	jobs := make(chan batchTask, len(sessions)*4)
	results := make(chan batchResult, len(sessions)*4)

	var wg sync.WaitGroup
	for workerID, session := range sessions {
		wg.Add(1)
		go func(id int, sess *ORTSession) {
			defer wg.Done()
			processed := 0
			for task := range jobs {
				pred, err := sess.Predict(task.input, len(task.patches), img.C, tile, tile, 2)
				processed++
				results <- batchResult{task: task, pred: pred, err: err, workerID: id}
			}
		}(workerID, session)
	}

	submittedCh := make(chan int, 1)
	go func() {
		defer close(jobs)
		submitted := 0
		batchPatches := make([]patchInfo, 0, batchSize)
		batchInput := make([]float32, 0, batchSize*img.C*tile*tile)

		flush := func() {
			if len(batchPatches) == 0 {
				return
			}
			inputCopy := make([]float32, len(batchInput))
			copy(inputCopy, batchInput)
			patchCopy := make([]patchInfo, len(batchPatches))
			copy(patchCopy, batchPatches)
			jobs <- batchTask{id: submitted, patches: patchCopy, input: inputCopy}
			submitted++
			batchPatches = batchPatches[:0]
			batchInput = batchInput[:0]
		}

		for _, p := range patches {
			patch, valid := extractPatch(img, p.x, p.y, tile)
			if isMostlyEmptyTile(
				patch, valid, img.C, tile, tile,
				float32(spec.EmptyTileValue),
				spec.EmptyTileWhiteRatio,
				spec.EmptyTileValidRatio,
			) {
				continue
			}
			batchPatches = append(batchPatches, p)
			batchInput = append(batchInput, patch...)
			if len(batchPatches) >= batchSize {
				flush()
			}
		}
		flush()
		submittedCh <- submitted
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	submitted := -1
	modelRuns := 0
	workerBatches := make(map[int]int)

	for {
		if submitted < 0 {
			select {
			case submitted = <-submittedCh:
				if submitted == 0 {
					break
				}
			case res, ok := <-results:
				if !ok {
					submitted = 0
					break
				}
				if res.err != nil {
					return nil, modelRuns, MaskDebugStats{}, res.err
				}
				modelRuns++
				workerBatches[res.workerID]++
				for bi, p := range res.task.patches {
					baseInt := bi * 2 * tile * tile
					baseBnd := baseInt + tile*tile
					accumulatePatchPrediction(img, tile, bound, p, res.pred, baseInt, baseBnd, interiorSum, boundarySum, counts)
				}
			}
		} else {
			break
		}
	}

	for res := range results {
		if res.err != nil {
			return nil, modelRuns, MaskDebugStats{}, res.err
		}
		modelRuns++
		workerBatches[res.workerID]++
		if modelRuns%50 == 0 {
		}
		for bi, p := range res.task.patches {
			baseInt := bi * 2 * tile * tile
			baseBnd := baseInt + tile*tile
			accumulatePatchPrediction(img, tile, bound, p, res.pred, baseInt, baseBnd, interiorSum, boundarySum, counts)
		}
	}

	interiorProb := make([]float32, img.W*img.H)
	boundaryProb := make([]float32, img.W*img.H)
	for i := 0; i < len(interiorProb); i++ {
		if counts[i] == 0 {
			continue
		}
		interiorProb[i] = interiorSum[i] / counts[i]
		boundaryProb[i] = boundarySum[i] / counts[i]
	}

	debug := MaskDebugStats{}
	debug.InteriorMax, debug.InteriorMean = probStats(interiorProb)
	debug.BoundaryMax, debug.BoundaryMean = probStats(boundaryProb)

	thr := spec.Threshold
	if thr <= 0 {
		thr = 0.5
	}
	interiorMask := thresholdMask(interiorProb, thr)
	debug.InteriorPixels = countMaskPixels(interiorMask)

	if debug.InteriorPixels == 0 && debug.InteriorMax > 0.05 {
		adaptive := debug.InteriorMax * 0.50
		if adaptive < 0.05 {
			adaptive = 0.05
		}
		if adaptive > thr {
			adaptive = thr
		}
		interiorMask = thresholdMask(interiorProb, adaptive)
		debug.InteriorPixels = countMaskPixels(interiorMask)
		debug.AdaptiveThreshold = adaptive
	}

	bthr := spec.BoundaryThreshold
	if bthr <= 0 {
		bthr = 0.35
	}
	boundaryMask := thresholdMask(boundaryProb, bthr)
	debug.BoundaryPixels = countMaskPixels(boundaryMask)

	mask := interiorMask
	if spec.BoundaryAware && debug.InteriorPixels > 0 {
		candidateBoundary := boundaryMask
		if debug.BoundaryPixels > 0 {
			candidateBoundary = dilateMask(candidateBoundary, img.W, img.H, 1)
		}
		candidate := andNotMask(interiorMask, candidateBoundary, img.ValidMask)
		candidatePixels := countMaskPixels(candidate)
		debug.PixelsAfterSubtract = candidatePixels

		if candidatePixels >= max(1, debug.InteriorPixels/20) {
			mask = candidate
		} else {
			debug.BoundaryDisabled = true
			mask = interiorMask
		}
	} else {
		debug.PixelsAfterSubtract = debug.InteriorPixels
	}
	applyValidMask(mask, img.ValidMask)

	beforeMorph := countMaskPixels(mask)
	if spec.MorphOpen > 0 {
		candidate := openMask(mask, img.W, img.H, spec.MorphOpen)
		if countMaskPixels(candidate) > 0 || beforeMorph == 0 {
			mask = candidate
		}
	}
	if spec.MorphClose > 0 {
		candidate := closeMask(mask, img.W, img.H, spec.MorphClose)
		if countMaskPixels(candidate) > 0 || countMaskPixels(mask) == 0 {
			mask = candidate
		}
	}
	debug.PixelsAfterMorphology = countMaskPixels(mask)

	beforeArea := countMaskPixels(mask)
	candidate := removeSmallComponents(mask, img.W, img.H, int(spec.MinArea))
	if countMaskPixels(candidate) > 0 || beforeArea == 0 {
		mask = candidate
	}
	applyValidMask(mask, img.ValidMask)
	debug.PixelsAfterAreaFilter = countMaskPixels(mask)

	outputMask := makeBoundaryOutputMask(mask, boundaryMask, img.W, img.H, spec)
	applyValidMask(outputMask, img.ValidMask)
	return outputMask, modelRuns, debug, nil
}

func accumulatePatchPrediction(
	img *OrthoImage,
	tile int,
	bound int,
	p patchInfo,
	pred []float32,
	baseInt int,
	baseBnd int,
	interiorSum []float32,
	boundarySum []float32,
	counts []float32,
) {
	y0, y1 := 0, tile
	x0, x1 := 0, tile
	if p.y > 0 {
		y0 = bound
	}
	if p.x > 0 {
		x0 = bound
	}
	if p.y+tile < img.H {
		y1 = tile - bound
	}
	if p.x+tile < img.W {
		x1 = tile - bound
	}

	for yy := y0; yy < y1; yy++ {
		sy := p.y + yy
		if sy >= img.H {
			continue
		}
		for xx := x0; xx < x1; xx++ {
			sx := p.x + xx
			if sx >= img.W {
				continue
			}
			idx := sy*img.W + sx
			if img.ValidMask[idx] == 0 {
				continue
			}
			interiorSum[idx] += sigmoid(pred[baseInt+yy*tile+xx])
			boundarySum[idx] += sigmoid(pred[baseBnd+yy*tile+xx])
			counts[idx] += 1
		}
	}
}

func makeBoundaryOutputMask(interiorMask []uint8, boundaryMask []uint8, w, h int, spec ModelSpec) []uint8 {
	mode := strings.ToLower(strings.TrimSpace(spec.ResultMode))

	// polygon / contour / parcels:
	// Возвращаем маску участков как площади. Polygonize по этой маске создаёт
	// замкнутые контуры отдельных участков. В интерфейсе такие polygon-геометрии
	// нужно рисовать без заливки, только outline.
	if mode == "" || mode == "polygon" || mode == "polygons" || mode == "contour" || mode == "contours" || mode == "parcel" || mode == "parcels" {
		out := make([]uint8, len(interiorMask))
		copy(out, interiorMask)
		return out
	}

	// boundary / lines:
	// Это диагностический режим. Он выводит тонкую boundary-маску, но такая
	// маска может быть разорванной и не является полноценным контуром участка.
	if mode == "boundary" || mode == "boundaries" || mode == "line" || mode == "lines" {
		if countMaskPixels(boundaryMask) > 0 {
			out := dilateMask(boundaryMask, w, h, 1)
			if countMaskPixels(out) > 0 {
				return out
			}
		}
		eroded := erodeMask(interiorMask, w, h, 1)
		out := make([]uint8, len(interiorMask))
		for i := range interiorMask {
			if interiorMask[i] != 0 && eroded[i] == 0 {
				out[i] = 1
			}
		}
		out = dilateMask(out, w, h, 1)
		return out
	}

	out := make([]uint8, len(interiorMask))
	copy(out, interiorMask)
	return out
}

func zipDir(srcDir, zipPath string) error {
	zf, err := os.Create(zipPath)
	if err != nil {
		return err
	}
	defer zf.Close()
	zw := zip.NewWriter(zf)
	defer zw.Close()
	return filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		hdr, err := zip.FileInfoHeader(info)
		if err != nil {
			return err
		}
		hdr.Name = rel
		hdr.Method = zip.Deflate
		w, err := zw.CreateHeader(hdr)
		if err != nil {
			return err
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = io.Copy(w, f)
		return err
	})
}

func countShapefileFeatures(shpPath string) (int, error) {
	g, err := LoadUnionGeometryFromVector(shpPath)
	if err != nil {
		return 0, err
	}
	defer DestroyGeometry(g)
	// Быстрый и грубый счёт по одному union geometry нам не подходит,
	// поэтому считаем через GDAL-открытие в отдельной функции ниже.
	return countVectorFeatures(shpPath)
}

func openORTSessions(app *AppConfig, spec ModelSpec, modelPath string) ([]*ORTSession, error) {
	deviceIDs := append([]int(nil), app.ORTDeviceIDs...)
	if len(deviceIDs) == 0 {
		deviceIDs = []int{app.ORTDeviceID}
	}
	if len(deviceIDs) == 0 {
		deviceIDs = []int{0}
	}
	log.Printf("ortophoto opening model sessions: model=%s path=%s input=%s output=%s device=%s gpus=%v", spec.Name, modelPath, spec.InputName, spec.OutputName, app.ORTDevice, deviceIDs)

	sessions := make([]*ORTSession, 0, len(deviceIDs))
	for _, gpuID := range deviceIDs {
		localCfg := *app
		localCfg.ORTDeviceID = gpuID
		sess, err := NewORTSession(&localCfg, modelPath, spec.InputName, spec.OutputName)
		if err != nil {
			closeORTSessions(sessions)
			return nil, err
		}
		sessions = append(sessions, sess)
	}
	return sessions, nil
}

func closeORTSessions(sessions []*ORTSession) {
	for _, s := range sessions {
		if s != nil {
			_ = s.Close()
		}
	}
}

func RunProcess(cfg ProcessConfig) (string, ProcessStats, error) {
	totalStart := time.Now()

	log.Printf("ortophoto loading search polygon: poligon=%s", cfg.SearchPolygon)
	allowedGeom, err := BuildAllowedGeometry(cfg.App, cfg.SearchPolygon, nil, nil)
	if err != nil {
		return "", ProcessStats{}, err
	}
	defer DestroyGeometry(allowedGeom)

	files, err := listOrthophotoFiles(cfg.App.Ortophoto, cfg.StartYear, cfg.EndYear)
	if err != nil {
		return "", ProcessStats{}, err
	}
	stats := ProcessStats{FilesSeen: len(files), Images: make([]ImageStat, 0, len(files))}
	if len(files) == 0 {
		return "", stats, fmt.Errorf("orthophoto files not found for selected years")
	}

	workDir, err := os.MkdirTemp(cfg.App.Tmp, "ortophoto-work-")
	if err != nil {
		return "", stats, err
	}
	defer os.RemoveAll(workDir)

	modelPath := cfg.App.ModelPath(cfg.Spec)
	log.Printf("ortophoto model configured: model=%s path=%s input=%s output=%s device=%s gpus=%v", cfg.Spec.Name, modelPath, cfg.Spec.InputName, cfg.Spec.OutputName, cfg.App.ORTDevice, cfg.App.ORTDeviceIDs)
	sessions, err := openORTSessions(cfg.App, cfg.Spec, modelPath)
	if err != nil {
		return "", stats, err
	}
	defer closeORTSessions(sessions)

	shpDir := filepath.Join(workDir, "Shp")
	if err := os.MkdirAll(shpDir, 0o755); err != nil {
		return "", stats, err
	}
	geojsonPath := filepath.Join(workDir, safeName(cfg.Spec.Name)+".geojson")
	if err := CreateEmptyGeoJSON(geojsonPath); err != nil {
		return "", stats, err
	}

	processedDurSum := time.Duration(0)
	for _, tif := range files {
		imgStart := time.Now()
		imgStat := ImageStat{File: tif}
		log.Printf("ortophoto image start: file=%s", filepath.Base(tif))

		img, err := ReadOrthophoto(tif)
		if err != nil {
			log.Printf("ortophoto read failed: file=%s error=%v", tif, err)
			imgStat.Skipped = true
			imgStat.SkipReason = err.Error()
			stats.Images = append(stats.Images, imgStat)
			continue
		}

		mask, runs, dbg, err := runModelOnImage(img, sessions, cfg.Spec)
		stats.ModelRuns += runs
		imgStat.InteriorMax = dbg.InteriorMax
		imgStat.InteriorMean = dbg.InteriorMean
		imgStat.BoundaryMax = dbg.BoundaryMax
		imgStat.BoundaryMean = dbg.BoundaryMean
		imgStat.InteriorPixels = dbg.InteriorPixels
		imgStat.BoundaryPixels = dbg.BoundaryPixels
		imgStat.PixelsAfterSubtract = dbg.PixelsAfterSubtract
		imgStat.PixelsAfterMorphology = dbg.PixelsAfterMorphology
		imgStat.PixelsAfterAreaFilter = dbg.PixelsAfterAreaFilter
		imgStat.BoundaryDisabled = dbg.BoundaryDisabled
		imgStat.AdaptiveThreshold = dbg.AdaptiveThreshold
		if err != nil {
			return "", stats, err
		}
		log.Printf("ortophoto inference: file=%s batches=%d int_pixels=%d bnd_pixels=%d out_pixels=%d", filepath.Base(tif), runs, dbg.InteriorPixels, dbg.BoundaryPixels, dbg.PixelsAfterAreaFilter)

		// Быстрый пропуск пустого результата
		nonZero := 0
		for _, v := range mask {
			if v != 0 {
				nonZero++
			}
		}
		if nonZero == 0 {
			imgStat.Skipped = true
			imgStat.SkipReason = "empty mask"
			imgStat.Duration = time.Since(imgStart)
			processedDurSum += imgStat.Duration
			stats.FilesProcessed++
			imgStat.AvgPerImage = processedDurSum / time.Duration(stats.FilesProcessed)
			stats.Images = append(stats.Images, imgStat)
			log.Printf("ortophoto image processed: file=%s polygons=0 duration=%s avg_per_image=%s", tif, imgStat.Duration.String(), imgStat.AvgPerImage.String())
			continue
		}

		maskTif := filepath.Join(workDir, safeName(filepath.Base(tif))+"_mask.tif")
		if err := WriteGeoTIFF1(maskTif, toFloatMask(mask), img.W, img.H, img.Geo, img.Proj); err != nil {
			return "", stats, err
		}

		fileShp := filepath.Join(workDir, safeName(strings.TrimSuffix(filepath.Base(tif), filepath.Ext(tif)))+"_tmp.shp")
		if err := PolygonizeMaskToShapefile(maskTif, fileShp, cfg.Spec.MinArea, cfg.Spec.Simplify); err != nil {
			log.Printf("ortophoto polygonize failed: file=%s error=%v", tif, err)
			imgStat.Skipped = true
			imgStat.SkipReason = err.Error()
			stats.Images = append(stats.Images, imgStat)
			continue
		}

		_ = os.Remove(maskTif)
		_, unclippedErr := countShapefileFeatures(fileShp)
		if unclippedErr != nil {
			log.Printf("ortophoto unclipped feature count failed: file=%s error=%v", tif, unclippedErr)
		}
		clippedShp := filepath.Join(workDir, safeName(strings.TrimSuffix(filepath.Base(tif), filepath.Ext(tif)))+"_clip_tmp.shp")
		if err := ClipShapefileToGeometry(fileShp, clippedShp, allowedGeom); err != nil {
			return "", stats, err
		}
		polyCount, err := countShapefileFeatures(clippedShp)
		if err != nil {
			return "", stats, err
		}
		log.Printf("ortophoto vectors: file=%s features=%d", filepath.Base(tif), polyCount)
		if polyCount > 0 {
			if err := AppendShapefileToGeoJSON(clippedShp, geojsonPath); err != nil {
				return "", stats, err
			}
		}
		removeShapefileSet(fileShp)
		removeShapefileSet(clippedShp)

		imgStat.Polygons = polyCount
		imgStat.Duration = time.Since(imgStart)
		processedDurSum += imgStat.Duration
		stats.FilesProcessed++
		stats.ResultPolygons += polyCount
		imgStat.AvgPerImage = processedDurSum / time.Duration(stats.FilesProcessed)
		stats.Images = append(stats.Images, imgStat)

		log.Printf("ortophoto image done: file=%s features=%d duration=%s avg=%s", filepath.Base(tif), polyCount, imgStat.Duration.String(), imgStat.AvgPerImage.String())
	}

	stats.TotalDuration = time.Since(totalStart)
	if stats.FilesProcessed > 0 {
		stats.AvgPerImage = processedDurSum / time.Duration(stats.FilesProcessed)
	}

	resultShp := filepath.Join(shpDir, safeName(cfg.Spec.Name)+".shp")
	if err := ExportGeoJSONToShapefile(geojsonPath, resultShp); err != nil {
		return "", stats, err
	}

	manifest := map[string]any{
		"model":           cfg.Spec.Name,
		"start_year":      cfg.StartYear,
		"end_year":        cfg.EndYear,
		"poligon":         cfg.SearchPolygon,
		"files_seen":      stats.FilesSeen,
		"files_processed": stats.FilesProcessed,
		"model_runs":      stats.ModelRuns,
		"result_polygons": stats.ResultPolygons,
		"total_duration":  stats.TotalDuration.String(),
		"avg_per_image":   stats.AvgPerImage.String(),
		"images":          stats.Images,
	}
	manifestPath := filepath.Join(workDir, "result.json")
	blob, _ := json.MarshalIndent(manifest, "", "  ")
	if err := os.WriteFile(manifestPath, blob, 0o644); err != nil {
		return "", stats, err
	}

	zipPath := filepath.Join(cfg.App.ResultsPath, safeName(cfg.OutputName)+".zip")
	if err := zipDir(workDir, zipPath); err != nil {
		return "", stats, err
	}
	return zipPath, stats, nil
}
