package main

import (
	"archive/zip"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

// ImageStat описывает обработку одного листа ортофото в result.json.
// Оставлены только поля, которые реально заполняются YOLO-конвейером;
// прежний набор (interior_max/mean, boundary_*, adaptive_threshold и т.д.)
// достался от UNet-конвейера sentinel-модуля и в result.json всегда
// содержал нули, вводя в заблуждение при отладке.
type ImageStat struct {
	File           string        `json:"file"`
	Polygons       int           `json:"polygons"`
	Duration       time.Duration `json:"duration"`
	AvgPerImage    time.Duration `json:"avg_per_image"`
	Skipped        bool          `json:"skipped"`
	SkipReason     string        `json:"skip_reason,omitempty"`
	DetectedPixels int           `json:"detected_pixels"`
	FinalPixels    int           `json:"final_pixels"`
}

type MaskDebugStats struct {
	// DetectedPixels — пиксели, записанные в маску candidate-масками YOLO
	// (до морфологии и фильтра площади).
	DetectedPixels int
	// FinalPixels — пиксели маски после морфологии, заливки дыр и фильтра
	// малых компонент (то, что уходит в полигонизацию).
	FinalPixels int
	// Candidates — диагностические записи по кандидатам (только при
	// spec.DumpCandidates).
	Candidates []candidateRecord
}

// writeCandidateCSV пишет диагностику кандидатов одного листа: координаты
// центра бокса в пикселях листа и в СК листа (через geotransform), сырые
// размеры бокса, score, класс, заполнение бокса маской и решение фильтров.
func writeCandidateCSV(path string, recs []candidateRecord, geo [6]float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	fmt.Fprintln(w, "patch_x,patch_y,cx_px,cy_px,geo_x,geo_y,raw_w,raw_h,score,class,fill,decision")
	for _, r := range recs {
		gx := geo[0] + r.cxPx*geo[1] + r.cyPx*geo[2]
		gy := geo[3] + r.cxPx*geo[4] + r.cyPx*geo[5]
		fmt.Fprintf(w, "%d,%d,%.1f,%.1f,%.3f,%.3f,%.1f,%.1f,%.4f,%d,%.4f,%s\n",
			r.patchX, r.patchY, r.cxPx, r.cyPx, gx, gy, r.rawW, r.rawH, r.score, r.classID, r.fill, r.decision)
	}
	return nil
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
				log.Printf("ortophoto scan entry failed: path=%s error=%v", path, err)
				return err
			}
			if d.IsDir() {
				return nil
			}
			// TrimSpace makes collection tolerant of accidental trailing spaces in
			// a filename after the extension. Log every discovered raster with its
			// full path so an allegedly missing leaf can be verified unambiguously.
			lower := strings.ToLower(strings.TrimSpace(d.Name()))
			if strings.HasSuffix(lower, ".tif") || strings.HasSuffix(lower, ".tiff") {
				files = append(files, path)
				log.Printf("ortophoto file discovered: path=%s", path)
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

// axisOffsets возвращает смещения тайлов вдоль одной оси растра.
//
// ВАЖНО (исправление пропуска сегментов зданий): последний тайл в ряду/столбце
// теперь ПРИЖИМАЕТСЯ к правому/нижнему краю растра (offset = size - tile),
// а не начинается на очередном шаге сетки и не свисает за край снимка.
//
// Раньше последний тайл выходил за границу растра, и недостающая часть
// заполнялась в extractPatch искусственным фоном (нулями). Модель YOLO11-seg
// обучалась на тайлах, целиком вырезанных из снимка, поэтому на таких
// «полупустых» тайлах систематически теряла объекты. При этом полоса растра
// шириной до step пикселей у правого/нижнего края покрывается ТОЛЬКО этим
// последним тайлом — в результате у зданий, попадающих в эту полосу
// (в т.ч. зданий на стыке соседних листов ортофото), пропадали целые
// прямоугольные сегменты с идеально ровными границами по линиям сетки тайлов.
//
// Теперь каждый тайл целиком лежит внутри растра (если растр не меньше тайла),
// модель всегда получает полностью «настоящие» пиксели, а перекрытие соседних
// тайлов у края лишь увеличивается, что безопасно: маски объединяются по OR.
func axisOffsets(size, tile, step int) []int {
	if size <= tile {
		return []int{0}
	}
	offsets := make([]int, 0, size/step+2)
	for v := 0; ; v += step {
		if v+tile >= size {
			last := size - tile
			if len(offsets) == 0 || offsets[len(offsets)-1] != last {
				offsets = append(offsets, last)
			}
			break
		}
		offsets = append(offsets, v)
	}
	return offsets
}

func collectPatches(img *OrthoImage, tile, step int) []patchInfo {
	xs := axisOffsets(img.W, tile, step)
	ys := axisOffsets(img.H, tile, step)
	patches := make([]patchInfo, 0, len(xs)*len(ys))
	for _, y := range ys {
		for _, x := range xs {
			patches = append(patches, patchInfo{x: x, y: y})
		}
	}
	return patches
}

func extractPatch(img *OrthoImage, x, y, tile int) ([]float32, []uint8) {
	data := make([]float32, img.C*tile*tile)
	valid := make([]uint8, tile*tile)
	// Область за пределами растра заполняем белым (1.0) — той же «краской»,
	// которой ReadOrthophoto помечает nodata-пиксели. Раньше заполнение было
	// нулями (чёрным): такой фон модель не видела при обучении, что снижало
	// качество детекции на краевых тайлах. После исправления axisOffsets
	// заполнение вообще требуется только когда растр меньше тайла.
	for i := range data {
		data[i] = 1
	}
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
	return runYOLOSegOnImage(img, sessions, spec)
}

func zipDir(srcDir, zipPath string) error {
	return zipDirExcluding(srcDir, zipPath, "")
}

// zipDirExcluding архивирует srcDir в zipPath, пропуская файл excludePath
// (если он не пустой). Используется чтобы не класть в итоговый архив
// технический result.json, который нужен только для отладки.
func zipDirExcluding(srcDir, zipPath, excludePath string) error {
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
		if excludePath != "" && path == excludePath {
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

func openORTSessions(app *AppConfig, spec ModelSpec, modelPath string) ([]*ORTSession, error) {
	deviceIDs := append([]int(nil), app.Compute.GPUs...)
	if len(deviceIDs) == 0 {
		deviceIDs = []int{app.Compute.DeviceID}
	}
	if len(deviceIDs) == 0 {
		deviceIDs = []int{0}
	}
	outputNames := append([]string(nil), spec.OutputNames...)
	if len(outputNames) == 0 {
		outputNames = []string{"output0", "output1"}
	}
	log.Printf("ortophoto opening model sessions: model=%s path=%s input=%s outputs=%v device=%s gpus=%v", spec.Name, modelPath, spec.InputName, outputNames, app.Compute.Device, deviceIDs)

	sessions := make([]*ORTSession, 0, len(deviceIDs))
	for _, gpuID := range deviceIDs {
		localCfg := *app
		localCfg.Compute.DeviceID = gpuID
		localCfg.Compute.GPUs = []int{gpuID}
		sess, err := NewORTSessionWithOutputs(&localCfg, modelPath, spec.InputName, outputNames)
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
	log.Printf("ortophoto YOLO11 model configured: model=%s path=%s input=%s outputs=%v device=%s gpus=%v", cfg.Spec.Name, modelPath, cfg.Spec.InputName, cfg.Spec.OutputNames, cfg.App.Compute.Device, cfg.App.Compute.GPUs)
	sessions, err := openORTSessions(cfg.App, cfg.Spec, modelPath)
	if err != nil {
		return "", stats, err
	}
	defer closeORTSessions(sessions)

	// Результаты разделяются ПО CRS ИСХОДНЫХ СНИМКОВ, а не дублируются в обе
	// системы координат. Снимки МСК-59 попадают только в MSK-накопитель;
	// geographic/другие немсковские CRS — только в WGS84-накопитель.
	// При этом метрические min_area_m2/simplify для любой исходной CRS считаются
	// во временной МСК-59 выбранной зоны (см. Append...ByGroup).
	shpDir := filepath.Join(workDir, "_internal_shp")
	if err := os.MkdirAll(shpDir, 0o755); err != nil {
		return "", stats, err
	}
	baseName := safeName(cfg.Spec.Name)
	mskResultShp := filepath.Join(shpDir, baseName+"_msk.shp")
	wgsResultShp := filepath.Join(shpDir, baseName+"_wgs.shp")
	searchEnv := GeometryEnvelope(allowedGeom)
	targetZone := MSK59ZoneForLongitude((searchEnv.MinX + searchEnv.MaxX) / 2)

	mskSRS, err := NewMSK59SRS(targetZone)
	if err != nil {
		return "", stats, fmt.Errorf("create target MSK-59 zone %d: %w", targetZone, err)
	}
	if err := CreateEmptyAccumulatorShapefile(mskResultShp, mskSRS); err != nil {
		DestroySRS(mskSRS)
		return "", stats, err
	}
	DestroySRS(mskSRS)

	wgsSRS, err := NewWGS84SRS()
	if err != nil {
		return "", stats, err
	}
	if err := CreateEmptyAccumulatorShapefile(wgsResultShp, wgsSRS); err != nil {
		DestroySRS(wgsSRS)
		return "", stats, err
	}
	DestroySRS(wgsSRS)

	seenMSKSource := false
	seenWGSSource := false
	mskResultPolygons := 0
	wgsResultPolygons := 0
	log.Printf("ortophoto metric CRS for filters: MSK-59 zone=%d search_center_lon=%.6f", targetZone, (searchEnv.MinX+searchEnv.MaxX)/2)

	processedDurSum := time.Duration(0)
	for _, tif := range files {
		imgStart := time.Now()
		imgStat := ImageStat{File: tif}
		log.Printf("ortophoto image start: file=%s path=%s", filepath.Base(tif), tif)

		info, err := ReadOrthophotoInfo(tif)
		if err != nil {
			log.Printf("ortophoto metadata read failed: file=%s error=%v", tif, err)
			imgStat.Skipped = true
			imgStat.SkipReason = err.Error()
			stats.Images = append(stats.Images, imgStat)
			continue
		}

		sourceGroup, sourceCRSLabel, err := ClassifySourceCRS(info.Proj)
		if err != nil {
			log.Printf("ortophoto CRS classify failed: file=%s error=%v", tif, err)
			imgStat.Skipped = true
			imgStat.SkipReason = "unsupported/unknown CRS: " + err.Error()
			stats.Images = append(stats.Images, imgStat)
			continue
		}
		log.Printf("ortophoto source CRS: file=%s group=%s detail=%s", filepath.Base(tif), sourceGroup, sourceCRSLabel)

		intersects, err := RasterIntersectsAllowedWGS84(info.Geo, info.W, info.H, info.Proj, allowedGeom)
		if err != nil {
			log.Printf("ortophoto spatial prefilter failed, image will be processed: file=%s error=%v", tif, err)
			intersects = true
		}
		if !intersects {
			imgStat.Skipped = true
			imgStat.SkipReason = "outside search polygon"
			imgStat.Duration = time.Since(imgStart)
			stats.Images = append(stats.Images, imgStat)
			log.Printf("ortophoto image skipped by spatial prefilter: file=%s path=%s", filepath.Base(tif), tif)
			continue
		}

		switch sourceGroup {
		case SourceCRSMSK59:
			seenMSKSource = true
		case SourceCRSWGS84:
			seenWGSSource = true
		}

		img, err := ReadOrthophoto(tif, cfg.Spec.Normalization)
		if err != nil {
			log.Printf("ortophoto read failed: file=%s error=%v", tif, err)
			imgStat.Skipped = true
			imgStat.SkipReason = err.Error()
			stats.Images = append(stats.Images, imgStat)
			continue
		}

		mask, runs, dbg, err := runModelOnImage(img, sessions, cfg.Spec)
		stats.ModelRuns += runs
		imgStat.DetectedPixels = dbg.DetectedPixels
		imgStat.FinalPixels = dbg.FinalPixels
		if err != nil {
			return "", stats, err
		}
		if cfg.Spec.DumpCandidates {
			csvPath := filepath.Join(workDir, safeName(filepath.Base(tif))+"_candidates.csv")
			if cerr := writeCandidateCSV(csvPath, dbg.Candidates, img.Geo); cerr != nil {
				log.Printf("ortophoto candidate dump failed: file=%s error=%v", tif, cerr)
			} else {
				log.Printf("ortophoto candidate dump: %s (%d candidates)", csvPath, len(dbg.Candidates))
			}
		}
		log.Printf("ortophoto inference: file=%s tiles=%d detected_pixels=%d final_pixels=%d", filepath.Base(tif), runs, dbg.DetectedPixels, dbg.FinalPixels)

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
		if err := WriteGeoTIFFMask(maskTif, mask, img.W, img.H, img.Geo, img.Proj); err != nil {
			return "", stats, err
		}

		fileShp := filepath.Join(workDir, safeName(strings.TrimSuffix(filepath.Base(tif), filepath.Ext(tif)))+"_tmp.shp")
		// Полигонизация выполняется в РОДНОЙ CRS снимка без метрических операций.
		// min_area_m2 и simplify применяются позже, после перевода геометрии в
		// целевую МСК-59, где единицы гарантированно являются метрами.
		if err := PolygonizeMaskToShapefile(maskTif, fileShp, 0, 0); err != nil {
			log.Printf("ortophoto polygonize failed: file=%s error=%v", tif, err)
			imgStat.Skipped = true
			imgStat.SkipReason = err.Error()
			stats.Images = append(stats.Images, imgStat)
			continue
		}

		_ = os.Remove(maskTif)
		clippedShp := filepath.Join(workDir, safeName(strings.TrimSuffix(filepath.Base(tif), filepath.Ext(tif)))+"_clip_tmp.shp")
		// Передаём img.Proj — полный WKT проекции снимка (BOUNDCRS с датум-сдвигом
		// Pulkovo1942->WGS84) напрямую, а не полагаемся на СК, перечитанную из
		// .prj файла fileShp: при записи в ESRI Shapefile GDAL теряет параметры
		// датум-трансформации, из-за чего сверка с шаблоном полигона поиска
		// (хранится в WGS84) ломается и результат становится пустым.
		if err := ClipShapefileToGeometryWithSRSWKT(fileShp, clippedShp, allowedGeom, img.Proj); err != nil {
			return "", stats, err
		}
		nativeCount, err := countVectorFeatures(clippedShp)
		if err != nil {
			return "", stats, err
		}
		polyCount := 0
		if nativeCount > 0 {
			dstAccumulator := wgsResultShp
			if sourceGroup == SourceCRSMSK59 {
				dstAccumulator = mskResultShp
			}
			polyCount, err = AppendShapefileToAccumulatorFilteredByGroup(
				clippedShp, dstAccumulator, img.Proj, targetZone, sourceGroup, cfg.Spec.MinAreaM2, cfg.Spec.Simplify,
			)
			if err != nil {
				return "", stats, err
			}
			if sourceGroup == SourceCRSMSK59 {
				mskResultPolygons += polyCount
			} else {
				wgsResultPolygons += polyCount
			}
		}
		log.Printf("ortophoto vectors: file=%s source_group=%s native_features=%d accepted_features=%d", filepath.Base(tif), sourceGroup, nativeCount, polyCount)
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

	// Публичные GeoJSON создаются только для тех групп CRS, которые реально
	// присутствовали среди обрабатываемых снимков. Один и тот же результат не
	// преобразуется и не копируется во второй файл: принадлежность определяется
	// исключительно CRS исходного GeoTIFF.
	generatedOutputs := make([]string, 0, 2)
	if seenMSKSource {
		mskGeoJSON := filepath.Join(workDir, fmt.Sprintf("%s_MSK59-%d.geojson", baseName, targetZone))
		if err := ExportShapefileToGeoJSON(mskResultShp, mskGeoJSON); err != nil {
			return "", stats, err
		}
		generatedOutputs = append(generatedOutputs, filepath.Base(mskGeoJSON))
	}
	if seenWGSSource {
		wgsGeoJSON := filepath.Join(workDir, baseName+"_WGS84_EPSG4326.geojson")
		// wgsResultShp уже содержит только результаты снимков WGS/geographic/
		// немсковских projected CRS, нормализованные в EPSG:4326.
		if err := ExportShapefileToGeoJSON(wgsResultShp, wgsGeoJSON); err != nil {
			return "", stats, err
		}
		generatedOutputs = append(generatedOutputs, filepath.Base(wgsGeoJSON))
	}
	if !seenMSKSource && !seenWGSSource {
		return "", stats, fmt.Errorf("no orthophoto source CRS groups were processed")
	}

	// Shapefile остаётся только внутренним транспортным форматом.
	if err := os.RemoveAll(shpDir); err != nil {
		return "", stats, err
	}
	log.Printf("ortophoto result groups: msk_source=%t msk_polygons=%d wgs_source=%t wgs_polygons=%d outputs=%v", seenMSKSource, mskResultPolygons, seenWGSSource, wgsResultPolygons, generatedOutputs)

	manifest := map[string]any{
		"model":               cfg.Spec.Name,
		"start_year":          cfg.StartYear,
		"end_year":            cfg.EndYear,
		"poligon":             cfg.SearchPolygon,
		"files_seen":          stats.FilesSeen,
		"files_processed":     stats.FilesProcessed,
		"model_runs":          stats.ModelRuns,
		"result_polygons":     stats.ResultPolygons,
		"msk_result_polygons": mskResultPolygons,
		"wgs_result_polygons": wgsResultPolygons,
		"has_msk_sources":     seenMSKSource,
		"has_wgs_sources":     seenWGSSource,
		"output_files":        generatedOutputs,
		"total_duration":      stats.TotalDuration.String(),
		"avg_per_image":       stats.AvgPerImage.String(),
		"images":              stats.Images,
	}
	// result.json пишем внутри workDir для удобства отладки (он будет удалён
	// вместе с workDir по defer), но НЕ включаем его в итоговый zip-архив —
	// в архиве находятся только GeoJSON тех CRS-групп, которые присутствовали
	// во входных снимках (и debug CSV, если включён dump_candidates).
	manifestPath := filepath.Join(workDir, "result.json")
	blob, _ := json.MarshalIndent(manifest, "", "  ")
	if err := os.WriteFile(manifestPath, blob, 0o644); err != nil {
		return "", stats, err
	}

	zipPath := filepath.Join(cfg.App.ResultsPath, safeName(cfg.OutputName)+".zip")
	if err := zipDirExcluding(workDir, zipPath, manifestPath); err != nil {
		return "", stats, err
	}
	return zipPath, stats, nil
}
