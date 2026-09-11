package main

import (
	"fmt"
	"math"
	"sync"
)

// semanticMergeWindow returns the trusted part of a semantic tile. Unlike the
// old core-only scheme, neighbouring tiles are intentionally allowed to overlap
// here; their probabilities are merged by max().  Only a small outer strip is
// ignored on internal tile borders because CNN predictions immediately at the
// edge of the input are the least reliable. Raster outer borders are kept.
func semanticMergeWindow(img *OrthoImage, p patchInfo, tile, edgeIgnore int) (x0, y0, x1, y1 int) {
	x0, y0, x1, y1 = 0, 0, tile, tile
	if edgeIgnore < 0 {
		edgeIgnore = 0
	}
	if 2*edgeIgnore >= tile {
		edgeIgnore = 0
	}
	if p.x > 0 {
		x0 = edgeIgnore
	}
	if p.y > 0 {
		y0 = edgeIgnore
	}
	if p.x+tile < img.W {
		x1 = tile - edgeIgnore
	}
	if p.y+tile < img.H {
		y1 = tile - edgeIgnore
	}
	if x1 < x0 {
		x1 = x0
	}
	if y1 < y0 {
		y1 = y0
	}
	return
}

func semanticAutoEdgeIgnore(bound int) int {
	if bound <= 0 {
		return 0
	}
	v := bound / 4
	if v < 4 {
		v = 4
	}
	if v > 16 {
		v = 16
	}
	return v
}

func semanticClassThreshold(spec ModelSpec, classID int) float32 {
	if spec.ClassThresholds != nil {
		if v, ok := spec.ClassThresholds[fmt.Sprintf("%d", classID)]; ok && v >= 0 && v <= 1 {
			return v
		}
	}
	return spec.SemanticThreshold
}

// runSemanticOnImage runs a true dense semantic segmentation model. Unlike
// YOLO-seg, there are no instances, bounding boxes, NMS or prototype masks:
// every output pixel receives a supervised logit from the ONNX graph.
//
// Input:  [1, img.C, tile, tile], float32 after the same ReadOrthophoto
// normalization used by the rest of the ortophoto module.
// Output: [1, NumClasses, tile, tile], logits.
func runSemanticOnImage(img *OrthoImage, sessions []*ORTSession, spec ModelSpec) ([]uint8, int, MaskDebugStats, error) {
	tile := spec.Tile
	if tile <= 0 {
		tile = 512
	}
	bound := spec.Bound
	if bound < 0 {
		bound = 0
	}
	if 2*bound >= tile {
		return nil, 0, MaskDebugStats{}, fmt.Errorf("semantic bound must be < tile/2: tile=%d bound=%d", tile, bound)
	}
	step := tile - 2*bound
	patches := collectPatches(img, tile, step)
	if len(sessions) == 0 {
		return nil, 0, MaskDebugStats{}, fmt.Errorf("no ONNX Runtime sessions opened")
	}
	if spec.NumClasses <= 0 {
		return nil, 0, MaskDebugStats{}, fmt.Errorf("semantic num_classes must be positive")
	}

	// Store the best probability per class for each raster pixel. IMPORTANT:
	// neighbouring semantic tiles overlap and are merged by max probability.
	// The previous implementation wrote only disjoint cores [bound,tile-bound],
	// which created hard vertical/horizontal seams: on one side of the seam the
	// decision came exclusively from tile A and on the other exclusively from
	// tile B. Dense segmentation outputs are not guaranteed to be identical for
	// the same ground pixel under different tile context. Overlap+max removes
	// this artificial discontinuity while preserving the calibrated threshold.
	plane := img.W * img.H
	probs := make([]float32, spec.NumClasses*plane)
	for i := range probs {
		probs[i] = -1
	}

	type job struct {
		idx int
		p   patchInfo
	}
	type workerResult struct {
		runs int
		err  error
	}
	jobs := make(chan job)
	results := make(chan workerResult, len(sessions))
	var mu sync.Mutex

	for workerID, sess := range sessions {
		go func(workerID int, sess *ORTSession) {
			res := workerResult{}
			for j := range jobs {
				patch, valid := extractPatch(img, j.p.x, j.p.y, tile)
				if spec.SkipEmptyTiles && !hasAnyValidPixel(valid) {
					continue
				}
				logits, err := sess.PredictSemantic(patch, img.C, tile, tile, spec.NumClasses)
				if err != nil {
					res.err = fmt.Errorf("gpu worker %d semantic patch %d (%d,%d): %w", workerID, j.idx, j.p.x, j.p.y, err)
					break
				}
				res.runs++
				expected := spec.NumClasses * tile * tile
				if len(logits) != expected {
					res.err = fmt.Errorf("semantic output size mismatch: got=%d expected=%d for [1,%d,%d,%d]", len(logits), expected, spec.NumClasses, tile, tile)
					break
				}

				edgeIgnore := spec.SemanticEdgeIgnore
				if edgeIgnore <= 0 {
					edgeIgnore = semanticAutoEdgeIgnore(bound)
				}
				x0, y0, x1, y1 := semanticMergeWindow(img, j.p, tile, edgeIgnore)
				mu.Lock()
				for yy := y0; yy < y1; yy++ {
					gy := j.p.y + yy
					if gy < 0 || gy >= img.H {
						continue
					}
					for xx := x0; xx < x1; xx++ {
						gx := j.p.x + xx
						if gx < 0 || gx >= img.W {
							continue
						}
						gidx := gy*img.W + gx
						if img.ValidMask[gidx] == 0 {
							continue
						}
						lidx := yy*tile + xx
						for c := 0; c < spec.NumClasses; c++ {
							v := logits[c*tile*tile+lidx]
							// Stable sigmoid for logits exported by BCEWithLogits-style training.
							var pv float32
							if v >= 0 {
								z := float32(math.Exp(float64(-v)))
								pv = 1 / (1 + z)
							} else {
								z := float32(math.Exp(float64(v)))
								pv = z / (1 + z)
							}
							pi := c*plane + gidx
							if pv > probs[pi] {
								probs[pi] = pv
							}
						}
					}
				}
				mu.Unlock()
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
	for _, r := range collected {
		runs += r.runs
		if r.err != nil && firstErr == nil {
			firstErr = r.err
		}
	}
	if firstErr != nil {
		return nil, runs, MaskDebugStats{}, firstErr
	}

	// Optional raw probability map for diagnosing tile/file seams. For a binary
	// model this is the merged hogweed probability. For multi-class models it is
	// the maximum class probability per pixel.
	var probMap []uint8
	if spec.DumpProbMap {
		probMap = make([]uint8, plane)
		for i := 0; i < plane; i++ {
			best := float32(0)
			for c := 0; c < spec.NumClasses; c++ {
				p := probs[c*plane+i]
				if p > best {
					best = p
				}
			}
			if best < 0 {
				best = 0
			}
			if best > 1 {
				best = 1
			}
			probMap[i] = uint8(best*255 + 0.5)
		}
	}

	// Convert probabilities to the same encoded class map used by YOLO path.
	// For binary hogweed NumClasses=1 this is simply thresholding. For future
	// multi-label semantic models, the highest probability among classes that
	// pass their threshold wins each pixel.
	classMap := make([]uint16, plane)
	detected := 0
	for i := 0; i < plane; i++ {
		if img.ValidMask[i] == 0 {
			continue
		}
		bestClass := -1
		bestProb := float32(-1)
		for c := 0; c < spec.NumClasses; c++ {
			p := probs[c*plane+i]
			if p < semanticClassThreshold(spec, c) {
				continue
			}
			if p > bestProb {
				bestProb = p
				bestClass = c
			}
		}
		if bestClass >= 0 {
			classMap[i] = uint16(bestClass + 1)
			detected++
		}
	}

	if classMergeRulesRelevant(spec.ClassMergeRules, spec.DetectionClasses) {
		applyClassMergeRules(classMap, img.W, img.H, img.Geo, img.Proj, spec.ClassMergeRules)
	}

	mask := make([]uint8, plane)
	for i, encoded := range classMap {
		if encoded == 0 || img.ValidMask[i] == 0 {
			continue
		}
		classID := int(encoded) - 1
		if classAllowed(classID, spec.DetectionClasses) {
			mask[i] = 1
		}
	}

	// For a true semantic mask morphology is safe for a single requested class.
	// For multi-class exclusive outputs keep the same conservative rule as YOLO.
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
	finalPixels := countMaskPixels(mask)

	debug := MaskDebugStats{DetectedPixels: detected, FinalPixels: finalPixels, ProbMap: probMap}
	return mask, runs, debug, nil
}
