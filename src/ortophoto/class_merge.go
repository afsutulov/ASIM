package main

import (
	"math"
	"strings"
)

// ClassMergeRule promotes whole connected components of SourceClasses to
// TargetClass when they touch or lie within DistanceM of the target class.
// The rule is applied to the common exclusive class map before a logical
// model extracts its requested output classes. Consequently the same rule
// makes the promoted geometry appear in the target search and disappear
// from the source-class search.
type ClassMergeRule struct {
	TargetClass   int     `json:"target_class"`
	SourceClasses []int   `json:"source_classes"`
	DistanceM     float64 `json:"distance_m,omitempty"`
}

func pixelSizeMetres(geo [6]float64, w, h int, proj string) float64 {
	// Для geographic CRS geotransform выражен в градусах. Переводим длины
	// базисных векторов пикселя в метры в центре снимка. Для небольших БПЛА
	// листов локальная аппроксимация WGS84 значительно точнее требуемых
	// расстояний class_merge_rules (десятки сантиметров).
	upper := strings.ToUpper(proj)
	geographic := strings.Contains(upper, "GEOGCRS[") && !strings.Contains(upper, "PROJCRS[")
	if geographic {
		cx, cy := float64(w)/2, float64(h)/2
		lat := geo[3] + cx*geo[4] + cy*geo[5]
		latRad := lat * math.Pi / 180
		metresPerDegLat := 111132.92 - 559.82*math.Cos(2*latRad) + 1.175*math.Cos(4*latRad)
		metresPerDegLon := 111412.84*math.Cos(latRad) - 93.5*math.Cos(3*latRad)
		sx := math.Hypot(geo[1]*metresPerDegLon, geo[4]*metresPerDegLat)
		sy := math.Hypot(geo[2]*metresPerDegLon, geo[5]*metresPerDegLat)
		if sx > 0 && sy > 0 {
			return (sx + sy) / 2
		}
	}

	// Projected CRS: basis vectors are expected in metres (МСК-59/UTM etc.).
	sx := math.Hypot(geo[1], geo[4])
	sy := math.Hypot(geo[2], geo[5])
	if sx <= 0 {
		sx = sy
	}
	if sy <= 0 {
		sy = sx
	}
	if sx <= 0 || sy <= 0 {
		return 1
	}
	return (sx + sy) / 2
}

func containsInt(values []int, value int) bool {
	for _, v := range values {
		if v == value {
			return true
		}
	}
	return false
}

// classMergeRulesRelevant reports whether merge rules can affect the classes
// requested by the current logical model. A target-class search needs the
// promotion, and a source-class search needs it so promoted source geometry is
// removed. Searches for unrelated classes can skip this post-processing pass.
func classMergeRulesRelevant(rules []ClassMergeRule, detectionClasses []int) bool {
	if len(rules) == 0 || len(detectionClasses) == 0 {
		return false
	}
	for _, rule := range rules {
		if containsInt(detectionClasses, rule.TargetClass) {
			return true
		}
		for _, sourceClass := range rule.SourceClasses {
			if containsInt(detectionClasses, sourceClass) {
				return true
			}
		}
	}
	return false
}

// applyClassMergeRules mutates classMap. Encoded class values are classID+1.
//
// Performance notes:
//   - no full-image dilation is built;
//   - target proximity is checked only around pixels of source components;
//   - queue traversal uses a head index instead of repeatedly reslicing;
//   - rules return immediately when either target or source is absent.
//
// The target snapshot is fixed for the duration of a rule, preserving the
// previous semantics: components promoted earlier by the same rule do not
// extend the target neighbourhood for later components.
func applyClassMergeRules(classMap []uint16, w, h int, geo [6]float64, proj string, rules []ClassMergeRule) {
	if len(classMap) != w*h || w <= 0 || h <= 0 || len(rules) == 0 {
		return
	}
	pixelSize := pixelSizeMetres(geo, w, h, proj)
	for _, rule := range rules {
		if rule.TargetClass < 0 || len(rule.SourceClasses) == 0 {
			continue
		}
		targetValue := uint16(rule.TargetClass + 1)
		radius := 0
		if rule.DistanceM > 0 && pixelSize > 0 {
			radius = int(math.Ceil(rule.DistanceM / pixelSize))
		}
		// The previous implementation first dilated the target by radius and
		// then checked the source pixel's one-pixel 8-neighbourhood. Its exact
		// effective Chebyshev distance was therefore radius+1. Preserve it.
		checkRadius := radius + 1

		targetSnapshot := make([]uint8, len(classMap))
		hasTarget := false
		hasSource := false
		for i, encoded := range classMap {
			if encoded == targetValue {
				targetSnapshot[i] = 1
				hasTarget = true
				continue
			}
			if encoded != 0 && containsInt(rule.SourceClasses, int(encoded)-1) {
				hasSource = true
			}
		}
		if !hasTarget || !hasSource {
			continue
		}

		visited := make([]uint8, len(classMap))
		queue := make([]int, 0, 1024)
		component := make([]int, 0, 1024)
		dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}

		for i, encoded := range classMap {
			if encoded == 0 || visited[i] != 0 {
				continue
			}
			classID := int(encoded) - 1
			if !containsInt(rule.SourceClasses, classID) {
				continue
			}

			queue = queue[:0]
			component = component[:0]
			queue = append(queue, i)
			visited[i] = 1
			touches := false

			for head := 0; head < len(queue); head++ {
				cur := queue[head]
				component = append(component, cur)
				x, y := cur%w, cur/w

				if !touches {
					for yy := max(0, y-checkRadius); yy <= min(h-1, y+checkRadius) && !touches; yy++ {
						row := yy * w
						for xx := max(0, x-checkRadius); xx <= min(w-1, x+checkRadius); xx++ {
							if targetSnapshot[row+xx] != 0 {
								touches = true
								break
							}
						}
					}
				}

				for _, d := range dirs {
					nx, ny := x+d[0], y+d[1]
					if nx < 0 || ny < 0 || nx >= w || ny >= h {
						continue
					}
					ni := ny*w + nx
					if visited[ni] != 0 || classMap[ni] != encoded {
						continue
					}
					visited[ni] = 1
					queue = append(queue, ni)
				}
			}

			if touches {
				for _, idx := range component {
					classMap[idx] = targetValue
				}
			}
		}
	}
}
