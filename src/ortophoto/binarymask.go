package main

import "math"

func makeUint8Mask(n int) []uint8 { return make([]uint8, n) }

func isMostlyEmptyTile(
	data []float32,
	valid []uint8,
	channels, width, height int,
	whiteThreshold float32,
	whiteRatio float64,
	minValidRatio float64,
) bool {
	if width*height == 0 {
		return true
	}
	validCount := 0
	whiteCount := 0
	for i := 0; i < width*height; i++ {
		if valid != nil && valid[i] == 0 {
			continue
		}
		validCount++
		allWhite := true
		for c := 0; c < channels; c++ {
			v := data[c*width*height+i]
			if v < whiteThreshold {
				allWhite = false
				break
			}
		}
		if allWhite {
			whiteCount++
		}
	}
	validRatio := float64(validCount) / float64(width*height)
	if validRatio < minValidRatio {
		return true
	}
	if validCount == 0 {
		return true
	}
	return float64(whiteCount)/float64(validCount) >= whiteRatio
}

func thresholdMask(probs []float32, thr float32) []uint8 {
	out := make([]uint8, len(probs))
	for i, v := range probs {
		if v >= thr {
			out[i] = 1
		}
	}
	return out
}

func andNotMask(a []uint8, b []uint8, valid []uint8) []uint8 {
	out := make([]uint8, len(a))
	for i := range a {
		if a[i] != 0 && b[i] == 0 {
			if valid == nil || valid[i] != 0 {
				out[i] = 1
			}
		}
	}
	return out
}

func applyValidMask(a []uint8, valid []uint8) {
	if valid == nil {
		return
	}
	for i := range a {
		if valid[i] == 0 {
			a[i] = 0
		}
	}
}

func dilateMask(src []uint8, w, h, radius int) []uint8 {
	if radius <= 0 {
		out := make([]uint8, len(src))
		copy(out, src)
		return out
	}
	dst := make([]uint8, len(src))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			val := uint8(0)
			for yy := max(0, y-radius); yy <= min(h-1, y+radius); yy++ {
				for xx := max(0, x-radius); xx <= min(w-1, x+radius); xx++ {
					if src[yy*w+xx] != 0 {
						val = 1
						goto done
					}
				}
			}
		done:
			dst[y*w+x] = val
		}
	}
	return dst
}

func erodeMask(src []uint8, w, h, radius int) []uint8 {
	if radius <= 0 {
		out := make([]uint8, len(src))
		copy(out, src)
		return out
	}
	dst := make([]uint8, len(src))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			val := uint8(1)
			for yy := max(0, y-radius); yy <= min(h-1, y+radius); yy++ {
				for xx := max(0, x-radius); xx <= min(w-1, x+radius); xx++ {
					if src[yy*w+xx] == 0 {
						val = 0
						goto done
					}
				}
			}
		done:
			dst[y*w+x] = val
		}
	}
	return dst
}

func openMask(src []uint8, w, h, radius int) []uint8 {
	return dilateMask(erodeMask(src, w, h, radius), w, h, radius)
}

func closeMask(src []uint8, w, h, radius int) []uint8 {
	return erodeMask(dilateMask(src, w, h, radius), w, h, radius)
}

func removeSmallComponents(src []uint8, w, h int, minArea int) []uint8 {
	if minArea <= 1 {
		out := make([]uint8, len(src))
		copy(out, src)
		return out
	}
	out := make([]uint8, len(src))
	visited := make([]uint8, len(src))
	queue := make([]int, 0, 1024)
	component := make([]int, 0, 1024)

	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for i := 0; i < len(src); i++ {
		if src[i] == 0 || visited[i] != 0 {
			continue
		}
		queue = queue[:0]
		component = component[:0]
		queue = append(queue, i)
		visited[i] = 1

		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			component = append(component, cur)
			x := cur % w
			y := cur / w
			for _, d := range dirs {
				nx := x + d[0]
				ny := y + d[1]
				if nx < 0 || ny < 0 || nx >= w || ny >= h {
					continue
				}
				ni := ny*w + nx
				if src[ni] == 0 || visited[ni] != 0 {
					continue
				}
				visited[ni] = 1
				queue = append(queue, ni)
			}
		}

		if len(component) >= minArea {
			for _, idx := range component {
				out[idx] = 1
			}
		}
	}
	return out
}

func toFloatMask(src []uint8) []float32 {
	out := make([]float32, len(src))
	for i, v := range src {
		if v != 0 {
			out[i] = 1.0
		}
	}
	return out
}

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
