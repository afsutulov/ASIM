package main

import "math"

func makeUint8Mask(n int) []uint8 { return make([]uint8, n) }

// hasAnyValidPixel returns true when a patch contains at least one pixel from
// the useful raster area. A patch with an entirely zero ValidMask cannot
// contribute a valid output polygon and can be skipped without changing the
// search result.
func hasAnyValidPixel(valid []uint8) bool {
	if valid == nil {
		return true
	}
	for _, value := range valid {
		if value != 0 {
			return true
		}
	}
	return false
}

// isMostlyEmptyTile решает, можно ли пропустить тайл без инференса.
//
// ВАЖНО (исправление пропуска сегментов зданий): пропускаем ТОЛЬКО практически
// пустые тайлы — те, в которых число валидных не-белых пикселей меньше
// minContentPixels (то есть тайл почти целиком состоит из nodata и/или белой
// каймы листа ортофото).
//
// Прежняя версия отбрасывала тайлы по ОТНОСИТЕЛЬНЫМ порогам: >=90% белых
// пикселей или <5% валидных. Для тайла 1536x1536 это до ~236 тыс. «настоящих»
// пикселей, которые молча выбрасывались вместе с тайлом. Если у края листа
// ортофото в такой тайл попадал фрагмент здания, он исчезал из результата
// целым прямоугольным сегментом с ровными границами по сетке тайлов.
func isMostlyEmptyTile(
	data []float32,
	valid []uint8,
	channels, width, height int,
	whiteThreshold float32,
	minContentPixels int,
) bool {
	n := width * height
	if n == 0 {
		return true
	}
	if minContentPixels < 1 {
		minContentPixels = 1
	}
	content := 0
	for i := 0; i < n; i++ {
		if valid != nil && valid[i] == 0 {
			continue
		}
		allWhite := true
		for c := 0; c < channels; c++ {
			v := data[c*n+i]
			if v < whiteThreshold {
				allWhite = false
				break
			}
		}
		if !allWhite {
			content++
			if content >= minContentPixels {
				// Тайл содержит достаточно реальных пикселей — обрабатываем.
				return false
			}
		}
	}
	return true
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

// dilateMask и erodeMask используют РАЗДЕЛИМУЮ реализацию: квадратный
// структурный элемент (2r+1)x(2r+1) раскладывается в одномерное окно по X и
// затем по Y, а само окно считается по префиксным суммам за O(1) на пиксель.
// Результат математически тождественен прежней наивной реализации (включая
// обрезку окна на границах растра), но стоимость перестаёт зависеть от
// радиуса: morph_close с радиусом 24-48 px становится применимым на листе
// 20000x20000, тогда как прежняя версия была O(r^2) на пиксель.
//
// Это нужно для «ковровых» классов: борщевик детектируется отдельными
// очагами, и чтобы получить полигон зоны зарастания, очаги надо сомкнуть
// морфологией на масштабе расстояния между ними.

// dilate1D выполняет одномерную дилатацию бинарной строки/столбца по окну
// [i-radius, i+radius], обрезанному границами.
func dilate1D(src []uint8, n, stride, radius int, dst []uint8, prefix []int32) {
	prefix[0] = 0
	for i := 0; i < n; i++ {
		v := int32(0)
		if src[i*stride] != 0 {
			v = 1
		}
		prefix[i+1] = prefix[i] + v
	}
	for i := 0; i < n; i++ {
		lo := i - radius
		if lo < 0 {
			lo = 0
		}
		hi := i + radius
		if hi > n-1 {
			hi = n - 1
		}
		if prefix[hi+1]-prefix[lo] > 0 {
			dst[i*stride] = 1
		} else {
			dst[i*stride] = 0
		}
	}
}

// erode1D — одномерная эрозия: пиксель остаётся, только если в окне нет нулей.
func erode1D(src []uint8, n, stride, radius int, dst []uint8, prefix []int32) {
	prefix[0] = 0
	for i := 0; i < n; i++ {
		v := int32(0)
		if src[i*stride] != 0 {
			v = 1
		}
		prefix[i+1] = prefix[i] + v
	}
	for i := 0; i < n; i++ {
		lo := i - radius
		if lo < 0 {
			lo = 0
		}
		hi := i + radius
		if hi > n-1 {
			hi = n - 1
		}
		if prefix[hi+1]-prefix[lo] == int32(hi-lo+1) {
			dst[i*stride] = 1
		} else {
			dst[i*stride] = 0
		}
	}
}

func separableMorph(src []uint8, w, h, radius int, dilate bool) []uint8 {
	if radius <= 0 {
		out := make([]uint8, len(src))
		copy(out, src)
		return out
	}
	op := erode1D
	if dilate {
		op = dilate1D
	}
	tmp := make([]uint8, len(src))
	dst := make([]uint8, len(src))
	maxN := w
	if h > maxN {
		maxN = h
	}
	prefix := make([]int32, maxN+1)
	for y := 0; y < h; y++ {
		op(src[y*w:], w, 1, radius, tmp[y*w:], prefix)
	}
	for x := 0; x < w; x++ {
		op(tmp[x:], h, w, radius, dst[x:], prefix)
	}
	return dst
}

func dilateMask(src []uint8, w, h, radius int) []uint8 {
	return separableMorph(src, w, h, radius, true)
}

func erodeMask(src []uint8, w, h, radius int) []uint8 {
	return separableMorph(src, w, h, radius, false)
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

// fillSmallHoles fills zero-components fully surrounded by mask pixels.
// If maxHoleArea <= 0, all internal holes are filled. Border-connected
// background is never filled.
func fillSmallHoles(src []uint8, w, h int, maxHoleArea int) []uint8 {
	out := make([]uint8, len(src))
	copy(out, src)
	visited := make([]uint8, len(src))
	queue := make([]int, 0, 1024)
	component := make([]int, 0, 1024)
	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for i := 0; i < len(src); i++ {
		if src[i] != 0 || visited[i] != 0 {
			continue
		}
		queue = queue[:0]
		component = component[:0]
		queue = append(queue, i)
		visited[i] = 1
		touchesBorder := false
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			component = append(component, cur)
			x := cur % w
			y := cur / w
			if x == 0 || y == 0 || x == w-1 || y == h-1 {
				touchesBorder = true
			}
			for _, d := range dirs {
				nx := x + d[0]
				ny := y + d[1]
				if nx < 0 || ny < 0 || nx >= w || ny >= h {
					continue
				}
				ni := ny*w + nx
				if src[ni] != 0 || visited[ni] != 0 {
					continue
				}
				visited[ni] = 1
				queue = append(queue, ni)
			}
		}
		if !touchesBorder && (maxHoleArea <= 0 || len(component) <= maxHoleArea) {
			for _, idx := range component {
				out[idx] = 1
			}
		}
	}
	return out
}
