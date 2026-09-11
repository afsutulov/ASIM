package main

import (
	"fmt"
	"log"
)

// СЛОЙ ПОКРЫТИЯ.
//
// Ортофотоплан обрабатывается ПОЛИСТНО. Лист может не дойти до модели по
// нескольким причинам, и все они сейчас видны только в журнале:
//
//   - ReadOrthophotoInfo не смог прочитать метаданные;
//   - ClassifySourceCRS не распознал СК листа;
//   - пространственный префильтр RasterIntersectsAllowedWGS84 счёл, что лист
//     не пересекается с полигоном поиска;
//   - ReadOrthophoto не смог прочитать растр (в т.ч. по памяти на крупных
//     листах);
//   - листа просто нет во входном каталоге, хотя в QGIS подложка его
//     показывает — подложка и вход АСИМ это разные вещи.
//
// Во всех этих случаях на карте получается ПРЯМОУГОЛЬНАЯ область без
// результата с идеально прямыми сторонами по границам листа. Отличить её от
// «модель здесь не сработала» по самому результату невозможно: и то и другое
// выглядит как отсутствие полигонов.
//
// Слой покрытия снимает этот вопрос за один взгляд. Если прямая граница
// результата совпадает с краем покрытия — виноват отсутствующий или
// пропущенный лист, и модель ни при чём. Если граница проходит ВНУТРИ
// покрытия — лист обработан, и разбираться надо с моделью или нарезкой.

// coverageAccumulator собирает контуры листов в две геометрии WGS84:
// обработанные и пропущенные.
type coverageAccumulator struct {
	processed Geometry
	skipped   Geometry
	nProc     int
	nSkip     int
}

// add добавляет контур одного листа. Ошибки перепроецирования не считаются
// фатальными: слой покрытия диагностический, ронять из-за него задачу нельзя.
func (c *coverageAccumulator) add(geo [6]float64, w, h int, projWKT string, processed bool) {
	rect := RectFromGeo(geo, w, h)
	if rect == nil {
		return
	}
	defer DestroyGeometry(rect)

	srs, err := projSRSFromWKT(projWKT)
	if err != nil || srs == nil {
		log.Printf("ortophoto coverage: cannot build SRS for sheet footprint, skipped: error=%v", err)
		return
	}
	defer DestroySRS(srs)

	wgs, err := reprojectGeometryToWGS84(rect, srs)
	if err != nil || wgs == nil {
		log.Printf("ortophoto coverage: cannot reproject sheet footprint, skipped: error=%v", err)
		return
	}
	defer DestroyGeometry(wgs)

	target := &c.processed
	if !processed {
		target = &c.skipped
	}
	merged := GeometryUnion(*target, wgs)
	if *target != nil {
		DestroyGeometry(*target)
	}
	*target = merged
	if processed {
		c.nProc++
	} else {
		c.nSkip++
	}
}

func (c *coverageAccumulator) close() {
	if c.processed != nil {
		DestroyGeometry(c.processed)
		c.processed = nil
	}
	if c.skipped != nil {
		DestroyGeometry(c.skipped)
		c.skipped = nil
	}
}

// write выгружает накопленные контуры в два GeoJSON рядом с результатом.
// Возвращает имена созданных файлов.
func (c *coverageAccumulator) write(processedPath, skippedPath string) []string {
	out := make([]string, 0, 2)
	if c.processed != nil && !GeometryIsEmpty(c.processed) {
		if err := WriteGeometryAsGeoJSON(c.processed, processedPath); err != nil {
			log.Printf("ortophoto coverage: write processed footprint failed: error=%v", err)
		} else {
			out = append(out, processedPath)
		}
	}
	if c.skipped != nil && !GeometryIsEmpty(c.skipped) {
		if err := WriteGeometryAsGeoJSON(c.skipped, skippedPath); err != nil {
			log.Printf("ortophoto coverage: write skipped footprint failed: error=%v", err)
		} else {
			out = append(out, skippedPath)
		}
	}
	log.Printf("ortophoto coverage: sheets_processed=%d sheets_skipped=%d files=%v", c.nProc, c.nSkip, out)
	return out
}

// coverageSummary — короткая строка для журнала и манифеста.
func (c *coverageAccumulator) summary() string {
	return fmt.Sprintf("processed=%d skipped=%d", c.nProc, c.nSkip)
}
