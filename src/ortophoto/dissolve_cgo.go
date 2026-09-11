package main

/*
#cgo pkg-config: gdal
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gdal.h"
#include "ogr_api.h"
#include "ogr_srs_api.h"

static void _gdal_init_dissolve_once() {
    GDALAllRegister();
    OGRRegisterAll();
}
static void _dissolve_set_traditional_gis_order(OGRSpatialReferenceH srs) {
    if (srs != NULL) {
        OSRSetAxisMappingStrategy(srs, OAMS_TRADITIONAL_GIS_ORDER);
    }
}
static OGRGeometryH _dissolve_new_multi_polygon() { return OGR_G_CreateGeometry(wkbMultiPolygon); }
*/
import "C"

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unsafe"
)

// ЗАЧЕМ ЭТОТ ШАГ.
//
// Ортофотоплан обрабатывается ПОЛИСТНО: каждый .tif отдельно даёт маску,
// полигонизуется и дописывается в накопитель. Заросль, пересекающая границу
// двух листов, попадает в результат ДВУМЯ объектами, состыкованными по прямой
// линии границы листа. QGIS рисует обводку каждому — на карте это выглядит как
// тёмная прямая поперёк массива.
//
// ПОЧЕМУ ПЕРВАЯ ВЕРСИЯ ЭТОГО ФАЙЛА ВЕШАЛА ЗАДАЧУ.
//
// Она делала OGR_G_UnionCascaded по ВСЕМУ накопителю целиком, а затем
// buffer(+d)/buffer(-d) по всей объединённой геометрии. Полигоны борщевика
// получаются полигонизацией растровой маски, то есть имеют вершину на каждом
// ребре пикселя: один объект — это десятки и сотни тысяч вершин. Каскадное
// объединение и тем более буферизация такой геометрии стоят минуты и растут
// нелинейно. При этом подавляющее большинство полигонов вообще не касается
// границ листов и в объединении не нуждается.
//
// ТЕКУЩАЯ ВЕРСИЯ.
//
// Объединяются только те объекты, которые реально могут оказаться половинками
// одного: кластеры, у которых огибающие прямоугольники (расширенные на
// seamClose) пересекаются. Кластеры ищутся заметающей прямой по X за
// O(n log n) на одних лишь огибающих, без обращения к геометрии. Одиночные
// объекты переписываются в результат КАК ЕСТЬ — без union, без buffer, без
// перестроения. Буферизация применяется только внутри кластеров и только если
// seamClose > 0.
//
// В итоге работа над всем накопителем превращается в работу над узкой полосой
// вдоль швов.

func bufferGeometry(g Geometry, dist float64) Geometry {
	if g == nil || dist == 0 {
		return nil
	}
	return C.OGR_G_Buffer(g, C.double(dist), C.int(8))
}

// dissolveEnvelope — огибающая одного объекта накопителя вместе с его FID.
type dissolveEnvelope struct {
	fid int64
	env Envelope
}

// envelopesTouch проверяет пересечение огибающих, расширенных на pad с каждой
// стороны. Это дешёвый предфильтр. Ложное срабатывание безвредно: внутри
// кластера union оставит несоприкасающиеся объекты раздельными частями
// мультиполигона, и они будут записаны отдельными объектами.
func envelopesTouch(a, b Envelope, pad float64) bool {
	return a.MinX-pad <= b.MaxX+pad && b.MinX-pad <= a.MaxX+pad &&
		a.MinY-pad <= b.MaxY+pad && b.MinY-pad <= a.MaxY+pad
}

// clusterByEnvelope группирует объекты в кластеры связности по пересечению
// огибающих. Заметающая прямая по MinX: объект сравнивается только с теми, чей
// MaxX ещё не ушёл влево за его MinX. Возвращает отображение fid -> корень
// кластера и размеры кластеров.
func clusterByEnvelope(items []dissolveEnvelope, pad float64) (map[int64]int, map[int]int) {
	n := len(items)
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	find := func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[rb] = ra
		}
	}

	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return items[order[a]].env.MinX < items[order[b]].env.MinX
	})

	active := make([]int, 0, 256)
	for _, i := range order {
		cur := items[i].env
		// Из активного набора выбрасывается всё, что уже не может пересечься.
		kept := active[:0]
		for _, j := range active {
			if items[j].env.MaxX+pad < cur.MinX-pad {
				continue
			}
			kept = append(kept, j)
			if envelopesTouch(cur, items[j].env, pad) {
				union(i, j)
			}
		}
		active = append(kept, i)
	}

	roots := make(map[int64]int, n)
	sizes := make(map[int]int, n)
	for i := range items {
		r := find(i)
		roots[items[i].fid] = r
		sizes[r]++
	}
	return roots, sizes
}

// featureWriter дописывает геометрии в накопитель, разбирая мультиполигоны на
// части: один объект на связную область.
type featureWriter struct {
	ds      C.GDALDatasetH
	layer   C.OGRLayerH
	defn    C.OGRFeatureDefnH
	dnIndex C.int
	written int
}

func newFeatureWriter(outShp string) (*featureWriter, error) {
	ds, layer, err := openLayerForAppend(outShp)
	if err != nil {
		return nil, err
	}
	defn := C.OGR_L_GetLayerDefn(layer)
	fdDN := C.CString("DN")
	defer C.free(unsafe.Pointer(fdDN))
	return &featureWriter{ds: ds, layer: layer, defn: defn, dnIndex: C.OGR_FD_GetFieldIndex(defn, fdDN)}, nil
}

func (w *featureWriter) close() {
	if w.ds != nil {
		C.GDALFlushCache(w.ds)
		C.GDALClose(w.ds)
		w.ds = nil
	}
}

func (w *featureWriter) writeOne(part Geometry) error {
	if part == nil || C.OGR_G_IsEmpty(part) != 0 {
		return nil
	}
	feat := C.OGR_F_Create(w.defn)
	if feat == nil {
		return fmt.Errorf("create feature")
	}
	defer C.OGR_F_Destroy(feat)
	if w.dnIndex >= 0 {
		C.OGR_F_SetFieldInteger(feat, w.dnIndex, C.int(1))
	}
	if C.OGR_F_SetGeometry(feat, part) != 0 {
		return fmt.Errorf("set geometry")
	}
	if C.OGR_L_CreateFeature(w.layer, feat) != 0 {
		return fmt.Errorf("write feature")
	}
	w.written++
	return nil
}

// writeParts разбирает мультиполигон на части и пишет каждую отдельным объектом.
func (w *featureWriter) writeParts(g Geometry) error {
	if g == nil {
		return nil
	}
	flat := C.OGR_GT_Flatten(C.OGR_G_GetGeometryType(g))
	if flat == C.wkbMultiPolygon || flat == C.wkbGeometryCollection {
		n := int(C.OGR_G_GetGeometryCount(g))
		for i := 0; i < n; i++ {
			if err := w.writeOne(C.OGR_G_GetGeometryRef(g, C.int(i))); err != nil {
				return err
			}
		}
		return nil
	}
	return w.writeOne(g)
}

// DissolveAccumulatorShapefile объединяет соприкасающиеся полигоны накопителя и
// переписывает шейп-файл на месте.
//
// seamClose — радиус морфологического замыкания в ЕДИНИЦАХ СК СЛОЯ: метры для
// МСК-59, градусы для WGS84. Применяется ТОЛЬКО к кластерам из нескольких
// объектов, то есть к окрестности швов. 0 отключает замыкание.
//
// При любой ошибке исходный файл остаётся нетронутым: вызывающий может
// продолжить с неразъединённым результатом.
func DissolveAccumulatorShapefile(shp string, seamClose float64) (int, error) {
	C._gdal_init_dissolve_once()
	started := time.Now()

	cp := C.CString(shp)
	defer C.free(unsafe.Pointer(cp))
	ds := C.GDALOpenEx(cp, C.GDAL_OF_VECTOR, nil, nil, nil)
	if ds == nil {
		return 0, fmt.Errorf("open accumulator: %s", shp)
	}
	layer := C.GDALDatasetGetLayer(ds, 0)
	if layer == nil {
		C.GDALClose(ds)
		return 0, fmt.Errorf("accumulator layer not found: %s", shp)
	}
	var srs C.OGRSpatialReferenceH
	if lsrs := C.OGR_L_GetSpatialRef(layer); lsrs != nil {
		srs = C.OSRClone(lsrs)
		C._dissolve_set_traditional_gis_order(srs)
	}
	if srs != nil {
		defer C.OSRDestroySpatialReference(srs)
	}

	// Проход 1: только огибающие. Геометрии не копируются — накопитель может
	// весить сотни мегабайт.
	items := make([]dissolveEnvelope, 0, 4096)
	C.OGR_L_ResetReading(layer)
	for {
		feat := C.OGR_L_GetNextFeature(layer)
		if feat == nil {
			break
		}
		g := C.OGR_F_GetGeometryRef(feat)
		if g != nil && C.OGR_G_IsEmpty(g) == 0 {
			items = append(items, dissolveEnvelope{fid: int64(C.OGR_F_GetFID(feat)), env: GeometryEnvelope(g)})
		}
		C.OGR_F_Destroy(feat)
	}
	if len(items) == 0 {
		C.GDALClose(ds)
		return 0, nil
	}

	roots, sizes := clusterByEnvelope(items, seamClose)
	clusters := 0
	inClusters := 0
	for _, sz := range sizes {
		if sz > 1 {
			clusters++
			inClusters += sz
		}
	}
	scanned := time.Since(started)

	base := strings.TrimSuffix(shp, filepath.Ext(shp))
	tmp := base + "_dissolved.shp"
	if err := CreateEmptyAccumulatorShapefile(tmp, srs); err != nil {
		C.GDALClose(ds)
		return 0, err
	}
	writer, err := newFeatureWriter(tmp)
	if err != nil {
		C.GDALClose(ds)
		CleanupShapefileSet(tmp)
		return 0, err
	}

	pending := make(map[int]Geometry, clusters)
	failed := func(e error) (int, error) {
		for _, g := range pending {
			DestroyGeometry(g)
		}
		writer.close()
		CleanupShapefileSet(tmp)
		return 0, e
	}

	// Проход 2: одиночные объекты переписываются как есть, члены кластеров
	// накапливаются по корню кластера.
	C.OGR_L_ResetReading(layer)
	for {
		feat := C.OGR_L_GetNextFeature(layer)
		if feat == nil {
			break
		}
		g := C.OGR_F_GetGeometryRef(feat)
		if g == nil || C.OGR_G_IsEmpty(g) != 0 {
			C.OGR_F_Destroy(feat)
			continue
		}
		fid := int64(C.OGR_F_GetFID(feat))
		root, ok := roots[fid]
		if !ok || sizes[root] <= 1 {
			werr := writer.writeParts(g)
			C.OGR_F_Destroy(feat)
			if werr != nil {
				C.GDALClose(ds)
				return failed(werr)
			}
			continue
		}
		multi, exists := pending[root]
		if !exists {
			multi = C._dissolve_new_multi_polygon()
			pending[root] = multi
		}
		flat := C.OGR_GT_Flatten(C.OGR_G_GetGeometryType(g))
		if flat == C.wkbMultiPolygon || flat == C.wkbGeometryCollection {
			n := int(C.OGR_G_GetGeometryCount(g))
			for i := 0; i < n; i++ {
				part := C.OGR_G_GetGeometryRef(g, C.int(i))
				if part != nil && C.OGR_GT_Flatten(C.OGR_G_GetGeometryType(part)) == C.wkbPolygon {
					C.OGR_G_AddGeometry(multi, part)
				}
			}
		} else if flat == C.wkbPolygon {
			C.OGR_G_AddGeometry(multi, g)
		}
		C.OGR_F_Destroy(feat)
	}
	C.GDALClose(ds)

	// Объединение внутри кластеров. Каждый кластер мал, поэтому UnionCascaded
	// здесь стоит доли секунды, а не минуты.
	for _, multi := range pending {
		merged := C.OGR_G_UnionCascaded(multi)
		DestroyGeometry(multi)
		if merged == nil {
			continue
		}
		if seamClose > 0 {
			grown := bufferGeometry(merged, seamClose)
			if grown != nil {
				shrunk := bufferGeometry(grown, -seamClose)
				DestroyGeometry(grown)
				if shrunk != nil && C.OGR_G_IsEmpty(shrunk) == 0 {
					DestroyGeometry(merged)
					merged = shrunk
				} else if shrunk != nil {
					DestroyGeometry(shrunk)
				}
			}
		}
		werr := writer.writeParts(merged)
		DestroyGeometry(merged)
		if werr != nil {
			return failed(werr)
		}
	}
	written := writer.written
	writer.close()

	// Подмена файлов только после успешной записи временного набора.
	tmpBase := strings.TrimSuffix(tmp, filepath.Ext(tmp))
	for _, ext := range []string{".shp", ".shx", ".dbf", ".prj", ".cpg"} {
		src := tmpBase + ext
		dst := base + ext
		if _, statErr := os.Stat(src); statErr != nil {
			_ = os.Remove(dst)
			continue
		}
		if rmErr := os.Remove(dst); rmErr != nil && !os.IsNotExist(rmErr) {
			return written, fmt.Errorf("replace %s: %w", dst, rmErr)
		}
		if mvErr := os.Rename(src, dst); mvErr != nil {
			return written, fmt.Errorf("rename %s -> %s: %w", src, dst, mvErr)
		}
	}
	log.Printf("ortophoto dissolve: file=%s input=%d clusters=%d features_in_clusters=%d output=%d scan=%s total=%s",
		filepath.Base(shp), len(items), clusters, inClusters, written, scanned, time.Since(started))
	return written, nil
}
