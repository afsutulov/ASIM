package main

/*
#cgo pkg-config: gdal
#include <stdlib.h>
#include "gdal.h"
#include "ogr_api.h"
#include "ogr_srs_api.h"

static void _gdal_init_vector_once() {
    GDALAllRegister();
    OGRRegisterAll();
}
static OGRSpatialReferenceH _clone_srs(OGRSpatialReferenceH srs) {
    if (srs == NULL) return NULL;
    OGRSpatialReferenceH c = OSRClone(srs);
    if (c != NULL) {
        OSRSetAxisMappingStrategy(c, OAMS_TRADITIONAL_GIS_ORDER);
    }
    return c;
}

static void _set_traditional_gis_order(OGRSpatialReferenceH srs) {
    if (srs != NULL) {
        OSRSetAxisMappingStrategy(srs, OAMS_TRADITIONAL_GIS_ORDER);
    }
}

*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

func countVectorFeatures(shpPath string) (int, error) {
	C._gdal_init_vector_once()
	cpath := C.CString(shpPath)
	defer C.free(unsafe.Pointer(cpath))
	ds := C.GDALOpenEx(cpath, C.GDAL_OF_VECTOR, nil, nil, nil)
	if ds == nil {
		return 0, fmt.Errorf("open vector: %s", shpPath)
	}
	defer C.GDALClose(ds)
	layer := C.GDALDatasetGetLayer(ds, 0)
	if layer == nil {
		return 0, fmt.Errorf("layer 0 not found: %s", shpPath)
	}
	return int(C.OGR_L_GetFeatureCount(layer, 1)), nil
}

func removeShapefileSet(outShp string) {
	base := strings.TrimSuffix(outShp, filepath.Ext(outShp))
	for _, ext := range []string{".shp", ".shx", ".dbf", ".prj", ".cpg"} {
		_ = osRemove(base + ext)
	}
}

func osRemove(path string) error {
	return os.Remove(path)
}

func transformWGS84GeometryToSRS(g Geometry, dstSRS C.OGRSpatialReferenceH) (Geometry, error) {
	if g == nil {
		return nil, nil
	}
	if dstSRS == nil {
		return cloneGeom(g), nil
	}
	src := C.OSRNewSpatialReference(nil)
	if src == nil {
		return nil, fmt.Errorf("create WGS84 SRS failed")
	}
	defer C.OSRDestroySpatialReference(src)
	if C.OSRImportFromEPSG(src, 4326) != 0 {
		return nil, fmt.Errorf("import EPSG:4326 failed")
	}
	C._set_traditional_gis_order(src)
	dst := C.OSRClone(dstSRS)
	if dst == nil {
		return nil, fmt.Errorf("clone destination SRS failed")
	}
	defer C.OSRDestroySpatialReference(dst)
	C._set_traditional_gis_order(dst)
	ct := C.OCTNewCoordinateTransformation(src, dst)
	if ct == nil {
		return nil, fmt.Errorf("create coordinate transformation WGS84->target failed")
	}
	defer C.OCTDestroyCoordinateTransformation(ct)
	cg := cloneGeom(g)
	if cg == nil {
		return nil, fmt.Errorf("clone clip geometry failed")
	}
	if C.OGR_G_Transform(cg, ct) != 0 {
		DestroyGeometry(cg)
		return nil, fmt.Errorf("transform clip geometry WGS84->target failed")
	}
	return cg, nil
}

// transformGeometryToSRS converts geometry directly from srcSRS to dstSRS using a single
// coordinate transformation (e.g. МСК 59-1 <-> МСК 59-2 <-> МСК 59-3, same Pulkovo 1942 datum).
// This intentionally avoids any intermediate hop through WGS84: ортофото-снимки и итоговые
// полигоны должны оставаться в МСК-59 (с учётом зоны), а лишний WGS84-roundtrip вносит
// систематическую погрешность из-за параметров Гельмерта (Pulkovo1942 -> WGS84), см. BUGFIX notes.
func transformGeometryToSRS(g Geometry, srcSRS, dstSRS C.OGRSpatialReferenceH) (Geometry, error) {
	if g == nil {
		return nil, nil
	}
	if srcSRS == nil || dstSRS == nil {
		return cloneGeom(g), nil
	}
	if C.OSRIsSame(srcSRS, dstSRS) != 0 {
		return cloneGeom(g), nil
	}
	C._set_traditional_gis_order(srcSRS)
	C._set_traditional_gis_order(dstSRS)
	ct := C.OCTNewCoordinateTransformation(srcSRS, dstSRS)
	if ct == nil {
		return nil, fmt.Errorf("create coordinate transformation between source CRS failed")
	}
	defer C.OCTDestroyCoordinateTransformation(ct)
	cg := cloneGeom(g)
	if cg == nil {
		return nil, fmt.Errorf("clone geometry failed")
	}
	if C.OGR_G_Transform(cg, ct) != 0 {
		DestroyGeometry(cg)
		return nil, fmt.Errorf("geometry transform between source CRS failed")
	}
	return cg, nil
}

func DestroySRS(srs C.OGRSpatialReferenceH) {
	if srs != nil {
		C.OSRDestroySpatialReference(srs)
	}
}

// ClipShapefileToGeometryWithSRSWKT работает как ClipShapefileToGeometry, но
// для трансформации clipGeom (WGS84) в СК снимка использует ЯВНО переданный
// rasterWKT — полный WKT проекции исходного растра (например img.Proj),
// а не СК, заново прочитанную из .prj файла shapefile.
//
// Это важно: при записи в ESRI Shapefile (.prj использует устаревший WKT1)
// GDAL ТЕРЯЕТ параметры датум-трансформации (BOUNDCRS/TOWGS84) — Pulkovo1942
// -> WGS84 (Helmert 7 параметров), которые присутствуют в исходном geotiff
// (МСК-59 хранится как BOUNDCRS с ABRIDGEDTRANSFORMATION к WGS84). После
// перечитывания .prj обратно получается "голая" PROJCS без датум-сдвига —
// поэтому трансформация шаблона полигона поиска (WGS84) в СК снимка либо
// падает с ошибкой, либо даёт грубо неверные координаты, и clip-результат
// оказывается пустым. Используя исходный WKT растра напрямую, мы сохраняем
// корректную привязку Pulkovo1942 <-> WGS84 на всём пути.
func ClipShapefileToGeometryWithSRSWKT(inShp string, outShp string, clipGeom Geometry, rasterWKT string) error {
	C._gdal_init_vector_once()
	removeShapefileSet(outShp)

	inC := C.CString(inShp)
	defer C.free(unsafe.Pointer(inC))
	ds := C.GDALOpenEx(inC, C.GDAL_OF_VECTOR, nil, nil, nil)
	if ds == nil {
		return fmt.Errorf("open vector: %s", inShp)
	}
	defer C.GDALClose(ds)
	layer := C.GDALDatasetGetLayer(ds, 0)
	if layer == nil {
		return fmt.Errorf("layer 0 not found: %s", inShp)
	}

	// СК для самого clip-преобразования берём из полного WKT растра (с датум-сдвигом),
	// а не из СК слоя shapefile (она урезана при сохранении в .prj).
	var rasterSRS C.OGRSpatialReferenceH
	if strings.TrimSpace(rasterWKT) != "" {
		rasterSRS = C.OSRNewSpatialReference(nil)
		cwkt := C.CString(rasterWKT)
		defer C.free(unsafe.Pointer(cwkt))
		pwkt := cwkt
		if C.OSRImportFromWkt(rasterSRS, (**C.char)(unsafe.Pointer(&pwkt))) != 0 {
			C.OSRDestroySpatialReference(rasterSRS)
			rasterSRS = nil
		}
	}
	if rasterSRS == nil {
		// Фолбэк на старое поведение, если WKT растра недоступен/некорректен.
		rasterSRS = C.OGR_L_GetSpatialRef(layer)
	} else {
		defer C.OSRDestroySpatialReference(rasterSRS)
	}

	var clippedGeom Geometry
	if clipGeom != nil {
		var err error
		clippedGeom, err = transformWGS84GeometryToSRS(clipGeom, rasterSRS)
		if err != nil {
			return fmt.Errorf("transform clipping geometry to layer SRS: %w", err)
		}
		defer DestroyGeometry(clippedGeom)
	}

	// Для записи выходного shapefile используем СК самого слоя (она совпадает
	// геометрически с rasterSRS, только без датум-сдвига, что для итоговых
	// shapefile-полигонов в той же зоне МСК-59 значения не имеет — координаты
	// уже посчитаны в этой проекции и трансформации больше не подвергаются).
	inSRS := C.OGR_L_GetSpatialRef(layer)

	drvName := C.CString("ESRI Shapefile")
	defer C.free(unsafe.Pointer(drvName))
	drv := C.OGRGetDriverByName(drvName)
	if drv == nil {
		return fmt.Errorf("OGR driver ESRI Shapefile not found")
	}
	outC := C.CString(outShp)
	defer C.free(unsafe.Pointer(outC))
	ods := C.OGR_Dr_CreateDataSource(drv, outC, nil)
	if ods == nil {
		return fmt.Errorf("create shp: %s", outShp)
	}
	defer C.OGR_DS_Destroy(ods)

	layerName := C.CString("result")
	defer C.free(unsafe.Pointer(layerName))
	srsCopy := C._clone_srs(inSRS)
	if srsCopy != nil {
		defer C.OSRDestroySpatialReference(srsCopy)
	}
	outLayer := C.OGR_DS_CreateLayer(ods, layerName, srsCopy, C.wkbPolygon, nil)
	if outLayer == nil {
		return fmt.Errorf("failed to create output layer")
	}

	fieldName := C.CString("DN")
	defer C.free(unsafe.Pointer(fieldName))
	fd := C.OGR_Fld_Create(fieldName, C.OFTInteger)
	defer C.OGR_Fld_Destroy(fd)
	if C.OGR_L_CreateField(outLayer, fd, 1) != 0 {
		return fmt.Errorf("failed to create DN field")
	}
	outDefn := C.OGR_L_GetLayerDefn(outLayer)

	if err := clipFeaturesInto(layer, outDefn, outLayer, clippedGeom); err != nil {
		return err
	}
	C.OGR_DS_SyncToDisk(ods)
	return nil
}

// clipFeaturesInto копирует все фичи из layer в outLayer (через outDefn), при
// необходимости пересекая каждую геометрию с clippedGeom. Общая часть для
// ClipShapefileToGeometry и ClipShapefileToGeometryWithSRSWKT.
func clipFeaturesInto(layer C.OGRLayerH, outDefn C.OGRFeatureDefnH, outLayer C.OGRLayerH, clippedGeom Geometry) error {
	C.OGR_L_ResetReading(layer)
	for {
		feat := C.OGR_L_GetNextFeature(layer)
		if feat == nil {
			break
		}
		geom := C.OGR_F_GetGeometryRef(feat)
		if geom == nil {
			C.OGR_F_Destroy(feat)
			continue
		}
		// If clipGeom is nil, copy as is
		useGeom := geom
		var inter C.OGRGeometryH
		if clippedGeom != nil {
			if C.OGR_G_Intersects(geom, clippedGeom) == 0 {
				C.OGR_F_Destroy(feat)
				continue
			}
			inter = C.OGR_G_Intersection(geom, clippedGeom)
			if inter == nil || C.OGR_G_IsEmpty(inter) != 0 {
				if inter != nil {
					C.OGR_G_DestroyGeometry(inter)
				}
				C.OGR_F_Destroy(feat)
				continue
			}
			useGeom = inter
		}
		outFeat := C.OGR_F_Create(outDefn)
		C.OGR_F_SetFieldInteger(outFeat, 0, C.OGR_F_GetFieldAsInteger(feat, 0))
		C.OGR_F_SetGeometry(outFeat, useGeom)
		C.OGR_L_CreateFeature(outLayer, outFeat)
		C.OGR_F_Destroy(outFeat)
		if inter != nil {
			C.OGR_G_DestroyGeometry(inter)
		}
		C.OGR_F_Destroy(feat)
	}
	return nil
}

func ClipShapefileToGeometry(inShp string, outShp string, clipGeom Geometry) error {
	C._gdal_init_vector_once()
	removeShapefileSet(outShp)

	inC := C.CString(inShp)
	defer C.free(unsafe.Pointer(inC))
	ds := C.GDALOpenEx(inC, C.GDAL_OF_VECTOR, nil, nil, nil)
	if ds == nil {
		return fmt.Errorf("open vector: %s", inShp)
	}
	defer C.GDALClose(ds)
	layer := C.GDALDatasetGetLayer(ds, 0)
	if layer == nil {
		return fmt.Errorf("layer 0 not found: %s", inShp)
	}
	inSRS := C.OGR_L_GetSpatialRef(layer)
	var clippedGeom Geometry
	if clipGeom != nil {
		var err error
		clippedGeom, err = transformWGS84GeometryToSRS(clipGeom, inSRS)
		if err != nil {
			return fmt.Errorf("transform clipping geometry to layer SRS: %w", err)
		}
		defer DestroyGeometry(clippedGeom)
	}

	drvName := C.CString("ESRI Shapefile")
	defer C.free(unsafe.Pointer(drvName))
	drv := C.OGRGetDriverByName(drvName)
	if drv == nil {
		return fmt.Errorf("OGR driver ESRI Shapefile not found")
	}
	outC := C.CString(outShp)
	defer C.free(unsafe.Pointer(outC))
	ods := C.OGR_Dr_CreateDataSource(drv, outC, nil)
	if ods == nil {
		return fmt.Errorf("create shp: %s", outShp)
	}
	defer C.OGR_DS_Destroy(ods)

	layerName := C.CString("result")
	defer C.free(unsafe.Pointer(layerName))
	srsCopy := C._clone_srs(inSRS)
	if srsCopy != nil {
		defer C.OSRDestroySpatialReference(srsCopy)
	}
	outLayer := C.OGR_DS_CreateLayer(ods, layerName, srsCopy, C.wkbPolygon, nil)
	if outLayer == nil {
		return fmt.Errorf("failed to create output layer")
	}

	// Field DN
	fieldName := C.CString("DN")
	defer C.free(unsafe.Pointer(fieldName))
	fd := C.OGR_Fld_Create(fieldName, C.OFTInteger)
	defer C.OGR_Fld_Destroy(fd)
	if C.OGR_L_CreateField(outLayer, fd, 1) != 0 {
		return fmt.Errorf("failed to create DN field")
	}
	outDefn := C.OGR_L_GetLayerDefn(outLayer)

	if err := clipFeaturesInto(layer, outDefn, outLayer, clippedGeom); err != nil {
		return err
	}
	C.OGR_DS_SyncToDisk(ods)
	return nil
}

// CreateEmptyAccumulatorShapefile создаёт пустой накопительный shapefile в заданной СК
// (СК первого обработанного снимка — МСК-59 нужной зоны). Все последующие снимки,
// даже если они относятся к другой зоне МСК-59, будут приведены к этой же СК
// напрямую (зона -> зона), без прохода через WGS84.
func CreateEmptyAccumulatorShapefile(path string, srs C.OGRSpatialReferenceH) error {
	C._gdal_init_vector_once()
	removeShapefileSet(path)

	drvName := C.CString("ESRI Shapefile")
	defer C.free(unsafe.Pointer(drvName))
	drv := C.OGRGetDriverByName(drvName)
	if drv == nil {
		return fmt.Errorf("OGR driver ESRI Shapefile not found")
	}
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	ds := C.OGR_Dr_CreateDataSource(drv, cp, nil)
	if ds == nil {
		return fmt.Errorf("create shp: %s", path)
	}
	defer C.OGR_DS_Destroy(ds)

	layerName := C.CString("result")
	defer C.free(unsafe.Pointer(layerName))
	srsCopy := C._clone_srs(srs)
	if srsCopy != nil {
		defer C.OSRDestroySpatialReference(srsCopy)
	}
	layer := C.OGR_DS_CreateLayer(ds, layerName, srsCopy, C.wkbMultiPolygon, nil)
	if layer == nil {
		return fmt.Errorf("create shp layer failed: %s", path)
	}
	fieldName := C.CString("DN")
	defer C.free(unsafe.Pointer(fieldName))
	fd := C.OGR_Fld_Create(fieldName, C.OFTInteger)
	defer C.OGR_Fld_Destroy(fd)
	if C.OGR_L_CreateField(layer, fd, 1) != 0 {
		return fmt.Errorf("failed to create DN field: %s", path)
	}
	C.OGR_DS_SyncToDisk(ds)
	return nil
}

// ReadLayerSRS открывает векторный файл (shapefile/GeoJSON) и возвращает копию
// пространственной привязки его первого слоя.
func ReadLayerSRS(path string) (C.OGRSpatialReferenceH, error) {
	C._gdal_init_vector_once()
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	ds := C.GDALOpenEx(cp, C.GDAL_OF_VECTOR, nil, nil, nil)
	if ds == nil {
		return nil, fmt.Errorf("open vector: %s", path)
	}
	defer C.GDALClose(ds)
	layer := C.GDALDatasetGetLayer(ds, 0)
	if layer == nil {
		return nil, fmt.Errorf("layer 0 not found: %s", path)
	}
	lsrs := C.OGR_L_GetSpatialRef(layer)
	if lsrs == nil {
		return nil, nil
	}
	return C._clone_srs(lsrs), nil
}

// AppendShapefileToAccumulatorFiltered переносит геометрии из native CRS снимка
// в фиксированную для задания зону МСК-59. После трансформации применяет
// метрические min_area_m2 и simplify. Полный WKT исходного GeoTIFF передаётся
// отдельно, чтобы не терять BOUNDCRS/датум-трансформацию через Shapefile .prj.
func AppendShapefileToAccumulatorFiltered(srcShp string, outShp string, sourceWKT string, targetMSK59Zone int, minAreaM2, simplifyMetres float64) (int, error) {
	C._gdal_init_vector_once()

	srcCP := C.CString(srcShp)
	defer C.free(unsafe.Pointer(srcCP))
	srcDS := C.GDALOpenEx(srcCP, C.GDAL_OF_VECTOR, nil, nil, nil)
	if srcDS == nil {
		return 0, fmt.Errorf("open source vector: %s", srcShp)
	}
	defer C.GDALClose(srcDS)

	srcLayer := C.GDALDatasetGetLayer(srcDS, 0)
	if srcLayer == nil {
		return 0, fmt.Errorf("source layer not found: %s", srcShp)
	}

	dstDS, dstLayer, err := openLayerForAppend(outShp)
	if err != nil {
		return 0, err
	}
	defer C.GDALClose(dstDS)

	dstDefn := C.OGR_L_GetLayerDefn(dstLayer)
	fdDN := C.CString("DN")
	defer C.free(unsafe.Pointer(fdDN))
	srcDNIndex := C.OGR_FD_GetFieldIndex(C.OGR_L_GetLayerDefn(srcLayer), fdDN)
	dstDNIndex := C.OGR_FD_GetFieldIndex(dstDefn, fdDN)

	// Используем полный WKT ИСХОДНОГО GeoTIFF, если он передан. Shapefile .prj
	// может потерять BOUNDCRS/параметры датум-трансформации МСК-59 -> WGS84.
	var srcSRS C.OGRSpatialReferenceH
	if strings.TrimSpace(sourceWKT) != "" {
		parsed, perr := projSRSFromWKT(sourceWKT)
		if perr != nil {
			return 0, fmt.Errorf("parse source raster CRS: %w", perr)
		}
		srcSRS = parsed
		defer C.OSRDestroySpatialReference(srcSRS)
	} else if lsrs := C.OGR_L_GetSpatialRef(srcLayer); lsrs != nil {
		srcSRS = C.OSRClone(lsrs)
		C._set_traditional_gis_order(srcSRS)
		defer C.OSRDestroySpatialReference(srcSRS)
	}
	// .prj Shapefile не сохраняет BOUNDCRS полностью. Для точной трансформации
	// используем полное определение целевой МСК-59, а не SRS, перечитанный из .prj.
	dstSRS, derr := NewMSK59SRS(targetMSK59Zone)
	if derr != nil {
		return 0, derr
	}
	defer C.OSRDestroySpatialReference(dstSRS)
	accepted := 0

	C.OGR_L_ResetReading(srcLayer)
	for {
		feat := C.OGR_L_GetNextFeature(srcLayer)
		if feat == nil {
			break
		}
		geom := C.OGR_F_GetGeometryRef(feat)
		if geom == nil {
			C.OGR_F_Destroy(feat)
			continue
		}
		tg, err := transformGeometryToSRS(geom, srcSRS, dstSRS)
		if err != nil || tg == nil || GeometryIsEmpty(tg) {
			if tg != nil {
				DestroyGeometry(tg)
			}
			C.OGR_F_Destroy(feat)
			continue
		}

		// ВАЖНО: tg уже находится в целевой МСК-59, поэтому OGR_G_Area
		// возвращает м², а simplifyMetres действительно измеряется в метрах
		// независимо от исходной CRS GeoTIFF (WGS84, МСК, UTM и т.п.).
		if minAreaM2 > 0 && float64(C.OGR_G_Area(tg)) < minAreaM2 {
			DestroyGeometry(tg)
			C.OGR_F_Destroy(feat)
			continue
		}
		if simplifyMetres > 0 {
			sg := C.OGR_G_SimplifyPreserveTopology(tg, C.double(simplifyMetres))
			if sg != nil {
				DestroyGeometry(tg)
				tg = sg
			}
		}
		normalized := toMultiPolygon(tg)
		if normalized != tg {
			DestroyGeometry(tg)
			tg = normalized
		}
		outFeat := C.OGR_F_Create(dstDefn)
		if outFeat != nil {
			if C.OGR_F_SetGeometry(outFeat, tg) == 0 {
				if srcDNIndex >= 0 && dstDNIndex >= 0 {
					C.OGR_F_SetFieldInteger(outFeat, dstDNIndex, C.OGR_F_GetFieldAsInteger(feat, srcDNIndex))
				}
				if C.OGR_L_CreateFeature(dstLayer, outFeat) == 0 {
					accepted++
				}
			}
			C.OGR_F_Destroy(outFeat)
		}
		DestroyGeometry(tg)
		C.OGR_F_Destroy(feat)
	}
	C.GDALFlushCache(dstDS)
	return accepted, nil
}

// AppendShapefileToAccumulator сохраняет прежний API для вспомогательного кода.
func AppendShapefileToAccumulator(srcShp string, outShp string) error {
	_, err := AppendShapefileToAccumulatorFiltered(srcShp, outShp, "", 2, 0, 0)
	return err
}
