package main

/*
#cgo pkg-config: gdal
#include <stdlib.h>
#include "ogr_srs_api.h"
static void _set_traditional_gis_order_msk(OGRSpatialReferenceH srs) {
    if (srs != NULL) OSRSetAxisMappingStrategy(srs, OAMS_TRADITIONAL_GIS_ORDER);
}
*/
import "C"

import (
	"fmt"
	"strings"
	"unsafe"
)

// MSK59ZoneForLongitude выбирает одну из трёх зон МСК-59 Пермского края.
// Центральные меридианы зон: 53.55, 56.55 и 59.55 градуса. Границы выбора
// проходят посередине между центральными меридианами. Для задания зона
// определяется по центру поискового полигона, а не по CRS конкретного TIFF.
func MSK59ZoneForLongitude(lon float64) int {
	if lon < 55.05 {
		return 1
	}
	if lon < 58.05 {
		return 2
	}
	return 3
}

// NewMSK59SRS создаёт полное BOUNDCRS МСК-59 с преобразованием Pulkovo 1942
// ГОСТ -> WGS84. Параметры соответствуют GeoTIFF МСК 59-2, используемым в
// проекте; между зонами меняются центральный меридиан и false easting.
func NewMSK59SRS(zone int) (C.OGRSpatialReferenceH, error) {
	if zone < 1 || zone > 3 {
		return nil, fmt.Errorf("unsupported MSK-59 zone: %d", zone)
	}
	lon0 := 53.55 + 3.0*float64(zone-1)
	falseEasting := 1250000.0 + 1000000.0*float64(zone-1)

	wkt := fmt.Sprintf(`BOUNDCRS[
SOURCECRS[PROJCRS["МСК 59-%d",
BASEGEOGCRS["Pulkovo 1942 ГОСТ",DATUM["unnamed",ELLIPSOID["Krassowsky 1940",6378245,298.3,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],
CONVERSION["unnamed",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",8801]]],PARAMETER["Longitude of natural origin",%.2f,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",8802]]],PARAMETER["Scale factor at natural origin",1,SCALEUNIT["unity",1,ID["EPSG",9201]]],PARAMETER["False easting",%.3f,LENGTHUNIT["metre",1,ID["EPSG",8806]]],PARAMETER["False northing",-5914743.504,LENGTHUNIT["metre",1,ID["EPSG",8807]]]],
CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]],
TARGETCRS[GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]],
ABRIDGEDTRANSFORMATION["Transformation from Pulkovo 1942 ГОСТ to WGS84",METHOD["Position Vector transformation (geog2D domain)",ID["EPSG",9606]],PARAMETER["X-axis translation",23.57,ID["EPSG",8605]],PARAMETER["Y-axis translation",-140.95,ID["EPSG",8606]],PARAMETER["Z-axis translation",-79.8,ID["EPSG",8607]],PARAMETER["X-axis rotation",0,ID["EPSG",8608]],PARAMETER["Y-axis rotation",0.35,ID["EPSG",8609]],PARAMETER["Z-axis rotation",0.79,ID["EPSG",8610]],PARAMETER["Scale difference",0.99999978,ID["EPSG",8611]]]]]`, zone, lon0, falseEasting)

	srs := C.OSRNewSpatialReference(nil)
	if srs == nil {
		return nil, fmt.Errorf("OSRNewSpatialReference failed")
	}
	cwkt := C.CString(wkt)
	defer C.free(unsafe.Pointer(cwkt))
	pwkt := cwkt
	if C.OSRImportFromWkt(srs, (**C.char)(unsafe.Pointer(&pwkt))) != 0 {
		C.OSRDestroySpatialReference(srs)
		return nil, fmt.Errorf("OSRImportFromWkt MSK-59 zone %d failed", zone)
	}
	C._set_traditional_gis_order_msk(srs)
	return srs, nil
}

// SourceCRSGroup определяет, в какой публичный набор результата должен попасть
// результат конкретного исходного GeoTIFF. ВАЖНО: это классификация ИСТОЧНИКА,
// а не желаемой системы координат общего задания.
type SourceCRSGroup string

const (
	SourceCRSMSK59 SourceCRSGroup = "msk59"
	SourceCRSWGS84 SourceCRSGroup = "wgs84"
)

// NewWGS84SRS создаёт EPSG:4326 с традиционным GIS-порядком осей lon,lat.
func NewWGS84SRS() (C.OGRSpatialReferenceH, error) {
	srs := C.OSRNewSpatialReference(nil)
	if srs == nil {
		return nil, fmt.Errorf("OSRNewSpatialReference WGS84 failed")
	}
	if C.OSRImportFromEPSG(srs, 4326) != 0 {
		C.OSRDestroySpatialReference(srs)
		return nil, fmt.Errorf("OSRImportFromEPSG(4326) failed")
	}
	C._set_traditional_gis_order_msk(srs)
	return srs, nil
}

// ClassifySourceCRS разделяет входные снимки на две группы результата:
//   - МСК-59 -> результат остаётся в МСК-59;
//   - geographic CRS (в т.ч. EPSG:4326) -> результат идёт в WGS84/EPSG:4326.
//
// Для других projected CRS (например UTM) используем WGS84-группу: такие
// координаты не являются МСК-59, а публичный международный результат всё равно
// нормализуется в EPSG:4326. Это позволяет принимать материалы разных подрядчиков.
func ClassifySourceCRS(wkt string) (SourceCRSGroup, string, error) {
	if strings.TrimSpace(wkt) == "" {
		return "", "", fmt.Errorf("empty raster CRS")
	}
	srs, err := projSRSFromWKT(wkt)
	if err != nil {
		return "", "", err
	}
	defer C.OSRDestroySpatialReference(srs)

	lower := strings.ToLower(wkt)
	isMSK59 := strings.Contains(lower, "мск 59") ||
		strings.Contains(lower, "мск-59") ||
		strings.Contains(lower, "msk 59") ||
		strings.Contains(lower, "msk-59")
	if isMSK59 {
		return SourceCRSMSK59, "MSK-59", nil
	}
	if C.OSRIsGeographic(srs) != 0 {
		return SourceCRSWGS84, "geographic->WGS84", nil
	}
	return SourceCRSWGS84, "projected-non-MSK59->WGS84", nil
}
