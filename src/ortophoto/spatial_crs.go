package main

/*
#cgo pkg-config: gdal
#include <stdlib.h>
#include <string.h>
#include "cpl_string.h"
#include "cpl_vsi.h"
#include "ogr_srs_api.h"
static void _set_traditional_gis_order_msk(OGRSpatialReferenceH srs) {
    if (srs != NULL) OSRSetAxisMappingStrategy(srs, OAMS_TRADITIONAL_GIS_ORDER);
}

// Параметры проекции, по которым МСК-59 распознаётся независимо от имени СК.
typedef struct {
    int ok;             // 1 — все параметры удалось прочитать
    int tmerc;          // 1 — проекция Transverse Mercator
    double lat0, lon0, k0, fe, fn;
    double semi_major, inv_flattening, linear_units;
} msk_proj_params;

static msk_proj_params _read_tm_params(OGRSpatialReferenceH srs) {
    msk_proj_params p;
    memset(&p, 0, sizeof(p));
    if (srs == NULL || !OSRIsProjected(srs)) return p;
    const char *proj = OSRGetAttrValue(srs, "PROJECTION", 0);
    p.tmerc = (proj != NULL && EQUAL(proj, SRS_PT_TRANSVERSE_MERCATOR)) ? 1 : 0;
    OGRErr e = OGRERR_NONE;
    int bad = 0;
    p.lat0 = OSRGetProjParm(srs, SRS_PP_LATITUDE_OF_ORIGIN, 0.0, &e);  bad |= (e != OGRERR_NONE);
    p.lon0 = OSRGetProjParm(srs, SRS_PP_CENTRAL_MERIDIAN, 0.0, &e);    bad |= (e != OGRERR_NONE);
    p.k0   = OSRGetProjParm(srs, SRS_PP_SCALE_FACTOR, 1.0, &e);        bad |= (e != OGRERR_NONE);
    p.fe   = OSRGetProjParm(srs, SRS_PP_FALSE_EASTING, 0.0, &e);       bad |= (e != OGRERR_NONE);
    p.fn   = OSRGetProjParm(srs, SRS_PP_FALSE_NORTHING, 0.0, &e);      bad |= (e != OGRERR_NONE);
    p.semi_major = OSRGetSemiMajor(srs, &e);                           bad |= (e != OGRERR_NONE);
    p.inv_flattening = OSRGetInvFlattening(srs, &e);                   bad |= (e != OGRERR_NONE);
    p.linear_units = OSRGetLinearUnits(srs, NULL);
    p.ok = bad ? 0 : 1;
    return p;
}
*/
import "C"

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"
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

// MSK59CentralMeridian и MSK59FalseEasting — параметры зоны МСК-59.
func MSK59CentralMeridian(zone int) float64 { return 53.55 + 3.0*float64(zone-1) }
func MSK59FalseEasting(zone int) float64    { return 1250000.0 + 1000000.0*float64(zone-1) }

const msk59FalseNorthing = -5914743.504

// MSK59WKT возвращает эталонный WKT2 (BOUNDCRS с ГОСТ-параметрами
// Pulkovo 1942 -> WGS84) для зоны МСК-59. Зона должна быть 1..3.
func MSK59WKT(zone int) string {
	lon0 := MSK59CentralMeridian(zone)
	falseEasting := MSK59FalseEasting(zone)
	return fmt.Sprintf(`BOUNDCRS[
SOURCECRS[PROJCRS["МСК 59-%d",
BASEGEOGCRS["Pulkovo 1942 ГОСТ",DATUM["unnamed",ELLIPSOID["Krassowsky 1940",6378245,298.3,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],
CONVERSION["unnamed",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",8801]]],PARAMETER["Longitude of natural origin",%.2f,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",8802]]],PARAMETER["Scale factor at natural origin",1,SCALEUNIT["unity",1,ID["EPSG",9201]]],PARAMETER["False easting",%.3f,LENGTHUNIT["metre",1,ID["EPSG",8806]]],PARAMETER["False northing",-5914743.504,LENGTHUNIT["metre",1,ID["EPSG",8807]]]],
CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]],
TARGETCRS[GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]],
ABRIDGEDTRANSFORMATION["Transformation from Pulkovo 1942 ГОСТ to WGS84",METHOD["Position Vector transformation (geog2D domain)",ID["EPSG",9606]],PARAMETER["X-axis translation",23.57,ID["EPSG",8605]],PARAMETER["Y-axis translation",-140.95,ID["EPSG",8606]],PARAMETER["Z-axis translation",-79.8,ID["EPSG",8607]],PARAMETER["X-axis rotation",0,ID["EPSG",8608]],PARAMETER["Y-axis rotation",0.35,ID["EPSG",8609]],PARAMETER["Z-axis rotation",0.79,ID["EPSG",8610]],PARAMETER["Scale difference",0.99999978,ID["EPSG",8611]]]]]`, zone, lon0, falseEasting)
}

// NewMSK59SRS создаёт полное BOUNDCRS МСК-59 с преобразованием Pulkovo 1942
// ГОСТ -> WGS84. Параметры соответствуют GeoTIFF МСК 59-2, используемым в
// проекте; между зонами меняются центральный меридиан и false easting.
func NewMSK59SRS(zone int) (C.OGRSpatialReferenceH, error) {
	if zone < 1 || zone > 3 {
		return nil, fmt.Errorf("unsupported MSK-59 zone: %d", zone)
	}
	wkt := MSK59WKT(zone)

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

// SourceCRSInfo — результат анализа CRS исходного GeoTIFF.
type SourceCRSInfo struct {
	// Group — в какой публичный набор результата попадает снимок.
	Group SourceCRSGroup
	// Detail — человекочитаемое пояснение для журнала.
	Detail string
	// MSKZone — зона МСК-59 (1..3), если она однозначно определена по
	// параметрам проекции; 0 — зона не определена.
	MSKZone int
	// WKT — CRS, которую нужно использовать для ВСЕХ координатных операций со
	// снимком (префильтр, обрезка по полигону поиска, перенос в накопитель,
	// слой покрытия). Для распознанной по параметрам МСК-59 это эталонный
	// BOUNDCRS с ГОСТ-параметрами датум-сдвига: во многих GeoTIFF МСК-59
	// записана «голой» PROJCS без TOWGS84 и под именем "unknown".
	WKT string
}

// Допуски сравнения параметров проекции. Значения МСК-59 заданы с точностью
// до миллиметра/стотысячной градуса, поэтому допуски заведомо меньше, чем
// различие между зонами, и больше, чем шум округления при записи GeoTIFF.
const (
	mskTolDeg   = 1e-6
	mskTolMetre = 0.01
	mskTolScale = 1e-9
)

func nearlyEqual(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// msk59ZoneByParams возвращает зону МСК-59 (1..3), если параметры проекции
// совпадают с одной из зон МСК-59, иначе 0. Имя CRS при этом не учитывается.
func msk59ZoneByParams(srs C.OGRSpatialReferenceH) (int, string) {
	p := C._read_tm_params(srs)
	if p.ok == 0 || p.tmerc == 0 {
		return 0, ""
	}
	if !nearlyEqual(float64(p.semi_major), 6378245.0, 0.5) ||
		!nearlyEqual(float64(p.inv_flattening), 298.3, 1e-6) {
		return 0, ""
	}
	if !nearlyEqual(float64(p.linear_units), 1.0, 1e-12) ||
		!nearlyEqual(float64(p.lat0), 0, mskTolDeg) ||
		!nearlyEqual(float64(p.k0), 1.0, mskTolScale) ||
		!nearlyEqual(float64(p.fn), msk59FalseNorthing, mskTolMetre) {
		return 0, ""
	}
	for zone := 1; zone <= 3; zone++ {
		if nearlyEqual(float64(p.lon0), MSK59CentralMeridian(zone), mskTolDeg) &&
			nearlyEqual(float64(p.fe), MSK59FalseEasting(zone), mskTolMetre) {
			return zone, fmt.Sprintf("lon0=%.2f fe=%.0f", float64(p.lon0), float64(p.fe))
		}
	}
	return 0, ""
}

// normalizeCRSName оставляет в строке только буквы и цифры в нижнем
// регистре: "МСК 59-2", "MSK59_zone2", "msk-59", "МСК–59" → "мск592"/"msk59zone2".
func normalizeCRSName(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// wktNamesMSK59 — запасной признак по имени (если параметры проекции
// нестандартны, но подрядчик явно назвал СК МСК-59).
func wktNamesMSK59(wkt string) bool {
	n := normalizeCRSName(wkt)
	return strings.Contains(n, "мск59") || strings.Contains(n, "msk59")
}

// DetectSourceCRS разделяет входные снимки на группы результата:
//   - МСК-59 -> результат остаётся в МСК-59;
//   - geographic CRS (в т.ч. EPSG:4326) -> результат идёт в WGS84/EPSG:4326;
//   - прочие projected CRS (например UTM) -> WGS84-группа.
//
// МСК-59 распознаётся В ПЕРВУЮ ОЧЕРЕДЬ ПО ПАРАМЕТРАМ ПРОЕКЦИИ (Transverse
// Mercator, эллипсоид Красовского, lon0 = 53.55/56.55/59.55, FE =
// 1250000/2250000/3250000, FN = -5914743.504), а не по имени СК. Имя СК в
// GeoTIFF ненадёжно: GDAL/PROJ и ПО подрядчиков записывают его как "unknown",
// "unnamed", "MSK59 zone 2", в CP1251 и т.п.
func DetectSourceCRS(wkt string) (SourceCRSInfo, error) {
	if strings.TrimSpace(wkt) == "" {
		return SourceCRSInfo{}, fmt.Errorf("empty raster CRS")
	}
	srs, err := projSRSFromWKT(wkt)
	if err != nil {
		return SourceCRSInfo{}, err
	}
	defer C.OSRDestroySpatialReference(srs)

	if zone, params := msk59ZoneByParams(srs); zone > 0 {
		return SourceCRSInfo{
			Group:   SourceCRSMSK59,
			Detail:  fmt.Sprintf("MSK-59 zone %d (by projection params: %s)", zone, params),
			MSKZone: zone,
			WKT:     MSK59WKT(zone),
		}, nil
	}
	if wktNamesMSK59(wkt) {
		// Имя говорит «МСК-59», но параметры не совпали ни с одной зоной.
		// Оставляем исходный WKT снимка без подмены.
		return SourceCRSInfo{
			Group:  SourceCRSMSK59,
			Detail: "MSK-59 (by CRS name, non-standard params, native WKT kept)",
			WKT:    wkt,
		}, nil
	}
	if C.OSRIsGeographic(srs) != 0 {
		return SourceCRSInfo{Group: SourceCRSWGS84, Detail: "geographic->WGS84", WKT: wkt}, nil
	}
	return SourceCRSInfo{Group: SourceCRSWGS84, Detail: "projected-non-MSK59->WGS84", WKT: wkt}, nil
}

// ClassifySourceCRS сохраняет прежний API.
func ClassifySourceCRS(wkt string) (SourceCRSGroup, string, error) {
	info, err := DetectSourceCRS(wkt)
	if err != nil {
		return "", "", err
	}
	return info.Group, info.Detail, nil
}

// MSK59WKT1 возвращает однострочный WKT1 зоны МСК-59 с TOWGS84 (ГОСТ).
// WKT1 выбран потому, что его одинаково читают GDAL/OGR, QGIS и ArcGIS.
func MSK59WKT1(zone int) (string, error) {
	srs, err := NewMSK59SRS(zone)
	if err != nil {
		return "", err
	}
	defer C.OSRDestroySpatialReference(srs)
	var out *C.char
	if C.OSRExportToWkt(srs, &out) != 0 || out == nil {
		return "", fmt.Errorf("export MSK-59 zone %d to WKT failed", zone)
	}
	defer C.VSIFree(unsafe.Pointer(out))
	return C.GoString(out), nil
}

// InjectGeoJSONCRS добавляет в GeoJSON член верхнего уровня
//
//	"crs": {"type": "name", "properties": {"name": "<WKT>"}}
//
// Драйвер GDAL GeoJSON записывает "crs" ТОЛЬКО для СК с кодом EPSG. У МСК-59
// кода EPSG нет, поэтому файл выходил вообще без СК, и QGIS/GDAL открывали
// его как WGS84: координаты ~(2 232 000; 516 000) трактовались как градусы,
// и полигоны «не находились» ни рядом с ортофото, ни где-либо на карте.
// Член "name" GDAL читает через SetFromUserInput, поэтому WKT в нём
// корректно распознаётся (проверено ogrinfo/ogr2ogr).
func InjectGeoJSONCRS(path string, crsWKT string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		return fmt.Errorf("parse GeoJSON %s: %w", path, err)
	}
	crs := map[string]any{
		"type":       "name",
		"properties": map[string]string{"name": crsWKT},
	}
	crsJSON, err := json.Marshal(crs)
	if err != nil {
		return err
	}
	if _, ok := top["crs"]; ok {
		top["crs"] = crsJSON
		// Порядок ключей не важен для GeoJSON-читателей.
		out, err := json.Marshal(top)
		if err != nil {
			return err
		}
		return os.WriteFile(path, out, 0o644)
	}
	// Вставляем "crs" сразу после открывающей скобки, сохраняя остальной
	// текст GDAL без изменений (порядок и форматирование координат).
	i := bytes.IndexByte(data, '{')
	if i < 0 {
		return fmt.Errorf("malformed GeoJSON: %s", path)
	}
	var buf bytes.Buffer
	buf.Grow(len(data) + len(crsJSON) + 16)
	buf.Write(data[:i+1])
	buf.WriteString("\n\"crs\": ")
	buf.Write(crsJSON)
	buf.WriteString(",")
	buf.Write(data[i+1:])
	return os.WriteFile(path, buf.Bytes(), 0o644)
}
