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
	C.OGR_DS_SyncToDisk(ods)
	return nil
}

func AppendShapefileToGeoJSON(srcShp string, outGeoJSON string) error {
	C._gdal_init_vector_once()

	if _, err := os.Stat(outGeoJSON); err != nil {
		if err := CreateEmptyGeoJSON(outGeoJSON); err != nil {
			return err
		}
	}

	srcCP := C.CString(srcShp)
	defer C.free(unsafe.Pointer(srcCP))
	srcDS := C.GDALOpenEx(srcCP, C.GDAL_OF_VECTOR, nil, nil, nil)
	if srcDS == nil {
		return fmt.Errorf("open source vector: %s", srcShp)
	}
	defer C.GDALClose(srcDS)

	srcLayer := C.GDALDatasetGetLayer(srcDS, 0)
	if srcLayer == nil {
		return fmt.Errorf("source layer not found: %s", srcShp)
	}

	dstDS, dstLayer, err := openLayerForAppend(outGeoJSON)
	if err != nil {
		return err
	}
	defer C.GDALClose(dstDS)

	dstDefn := C.OGR_L_GetLayerDefn(dstLayer)
	fdDN := C.CString("DN")
	defer C.free(unsafe.Pointer(fdDN))
	srcDNIndex := C.OGR_FD_GetFieldIndex(C.OGR_L_GetLayerDefn(srcLayer), fdDN)
	dstDNIndex := C.OGR_FD_GetFieldIndex(dstDefn, fdDN)

	var srcSRS C.OGRSpatialReferenceH
	if lsrs := C.OGR_L_GetSpatialRef(srcLayer); lsrs != nil {
		srcSRS = C.OSRClone(lsrs)
		C._set_traditional_gis_order(srcSRS)
		defer C.OSRDestroySpatialReference(srcSRS)
	}

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
		wg, err := reprojectGeometryToWGS84(geom, srcSRS)
		if err != nil || wg == nil || GeometryIsEmpty(wg) {
			if wg != nil {
				DestroyGeometry(wg)
			}
			C.OGR_F_Destroy(feat)
			continue
		}
		outFeat := C.OGR_F_Create(dstDefn)
		if outFeat != nil {
			if C.OGR_F_SetGeometry(outFeat, wg) == 0 {
				if srcDNIndex >= 0 && dstDNIndex >= 0 {
					C.OGR_F_SetFieldInteger(outFeat, dstDNIndex, C.OGR_F_GetFieldAsInteger(feat, srcDNIndex))
				}
				C.OGR_L_CreateFeature(dstLayer, outFeat)
			}
			C.OGR_F_Destroy(outFeat)
		}
		DestroyGeometry(wg)
		C.OGR_F_Destroy(feat)
	}
	C.GDALFlushCache(dstDS)
	return nil
}

func ExportGeoJSONToShapefile(srcGeoJSON string, dstShp string) error {
	C._gdal_init_vector_once()
	removeShapefileSet(dstShp)

	srcCP := C.CString(srcGeoJSON)
	defer C.free(unsafe.Pointer(srcCP))
	srcDS := C.GDALOpenEx(srcCP, C.GDAL_OF_VECTOR, nil, nil, nil)
	if srcDS == nil {
		return fmt.Errorf("open geojson: %s", srcGeoJSON)
	}
	defer C.GDALClose(srcDS)

	srcLayer := C.GDALDatasetGetLayer(srcDS, 0)
	if srcLayer == nil {
		return fmt.Errorf("geojson layer not found: %s", srcGeoJSON)
	}

	drvName := C.CString("ESRI Shapefile")
	defer C.free(unsafe.Pointer(drvName))
	drv := C.OGRGetDriverByName(drvName)
	if drv == nil {
		return fmt.Errorf("OGR driver ESRI Shapefile not found")
	}

	dstCP := C.CString(dstShp)
	defer C.free(unsafe.Pointer(dstCP))
	dstDS := C.OGR_Dr_CreateDataSource(drv, dstCP, nil)
	if dstDS == nil {
		return fmt.Errorf("create shp: %s", dstShp)
	}
	defer C.OGR_DS_Destroy(dstDS)

	layerName := C.CString("result")
	defer C.free(unsafe.Pointer(layerName))

	wgs84 := C.OSRNewSpatialReference(nil)
	defer C.OSRDestroySpatialReference(wgs84)
	wgsName := C.CString("WGS84")
	defer C.free(unsafe.Pointer(wgsName))
	C.OSRSetWellKnownGeogCS(wgs84, wgsName)
	C._set_traditional_gis_order(wgs84)

	dstLayer := C.OGR_DS_CreateLayer(dstDS, layerName, wgs84, C.wkbPolygon, nil)
	if dstLayer == nil {
		return fmt.Errorf("create shp layer failed")
	}

	fieldName := C.CString("DN")
	defer C.free(unsafe.Pointer(fieldName))
	fd := C.OGR_Fld_Create(fieldName, C.OFTInteger)
	defer C.OGR_Fld_Destroy(fd)
	C.OGR_L_CreateField(dstLayer, fd, 1)

	dstDefn := C.OGR_L_GetLayerDefn(dstLayer)
	srcDefn := C.OGR_L_GetLayerDefn(srcLayer)
	srcDN := C.OGR_FD_GetFieldIndex(srcDefn, fieldName)

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
		outFeat := C.OGR_F_Create(dstDefn)
		C.OGR_F_SetGeometry(outFeat, geom)
		if srcDN >= 0 {
			C.OGR_F_SetFieldInteger(outFeat, 0, C.OGR_F_GetFieldAsInteger(feat, srcDN))
		} else {
			C.OGR_F_SetFieldInteger(outFeat, 0, 1)
		}
		C.OGR_L_CreateFeature(dstLayer, outFeat)
		C.OGR_F_Destroy(outFeat)
		C.OGR_F_Destroy(feat)
	}
	C.OGR_DS_SyncToDisk(dstDS)
	return nil
}
