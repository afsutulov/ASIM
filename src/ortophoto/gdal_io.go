package main

import (
	"fmt"
	"log"
	"math"

	"github.com/lukeroth/gdal"
)

type OrthoImage struct {
	Data      []float32 // CHW normalized to [0,1]
	ValidMask []uint8   // 1 = valid pixel, 0 = nodata/empty
	W         int
	H         int
	C         int
	Geo       [6]float64
	Proj      string
}

// fixedScaleForDataType возвращает делитель для режима нормализации "fixed"
// по типу данных растра. Второй результат false — тип не поддерживается
// фиксированной шкалой (например, Float32), нужен фолбэк.
func fixedScaleForDataType(dt gdal.DataType) (float32, bool) {
	switch dt {
	case gdal.Byte:
		return 255, true
	case gdal.UInt16:
		return 65535, true
	case gdal.Int16:
		return 32767, true
	default:
		return 0, false
	}
}

// ReadOrthophoto читает лист ортофото и приводит его к float32 CHW в [0,1].
//
// normalization — режим нормализации (см. ModelSpec.Normalization):
//
//	"minmax" — растяжка по глобальным min/max валидных пикселей листа
//	           (прежнее поведение). ВНИМАНИЕ: контраст входа модели при этом
//	           зависит от содержимого конкретного листа и может отличаться
//	           от препроцессинга при обучении;
//	"fixed"  — деление на диапазон типа данных (Byte -> /255 и т.п.);
//	"none"   — данные уже в [0,1], только клиппинг.
//
// Невалидные пиксели (nodata) закрашиваются белым (1.0) — эта же конвенция
// используется в extractPatch для областей за пределами растра.
//
// Определение nodata:
//  1. если есть альфа-канал (4-я полоса с color interpretation Alpha, либо
//     просто 4-я полоса при 4 полосах — прежнее поведение), пиксели с alpha<=0
//     считаются невалидными;
//  2. дополнительно читается NoDataValue из метаданных GDAL: пиксель невалиден,
//     если во всех трёх каналах значение равно nodata своего канала.
//     Ранее 3-канальные листы вообще не имели детекции nodata, и «пустая»
//     кайма листа обрабатывалась как валидные данные.
func ReadOrthophoto(path string, normalization string) (*OrthoImage, error) {
	ds, err := gdal.Open(path, gdal.ReadOnly)
	if err != nil {
		return nil, err
	}
	defer ds.Close()

	w, h := ds.RasterXSize(), ds.RasterYSize()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("invalid raster size")
	}
	bands := ds.RasterCount()
	if bands < 3 {
		return nil, fmt.Errorf("orthophoto must have at least 3 bands")
	}

	c := 3
	n := w * h
	data := make([]float32, c*n)
	valid := make([]uint8, n)
	for i := range valid {
		valid[i] = 1
	}

	// Альфа-канал: полоса с color interpretation Alpha среди полос 4..bands,
	// либо (как раньше) просто 4-я полоса, если интерпретация не размечена.
	alphaBand := 0
	if bands >= 4 {
		alphaBand = 4
		for bi := 4; bi <= bands; bi++ {
			if ds.RasterBand(bi).ColorInterp() == gdal.CI_AlphaBand {
				alphaBand = bi
				break
			}
		}
	}
	var alpha []float32
	if alphaBand > 0 {
		alpha = make([]float32, n)
		if err := ds.RasterBand(alphaBand).IO(gdal.Read, 0, 0, w, h, alpha, w, h, 0, 0); err != nil {
			return nil, err
		}
	}

	// NoDataValue из метаданных по каждому из трёх каналов.
	nodataVal := [3]float32{}
	nodataSet := [3]bool{}
	dtype := ds.RasterBand(1).RasterDataType()
	for bi := 0; bi < c; bi++ {
		if v, ok := ds.RasterBand(bi + 1).NoDataValue(); ok {
			nodataVal[bi] = float32(v)
			nodataSet[bi] = true
		}
	}
	anyNodata := nodataSet[0] || nodataSet[1] || nodataSet[2]

	// Читаем каналы напрямую в целевой срез (без промежуточного буфера —
	// экономия одного полного канала памяти на больших листах).
	for bi := 0; bi < c; bi++ {
		if err := ds.RasterBand(bi+1).IO(gdal.Read, 0, 0, w, h, data[bi*n:(bi+1)*n], w, h, 0, 0); err != nil {
			return nil, err
		}
	}

	globalMin := float32(math.MaxFloat32)
	globalMax := float32(-math.MaxFloat32)
	for i := 0; i < n; i++ {
		if alpha != nil && alpha[i] <= 0 {
			valid[i] = 0
			continue
		}
		if anyNodata {
			isNodata := true
			for bi := 0; bi < c; bi++ {
				if !nodataSet[bi] || data[bi*n+i] != nodataVal[bi] {
					isNodata = false
					break
				}
			}
			if isNodata {
				valid[i] = 0
				continue
			}
		}
		for bi := 0; bi < c; bi++ {
			v := data[bi*n+i]
			if v < globalMin {
				globalMin = v
			}
			if v > globalMax {
				globalMax = v
			}
		}
	}

	mode := normalization
	if mode == "" {
		mode = "minmax"
	}
	var scale, offset float32
	switch mode {
	case "fixed":
		if s, ok := fixedScaleForDataType(dtype); ok {
			scale, offset = s, 0
		} else {
			log.Printf("ortophoto normalization=fixed unsupported for data type %s, falling back to minmax: file=%s", dtype.Name(), path)
			mode = "minmax"
		}
	case "none":
		scale, offset = 1, 0
	}
	if mode == "minmax" {
		if globalMax > globalMin {
			scale, offset = globalMax-globalMin, globalMin
		} else {
			// Лист без вариации значений — данные неинформативны.
			for i := range data {
				data[i] = 0
			}
			scale = 0
		}
	}
	if scale > 0 {
		for cidx := 0; cidx < c; cidx++ {
			base := cidx * n
			for i := 0; i < n; i++ {
				if valid[i] == 0 {
					data[base+i] = 1.0
					continue
				}
				v := (data[base+i] - offset) / scale
				if v < 0 {
					v = 0
				}
				if v > 1 {
					v = 1
				}
				data[base+i] = v
			}
		}
	}

	gt := ds.GeoTransform()
	var geo [6]float64
	copy(geo[:], gt[:])
	return &OrthoImage{
		Data:      data,
		ValidMask: valid,
		W:         w,
		H:         h,
		C:         c,
		Geo:       geo,
		Proj:      ds.Projection(),
	}, nil
}

// WriteGeoTIFFMask пишет бинарную маску (0/1) как однополосный Byte GeoTIFF
// со сжатием DEFLATE. Прежняя реализация писала несжатый Float32 —
// в 4 раза больше памяти/диска на промежуточный файл без какой-либо пользы:
// GDALPolygonize прекрасно работает с Byte.
func WriteGeoTIFFMask(path string, mask []uint8, w, h int, geo [6]float64, proj string) error {
	drv, err := gdal.GetDriverByName("GTiff")
	if err != nil {
		return err
	}
	ds := drv.Create(path, w, h, 1, gdal.Byte, []string{"COMPRESS=DEFLATE", "TILED=YES"})
	defer ds.Close()
	ds.SetGeoTransform(geo)
	if proj != "" {
		ds.SetProjection(proj)
	}
	band := ds.RasterBand(1)
	return band.IO(gdal.Write, 0, 0, w, h, mask, w, h, 0, 0)
}

type OrthoImageInfo struct {
	W    int
	H    int
	Geo  [6]float64
	Proj string
}

// ReadOrthophotoInfo reads only raster metadata. It is used for a cheap
// spatial pre-filter before loading all pixels and running YOLO inference.
func ReadOrthophotoInfo(path string) (*OrthoImageInfo, error) {
	ds, err := gdal.Open(path, gdal.ReadOnly)
	if err != nil {
		return nil, err
	}
	defer ds.Close()
	w, h := ds.RasterXSize(), ds.RasterYSize()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("invalid raster size")
	}
	gt := ds.GeoTransform()
	var geo [6]float64
	copy(geo[:], gt[:])
	return &OrthoImageInfo{W: w, H: h, Geo: geo, Proj: ds.Projection()}, nil
}
