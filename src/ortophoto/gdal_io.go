package main

import (
	"fmt"
	"github.com/lukeroth/gdal"
	"math"
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

func ReadOrthophoto(path string) (*OrthoImage, error) {
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
	data := make([]float32, c*w*h)
	valid := make([]uint8, w*h)
	for i := range valid {
		valid[i] = 1
	}

	hasAlpha := bands >= 4
	var alpha []float32
	if hasAlpha {
		alpha = make([]float32, w*h)
		if err := ds.RasterBand(4).IO(gdal.Read, 0, 0, w, h, alpha, w, h, 0, 0); err != nil {
			return nil, err
		}
	}

	globalMin := float32(math.MaxFloat32)
	globalMax := float32(-math.MaxFloat32)
	for bi := 0; bi < c; bi++ {
		buf := make([]float32, w*h)
		if err := ds.RasterBand(bi+1).IO(gdal.Read, 0, 0, w, h, buf, w, h, 0, 0); err != nil {
			return nil, err
		}
		copy(data[bi*w*h:(bi+1)*w*h], buf)
		for i, v := range buf {
			if hasAlpha && alpha[i] <= 0 {
				valid[i] = 0
				continue
			}
			if v < globalMin {
				globalMin = v
			}
			if v > globalMax {
				globalMax = v
			}
		}
	}
	// normalize only by global min/max of valid pixels
	if globalMax > globalMin {
		denom := globalMax - globalMin
		for cidx := 0; cidx < c; cidx++ {
			base := cidx * w * h
			for i := 0; i < w*h; i++ {
				if valid[i] == 0 {
					data[base+i] = 1.0
					continue
				}
				data[base+i] = (data[base+i] - globalMin) / denom
				if data[base+i] < 0 {
					data[base+i] = 0
				}
				if data[base+i] > 1 {
					data[base+i] = 1
				}
			}
		}
	} else {
		for i := range data {
			data[i] = 0
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

func WriteGeoTIFF1(path string, data []float32, w, h int, geo [6]float64, proj string) error {
	drv, err := gdal.GetDriverByName("GTiff")
	if err != nil {
		return err
	}
	ds := drv.Create(path, w, h, 1, gdal.Float32, nil)
	defer ds.Close()
	ds.SetGeoTransform(geo)
	if proj != "" {
		ds.SetProjection(proj)
	}
	band := ds.RasterBand(1)
	return band.IO(gdal.Write, 0, 0, w, h, data, w, h, 0, 0)
}
