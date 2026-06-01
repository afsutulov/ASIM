package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// SafeTile оставлен для совместимости с функциями геометрии,
// перенесёнными из sentinel. В orthophoto-пайплайне SAFE-тайлы не используются.
type SafeTile struct {
	Envelope [4]float64
}

// RasterCube нужен только в сигнатуре совместимости.
// Реальный orthophoto-пайплайн его не использует.
type RasterCube struct {
	Data []float32
	W    int
	H    int
	C    int
	Geo  [6]float64
	Proj string
}

// ReadSentinelTileCube в модуле ortophoto не поддерживается.
// Функция оставлена только чтобы проект собирался.
func ReadSentinelTileCube(tile SafeTile, channels []string, cloudMaskMode string) (*RasterCube, error) {
	return nil, fmt.Errorf("ReadSentinelTileCube is not supported in ortophoto module")
}

// CopyShapefileSet копирует комплект файлов shape: .shp/.shx/.dbf/.prj/.cpg
func CopyShapefileSet(srcShp, dstShp string) error {
	srcShp = filepath.Clean(srcShp)
	dstShp = filepath.Clean(dstShp)

	srcBase := strings.TrimSuffix(srcShp, filepath.Ext(srcShp))
	dstBase := strings.TrimSuffix(dstShp, filepath.Ext(dstShp))

	if err := os.MkdirAll(filepath.Dir(dstShp), 0o755); err != nil {
		return err
	}

	for _, ext := range []string{".shp", ".shx", ".dbf", ".prj", ".cpg"} {
		src := srcBase + ext
		if _, err := os.Stat(src); err != nil {
			if ext == ".prj" || ext == ".cpg" {
				continue
			}
			return err
		}

		dst := dstBase + ext
		in, err := os.Open(src)
		if err != nil {
			return err
		}
		out, err := os.Create(dst)
		if err != nil {
			in.Close()
			return err
		}
		if _, err := io.Copy(out, in); err != nil {
			out.Close()
			in.Close()
			return err
		}
		if err := out.Close(); err != nil {
			in.Close()
			return err
		}
		if err := in.Close(); err != nil {
			return err
		}
	}
	return nil
}
