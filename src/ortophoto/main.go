package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

func printUsage(configPath string) {
	fmt.Fprintln(os.Stderr, "ГБУ ПК \"Центр информационного развития Пермского края\". 2026 год")
	fmt.Fprintln(os.Stderr, "Модуль поиска границ земельных участков по ортофотоснимкам")
	fmt.Fprintln(os.Stderr)
	fmt.Fprintf(os.Stderr, "Пример: ./ortophoto --config %s --model uchastki --poligon kray --start 2023 --end 2024 --out run1\n", configPath)
}

func initLogger(logDir string) (ioWriteCloser, error) {
	if logDir == "" {
		return nil, nil
	}
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, err
	}
	fp := filepath.Join(logDir, "asim.log")
	f, err := os.OpenFile(fp, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	mw := NewMultiWriteCloser(os.Stdout, f)
	log.SetOutput(mw)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	return mw, nil
}

type multiWriteCloser struct {
	writers []ioWriteCloser
}
type ioWriteCloser interface {
	Write([]byte) (int, error)
	Close() error
}
type nopWriteCloser struct{ *os.File }

func (n nopWriteCloser) Close() error { return nil }
func NewMultiWriteCloser(stdout *os.File, file *os.File) *multiWriteCloser {
	return &multiWriteCloser{writers: []ioWriteCloser{nopWriteCloser{stdout}, file}}
}
func (m *multiWriteCloser) Write(p []byte) (int, error) {
	for _, w := range m.writers {
		if _, err := w.Write(p); err != nil {
			return 0, err
		}
	}
	return len(p), nil
}
func (m *multiWriteCloser) Close() error {
	var first error
	for _, w := range m.writers {
		if err := w.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func main() {
	configPath := flag.String("config", DefaultConfigPath(), "Путь к config.json")
	modelName := flag.String("model", "", "Название модели из config.json с preprocess=ortophoto")
	poligon := flag.String("poligon", "kray", "Идентификатор полигона поиска из poligons.json")
	startYear := flag.Int("start", 0, "Год начала поиска")
	endYear := flag.Int("end", 0, "Год завершения поиска")
	outputName := flag.String("out", "", "Имя выходного архива zip без пути")
	flag.Parse()

	if *modelName == "" || *startYear == 0 || *endYear == 0 || *outputName == "" {
		printUsage(*configPath)
		os.Exit(2)
	}
	if *startYear > *endYear {
		fmt.Fprintln(os.Stderr, "--start не может быть больше --end")
		os.Exit(2)
	}

	cfg, err := LoadAppConfig(*configPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка загрузки config.json:", err)
		printUsage(*configPath)
		os.Exit(1)
	}

	logFile, err := initLogger(cfg.Logs)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка инициализации лога:", err)
		os.Exit(1)
	}
	if logFile != nil {
		defer logFile.Close()
	}

	spec, ok := cfg.Models[*modelName]
	if !ok {
		fmt.Fprintln(os.Stderr, "Неизвестная модель ortophoto:", *modelName)
		os.Exit(2)
	}
	if err := os.MkdirAll(cfg.ResultsPath, 0o755); err != nil {
		log.Printf("mkdir results failed: %v", err)
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := os.MkdirAll(cfg.Tmp, 0o755); err != nil {
		log.Printf("mkdir tmp failed: %v", err)
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	log.Printf("ortophoto process started: model=%s output=%s years=%d..%d poligon=%s ortophoto=%s device=%s gpus=%v", spec.Name, *outputName, *startYear, *endYear, *poligon, cfg.Ortophoto, cfg.Compute.Device, cfg.Compute.GPUs)
	zipPath, stats, err := RunProcess(ProcessConfig{
		App:           cfg,
		Spec:          spec,
		SearchPolygon: *poligon,
		StartYear:     *startYear,
		EndYear:       *endYear,
		OutputName:    *outputName,
	})
	if err != nil {
		log.Printf("ortophoto failed: %v; files_seen=%d files_processed=%d model_runs=%d result_polygons=%d", err, stats.FilesSeen, stats.FilesProcessed, stats.ModelRuns, stats.ResultPolygons)
		log.Printf("ortophoto process stopped: status=error")
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	log.Printf("ortophoto completed: %s files_seen=%d files_processed=%d model_runs=%d result_polygons=%d avg_image=%s total=%s", zipPath, stats.FilesSeen, stats.FilesProcessed, stats.ModelRuns, stats.ResultPolygons, stats.AvgPerImage.String(), stats.TotalDuration.String())
	log.Printf("ortophoto process stopped: status=ok")
	fmt.Println(zipPath)
}
