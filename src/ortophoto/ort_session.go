package main

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var ortOnce sync.Once
var ortInitErr error

func initORTWithConfig(cfg *AppConfig) error {
	ortOnce.Do(func() {
		lib := strings.TrimSpace(os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"))
		if lib == "" && cfg != nil {
			lib = strings.TrimSpace(cfg.ORTLibraryPath)
		}
		if lib != "" {
			ort.SetSharedLibraryPath(lib)
			log.Printf("onnxruntime shared library: %s", lib)
		} else {
			log.Printf("onnxruntime shared library: default lookup")
		}
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

type ORTSession struct {
	sess        *ort.DynamicAdvancedSession
	device      string
	inputName   string
	outputNames []string
	// Формы выходов, прочитанные из метаданных ONNX-модели при открытии
	// сессии (nil, если модель экспортирована с динамическими размерами).
	// Используются вместо жёстко зашитых предположений о страйдах головы
	// детектора (8/16/32) и страйде proto-выхода.
	detShape   []int64 // [1, attrs, detCount]
	protoShape []int64 // [1, maskDim, protoH, protoW]
}

func NewORTSession(cfg *AppConfig, modelPath string, inputName, outputName string) (*ORTSession, error) {
	return NewORTSessionWithOutputs(cfg, modelPath, inputName, []string{outputName})
}

func NewORTSessionWithOutputs(cfg *AppConfig, modelPath string, inputName string, outputNames []string) (*ORTSession, error) {
	if err := initORTWithConfig(cfg); err != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	_ = opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)

	device := strings.ToLower(strings.TrimSpace(cfg.Compute.Device))
	if device == "" {
		device = "cuda"
	}
	if device == "cuda" {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("create CUDA provider options: %w", err)
		}
		defer cudaOpts.Destroy()
		if err := cudaOpts.Update(map[string]string{
			"device_id": fmt.Sprintf("%d", cfg.Compute.DeviceID),
		}); err != nil {
			return nil, fmt.Errorf("update CUDA provider options: %w", err)
		}
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			return nil, fmt.Errorf("append CUDA execution provider: %w", err)
		}
		log.Printf("onnxruntime session configured for CUDA: device_id=%d model=%s", cfg.Compute.DeviceID, modelPath)
	} else {
		log.Printf("onnxruntime session configured for CPU: model=%s", modelPath)
	}

	if len(outputNames) == 0 {
		return nil, fmt.Errorf("no output names configured")
	}
	s, err := ort.NewDynamicAdvancedSession(modelPath, []string{inputName}, outputNames, opts)
	if err != nil {
		return nil, err
	}
	sess := &ORTSession{sess: s, device: device, inputName: inputName, outputNames: append([]string(nil), outputNames...)}
	sess.detShape, sess.protoShape = readYOLOSegOutputShapes(modelPath, outputNames)
	return sess, nil
}

// readYOLOSegOutputShapes читает формы выходов YOLO-seg модели из метаданных
// ONNX-файла. Возвращает (nil, nil), если метаданные недоступны или размеры
// динамические — тогда PredictYOLOSeg вычисляет формы по классическим
// страйдам (8/16/32 и proto stride). Раньше страйды были зашиты жёстко:
// модель с нестандартным экспортом (головы P2/P6, иной proto stride) падала
// на Run() с малопонятной ошибкой несоответствия формы.
func readYOLOSegOutputShapes(modelPath string, outputNames []string) ([]int64, []int64) {
	_, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		log.Printf("onnx model metadata unavailable, YOLO output shapes fall back to stride formulas: model=%s error=%v", modelPath, err)
		return nil, nil
	}
	byName := make(map[string][]int64, len(outputs))
	for _, o := range outputs {
		byName[o.Name] = append([]int64(nil), o.Dimensions...)
	}
	staticShape := func(name string, dims int) []int64 {
		s, ok := byName[name]
		if !ok || len(s) != dims {
			return nil
		}
		for _, d := range s {
			if d <= 0 { // динамический размер
				return nil
			}
		}
		return s
	}
	if len(outputNames) < 2 {
		return nil, nil
	}
	det := staticShape(outputNames[0], 3)
	proto := staticShape(outputNames[1], 4)
	if det == nil || proto == nil {
		return nil, nil
	}
	log.Printf("onnx model output shapes: det=%v proto=%v model=%s", det, proto, modelPath)
	return det, proto
}

func (s *ORTSession) Close() error {
	if s.sess != nil {
		return s.sess.Destroy()
	}
	return nil
}

func (s *ORTSession) Predict(input []float32, batch, channels, height, width, outChannels int) ([]float32, error) {
	inShape := ort.NewShape(int64(batch), int64(channels), int64(height), int64(width))
	inTensor, err := ort.NewTensor[float32](inShape, input)
	if err != nil {
		return nil, err
	}
	defer inTensor.Destroy()
	outShape := ort.NewShape(int64(batch), int64(outChannels), int64(height), int64(width))
	outTensor, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, err
	}
	defer outTensor.Destroy()
	if err := s.sess.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return nil, err
	}
	out := outTensor.GetData()
	cpy := make([]float32, len(out))
	copy(cpy, out)
	return cpy, nil
}

func yoloDetectionCount(tile int) int64 {
	// YOLO detector heads at strides 8, 16 and 32.
	// For 1536: 192*192 + 96*96 + 48*48 = 48384.
	n := int64(0)
	for _, stride := range []int{8, 16, 32} {
		g := (tile + stride - 1) / stride
		n += int64(g * g)
	}
	return n
}

func (s *ORTSession) PredictYOLOSeg(input []float32, channels, height, width, numClasses, maskDim, protoStride int) ([]float32, []float32, int, int, int, error) {
	if len(s.outputNames) < 2 {
		return nil, nil, 0, 0, 0, fmt.Errorf("YOLO-seg ONNX session requires two outputs, got %d", len(s.outputNames))
	}
	if numClasses <= 0 {
		return nil, nil, 0, 0, 0, fmt.Errorf("numClasses must be positive")
	}
	if maskDim <= 0 {
		maskDim = 32
	}
	if protoStride <= 0 {
		protoStride = 4
	}
	attrs := 4 + numClasses + maskDim
	detCount := int(yoloDetectionCount(width))
	protoH := height / protoStride
	protoW := width / protoStride
	// Если формы выходов известны из метаданных модели — используем их
	// (и сверяем attrs с конфигурацией, чтобы ошибка num_classes/yolo_mask_dim
	// диагностировалась внятным сообщением, а не падением по форме тензора).
	if len(s.detShape) == 3 && len(s.protoShape) == 4 {
		if int(s.detShape[1]) != attrs {
			return nil, nil, 0, 0, 0, fmt.Errorf(
				"YOLO-seg model output %q has %d attributes per detection, but config expects %d (4 + num_classes=%d + yolo_mask_dim=%d); check num_classes/yolo_mask_dim in config.json",
				s.outputNames[0], s.detShape[1], attrs, numClasses, maskDim)
		}
		detCount = int(s.detShape[2])
		if int(s.protoShape[1]) != maskDim {
			return nil, nil, 0, 0, 0, fmt.Errorf(
				"YOLO-seg model proto output %q has mask_dim=%d, but config expects yolo_mask_dim=%d",
				s.outputNames[1], s.protoShape[1], maskDim)
		}
		protoH = int(s.protoShape[2])
		protoW = int(s.protoShape[3])
	}
	if protoH <= 0 || protoW <= 0 {
		return nil, nil, 0, 0, 0, fmt.Errorf("invalid YOLO proto size for input %dx%d stride=%d", width, height, protoStride)
	}

	inShape := ort.NewShape(1, int64(channels), int64(height), int64(width))
	inTensor, err := ort.NewTensor[float32](inShape, input)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	defer inTensor.Destroy()

	detShape := ort.NewShape(1, int64(attrs), int64(detCount))
	detTensor, err := ort.NewEmptyTensor[float32](detShape)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	defer detTensor.Destroy()

	protoShape := ort.NewShape(1, int64(maskDim), int64(protoH), int64(protoW))
	protoTensor, err := ort.NewEmptyTensor[float32](protoShape)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	defer protoTensor.Destroy()

	if err := s.sess.Run([]ort.Value{inTensor}, []ort.Value{detTensor, protoTensor}); err != nil {
		return nil, nil, 0, 0, 0, err
	}
	det := detTensor.GetData()
	proto := protoTensor.GetData()
	detCopy := make([]float32, len(det))
	protoCopy := make([]float32, len(proto))
	copy(detCopy, det)
	copy(protoCopy, proto)
	return detCopy, protoCopy, detCount, protoH, protoW, nil
}
