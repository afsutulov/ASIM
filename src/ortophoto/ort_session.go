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
	sess   *ort.DynamicAdvancedSession
	device string
}

func NewORTSession(cfg *AppConfig, modelPath string, inputName, outputName string) (*ORTSession, error) {
	if err := initORTWithConfig(cfg); err != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	_ = opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)

	device := strings.ToLower(strings.TrimSpace(cfg.ORTDevice))
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
			"device_id": fmt.Sprintf("%d", cfg.ORTDeviceID),
		}); err != nil {
			return nil, fmt.Errorf("update CUDA provider options: %w", err)
		}
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			return nil, fmt.Errorf("append CUDA execution provider: %w", err)
		}
		log.Printf("onnxruntime session configured for CUDA: device_id=%d model=%s", cfg.ORTDeviceID, modelPath)
	} else {
		log.Printf("onnxruntime session configured for CPU: model=%s", modelPath)
	}

	s, err := ort.NewDynamicAdvancedSession(modelPath, []string{inputName}, []string{outputName}, opts)
	if err != nil {
		return nil, err
	}
	return &ORTSession{sess: s, device: device}, nil
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
