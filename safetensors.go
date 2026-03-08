package goformer

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// tensorMeta holds the metadata for a single tensor in a safetensors file.
type tensorMeta struct {
	DType   string `json:"dtype"`
	Shape   []int  `json:"shape"`
	Offsets [2]int `json:"data_offsets"`
}

// weightMap holds all tensors loaded from a safetensors file.
type weightMap struct {
	tensors map[string]*tensor
}

// loadSafetensors reads a safetensors file and returns all tensors.
func loadSafetensors(path string) (*weightMap, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	if len(data) < 8 {
		return nil, fmt.Errorf("safetensors: file too short")
	}

	headerLen := binary.LittleEndian.Uint64(data[:8])
	if 8+headerLen > uint64(len(data)) {
		return nil, fmt.Errorf("safetensors: header length exceeds file size")
	}

	headerJSON := data[8 : 8+headerLen]
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("safetensors: invalid header JSON: %w", err)
	}

	dataStart := 8 + headerLen
	wm := &weightMap{tensors: make(map[string]*tensor)}

	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var meta tensorMeta
		if err := json.Unmarshal(raw, &meta); err != nil {
			return nil, fmt.Errorf("safetensors: invalid tensor metadata for %q: %w", name, err)
		}

		if meta.DType != "F32" {
			// Convert F16 to F32 at load time.
			if meta.DType == "F16" {
				t, err := loadF16Tensor(data[dataStart:], meta)
				if err != nil {
					return nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
				}
				wm.tensors[name] = t
				continue
			}
			// Skip non-float tensors (e.g. position_ids is I64).
			if meta.DType == "I64" || meta.DType == "I32" {
				continue
			}
			return nil, fmt.Errorf("safetensors: unsupported dtype %q for tensor %q", meta.DType, name)
		}

		tensorData := data[dataStart+uint64(meta.Offsets[0]) : dataStart+uint64(meta.Offsets[1])]
		numFloats := len(tensorData) / 4
		floats := make([]float32, numFloats)
		for i := 0; i < numFloats; i++ {
			floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(tensorData[i*4:]))
		}

		shape := make([]int, len(meta.Shape))
		copy(shape, meta.Shape)
		wm.tensors[name] = &tensor{data: floats, shape: shape}
	}

	return wm, nil
}

// loadF16Tensor converts F16 tensor data to F32.
func loadF16Tensor(fileData []byte, meta tensorMeta) (*tensor, error) {
	tensorData := fileData[meta.Offsets[0]:meta.Offsets[1]]
	numFloats := len(tensorData) / 2
	floats := make([]float32, numFloats)
	for i := 0; i < numFloats; i++ {
		bits := binary.LittleEndian.Uint16(tensorData[i*2:])
		floats[i] = float16ToFloat32(bits)
	}
	shape := make([]int, len(meta.Shape))
	copy(shape, meta.Shape)
	return &tensor{data: floats, shape: shape}, nil
}

// float16ToFloat32 converts an IEEE 754 half-precision float to float32.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff

	switch {
	case exp == 0:
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: normalise.
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x3ff
		return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
	case exp == 0x1f:
		if frac == 0 {
			return math.Float32frombits((sign << 31) | (0xff << 23))
		}
		return math.Float32frombits((sign << 31) | (0xff << 23) | (frac << 13))
	default:
		return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
	}
}

// get returns a tensor by its full name.
func (wm *weightMap) get(name string) (*tensor, error) {
	t, ok := wm.tensors[name]
	if !ok {
		return nil, fmt.Errorf("weight %q not found", name)
	}
	return t, nil
}

// findSafetensorsFile finds the safetensors file in a model directory.
func findSafetensorsFile(dir string) (string, error) {
	// Try model.safetensors first.
	p := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(p); err == nil {
		return p, nil
	}

	// Look for any .safetensors file.
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("reading model directory: %w", err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") {
			return filepath.Join(dir, e.Name()), nil
		}
	}
	return "", fmt.Errorf("no safetensors file found in %s", dir)
}
