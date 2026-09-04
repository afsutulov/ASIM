package main

import "strings"

// ModelSpec describes one orthophoto detection task.
// Ortophoto module is YOLO11-seg only. Sentinel/UNet models are handled by the sentinel module.
type ModelSpec struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	ONNXFile    string `json:"onnx_file"`
	Preprocess  string `json:"preprocess"`
	Tile        int    `json:"tile,omitempty"`

	// Bound задаётся в JSON через указатель, чтобы отличать «не задано»
	// (nil -> значение по умолчанию 128) от явного "bound": 0 (без перекрытия).
	// Ранее явный ноль был невозможен: он молча заменялся на 128 — та же
	// проблема «0 vs отсутствует», что была у reject-фильтров.
	BoundJSON *int `json:"bound,omitempty"`
	Bound     int  `json:"-"`

	Threshold float32 `json:"threshold,omitempty"`
	// ClassThresholds задаёт порог уверенности отдельно для каждого класса.
	// Ключ — номер класса строкой (например "2": 0.70). Если порог не задан,
	// используется общий threshold. Параметр полностью модель-независимый.
	ClassThresholds map[string]float32 `json:"class_thresholds,omitempty"`
	// AgnosticNMS подавляет перекрывающиеся bbox независимо от класса.
	AgnosticNMS bool `json:"agnostic_nms,omitempty"`
	// ExclusiveClasses формирует единую карту классов: каждый пиксель может
	// принадлежать не более чем одному классу. Для многоклассовой сегментации
	// этот режим должен быть включён.
	ExclusiveClasses bool  `json:"exclusive_classes,omitempty"`
	NumClasses       int   `json:"num_classes"`
	DetectionClasses []int `json:"classes"`

	// MinArea — минимальная площадь компоненты маски в ПИКСЕЛЯХ
	// (используется в removeSmallComponents до полигонизации).
	MinArea float64 `json:"min_area,omitempty"`
	// MinAreaM2 — минимальная площадь итогового полигона в КВАДРАТНЫХ МЕТРАХ
	// (в СК снимка, МСК-59; используется при полигонизации). Ранее оба фильтра
	// управлялись одним параметром min_area в РАЗНЫХ единицах. Если min_area_m2
	// не задан, для обратной совместимости берётся значение min_area.
	MinAreaM2 float64 `json:"min_area_m2,omitempty"`
	Simplify  float64 `json:"simplify,omitempty"`

	// Normalization — способ нормализации снимка перед подачей в модель:
	//   "minmax" (по умолчанию) — растяжка по глобальным min/max листа
	//             (прежнее поведение; контраст зависит от содержимого листа);
	//   "fixed"  — деление на диапазон типа данных (Byte -> /255,
	//             UInt16 -> /65535). Рекомендуется, если снимки уже
	//             нормализованы и обучение шло на данных вида pixel/255;
	//   "none"   — данные уже в [0,1], только клиппинг.
	// ВАЖНО: режим должен совпадать с препроцессингом при обучении модели.
	Normalization string `json:"normalization,omitempty"`

	// Usually stable for ONNX files exported by tools/convert_yolo11_best_to_onnx.py.
	// Keep them configurable only for non-standard exports.
	InputName       string   `json:"input_name,omitempty"`
	OutputNames     []string `json:"output_names,omitempty"`
	YoloMaskDim     int      `json:"yolo_mask_dim,omitempty"`
	YoloProtoStride int      `json:"yolo_proto_stride,omitempty"`
	NMSThreshold    float32  `json:"nms_threshold,omitempty"`
	MaskThreshold   float32  `json:"mask_threshold,omitempty"`
	MorphOpen       int      `json:"morph_open,omitempty"`
	MorphClose      int      `json:"morph_close,omitempty"`
	FillHoles       bool     `json:"fill_holes,omitempty"`
	MaxHoleArea     int      `json:"max_hole_area,omitempty"`

	// Filters against YOLO-seg tile artefacts: sometimes a whole 1536x1536 patch
	// is returned as one very rectangular object. Defaults are conservative and
	// can be overridden per model in config.json.
	RejectTileMaskRatio   float32 `json:"reject_tile_mask_ratio,omitempty"`
	RejectTileEdgeTouches int     `json:"reject_tile_edge_touches,omitempty"`
	RejectRectangularity  float32 `json:"reject_rectangularity,omitempty"`
	RejectMinBoxTileRatio float32 `json:"reject_min_box_tile_ratio,omitempty"`

	// Фильтр вырожденных «квадратных» детекций у краёв тайла.
	//
	// На однородных текстурах (пашня, поле) YOLO у границ тайла иногда выдаёт
	// вырожденные детекции: DFL-распределение регрессии почти равномерно, и
	// bbox получает характерный фиксированный размер (~15*stride), а маска
	// заливает bbox ЦЕЛИКОМ (заполнение = 1.0). Пока маски записывались только
	// внутри окна [bound, tile-bound], эти артефакты срезались этим окном;
	// после перехода на полную запись масок (лечение «ровных обрезов» зданий
	// на швах) их нужно фильтровать явно.
	//
	// Правила (применяются только к детекциям со score < reject_boxfill_score):
	//  1) центр bbox в краевой полосе тайла (в пределах bound от внутренней,
	//     не растровой, границы) и заполнение bbox маской >= reject_edge_boxfill
	//     -> отбросить. Безопасно: центр такой детекции по построению сетки
	//     лежит в «ядре» соседнего тайла, и реальный объект будет записан им;
	//  2) (ОПЦИОНАЛЬНО, по умолчанию ВЫКЛЮЧЕНО) где угодно в тайле: заполнение
	//     >= reject_boxfill при bbox не меньше 48 px по стороне -> отбросить.
	//     Правило НЕ безопасно: заполнение считается по экстенту маски, и
	//     сплошная прямоугольная маска реального здания тоже даёт ~1.0.
	//     Включать только если после краевого правила на полях остаются
	//     одиночные вырожденные квадраты вне краевых полос (например,
	//     "reject_boxfill": 0.995).
	//
	// Значения — доли заполнения bbox (0..1]. Явный 0 в конфиге отключает
	// правило (поля с указателем, поэтому «0» и «не задано» различаются).
	// По умолчанию: reject_edge_boxfill=0.95, reject_boxfill=0 (выкл),
	// reject_boxfill_score=0.5.
	RejectEdgeBoxfillJSON *float32 `json:"reject_edge_boxfill,omitempty"`
	RejectEdgeBoxfill     float32  `json:"-"`
	RejectBoxfillJSON     *float32 `json:"reject_boxfill,omitempty"`
	RejectBoxfill         float32  `json:"-"`
	RejectBoxfillScore    float32  `json:"reject_boxfill_score,omitempty"`

	// Допуск сигнатуры DFL-коллапса (см. isDFLCollapseSize): вырожденные
	// детекции на однородных текстурах имеют бокс размером ровно 15*stride
	// (120/240/480 px) до клиппинга. Детекция с таким размером по обеим осям
	// и маской, заливающей бокс, отбрасывается при ЛЮБОМ score.
	// Явный 0 отключает сигнатуру. По умолчанию 0.06 (±6%).
	RejectDFLSizeTolJSON *float32 `json:"reject_dfl_size_tol,omitempty"`
	RejectDFLSizeTol     float32  `json:"-"`

	// Частотный фильтр повторяющихся размеров бокса. Вырожденные детекции
	// имеют КОНСТАНТНЫЙ (модельно-специфичный, не обязательно 15*stride)
	// размер бокса, поэтому на листе они выдают себя десятками почти
	// квадратных, залитых маской боксов ОДНОГО размера (с точностью до
	// ~2 px). Реальные здания одного проекта такой точности совпадения
	// боксов не дают. Кандидаты с заполнением >= reject_edge_boxfill,
	// почти квадратные (|w-h| <= max(2, 2% стороны)) и размером 60..600 px
	// откладываются; после обработки всех тайлов листа размеры, встретившиеся
	// >= reject_repeat_count раз (бины по 2 px), отбрасываются, остальные
	// записываются как обычно. Явный 0 отключает фильтр. По умолчанию 6.
	RejectRepeatCountJSON *int `json:"reject_repeat_count,omitempty"`
	RejectRepeatCount     int  `json:"-"`

	// DumpCandidates: писать в рабочий каталог CSV со всеми кандидатами
	// (позиция, размеры до клиппинга, score, класс, заполнение бокса, решение
	// фильтров) — для диагностики ложных срабатываний.
	DumpCandidates bool `json:"dump_candidates,omitempty"`

	// SkipEmptyTiles enables the optional optimisation that avoids ONNX inference
	// on almost completely nodata/white patches. Disabled by default because
	// small valid fragments at a raster edge may contain real buildings.
	SkipEmptyTiles   bool `json:"skip_empty_tiles,omitempty"`
	MinContentPixels int  `json:"min_content_pixels,omitempty"`

	// ClassMergeRules are applied to the common exclusive class map before
	// extracting the classes requested by this logical model.
	ClassMergeRules []ClassMergeRule `json:"class_merge_rules,omitempty"`
}

func FinalizeSpec(m ModelSpec) ModelSpec {
	if m.ONNXFile == "" && m.Name != "" {
		m.ONNXFile = m.Name + ".onnx"
	}
	if m.Preprocess == "" {
		m.Preprocess = "ortophoto"
	}
	if m.Tile <= 0 {
		m.Tile = 1536
	}
	if m.BoundJSON != nil {
		b := *m.BoundJSON
		if b < 0 {
			b = 0
		}
		m.Bound = b
	} else if m.Bound <= 0 {
		m.Bound = 128
	}
	if m.Threshold == 0 {
		m.Threshold = 0.25
	}
	if m.NumClasses <= 0 {
		m.NumClasses = 1
	}
	if m.MinArea <= 0 {
		m.MinArea = 10
	}
	// Обратная совместимость: если фильтр площади полигона в м² не задан,
	// сохраняем прежнее поведение (использовалось значение min_area).
	if m.MinAreaM2 <= 0 {
		m.MinAreaM2 = m.MinArea
	}
	switch strings.ToLower(strings.TrimSpace(m.Normalization)) {
	case "", "minmax":
		m.Normalization = "minmax"
	case "fixed":
		m.Normalization = "fixed"
	case "none":
		m.Normalization = "none"
	default:
		// Неизвестное значение не должно молча менять поведение.
		m.Normalization = "minmax"
	}
	if m.InputName == "" {
		m.InputName = "images"
	}
	if len(m.OutputNames) == 0 {
		m.OutputNames = []string{"output0", "output1"}
	}
	if m.YoloMaskDim <= 0 {
		m.YoloMaskDim = 32
	}
	if m.YoloProtoStride <= 0 {
		m.YoloProtoStride = 4
	}
	if m.NMSThreshold <= 0 {
		m.NMSThreshold = 0.45
	}
	if m.MaskThreshold <= 0 {
		m.MaskThreshold = 0.5
	}
	if m.MorphOpen < 0 {
		m.MorphOpen = 0
	}
	if m.MorphClose < 0 {
		m.MorphClose = 0
	}
	if m.MaxHoleArea < 0 {
		m.MaxHoleArea = 0
	}
	if m.MinContentPixels <= 0 {
		m.MinContentPixels = 256
	}
	// Tile-artefact rejection must be opt-in.
	// Important: JSON numeric fields cannot distinguish an omitted value from an
	// explicit zero. Earlier versions silently replaced zero values with defaults,
	// so a config like "reject_tile_mask_ratio": 0 did NOT disable the filter.
	// That could cut off legitimate large buildings.
	// Now zero means disabled; set positive values in config to enable rejection.
	if m.RejectTileMaskRatio < 0 {
		m.RejectTileMaskRatio = 0
	}
	if m.RejectTileEdgeTouches < 0 {
		m.RejectTileEdgeTouches = 0
	}
	if m.RejectRectangularity < 0 {
		m.RejectRectangularity = 0
	}
	if m.RejectMinBoxTileRatio < 0 {
		m.RejectMinBoxTileRatio = 0
	}
	if m.RejectEdgeBoxfillJSON != nil {
		m.RejectEdgeBoxfill = *m.RejectEdgeBoxfillJSON
		if m.RejectEdgeBoxfill < 0 {
			m.RejectEdgeBoxfill = 0
		}
	} else if m.RejectEdgeBoxfill == 0 {
		m.RejectEdgeBoxfill = 0.95
	}
	if m.RejectBoxfillJSON != nil {
		m.RejectBoxfill = *m.RejectBoxfillJSON
		if m.RejectBoxfill < 0 {
			m.RejectBoxfill = 0
		}
	}
	if m.RejectBoxfillScore <= 0 {
		m.RejectBoxfillScore = 0.5
	}
	if m.RejectDFLSizeTolJSON != nil {
		m.RejectDFLSizeTol = *m.RejectDFLSizeTolJSON
		if m.RejectDFLSizeTol < 0 {
			m.RejectDFLSizeTol = 0
		}
	} else if m.RejectDFLSizeTol == 0 {
		m.RejectDFLSizeTol = 0.06
	}
	if m.RejectRepeatCountJSON != nil {
		m.RejectRepeatCount = *m.RejectRepeatCountJSON
		if m.RejectRepeatCount < 0 {
			m.RejectRepeatCount = 0
		}
	} else if m.RejectRepeatCount == 0 {
		m.RejectRepeatCount = 6
	}
	return m
}
