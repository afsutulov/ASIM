# ASIM --- Analytical System of Intelligent Monitoring

**ASIM (Analytical System of Intelligent Monitoring)** is a GIS/AI
platform for automated analysis of remote-sensing imagery. It combines
Sentinel-2 acquisition and processing, UAV orthophoto analysis,
GPU-accelerated ONNX inference, geospatial post-processing, task
execution, result packaging, and a web interface.

Repository: https://github.com/afsutulov/asim

ASIM is designed for practical regional-scale monitoring workflows. The
current codebase supports two substantially different computer-vision
paths:

-   **Sentinel-2 multispectral analysis** for vegetation, land, water,
    forest and other remote-sensing tasks.
-   **UAV orthophoto analysis** using either **YOLO11 instance
    segmentation** for compact objects or **dense semantic
    segmentation** for continuous area-like phenomena such as
    Sosnowsky's hogweed infestation.

> **Important:** trained `.onnx` model files, production imagery,
> credentials and operational datasets may be excluded from the public
> repository because of their size or distribution restrictions.

------------------------------------------------------------------------

## Table of contents

1.  [Main capabilities](#main-capabilities)
2.  [Architecture](#architecture)
3.  [Repository layout](#repository-layout)
4.  [Runtime data layout](#runtime-data-layout)
5.  [Requirements](#requirements)
6.  [Building the Go modules](#building-the-go-modules)
7.  [Configuration](#configuration)
8.  [Orthophoto model types](#orthophoto-model-types)
9.  [Semantic segmentation for
    hogweed](#semantic-segmentation-for-hogweed)
10. [YOLO11 segmentation for buildings and compact
    objects](#yolo11-segmentation-for-buildings-and-compact-objects)
11. [Orthophoto tiling and seam
    handling](#orthophoto-tiling-and-seam-handling)
12. [Coordinate reference systems](#coordinate-reference-systems)
13. [Orthophoto output format](#orthophoto-output-format)
14. [GeoTIFF integrity and repair](#geotiff-integrity-and-repair)
15. [Sentinel-2 downloader](#sentinel-2-downloader)
16. [Sentinel-2 processing](#sentinel-2-processing)
17. [Task queue](#task-queue)
18. [Web application](#web-application)
19. [systemd deployment](#systemd-deployment)
20. [Manual execution](#manual-execution)
21. [Model tuning reference](#model-tuning-reference)
22. [Diagnostics and troubleshooting](#diagnostics-and-troubleshooting)
23. [Operational recommendations](#operational-recommendations)
24. [Security notes](#security-notes)

------------------------------------------------------------------------

## Main capabilities

ASIM provides the following building blocks:

-   automated Sentinel-2 L1C/L2A discovery and download;
-   local indexing and periodic data maintenance;
-   Sentinel-2 mosaic preparation;
-   ONNX Runtime inference on CPU or NVIDIA CUDA GPUs;
-   multi-GPU execution;
-   analysis of large UAV GeoTIFF orthomosaics without resampling the
    source image before inference;
-   YOLO11 instance segmentation;
-   dense semantic segmentation with per-pixel probability maps;
-   overlapping-tile inference for large rasters;
-   class-specific confidence thresholds;
-   mask post-processing and polygonization;
-   filtering by minimum area in pixels and square metres;
-   geometry simplification;
-   optional morphological opening/closing;
-   optional hole filling;
-   result dissolve across image/tile boundaries;
-   source-CRS-aware result grouping;
-   search-area clipping using configured GeoJSON polygons;
-   ZIP packaging of final GeoJSON results;
-   JSON-backed task queue;
-   Flask-based web UI;
-   systemd deployment files.

The exact set of available monitoring tasks is controlled by
`data/config.json` and by the ONNX files installed in `models_path`.

------------------------------------------------------------------------

## Architecture

``` text
                              ┌──────────────────────┐
                              │     Web / Flask      │
                              │       app.py         │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │ data/prosessing.json │
                              │      task queue      │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │      task (Go)       │
                              └───────┬───────┬──────┘
                                      │       │
                         Sentinel-2   │       │   UAV / orthophoto
                                      ▼       ▼
                            ┌─────────────┐ ┌──────────────┐
                            │  sentinel   │ │  ortophoto   │
                            │ ONNX models │ │ ONNX Runtime │
                            └──────┬──────┘ └──────┬───────┘
                                   │               │
                                   └───────┬───────┘
                                           ▼
                                  ┌────────────────┐
                                  │ GeoJSON / ZIP  │
                                  │ results.json   │
                                  └────────────────┘

       downloader ──► Sentinel-2 storage
       cron       ──► periodic maintenance
       mosaic     ──► Sentinel-2 mosaic preparation
```

The orthophoto module contains two inference branches:

``` text
                           Orthophoto GeoTIFF
                                  │
                         native source pixels
                                  │
                  ┌───────────────┴────────────────┐
                  │                                │
             model_kind                       model_kind
             "yolo_seg"                       "semantic"
                  │                                │
          YOLO detections +                    dense logits
          instance masks                          │
                  │                             sigmoid
                  │                                │
                  └───────────────┬────────────────┘
                                  ▼
                         common class mask
                                  │
                    morphology / hole handling
                                  │
                            polygonization
                                  │
                      CRS-aware vector pipeline
                                  │
                       dissolve / GeoJSON / ZIP
```

------------------------------------------------------------------------

## Repository layout

``` text
asim/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── config.json
│   ├── users.json
│   ├── results.json
│   ├── poligons.json
│   └── prosessing.json
├── deploy/
│   ├── asim-task.service
│   ├── asim-web.service
│   └── contractor
├── models/
├── poligons/
│   ├── kray.geojson
│   ├── perm.geojson
│   ├── permsky.geojson
│   └── ...
├── src/
│   ├── cron/
│   ├── downloader/
│   ├── mosaic/
│   ├── ortophoto/
│   ├── sentinel/
│   └── task/
├── static/
└── templates/
```

### Module responsibilities

  -----------------------------------------------------------------------
  Component                           Purpose
  ----------------------------------- -----------------------------------
  `src/downloader`                    Sentinel-2 L1C/L2A discovery and
                                      download

  `src/cron`                          periodic data operations

  `src/mosaic`                        Sentinel-2 mosaic generation

  `src/sentinel`                      multispectral inference and vector
                                      post-processing

  `src/ortophoto`                     UAV GeoTIFF inference, masks,
                                      polygons and CRS-aware export

  `src/task`                          execution of queued analysis jobs

  `app.py`                            Flask web application

  `poligons/`                         search boundaries and
                                      administrative polygons

  `data/config.json`                  main runtime configuration

  `data/prosessing.json`              queued/running jobs

  `data/results.json`                 completed result metadata

  `deploy/`                           systemd units and deployment
                                      helpers
  -----------------------------------------------------------------------

The spelling `poligons` and `prosessing` is retained for compatibility
with the existing project paths and configuration.

------------------------------------------------------------------------

## Runtime data layout

A typical production installation uses:

``` text
/opt/asim/
├── app.py
├── data/
├── models/
├── poligons/
├── downloader
├── cron
├── mosaic
├── sentinel
├── ortophoto
└── task

/mnt/share/newest/
├── Sentinel2_L1C/
├── Sentinel2_L2A/
└── Ortophoto/
    ├── 2025/
    │   └── <territory>/
    └── 2026/
        └── <territory>/

/mnt/share/results/
└── <task-id>.zip

/var/log/asim/
└── asim.log
```

All paths are configurable.

------------------------------------------------------------------------

## Requirements

### Base software

-   Linux;
-   Go 1.22 or newer;
-   Python 3.10 or newer;
-   GDAL/OGR including development headers;
-   PROJ;
-   `pkg-config`;
-   ONNX Runtime;
-   NVIDIA driver for GPU mode;
-   CUDA/cuDNN compatible with the installed ONNX Runtime build;
-   Python packages from `requirements.txt`.

The Go inference modules use, among other dependencies:

-   `github.com/lukeroth/gdal`
-   `github.com/yalue/onnxruntime_go`

### GPU mode

Example:

``` json
"compute": {
  "device": "gpu",
  "gpus": [0, 1]
}
```

CPU execution is also supported where configured:

``` json
"compute": {
  "device": "cpu",
  "gpus": []
}
```

Monitor GPU memory during large orthophoto jobs:

``` bash
watch -n 1 nvidia-smi
```

ONNX Runtime sessions can consume substantial VRAM. The orthophoto
implementation should reuse sessions instead of creating a new CUDA
session for every patch.

------------------------------------------------------------------------

## Building the Go modules

Each component is a separate Go module.

Example:

``` bash
cd src/ortophoto
go build -o /opt/asim/ortophoto .

cd ../sentinel
go build -o /opt/asim/sentinel .

cd ../task
go build -o /opt/asim/task .

cd ../downloader
go build -o /opt/asim/downloader .

cd ../cron
go build -o /opt/asim/cron .

cd ../mosaic
go build -o /opt/asim/mosaic .
```

For `sentinel` and `ortophoto`, GDAL and ONNX Runtime libraries must be
visible to the compiler/linker and at runtime.

On isolated servers, pre-populate the Go module cache or use an approved
vendoring process before moving the source to the offline environment.

------------------------------------------------------------------------

## Configuration

The main configuration file is:

``` text
data/config.json
```

Typical global keys include:

``` json
{
  "logs": "/var/log/asim",
  "path": "/opt/asim",
  "satellite": "/mnt/share/newest",
  "sentinel": "/mnt/share/newest/Sentinel2_L1C",
  "sentinel2A": "/mnt/share/newest/Sentinel2_L2A",
  "tmp": "/mnt/data",
  "users": "/opt/asim/data/users.json",
  "poli_path": "/opt/asim/poligons",
  "poligons": "/opt/asim/data/poligons.json",
  "results_path": "/mnt/share/results",
  "results": "/opt/asim/data/results.json",
  "prosessing": "/opt/asim/data/prosessing.json",
  "models_path": "/opt/asim/models",
  "ortophoto": "/mnt/share/newest/Ortophoto",
  "compute": {
    "device": "gpu",
    "gpus": [0, 1]
  },
  "models": {}
}
```

### Model selection

Every logical task is defined under `models`.

`preprocess` selects the processing pipeline. Orthophoto models use:

``` json
"preprocess": "ortophoto"
```

For orthophoto models, `model_kind` selects the inference type:

``` json
"model_kind": "yolo_seg"
```

or:

``` json
"model_kind": "semantic"
```

If `model_kind` is omitted, the orthophoto module preserves the legacy
YOLO-seg behaviour.

------------------------------------------------------------------------

## Orthophoto model types

### `yolo_seg`

Use for **compact, instance-like objects**:

-   buildings;
-   pavilions;
-   isolated structures;
-   pavement objects;
-   other targets naturally represented by a bounding box plus instance
    mask.

YOLO segmentation is not the preferred representation for very large
continuous vegetation fields because an instance mask is coupled to a
detected bounding box.

### `semantic`

Use for **continuous area-like classes**:

-   large hogweed infestations;
-   vegetation cover;
-   continuous surface classes;
-   phenomena where a single region can extend across a substantial part
    of an orthomosaic.

A semantic ONNX model produces dense logits:

``` text
input : images [1, 3, H, W] float32
output: output [1, C, H, W] float32
```

For the current hogweed model:

``` text
input : images [1, 3, 512, 512]
output: output [1, 1, 512, 512]
```

ASIM applies sigmoid and `semantic_threshold` itself.

------------------------------------------------------------------------

## Semantic segmentation for hogweed

The current hogweed workflow uses dense semantic segmentation rather
than YOLO instance segmentation. This change is important for large
infestations: the model predicts a probability for every pixel instead
of requiring the vegetation patch to fit into a YOLO bounding box.

### Recommended precision-oriented configuration

The following profile exposes the raw model geometry with minimal
artificial filling/smoothing:

``` json
"hogweed_bpla": {
  "description": "Sosnowsky's hogweed infestation (UAV)",
  "onnx_file": "hogweed_bpla.onnx",
  "preprocess": "ortophoto",
  "model_kind": "semantic",

  "num_classes": 1,
  "classes": [0],

  "tile": 512,
  "bound": 64,
  "semantic_edge_ignore": 16,

  "normalization": "fixed",
  "semantic_threshold": 0.6,
  "exclusive_classes": false,

  "input_name": "images",
  "output_names": ["output"],

  "min_area": 25,
  "min_area_m2": 2,

  "morph_open": 0,
  "morph_close": 0,

  "fill_holes": false,
  "max_hole_area": 0,

  "dissolve_result": true,
  "dissolve_seam_close_m": 0,

  "dump_coverage": false,
  "simplify": 0,

  "skip_empty_tiles": true,
  "min_content_pixels": 1,

  "class_merge_rules": []
}
```

This is a **precision-oriented starting point**, not a universal
optimum. Tune the threshold on a validation territory that was not used
for training.

### `semantic_threshold`

This is the primary probability cutoff:

``` json
"semantic_threshold": 0.6
```

Higher values make ASIM more conservative:

-   fewer false positives;
-   potentially more false negatives;
-   smaller/tighter polygons.

Lower values increase sensitivity:

-   more complete vegetation coverage;
-   more false positives.

Typical experiments:

``` text
0.50 → balanced baseline
0.60 → more conservative
0.65–0.75 → increasingly strict
```

Do not assume that a change from 0.50 to 0.60 must visibly alter every
result. If most predicted hogweed pixels already have probabilities well
above 0.60, the difference can be small.

### `fill_holes`

Controls whether holes inside a detected region are filled.

For the most faithful model contour:

``` json
"fill_holes": false,
"max_hole_area": 0
```

For intentionally solid polygons:

``` json
"fill_holes": true,
"max_hole_area": 4000
```

Use hole filling carefully: it can convert real non-hogweed gaps inside
a large infestation into positive area.

### `morph_close`

Morphological closing connects nearby positive regions and closes small
gaps.

``` json
"morph_close": 0
```

preserves the model output most closely.

A small positive value can improve continuity but may bridge real gaps.

### `morph_open`

Morphological opening removes small protrusions/noise:

``` json
"morph_open": 0
```

Increase only when there is a clear need to suppress small isolated mask
noise.

### `simplify`

Geometry simplification tolerance is applied in metric processing.

For maximum boundary fidelity:

``` json
"simplify": 0
```

A positive value reduces vertex count and GeoJSON size at the cost of
geometric detail.

### `dissolve_result`

``` json
"dissolve_result": true
```

merges adjacent result polygons. This is especially useful where an
infestation crosses boundaries between orthophoto source files. Without
dissolve, QGIS may draw the border of each polygon and create an
apparent straight line through a continuous infestation even though both
sides were detected.

### `dissolve_seam_close_m`

Optional metric closing before/while dissolving source-sheet seams:

``` json
"dissolve_seam_close_m": 0.2
```

can bridge very narrow nodata seams between neighbouring source images.

Use:

``` json
"dissolve_seam_close_m": 0
```

when preserving real narrow gaps is more important than closing
source-sheet seams.

------------------------------------------------------------------------

## YOLO11 segmentation for buildings and compact objects

A typical building configuration is:

``` json
"buildings": {
  "description": "Buildings",
  "onnx_file": "buildings.onnx",
  "preprocess": "ortophoto",

  "num_classes": 5,
  "classes": [0],

  "tile": 1536,
  "bound": 128,
  "normalization": "fixed",

  "threshold": 0.25,
  "class_thresholds": {
    "0": 0.4,
    "1": 0.4,
    "2": 0.7,
    "3": 0.45,
    "4": 0.4
  },

  "nms_threshold": 0.45,
  "agnostic_nms": false,
  "exclusive_classes": true,

  "mask_threshold": 0.5,
  "yolo_mask_dim": 32,
  "yolo_proto_stride": 4,

  "input_name": "images",
  "output_names": ["output0", "output1"],

  "min_area": 10,
  "min_area_m2": 10,

  "morph_open": 0,
  "morph_close": 0,
  "fill_holes": false,
  "max_hole_area": 0,

  "simplify": 0,

  "reject_tile_mask_ratio": 0,
  "reject_tile_edge_touches": 0,
  "reject_rectangularity": 0,
  "reject_min_box_tile_ratio": 0,
  "reject_edge_boxfill": 0,
  "reject_boxfill": 0,
  "reject_boxfill_score": 0,
  "reject_dfl_size_tol": 0,
  "reject_repeat_count": 0,

  "skip_empty_tiles": true,
  "min_content_pixels": 1,

  "class_merge_rules": [
    {
      "target_class": 0,
      "source_classes": [1],
      "distance_m": 0.3
    }
  ]
}
```

### YOLO-specific parameters

  -----------------------------------------------------------------------
  Parameter                           Meaning
  ----------------------------------- -----------------------------------
  `threshold`                         default detection confidence

  `class_thresholds`                  confidence override per class

  `nms_threshold`                     IoU threshold for non-maximum
                                      suppression

  `agnostic_nms`                      suppress overlapping boxes across
                                      classes

  `exclusive_classes`                 force each pixel to belong to at
                                      most one class

  `mask_threshold`                    instance-mask binarization
                                      threshold

  `yolo_mask_dim`                     YOLO mask prototype dimension

  `yolo_proto_stride`                 prototype-map stride

  `reject_*`                          optional filters for degenerate
                                      tile/bbox artefacts

  `dump_candidates`                   diagnostic candidate output in the
                                      temporary work directory
  -----------------------------------------------------------------------

The `reject_*` controls are primarily diagnostic/defensive tools for
YOLO artefacts. Keep them disabled unless a reproducible artefact
requires them; aggressive rejection can remove real objects.

------------------------------------------------------------------------

## Orthophoto tiling and seam handling

Large GeoTIFF files are processed as overlapping model tiles.

### Semantic path

Important parameters:

``` json
"tile": 512,
"bound": 64,
"semantic_edge_ignore": 16
```

-   `tile` must match the ONNX model input size.
-   `bound` provides overlap/context between neighbouring inference
    windows.
-   `semantic_edge_ignore` discards only the least reliable edge strip
    of interior predictions.
-   overlapping semantic predictions are combined so that the image is
    covered without the hard gaps created by naïve
    non-overlapping/core-only stitching.

Do **not** set `tile=512` merely for performance if the exported model
requires `1536x1536`. `tile` must follow the actual ONNX input contract.

For the current semantic hogweed model, the model itself was
trained/exported for `512x512`, so `tile=512` is correct.

### Why semantic segmentation is used for hogweed

YOLO-seg remains excellent for compact objects, but very large
continuous vegetation regions are a poor match for an instance detector
whose mask is associated with a bounding box. Dense semantic
segmentation removes that constraint and predicts the entire raster
surface pixel by pixel.

This is why the two pipelines coexist:

``` text
Buildings / structures  → YOLO11-seg
Large hogweed fields    → semantic segmentation
```

------------------------------------------------------------------------

## Coordinate reference systems

UAV imagery can come from different contractors, years and processing
pipelines. ASIM therefore does **not** assume that every orthophoto uses
the same CRS.

Supported inputs include, where GDAL/PROJ can interpret them:

-   MSK-59;
-   WGS84 / EPSG:4326;
-   other valid projected coordinate systems.

### Key design rule

ASIM does **not** reproject the RGB raster before neural-network
inference.

``` text
source GeoTIFF
      │
      ├── original pixels → neural network
      │
      └── georeferencing → vector coordinate processing
```

This avoids unnecessary resampling and keeps inference consistent with
the source image.

### Source-CRS grouping

Results are separated according to the CRS group of the **source
orthophoto**.

-   polygons originating from **MSK-59 source images** go only to the
    MSK-59 result;
-   polygons originating from **WGS84/geographic/non-MSK projected
    source images** go only to the WGS84 result;
-   the same detection is not duplicated into both result files.

Therefore:

``` text
only MSK-59 input images
    → one *_MSK59-<zone>.geojson

only WGS84/other non-MSK input images
    → one *_WGS84_EPSG4326.geojson

mixed input dataset
    → two GeoJSON files, each containing only its own source group
```

This behaviour is intentional. The two output files are **not two
coordinate representations of the same detections**.

### MSK-59 target zone

ASIM determines the appropriate MSK-59 zone from the search-area
longitude. The implementation supports the Perm Krai MSK-59 zones with
central meridians around:

``` text
53.55°
56.55°
59.55°
```

Metric operations such as `min_area_m2` and metric geometry
simplification are performed in an appropriate metric CRS.

### Axis order

GeoTIFF metadata may report axis mappings such as:

``` text
Data axis to CRS axis mapping: 2,1
```

for EPSG:4326. Do not infer a processing failure from that line alone.
GDAL/OGR CRS handling must be used consistently rather than manually
swapping coordinates based only on displayed WKT axis definitions.

------------------------------------------------------------------------

## Orthophoto output format

The final public ZIP archive contains **only result GeoJSON files**.

Possible contents:

``` text
<model>_MSK59-<zone>.geojson
```

or:

``` text
<model>_WGS84_EPSG4326.geojson
```

or both when both source CRS groups were processed.

Diagnostic files such as coverage layers, probability maps, candidate
dumps and `result.json` are temporary diagnostic artifacts and are not
part of the public result ZIP in the current pipeline.

Shapefile is used only as an internal transport/intermediate format. The
user-facing vector result is GeoJSON.

------------------------------------------------------------------------

## GeoTIFF integrity and repair

QGIS being able to display a TIFF does **not** guarantee that every
compressed TIFF tile can be decoded by GDAL during full-resolution
automated processing.

Typical corruption symptoms:

``` text
TIFFReadEncodedTile() failed
IReadBlock failed
LZWDecode: Not enough data
LZWDecode: ... not terminated with EOI code
Using code not yet in table
```

ASIM may log:

``` text
ortophoto read failed: file=... error=Failure Error
```

### Full integrity test

Always test the complete raster, not only metadata:

``` bash
gdalinfo -checksum "Ортофото-0-3.tif"
```

A healthy file should complete without `IReadBlock`,
`TIFFReadEncodedTile` or `LZWDecode` errors.

### Repairing a partially damaged LZW GeoTIFF

If the original source cannot be re-exported, a practical recovery
method is:

``` bash
mkdir -p repaired

gdal_translate \
  --config GTIFF_IGNORE_READ_ERRORS TRUE \
  -of GTiff \
  -co TILED=YES \
  -co BLOCKXSIZE=256 \
  -co BLOCKYSIZE=256 \
  -co COMPRESS=DEFLATE \
  -co BIGTIFF=IF_SAFER \
  "Ортофото-0-3.tif" \
  "repaired/Ортофото-0-3.tif"
```

Then verify the repaired file:

``` bash
gdalinfo -checksum "repaired/Ортофото-0-3.tif" \
  2>&1 | tee /tmp/repaired-check.log

grep -Ei "ERROR|failed|IReadBlock|TIFFRead|LZWDecode" \
  /tmp/repaired-check.log
```

If `grep` returns nothing and all bands receive valid checksums, the
rewritten TIFF is readable end-to-end.

Optional overviews:

``` bash
gdaladdo -r average "repaired/Ортофото-0-3.tif" 2 4 8 16 32 64
```

### Important limitation

`GTIFF_IGNORE_READ_ERRORS=TRUE` cannot reconstruct compressed image data
that is physically missing or corrupt. The rewritten file becomes
structurally readable, but damaged blocks may contain missing/invalid
pixels. If an original undamaged orthophoto can be obtained from the
producer, replacing the damaged source is preferable.

------------------------------------------------------------------------

## Sentinel-2 downloader

The downloader supports L1C, L2A or both.

Automatic 15-day window:

``` bash
/opt/asim/downloader \
  --config /opt/asim/data/config.json \
  --run \
  --type all
```

Explicit date range:

``` bash
/opt/asim/downloader \
  --config /opt/asim/data/config.json \
  --start 2026-08-01 \
  --end 2026-08-31 \
  --type all
```

Concurrency can be configured on the command line:

``` bash
--list-workers 8
--download-workers 8
```

Available product types:

``` text
L1C
L2A
all
```

### Network troubleshooting

The downloader requires HTTPS connectivity to the remote storage
endpoint. If discovery hangs immediately after `DISCOVER[...]`, test DNS
and TCP/443 separately.

DNS:

``` bash
getent hosts storage.googleapis.com
```

TCP:

``` bash
timeout 5 bash -c '</dev/tcp/storage.googleapis.com/443' \
  && echo "TCP 443 OK" \
  || echo "TCP 443 FAIL"
```

TLS:

``` bash
timeout 10 openssl s_client \
  -connect storage.googleapis.com:443 \
  -servername storage.googleapis.com \
  </dev/null
```

HTTP:

``` bash
curl -v \
  --connect-timeout 5 \
  --max-time 15 \
  https://storage.googleapis.com/
```

If DNS resolves but every TCP connection to port 443 times out, the
failure is below the HTTP/application layer and should be investigated
in routing, firewall, proxy or UserGate policy.

Operationally, avoid allowing cron to start unlimited overlapping
downloader instances when a previous run is hung. Use a lock (`flock`)
or a systemd timer/service model with appropriate timeouts.

------------------------------------------------------------------------

## Sentinel-2 processing

The Sentinel path handles multispectral models separately from
orthophoto RGB models.

Configured tasks may include, depending on installed model files:

-   Sosnowsky's hogweed;
-   fire/burned areas;
-   quarries;
-   landfills;
-   arable land;
-   windfall;
-   forest pathology;
-   overgrowth;
-   water;
-   reforestation;
-   crop types.

Model availability is defined by `data/config.json` and the files in
`models_path`.

The Sentinel modules contain dedicated band loading, tile caching, ONNX
inference, mask filtering, polygonization and geometry finalization
logic.

------------------------------------------------------------------------

## Task queue

`src/task` executes jobs described by:

``` text
data/prosessing.json
```

and writes result metadata to:

``` text
data/results.json
```

Run:

``` bash
/opt/asim/task \
  --config /opt/asim/data/config.json \
  --run
```

The task runner selects the `sentinel` or `ortophoto` executable
according to the requested model/pipeline.

Because the queue is file-backed, production deployments should avoid
manually editing the JSON files while the service is updating them.

------------------------------------------------------------------------

## Web application

The web application is implemented in:

``` text
app.py
```

with templates under:

``` text
templates/
```

and static assets under:

``` text
static/
```

Install Python dependencies:

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use the supplied systemd unit or the deployment method appropriate to
the target environment.

User/account information is stored in the configured `users.json`; do
not commit production credentials or password material to a public
repository.

------------------------------------------------------------------------

## systemd deployment

Example unit files are included under:

``` text
deploy/
├── asim-task.service
└── asim-web.service
```

After installation or changes:

``` bash
sudo systemctl daemon-reload
sudo systemctl enable --now asim-task.service
sudo systemctl enable --now asim-web.service
```

Status:

``` bash
systemctl status asim-task.service
systemctl status asim-web.service
```

Logs:

``` bash
journalctl -u asim-task.service -f
journalctl -u asim-web.service -f
```

ASIM also writes application logs to the directory configured by `logs`.

------------------------------------------------------------------------

## Manual execution

### Orthophoto

General syntax:

``` bash
/opt/asim/ortophoto \
  --config /opt/asim/data/config.json \
  --model <model-name> \
  --poligon <search-polygon> \
  --start <year> \
  --end <year> \
  --out <output-name>
```

Example:

``` bash
/opt/asim/ortophoto \
  --config /opt/asim/data/config.json \
  --model hogweed_bpla \
  --poligon kray \
  --start 2026 \
  --end 2026 \
  --out hogweed_2026
```

The resulting ZIP is written to `results_path`.

### Downloader

``` bash
/opt/asim/downloader --config /opt/asim/data/config.json --run --type all
```

### Task runner

``` bash
/opt/asim/task --config /opt/asim/data/config.json --run
```

------------------------------------------------------------------------

## Model tuning reference

### Common orthophoto parameters

  -------------------------------------------------------------------------
  Parameter                 Applies to              Description
  ------------------------- ----------------------- -----------------------
  `onnx_file`               all                     model file relative to
                                                    `models_path`

  `preprocess`              all                     processing pipeline

  `model_kind`              orthophoto              `yolo_seg` or
                                                    `semantic`

  `num_classes`             all                     model class count

  `classes`                 all                     classes exported by the
                                                    logical task

  `tile`                    all                     ONNX input tile size

  `bound`                   orthophoto              overlap/context around
                                                    neighbouring tiles

  `normalization`           all                     input normalization
                                                    mode

  `min_area`                masks                   minimum connected
                                                    component area in
                                                    pixels

  `min_area_m2`             vectors                 minimum final polygon
                                                    area in m²

  `morph_open`              masks                   removes small
                                                    foreground structures

  `morph_close`             masks                   closes small
                                                    gaps/connects nearby
                                                    foreground

  `fill_holes`              masks                   fills holes inside
                                                    positive regions

  `max_hole_area`           masks                   maximum hole size
                                                    eligible for filling

  `simplify`                vectors                 geometry simplification
                                                    tolerance

  `skip_empty_tiles`        orthophoto              skip nearly
                                                    empty/nodata tiles

  `min_content_pixels`      orthophoto              minimum content needed
                                                    to run inference

  `dissolve_result`         vectors                 merge
                                                    touching/overlapping
                                                    result polygons

  `dissolve_seam_close_m`   vectors                 close narrow
                                                    source-sheet seams
                                                    before/while dissolving

  `dump_coverage`           diagnostics             generate temporary
                                                    processed/skipped
                                                    coverage layers

  `dump_prob_map`           semantic diagnostics    generate temporary
                                                    semantic probability
                                                    map
  -------------------------------------------------------------------------

### Semantic-only parameters

  -----------------------------------------------------------------------
  Parameter                           Description
  ----------------------------------- -----------------------------------
  `semantic_threshold`                per-pixel probability threshold
                                      after sigmoid

  `semantic_edge_ignore`              unsafe border ignored at interior
                                      tile edges

  `output_names: ["output"]`          expected dense output name for
                                      current hogweed model
  -----------------------------------------------------------------------

### YOLO-only parameters

  Parameter             Description
  --------------------- --------------------------------------
  `threshold`           default confidence threshold
  `class_thresholds`    class-specific confidence thresholds
  `nms_threshold`       NMS IoU threshold
  `agnostic_nms`        class-agnostic NMS
  `mask_threshold`      mask binarization threshold
  `yolo_mask_dim`       prototype dimension
  `yolo_proto_stride`   prototype stride
  `reject_*`            optional artefact filters
  `dump_candidates`     temporary candidate diagnostics

### Normalization

The inference normalization **must match training**.

For ordinary 8-bit RGB models trained with:

``` text
pixel / 255
```

use:

``` json
"normalization": "fixed"
```

Do not train with ImageNet mean/std normalization and then infer with
simple `/255` unless the mean/std operation is embedded in the ONNX
graph or implemented identically in ASIM.

------------------------------------------------------------------------

## Diagnostics and troubleshooting

### No polygons are found

Check, in this order:

1.  ONNX input/output names and dimensions.
2.  `model_kind`.
3.  normalization.
4.  probability/confidence threshold.
5.  source GeoTIFF readability.
6.  search polygon coverage.
7.  model logs and `model_runs`.
8.  whether the source area was actually processed.

For semantic models, a YOLO exporter cannot load the semantic `best.pt`
checkpoint unless that checkpoint was saved in Ultralytics YOLO format.
A custom semantic training checkpoint must be exported using the
corresponding PyTorch model definition.

### Semantic model works on validation but poorly in ASIM

The most common causes are:

-   training/inference normalization mismatch;
-   wrong ONNX input size;
-   sigmoid applied twice or not applied at all;
-   incorrect output name;
-   different RGB channel order;
-   damaged source raster;
-   threshold tuned on training rather than independent validation data.

### Straight horizontal/vertical lines in result

Possible causes include:

-   polygon boundaries between neighbouring source files;
-   nodata seam between source files;
-   an unreadable/corrupt source TIFF that was skipped;
-   incorrect tile stitching;
-   insufficient overlap;
-   an actual sharp probability transition.

`dissolve_result` removes artificial polygon boundaries between adjacent
detections. `dissolve_seam_close_m` can bridge very narrow nodata seams.

### A whole part of a territory has no detections

Before changing the model, verify the source TIFFs:

``` bash
gdalinfo -checksum *.tif
```

A raster can look normal in QGIS while containing unreadable LZW tiles.
A skipped image can produce a perfectly straight geographic edge in the
final result.

### Excessive false positives for hogweed

Preferred order of action:

1.  add representative negative training tiles from the actual failure
    environments;
2.  ensure train/validation are split by territory/orthomosaic rather
    than neighbouring tiles;
3.  tune `semantic_threshold` on an independent validation area;
4.  only then add post-processing filters.

Negative semantic samples should have valid images and fully empty
masks.

### Training split

Training and validation lists must be disjoint.

A file in `val` must **not** also be present in `train`.

More importantly, neighbouring tiles from the same orthomosaic should
generally not be split across train and validation because spatial
leakage can make validation metrics unrealistically optimistic. Prefer
splitting by complete territory/flight/orthomosaic.

------------------------------------------------------------------------

## Operational recommendations

### Orthophoto source validation

Before placing a new contractor delivery into the production directory:

``` bash
for f in *.tif; do
  echo "Checking: $f"
  gdalinfo -checksum "$f" >/tmp/gdal-check.log 2>&1 || {
    echo "FAILED: $f"
    cat /tmp/gdal-check.log
  }
done
```

For large deliveries, retain a machine-readable validation report.

### Model validation

Do not judge a new model only on imagery used during training. Maintain:

-   training territories;
-   validation territories;
-   a final test territory;
-   negative examples representing forests, grassland, agricultural
    fields and other common confusers.

### Configuration changes

Change one parameter group at a time and preserve the previous
production configuration. For semantic segmentation, first inspect the
raw model behaviour with minimal post-processing:

``` json
"morph_open": 0,
"morph_close": 0,
"fill_holes": false,
"max_hole_area": 0,
"simplify": 0,
"dissolve_seam_close_m": 0
```

Then add only the post-processing operations that demonstrably improve
independent test imagery.

### Result size and QGIS performance

`dissolve_result=true` can significantly reduce the number of adjacent
polygons and improve QGIS rendering. A small non-zero `simplify` can
further reduce GeoJSON size, but it should be introduced only after
confirming that the loss of boundary detail is acceptable.

------------------------------------------------------------------------

## Security notes

ASIM may be deployed in restricted or isolated infrastructure.
Recommended practices:

-   do not commit credentials;
-   do not commit production `users.json`;
-   do not expose the Flask development server directly to untrusted
    networks;
-   restrict write access to `/opt/asim`, model files and configuration;
-   restrict access to result and imagery shares;
-   validate third-party ONNX models before deployment;
-   pin/verify dependencies in controlled environments;
-   use least-privilege service accounts;
-   protect task/result JSON files from concurrent manual modification;
-   monitor disk usage in temporary and result directories;
-   use network allow-lists for external Sentinel download endpoints
    where required.

------------------------------------------------------------------------

## Notes for contributors

When changing the orthophoto pipeline, preserve the separation between
the two model families:

``` text
YOLO instance segmentation → compact objects
dense semantic segmentation → continuous area classes
```

Do not fix a semantic-model problem by changing the YOLO path globally,
because buildings and other existing YOLO tasks depend on the current
instance-segmentation behaviour.

Likewise, CRS handling belongs to the vector/geospatial stage rather
than the neural-network RGB preprocessing stage. Reprojecting source
orthophoto pixels before inference should be avoided unless there is a
model-specific, measured reason to do so.

Changes affecting tile overlap, semantic edge handling, polygon
dissolve, source CRS grouping or GeoTIFF reads should be tested on:

-   a complete MSK-59 orthophoto;
-   a complete WGS84 orthophoto;
-   a mixed-CRS date range;
-   a source-sheet boundary;
-   a large continuous hogweed infestation;
-   small isolated hogweed patches;
-   building/structure models;
-   a deliberately damaged or unreadable TIFF.

------------------------------------------------------------------------

## Project status

ASIM is an actively developed applied monitoring system. Configuration
values, model files and inference behaviour should be treated as
versioned parts of the deployment. When updating a production
installation, deploy the executable, `config.json`, ONNX model and
documentation as one tested set.
