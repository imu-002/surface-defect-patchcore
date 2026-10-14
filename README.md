# Low-cost surface-defect inspection with PatchCore (CPU, no defect labels)

Detects surface defects on textured materials (carpet, leather, wood, tile, grid, fabric), **locates** them with a
heat-map, and **identifies their type**, using only defect-free images for training. Runs on CPU with ONNX Runtime.

| Requirement in the problem statement | Where it is handled |
|---|---|
| Detect defects without large labelled defect collections | PatchCore trains on **defect-free images only** (`sdd/patchcore.py`) |
| Locate defects precisely | Patch-level anomaly map -> heat-map and region mask; pixel AUROC and AU-PRO (`sdd/metrics.py`) |
| Identify the defect type | Region descriptors + kNN / SVM / Random Forest (`sdd/classify.py`) |
| Handle new, unseen defect types | Leave-one-defect-type-out comparison against supervised SVM / RF (`scripts/run_experiments.py`) |
| Low cost, real time | CPU-only ONNX Runtime; speed/accuracy trade-off in `scripts/ablation.py` |
| MVTec AD + our own fabric/leather/wood photos | Same folder layout for both (`sdd/data.py`) |

## Setup

```bash
pip install -r requirements.txt
python scripts/01_build_backbone.py     # downloads ImageNet WideResNet50-2, converts to ONNX, self-checks
```
No PyTorch is needed anywhere. The weights are the timm `wide_resnet50_racm` ImageNet checkpoint
(SHA-256 verified). Converted weights are checked by classifying two sample photos.

## Run on Google Colab (no installation)

1. Create a GitHub repository and upload the contents of this folder (the `.gitignore` keeps large files out), or skip GitHub and upload the zip inside Colab.
2. Open `colab_run.ipynb` in Colab: <https://colab.research.google.com/github/imu-002/surface-defect-patchcore/blob/main/colab_run.ipynb>
   (or upload the notebook via File > Upload notebook).
3. Run the cells top to bottom. A GPU is not needed.

## 1. Evaluate on MVTec AD

Download MVTec AD from <https://www.mvtec.com/company/research/datasets/mvtec-ad> (licence CC BY-NC-SA 4.0,
non-commercial) and unpack it. The relevant categories are `carpet leather wood tile grid`.

```bash
python scripts/run_experiments.py --data /path/to/mvtec_anomaly_detection \
       --categories carpet leather wood tile grid --out results/mvtec
python scripts/aggregate.py --runs results/mvtec --names MVTec --out results/mvtec_report
```
Per category this writes metrics (`<cat>.json`), a qualitative figure, the fitted model and a type classifier.
Expect roughly 3-15 minutes per category on a 2-core CPU (the greedy coreset dominates; use `--coreset-ratio 0.05`
or `--pca-dim 128` for faster runs).

## 2. Your own materials (fabric / leather / wood photos)

Use the same layout (masks are optional; without them only image-level metrics are computed):

```
my_materials/fabric/train/good/*.jpg           # 30-200 defect-free photos
my_materials/fabric/test/good/*.jpg            # held-out good photos
my_materials/fabric/test/<defect_type>/*.jpg   # defective photos, one folder per type
my_materials/fabric/ground_truth/<defect_type>/<name>_mask.png   # optional
```
```bash
python scripts/run_experiments.py --data my_materials --out results/local
```
Or, for deployment, without any evaluation:

```bash
python scripts/fit_model.py --good-dir photos/fabric_good --out models/fabric.pkl
python scripts/fit_type_classifier.py --model models/fabric.pkl --defect-root photos/fabric_defects --out models/fabric_typeclf.pkl   # optional
python scripts/infer.py --model models/fabric.pkl --typeclf models/fabric_typeclf.pkl --input new_photos/ --out inspected/
```
`infer.py` writes an annotated image per input (heat-map, outlined region, verdict) and `results.jsonl`.
Photography matters more than any parameter: fixed camera, fixed lighting, same distance for training and inspection.

## 3. Offline pipeline check (no dataset needed)

```bash
python scripts/00_make_standin_dataset.py --out data/standin
python scripts/00_make_standin_dataset.py --out data/standin_subtle --strength 0.35
python scripts/run_experiments.py --data data/standin --out results/standin
```
The stand-in set is **synthetic** (real + procedural textures with painted defects). It validates that the code works;
its numbers say nothing about MVTec AD performance and must not be reported as such.

## Method in brief

1. Backbone: ImageNet WideResNet50-2, layers 2 and 3, 3x3 local-average pooling, layer 3 upsampled and concatenated
   -> 1536-dim embedding per 8x8-pixel patch (as in PatchCore / anomalib).
2. Training: embed all patches of the defect-free images; keep a 10 % greedy k-center coreset as the memory bank
   (random-projection accelerated); hold out ~15 % of the images to calibrate the decision threshold
   (mean + 3 sigma of their scores).
3. Inference: each patch's score is its distance to the nearest memory-bank patch. Image score = maximum patch score;
   heat-map = bilinear upsampling + Gaussian smoothing.
4. Type identification: the localised region is described by (a) the mean deep embedding inside it and
   (b) 18 shape and colour/contrast features; kNN, SVM or Random Forest classify it.

## Layout

```
sdd/            library: backbone.py (ONNX export), patchcore.py, classify.py, metrics.py, data.py, synth.py, viz.py
scripts/        01_build_backbone, 00_make_standin_dataset, run_experiments, ablation, aggregate, fit_model,
                fit_type_classifier, infer
results/        figures, tables and per-category JSON from the runs reported in REPORT.md
```
