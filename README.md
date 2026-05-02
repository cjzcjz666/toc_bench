# TOC-Bench: Benchmark Construction & Evaluation

This repository contains (i) a **multi-stage pipeline** for curating video-centric question–answer items from heterogeneous sources, and (ii) a **standardized evaluation harness** for scoring model predictions under a fixed prompting protocol.

All concrete filesystem locations below are written as **placeholders**; substitute them with your own deployment paths and secrets.

---

## 1. Repository layout

| Path (relative to `<PROJECT_ROOT>`) | Role |
|--------------------------------------|------|
| `configs/config.py` | Single source of truth: directory layout, data-source metadata, VLM / SAM / event / filter / QA-dimension / format definitions, and sampling-related constants. |
| `scripts/` | Executable stages of the benchmark construction pipeline (numbered steps and utilities). |
| `eval_runner.py` | Runs models on a released benchmark JSON and writes **JSONL** predictions (resumable). |
| `answer_extractor.py` | Deterministic post-processing: maps free-form model text to structured answers per format. |
| `compute_metrics.py` | Aggregates accuracy, tier-weighted metrics, hallucination-related breakdowns, and **invalid extraction rate**. |
| `data_repair.py` | Small, auditable fixes applied to an exported clean benchmark before scoring (produces a versioned JSON used by `eval_runner.py`). |

Auxiliary scripts under `scripts/` (e.g., diagnostics, duplicate-label repair, referring-expression helpers) support QA quality control and analysis; they are optional for the main linear pipeline.

---

## 2. Environment

### 2.1 Python

Create an isolated environment (recommended):

```bash
cd <PROJECT_ROOT>
python -m venv <VENV_DIR>
source <VENV_DIR>/bin/activate   # or equivalent on your OS
pip install -r requirements.txt
```

**Construction stages** may require additional packages depending on which steps you run (e.g., vision-language APIs, segmenters, trackers). Consult the docstring at the top of each `scripts/step*.py` for that step’s dependencies.

**Evaluation** typically needs:

- `torch`, `transformers`, `accelerate`, `Pillow`, and (for API models) `openai` / vendor SDKs as applicable.
- For **local OpenAI-compatible servers** (e.g., vLLM), install the server stack separately; `eval_runner.py` only issues HTTP requests to `<BASE_URL>`.

### 2.2 Global data root

Pipeline scripts resolve storage via the environment variable:

- **`TOC_BENCH_ROOT`** — base directory under which `videos/`, `tracks/`, `events/`, `filtered/`, `qa/`, etc. are created (see `configs/config.py`).

If unset, a default relative path defined in `configs/config.py` is used; **set `TOC_BENCH_ROOT` explicitly** on shared clusters to avoid writing under the repository tree.

### 2.3 Model weights & Hugging Face cache

Set a **single** Hugging Face home (example placeholder):

```bash
export HF_HOME=<HF_CACHE_ROOT>
```

Avoid mixing legacy variables (e.g., `TRANSFORMERS_CACHE` pointing elsewhere) with `HF_HOME`, or you may observe redundant downloads. The evaluation entrypoint normalizes some legacy variables when invoked; keeping shell configuration consistent is still best practice.

### 2.4 API keys (evaluation only)

Closed-weight or hosted models expect credentials via **environment variables** named in each model entry inside `eval_runner.py` (e.g., `<OPENAI_API_KEY>`, `<ANTHROPIC_API_KEY>`, `<GOOGLE_API_KEY>`, or provider-specific keys). Do not commit secrets; inject them at runtime.

---

## 3. Configuration (`configs/config.py`)

`configs/config.py` centralizes:

1. **Directory layout** — all intermediate and final artifact roots under `TOC_BENCH_ROOT`.
2. **`DATA_SOURCES`** — per-source acquisition policy (auto-download vs. manual), target counts, and file conventions.
3. **`VLM_CONFIG`** — parameters for optional VLM-guided object listing (model id/path, device, frame budget, prompts).
4. **`SAM3_CONFIG` / `SAM3_POSTPROC_CONFIG`** — tracker behavior and post-hoc instance repair / confidence scoring.
5. **`EVENT_CONFIG` / `EVENT_STATS_CONFIG`** — event taxonomy and statistics feeding downstream QA preconditions.
6. **`FILTER_CONFIG`** — minimum structural richness for a clip to proceed.
7. **`DIMENSIONS` / `FORMATS` / bucket definitions** — the public **11-dimension**, **5-format** schema (tiers, hallucination sensitivity, etc.).
8. **Filtering, skeleton caps, surface-realization locks** — additional dicts (see file body) consumed by Steps 3–5.

Tuning the benchmark **without** editing step logic is usually done by editing this file and re-running the affected downstream stages.

---

## 4. Benchmark construction pipeline (`scripts/`)

The pipeline is **modular**: later steps assume earlier artifacts exist under the paths declared in `configs/config.py`. Typical order:

### Step 1a — Ingest raw media (`step1a_download_videos.py`)

- Pulls or unpacks videos per `DATA_SOURCES`.
- Some sources support automatic download; others require manual placement into the configured subdirectory.

### Step 1b — Track objects (`step1b_track_objects.py`)

- **Phase A:** optional VLM pass over sampled frames to propose object noun phrases (configured in `VLM_CONFIG`).
- **Phase B:** text-prompted dense tracking (SAM family, configured in `SAM3_CONFIG`).

### Step 1b (post) — Stabilize tracks (`step1b_postprocess.py`)

- Merges fragmentary IDs, flags swaps, assigns **per-instance confidence** used to gate identity-critical dimensions downstream.

### Step 1c — Events & coarse filter (`step1c_detect_events.py`)

- Emits structured **event timelines** and per-clip **phenomenon profiles**.
- Applies `FILTER_CONFIG` thresholds; surviving clips feed Step 2.

### Step 2 — Phenomenon-balanced selection (`step2_select_videos.py`)

- Greedy subset selection toward coverage targets (phenomenon / duration / density / source balance).

### Step 3a — Reasoning units (`step3a_extract_units.py`)

- One **unit** ≈ one candidate QA anchor (dimension-specific preconditions enforced here).

### Step 3b — Skeletons + controlled distractors (`step3b_build_skeletons.py`)

- Materializes units into fully specified items (options, correct labels, hallucination variants per design rules).

### Step 3c — Surface realization (`step3c_surface_realize.py`)

- Optional LLM rewrite of **surface text** under hard constraints (locked tokens / no semantic drift of keyed fields).

### Step 4 — Temporal necessity filter (`step4_temporal_filter.py`)

- Three **orthogonal** probe layers (text-only, single-frame, temporal-shuffle) using a configured VLM; intersection defines high-confidence items.

### Step 4b — Stratified sampling (`step4b_sample_for_verification.py`)

- Splits filtered pool into **verified**, **large-scale**, and **quality-estimation** subsets with documented seeds and caps.

### Step 5a — Export release bundle (`step5a_export_for_humans.py`)

- Produces a **clean benchmark JSON**, annotator spreadsheets, and a helper script to consolidate media for distribution.

**Representative invocation pattern** (adapt paths and flags per step docstrings):

```bash
export TOC_BENCH_ROOT=<BENCH_DATA_ROOT>
python scripts/step1a_download_videos.py --source <SOURCE_NAME>
python scripts/step1b_track_objects.py --device <CUDA_DEVICE>
python scripts/step1b_postprocess.py
python scripts/step1c_detect_events.py
python scripts/step2_select_videos.py
python scripts/step3a_extract_units.py
python scripts/step3b_build_skeletons.py
python scripts/step3c_surface_realize.py --model <REALIZER_MODEL_ID>
python scripts/step4_temporal_filter.py
python scripts/step4b_sample_for_verification.py
python scripts/step5a_export_for_humans.py
```

Intermediate JSON locations follow the directory constants in `configs/config.py` (e.g., `<BENCH_DATA_ROOT>/qa/...`).

---

## 5. Post-export repair & frozen benchmark JSON

After `step5a_export_for_humans.py` (or any equivalent export), run **`data_repair.py`** to apply **versioned, logged** fixes (e.g., label hygiene, answer-pair balance) and emit the JSON consumed by evaluation:

```bash
python data_repair.py \
  --in <EXPORTED_CLEAN_JSON> \
  --out <REPAIRED_BENCHMARK_JSON> \
  --seed <INT_SEED>
```

The evaluation harness docstring refers to this repaired file as the canonical `--bench` input.

---

## 6. Evaluation

### 6.1 Inputs

- **`<REPAIRED_BENCHMARK_JSON>`** — items with `qa_id`, `format`, `correct_answer` / `correct_order`, `video_id`, and metadata.
- **`<VIDEO_ROOT>`** — directory tree or symlink layout resolvable to actual media files.
- **Optional `<VIDEO_REGISTRY_JSON>`** — maps `video_id` to absolute paths (recommended for frame-folder sources).

### 6.2 Running inference (`eval_runner.py`)

Each model is keyed in `MODEL_REGISTRY` inside `eval_runner.py`. Kinds include:

- **HTTP OpenAI-compatible** video chat (local vLLM or hosted APIs),
- **Google native video** (Gemini SDK path),
- **Transformers local** fallbacks for selected architectures,
- **Logic baselines** (no GPU).

Example (placeholders):

```bash
# Local server must already be listening at <BASE_URL> if using an openai_video model.
python eval_runner.py \
  --bench <REPAIRED_BENCHMARK_JSON> \
  --videos-root <VIDEO_ROOT> \
  --video-registry <VIDEO_REGISTRY_JSON> \
  --model <MODEL_KEY> \
  --frames <FRAME_COUNT> \
  --concurrency <N_WORKERS> \
  --out <PREDICTIONS_JSONL>
```

**Resuming:** existing `qa_id` lines in `<PREDICTIONS_JSONL>` are skipped on restart.

**Frame count:** defaults per model can be overridden with `--frames` (used for ablations such as single-frame baselines).

### 6.3 Scoring (`compute_metrics.py`)

```bash
python compute_metrics.py \
  --bench <REPAIRED_BENCHMARK_JSON> \
  --preds <PREDICTIONS_JSONL> [<MORE_JSONL> ...] \
  --save-summary <OPTIONAL_SUMMARY_JSON>
```

Metrics include overall accuracy, macro-by-dimension, tier-weighted accuracy, hallucination-bucket accuracies, and **invalid extraction rate** (`extracted_answer is None` among scored items).

---

## 7. Reproducibility checklist

1. Fix `TOC_BENCH_ROOT`, `HF_HOME`, API key env vars, and CUDA device indices for your machine.
2. Pin **package versions** (transformers / vLLM / segmenter) when reporting numbers; several model-specific loaders are sensitive to minor version skew.
3. Record **seeds** passed to stochastic stages (`step4b_sample_for_verification.py`, `data_repair.py`, and any LLM-based realization).
4. Archive **exact** `<REPAIRED_BENCHMARK_JSON>` and model prediction JSONLs alongside submitted results.

---

## 8. Citation

If you use this benchmark or codebase, please cite the **TOC-Bench** paper (bib entry to be added upon publication). Until then, cite the repository URL and commit hash used in your experiments.

---

## 9. License & third-party data

Individual sources in `DATA_SOURCES` carry their own licenses and access procedures. Users are responsible for complying with each provider’s terms when downloading or redistributing video material.
