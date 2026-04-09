# WireSegHR Autoresearch

This repo is for speeding up and modernizing WireSegHR on a single RTX 4090
while preserving or improving segmentation quality.

The modernization plan in `MODERNIZATION_PLAN.md` is the source of truth for
what to try first.

## Setup

1. Create a fresh branch for the run, using a date-based tag:
   - `autoresearch/<tag>`
2. Read these files for context:
   - `README.md`
   - `MODERNIZATION_PLAN.md`
   - `paper-tex/sections/method_yq.tex`
3. Ensure the benchmark ledger exists:
   - `results.tsv`
4. Use the launcher script for non-interactive runs:
   - `scripts/autoresearch_codex.sh`
5. After each benchmark, append a row with:
   - `scripts/autoresearch_record.py`

## Scope

Treat the following as the normal search surface:

- `train.py`
- `infer.py`
- `configs/default.yaml`
- `src/wireseghr/data/transforms.py`
- `src/wireseghr/data/sampler.py`
- `src/wireseghr/model/condition.py`
- `src/wireseghr/model/encoder.py`
- `src/wireseghr/model/decoder.py`

Prefer small, isolated changes. Do not mix unrelated experiments in one run.

Do not edit `paper-tex/` unless you are explicitly updating documentation to
match a code change.

## Experiment Order

Follow the modernization plan order:

1. Real augmentation
2. Stronger loss
3. Weighted stitching / better tile fusion
4. Routing improvements
5. Encoder upgrades
6. Conditioning upgrades
7. Topology-aware supervision

## Verify

Use the benchmark wrapper:

```bash
scripts/autoresearch_verify.sh
```

That should call the existing benchmark path in `infer.py` and write a JSON
report under `.autoresearch/`.

Primary metric:

- fine Wire F1

Secondary metrics:

- fine Wire IoU
- average total latency
- peak GPU memory

## Loop Rules

1. Make one change.
2. Run the verify script.
3. If the result is better or meaningfully simpler for the same score, keep it.
4. Otherwise revert it.
5. Append a row to `results.tsv` with the commit, metrics, status, and a short
   description.
6. Keep going without asking the human whether to continue.

If a run crashes because of a small bug, fix the bug and rerun. If the idea is
fundamentally bad, discard it and move to the next one.

## Results Format

Use one tab-separated row per experiment.

Recommended columns:

- commit
- fine_iou
- fine_f1
- coarse_iou
- coarse_f1
- avg_coarse_ms
- avg_fine_ms
- avg_total_ms
- peak_allocated_gb
- status
- description

## Reporting

When you finish a run, report:

- commit hash
- fine IoU / F1
- coarse IoU / F1
- average total latency
- peak GPU memory
- status
- what changed
