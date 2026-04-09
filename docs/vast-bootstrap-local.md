# Vast Bootstrap: Local-Side Checklist

This file captures the part we do from the local machine when a new Vast.ai
instance is provisioned.

## Current SSH Target

- Alias: `vast-wireseghr`
- Host: `77.33.143.182`
- Port: `10136`
- User: `root`
- SSH key: `~/.ssh/vastai_wireseghr`

Use the alias when possible:

```bash
ssh vast-wireseghr
```

If the alias is missing, use the direct command:

```bash
ssh -i ~/.ssh/vastai_wireseghr -p 10136 root@77.33.143.182
```

## What We Do Locally

1. Confirm the SSH target is alive.
2. Copy the Codex auth/config state from `~/.codex/` to `/root/.codex/` on the
   instance, excluding the large log/cache directories.
3. Install Codex on the instance if it is missing:
   `npm i -g @openai/codex`
4. Copy the Google Drive service account key from
   `/home/maksym/Downloads/g-account-drive-key.json` to the repo checkout on the
   instance as `secrets/drive-json.json`.
5. Install the Python dependencies inside the remote `/venv/main` environment:
   `uv pip install -r requirements.txt`
6. Pull and preprocess the datasets.
7. Run the benchmark wrapper and record the result in `results.tsv`.

## Remote Shell Notes

The Vast image auto-activates `/venv/main` for login shells. For non-interactive
commands, source it explicitly:

```bash
source /venv/main/bin/activate
```

That shell already has:

- `npm`
- `uv`
- `python3`

but it does not yet have all project Python packages. The missing ones that
matter for this repo are `pydrive2`, `gdown`, and `transformers`, so the
`uv pip install -r requirements.txt` step is still required.

## Dataset Prep

The repo’s data pipeline lives in `scripts/setup_script.sh` and
`scripts/pull_and_preprocess_wireseghr_dataset.py`.

Recommended order:

```bash
cd /workspace/wire-seg-hr-impl
source /venv/main/bin/activate
uv pip install -r requirements.txt
python3 scripts/pull_and_preprocess_wireseghr_dataset.py pull \
  --output-dir dataset/wireseghr_raw \
  --service-account secrets/drive-json.json
bash scripts/pull_ttpla.sh "" "" ttpla_dataset
bash scripts/setup_script.sh
```

If you want to use different TTPLA paths, update the environment variables in
`scripts/setup_script.sh` before running it.

## Benchmark / Experiment Loop

Run the benchmark wrapper:

```bash
scripts/autoresearch_verify.sh
```

Then append a ledger row:

```bash
python3 scripts/autoresearch_record.py \
  --report .autoresearch/bench.json \
  --status keep \
  --description "baseline after bootstrap"
```

## What To Repeat On The Next Instance

1. Update the SSH alias to the new `HostName` and `Port`.
2. Re-copy `~/.codex/` to the new instance.
3. Re-copy the Google Drive key.
4. Re-run `uv pip install -r requirements.txt`.
5. Re-run the dataset prep and benchmark steps above.

