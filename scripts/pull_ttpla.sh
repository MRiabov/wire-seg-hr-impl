#!/usr/bin/env bash
set -euo pipefail

# Download TTPLA zip and extract
# Usage:
#   scripts/pull_ttpla.sh [FILE_ID_OR_URL] [ZIP_NAME] [OUT_DIR]
# Defaults replicate the manual commands you ran.

FILE_ID_OR_URL="${1:-1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0}"
ZIP_NAME="${2:-data_original_size_v1.zip}"
OUT_DIR="${3:-dataset/ttpla_dataset}"

# Work inside dataset/ like you did manually
mkdir -p dataset
cd dataset

echo "Downloading TTPLA: ${FILE_ID_OR_URL} -> ${ZIP_NAME}"
gdown "${FILE_ID_OR_URL}" -O "${ZIP_NAME}" --fuzzy

echo "Unzipping ${ZIP_NAME} -> ${OUT_DIR}"
mkdir -p "${OUT_DIR}"
unzip -q -o "${ZIP_NAME}" -d "${OUT_DIR}"

echo "Done. Contents extracted under: ${OUT_DIR}"
