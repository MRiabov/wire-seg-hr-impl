#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${CODEX_MODEL:-o3}"
PROFILE="${CODEX_PROFILE:-}"
SANDBOX="${CODEX_SANDBOX:-workspace-write}"
LAST_MESSAGE_PATH="${CODEX_LAST_MESSAGE_PATH:-$ROOT/.autoresearch/last_message.md}"

mkdir -p "$ROOT/.autoresearch"

cmd=(
  codex exec
  --full-auto
  --cd "$ROOT"
  --model "$MODEL"
  --sandbox "$SANDBOX"
  --output-last-message "$LAST_MESSAGE_PATH"
)

if [[ -n "$PROFILE" ]]; then
  cmd+=(--profile "$PROFILE")
fi

exec "${cmd[@]}" < "$ROOT/program.md"
