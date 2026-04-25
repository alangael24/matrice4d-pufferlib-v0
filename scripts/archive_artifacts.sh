#!/usr/bin/env bash
set -euo pipefail

BATCH_ROOT="${1:-runs}"
OUT_DIR="${2:-artifacts}"

if [[ ! -d "$BATCH_ROOT" ]]; then
  echo "Missing batch root: $BATCH_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

BATCH_ABS="$(cd "$(dirname "$BATCH_ROOT")" && pwd)/$(basename "$BATCH_ROOT")"
BATCH_NAME="$(basename "$BATCH_ABS")"
ARCHIVE_PATH="${OUT_DIR}/${BATCH_NAME}.tgz"
MANIFEST_PATH="${BATCH_ABS}/artifact_manifest.txt"

{
  echo "artifact_manifest_for=${BATCH_NAME}"
  echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  find "$BATCH_ABS" -type f -printf '%s %p\n' | sort -k2
} > "$MANIFEST_PATH"

tar -czf "$ARCHIVE_PATH" -C "$(dirname "$BATCH_ABS")" "$BATCH_NAME"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"
fi

echo "Created archive: $ARCHIVE_PATH"
if [[ -f "${ARCHIVE_PATH}.sha256" ]]; then
  cat "${ARCHIVE_PATH}.sha256"
fi
