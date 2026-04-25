#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-checkpoints}"

if [[ ! -d "$ROOT" ]]; then
  echo "No checkpoint directory found: $ROOT" >&2
  exit 0
fi

find "$ROOT" -type f \( \
    -name '*.bin' -o \
    -name '*.pt' -o \
    -name '*.pth' -o \
    -name '*.ckpt' -o \
    -name '*.safetensors' \
  \) -printf '%T@ %s %p\n' \
  | sort -n \
  | while read -r mtime bytes file; do
      if command -v sha256sum >/dev/null 2>&1; then
        sha="$(sha256sum "$file" | awk '{print $1}')"
      else
        sha="sha256sum_unavailable"
      fi
      printf '%s bytes=%s sha256=%s path=%s\n' "$mtime" "$bytes" "$sha" "$file"
    done
