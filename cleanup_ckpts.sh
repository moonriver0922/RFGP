#!/bin/bash
# Keep only the newest .tar checkpoint in a directory.
# Usage: ./cleanup_ckpts.sh [CKPTS_DIR]
# Env:   LOC_CKPTS_DIR  (used when no argument is given)

set -euo pipefail

CKPTS_DIR="${1:-${LOC_CKPTS_DIR:-}}"
if [[ -z "${CKPTS_DIR}" ]]; then
  echo "Usage: $0 <ckpts_dir>   or set LOC_CKPTS_DIR" >&2
  exit 1
fi

cd "$CKPTS_DIR" || exit 1

mapfile -t files < <(ls -1v *.tar 2>/dev/null || true)
num_files=${#files[@]}

if [[ "$num_files" -gt 1 ]]; then
  largest_file="${files[$((num_files - 1))]}"
  for file in "${files[@]}"; do
    if [[ "$file" != "$largest_file" ]]; then
      rm -f "$file"
    fi
  done
  echo "Kept $largest_file (removed $((num_files - 1)) older checkpoints)"
else
  echo "Nothing to clean ($num_files checkpoint(s) in $CKPTS_DIR)"
fi
