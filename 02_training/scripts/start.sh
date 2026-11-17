#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/essentials_in_text_and_speech/02_training
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/root/autodl-tmp/essentials_in_text_and_speech/02_training/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
exec environment/bin/python -B -m src.runner start
