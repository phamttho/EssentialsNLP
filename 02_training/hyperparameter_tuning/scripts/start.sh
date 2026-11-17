#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/essentials_in_text_and_speech/02_training/hyperparameter_tuning
export PYTHONDONTWRITEBYTECODE=1
exec ../environment/bin/python -B run.py start
