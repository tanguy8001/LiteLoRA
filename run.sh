#!/bin/bash
# Minimal launcher for a single LiteLoRA run.
# Usage: bash run.sh [path/to/config.json]
# Example: bash run.sh exps/litelora_inr.json
set -e

CONFIG="${1:-exps/litelora_inr.json}"

python3 main.py --config="${CONFIG}"
