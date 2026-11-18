#!/usr/bin/env bash
# The chain on the slim files of data/slim: event stage, then the PFParticle set flow on top with the event stage frozen.
# Each step is skipped when its output exists. Launch as a capped unit:
#   systemd-run --user --unit xsum-ub-train -p MemoryMax=16G --working-directory=$PWD bash scripts/train/train_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/lwan/AI/Diffusion/.venv/bin/python}; OUT=${OUT:-reports}; STEMS=${STEMS:-data/slim/BNB_All_NoWire_00}; mkdir -p "$OUT/logs"
[ -f "$OUT/event00/metrics.json" ]   || $PY -u scripts/train/train_event.py "$OUT/event00" --stems $STEMS --epochs 30 --bs 512 > "$OUT/logs/train_event00.log" 2>&1
[ -f "$OUT/pfp00/metrics_pfp.json" ] || $PY -u scripts/train/train_pfp.py "$OUT/pfp00" --init "$OUT/event00/model.pt" --stems $STEMS --epochs 120 --bs 512 --only-pfp > "$OUT/logs/train_pfp00.log" 2>&1
echo "done: $OUT/event00 $OUT/pfp00"
