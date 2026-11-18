#!/usr/bin/env bash
# File 00 end to end: wait for the verified download, slim it (event level) into data/slim/, measure it. Each step is
# skipped when its output exists, so the script can be re-run after the nightly reboot.
# Launch: systemd-run --user --unit xsum-ub-file00 -p MemoryMax=8G --working-directory=$PWD bash scripts/curate/pipeline_file00.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
H5=data/h5/BNB_All_NoWire_00.h5; STEM=data/slim/BNB_All_NoWire_00; LOG=reports/logs/file00.log; mkdir -p data/slim reports/logs
run() { echo "[$(date '+%F %T')] $*" >> $LOG; "$@" >> $LOG 2>&1; }
until [ -f $H5 ]; do sleep 60; done                       # fetch_zenodo.py renames .part -> .h5 only after the md5 check
[ -f $STEM.reco.parquet ] || run python3 -u scripts/curate/slim.py $H5 $STEM --chunk 5000
[ -f reports/file00_summary.md ] || run python3 -u scripts/curate/h5_summary.py $H5 --events 42674 --out reports/file00_summary.md
echo "[$(date '+%F %T')] done" >> $LOG
