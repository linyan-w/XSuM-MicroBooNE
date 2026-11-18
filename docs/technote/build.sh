#!/usr/bin/env bash
# Build docs/technote/technote.pdf: the schematic (standalone) first, then the technote (three passes: references and the longtable).
# Prints the second pass's overfull boxes and unresolved references, if any.
set -euo pipefail
cd "$(dirname "$0")"
python3 h5_content_table.py > h5_content_table.tex
( cd schematic && pdflatex -interaction=nonstopmode -halt-on-error schematic.tex > /dev/null && rm -f schematic.aux schematic.log )
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error technote.tex > /dev/null; done
pdflatex -interaction=nonstopmode -halt-on-error technote.tex > /dev/null
grep -E "Overfull|Warning: (Reference|Citation|Label)|undefined" technote.log || true
rm -f technote.aux technote.log technote.out technote.toc
echo "built $(pwd)/technote.pdf"
