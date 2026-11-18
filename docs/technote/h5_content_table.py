#!/usr/bin/env python
"""Print the column documentation of the open-sample HDF5 files (uboone/OpenSamples v02, file-content-hdf5.md, CC BY 4.0)
as a LaTeX longtable, for the technote's appendix. The source is h5_content.md next to this script."""
import pathlib, re

SRC = pathlib.Path(__file__).resolve().parent / "h5_content.md"


def tex(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                 ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"), ("<", "\\textless{}"), (">", "\\textgreater{}")]:
        s = s.replace(a, b)
    s = re.sub(r"https?://[^\s)]+", lambda m: "\\url{" + m.group(0).replace("\\_", "_").replace("\\#", "#").replace("\\%", "%") + "}", s)
    return s


rows = []
for line in SRC.read_text().splitlines():
    if not line.startswith("/") and not line.startswith("/ |"):
        continue
    cells = [c.strip() for c in line.split("|")]
    if len(cells) < 4:
        continue
    rows.append(cells[:4])

print("\\begin{longtable}{@{}p{34mm}p{10mm}p{7mm}p{100mm}@{}}")
print("\\caption{Content of the open-sample HDF5 files, as documented by the collaboration (uboone/OpenSamples v02, \\texttt{file-content-hdf5.md}; the \\texttt{\\#CHECK} marks are theirs). The \\texttt{wire\\_table} is absent from the NoWire files.}\\label{tab:h5content}\\\\")
print("\\toprule entry & type & $n$ & description\\\\ \\midrule \\endfirsthead")
print("\\toprule entry & type & $n$ & description\\\\ \\midrule \\endhead")
print("\\bottomrule \\endfoot")
for entry, typ, n, desc in rows:
    name = entry.split("/")[-1] or "/"
    if typ == "Group":
        print(f"\\addlinespace[3pt]\\multicolumn{{4}}{{@{{}}p{{151mm}}@{{}}}}{{\\textbf{{\\texttt{{{tex(entry)}}}}} \\quad {tex(desc)}}}\\\\")
    else:
        print(f"\\quad\\texttt{{{tex(name)}}} & {tex(typ).lower()} & {tex(n)} & {tex(desc)}\\\\")
print("\\end{longtable}")
