#!/usr/bin/env python
"""Download files of the Zenodo record into a directory, one at a time, verifying the md5 and refusing to start a file
that would leave less than --reserve-gb free. Resumes a partial download with an HTTP Range request.

Usage: scripts/curate/fetch_zenodo.py DEST [--files 00 01] [--manifest configs/zenodo_8370883_files.txt] [--reserve-gb 20]
       scripts/curate/fetch_zenodo.py --manifest-update          # rewrite the manifest from the Zenodo API
Each NoWire file is about 11 GB and holds about 42k events; the whole sample is 195 GB (18 files).
"""
import argparse, hashlib, json, os, pathlib, shutil, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECORD = "8370883"


def read_manifest(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, size, md5, url = line.split()
        rows.append((name, int(float(size)), md5, url))
    return rows


def update_manifest(path):
    r = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}"))
    files = sorted(r["files"], key=lambda f: f["key"])
    with open(path, "w") as out:
        out.write(f"# {r['metadata']['title']}: Zenodo record {RECORD} (doi {r['doi']}), {r['metadata']['license']['id']}.\n")
        out.write(f"# Generated from https://zenodo.org/api/records/{RECORD} by scripts/curate/fetch_zenodo.py --manifest-update. Columns: name size_bytes md5 url\n")
        for f in files:
            out.write(f"{f['key']}  {f['size']}  {f['checksum'].split(':')[1]}  {f['links']['self']}\n")
    print(f"{len(files)} files, {sum(f['size'] for f in files)/1e9:.1f} GB -> {path}")


def md5sum(path, chunk=16 * 2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name, size, md5, url, dest, reserve_gb):
    out = dest / name
    if out.exists() and out.stat().st_size == size:
        ok = md5sum(out) == md5
        print(f"{name}: present, md5 {'ok' if ok else 'MISMATCH'}")
        return ok
    part = dest / (name + ".part")
    have = part.stat().st_size if part.exists() else 0
    free = shutil.disk_usage(dest).free
    if free - (size - have) < reserve_gb * 1e9:
        print(f"{name}: need {(size-have)/1e9:.1f} GB, only {free/1e9:.1f} GB free with a {reserve_gb} GB reserve; skipped")
        return False
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    t0 = time.time()
    with urllib.request.urlopen(req) as r, open(part, "ab" if have else "wb") as f:
        if have and r.status != 206:
            f.seek(0); f.truncate(); have = 0
        done = have
        for b in iter(lambda: r.read(16 * 2**20), b""):
            f.write(b); done += len(b)
            if done % (512 * 2**20) < 16 * 2**20:
                print(f"\r{name}: {done/1e9:6.2f}/{size/1e9:.2f} GB  {done/1e6/(time.time()-t0):.0f} MB/s", end="", flush=True)
    print()
    if part.stat().st_size != size:
        print(f"{name}: size {part.stat().st_size} != {size}; kept as .part for resume"); return False
    if md5sum(part) != md5:
        print(f"{name}: md5 mismatch; removed"); part.unlink(); return False
    part.rename(out)
    print(f"{name}: done, md5 ok, {time.time()-t0:.0f} s")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", nargs="?")
    ap.add_argument("--files", nargs="*", help="two-digit indices, e.g. 00 05; default: every h5 file")
    ap.add_argument("--manifest", default=str(ROOT / "configs/zenodo_8370883_files.txt"))
    ap.add_argument("--reserve-gb", type=float, default=20.0)
    ap.add_argument("--manifest-update", action="store_true")
    a = ap.parse_args()
    if a.manifest_update:
        update_manifest(a.manifest); sys.exit(0)
    if not a.dest:
        ap.error("DEST is required")
    dest = pathlib.Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_manifest(a.manifest) if r[0].endswith(".h5")]
    if a.files:
        rows = [r for r in rows if r[0][-5:-3] in a.files]
    ok = all([fetch(*r, dest, a.reserve_gb) for r in rows])
    sys.exit(0 if ok else 1)
