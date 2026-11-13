#!/usr/bin/env python3
"""Download the depth-relevant subset of Skoltech3D (CVPR 2023).

Dry-run by default: prints the chunk list, expected sizes and licence, and
downloads NOTHING unless both ``--yes`` and ``--accept-license`` are given.

Dataset facts (verified from github.com/Skoltech-3D/sk3d_data README, its
``examples/conf.tsv``, ``src/sk3d/data/make_links.py`` and the public
"Skoltech3D data" spreadsheet, 2026-11-13):
  * Licence: data is CC BY-NC 4.0 (project page skoltech3d.appliedai.tech);
    the sk3d_data *code* repo is MIT.  Non-commercial use only, attribution
    required.  Cite Voynov et al., CVPR 2023.
  * Chunks are ZIPs <= 25 GB at
      https://storage.yandexcloud.net/skoltech3d/<chunk>.zip   (default)
      https://skoltech3d.appliedai.tech/data/<chunk>.zip       (slower mirror)
    Chunk names containing ``.all.`` hold ALL 107 scenes and cannot be
    subset; names containing ``{scene}`` are per scene.
  * No checksums are published by the authors (none in README/spreadsheet).
    We therefore verify (a) Content-Length from a HEAD request is matched by
    the finished file size and (b) ``unzip -tq`` integrity.

Depth-capable sensors selected here (camera names from the README):
  kinect_v2 (ToF), phone_right / phone_left (Huawei ToF), real_sense (D435
  stereo).  Plus the reference structured-light scan and calibration.
  RGB / TIS / raw / IR chunks are deliberately excluded (opt-in flags below).

Layout produced under ``$SENSORFAULT_DATA/skoltech3d`` (default ./data):
  _zips/                       downloaded archives (can be deleted after unzip)
  dataset/{scene}/{cam}/depth/undistorted/...
  dataset/{scene}/stl/reconstruction/cleaned.ply
  dataset/calibration/{cam}/{mode}/cameras.txt
  dataset/{scene}/{cam}/{mode}/images.txt          (poses, COLMAP text)
TODO(verify): that each zip's internal root is ``dataset/`` / ``raw/`` /
``addons/`` exactly as in the README path table (assumed, extracted into the
skoltech3d/ root); check with ``unzip -l`` on the 43 MB calibration zip first.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

SERVERS = {
    "storage.yandexcloud.net": "https://storage.yandexcloud.net/skoltech3d",
    "skoltech3d.appliedai.tech": "https://skoltech3d.appliedai.tech/data",
}
REPO_URL = "https://github.com/Skoltech-3D/sk3d_data"
LICENSE_TXT = ("Skoltech3D data: Creative Commons Attribution-NonCommercial 4.0 "
               "International (CC BY-NC 4.0). Non-commercial use only; cite "
               "Voynov et al., 'Multi-sensor large-scale dataset for multi-view 3D "
               "reconstruction', CVPR 2023. Terms: https://skoltech3d.appliedai.tech/")

# (chunk pattern, approx GiB all-scenes total [from 'Skoltech3D data' sheet], per-scene GiB or None, note)
# per-scene chunks: size column in the sheet is the 107-scene total, so
# per-scene estimate = total / 107.
CORE = [
    ("dataset.all.kinect_v2.depth.undistorted.zip", 5.0, None, "Kinect v2 ToF depth, ALL scenes"),
    ("dataset.all.phone_right.depth.undistorted.zip", 1.0, None, "phone R ToF depth, ALL scenes"),
    ("dataset.all.phone_left.depth.undistorted.zip", 1.0, None, "phone L ToF depth, ALL scenes"),
    ("dataset.{scene}.real_sense.depth.undistorted.zip", 279.0, 279.0 / 107, "RealSense D435 depth, per scene (12 lightings)"),
    ("dataset.all.stl.reconstruction.cleaned.zip", 3.0, None, "reference SL scan (cleaned PLY), ALL scenes"),
    ("dataset.all.calibration.zip", 0.04, None, "calibration + poses (43 MB, HEAD-verified)"),
]
OPT_RAW = [
    ("raw.all.kinect_v2.depth.raw.zip", 5.0, None, "Kinect raw depth"),
    ("raw.all.phone_right.depth.raw.zip", 0.5, None, "phone R raw depth (sheet says 0; rounded)"),
    ("raw.all.phone_left.depth.raw.zip", 0.5, None, "phone L raw depth (sheet says 0; rounded)"),
    ("raw.{scene}.real_sense.depth.raw.zip", 45.0, 45.0 / 107, "RealSense raw depth"),
]
OPT_IR = [
    ("dataset.all.kinect_v2.ir.undistorted.zip", None, None, "Kinect IR (size TODO(verify) from sheet)"),
    ("dataset.all.phone_right.ir.undistorted.zip", None, None, "phone R IR (size TODO(verify))"),
    ("dataset.all.phone_left.ir.undistorted.zip", None, None, "phone L IR (size TODO(verify))"),
    ("dataset.{scene}.real_sense.ir.undistorted.zip", None, None, "RealSense IR (size TODO(verify))"),
]
OPT_SLDEPTH = [  # SL reference depth reprojected onto each depth sensor's IR view (addons)
    ("addons.all.proj_depth.stl.clean_rec@kinect_v2_ir.undist.zip", 0.5, None, "SL ref depth at Kinect IR"),
    ("addons.all.proj_depth.stl.clean_rec@real_sense_ir.undist.zip", 1.0, None, "SL ref depth at RealSense IR"),
    ("addons.all.proj_depth.stl.clean_rec@phone_right_ir.undist.zip", 0.5, None, "SL ref depth at phone R IR"),
    ("addons.all.proj_depth.stl.clean_rec@phone_left_ir.undist.zip", 0.5, None, "SL ref depth at phone L IR"),
]


def data_root() -> Path:
    return Path(os.environ.get("SENSORFAULT_DATA", "./data")).resolve() / "skoltech3d"


def known_scenes(tool_dir: Path) -> list[str] | None:
    f = tool_dir / "src/sk3d/data/dataset/scenes.py"
    if not f.is_file():
        return None
    return re.findall(r"^\s*\d+\s*:\s*'([^']+)'", f.read_text(), flags=re.M)


def build_chunks(scenes: list[str], raw: bool, ir: bool, sl: bool):
    rows = list(CORE) + (OPT_RAW if raw else []) + (OPT_IR if ir else []) + (OPT_SLDEPTH if sl else [])
    out = []
    for pat, total, per_scene, note in rows:
        if "{scene}" in pat:
            for s in scenes:
                out.append((pat.format(scene=s), per_scene, note))
        else:
            out.append((pat, total, note))
    return out


def write_conf(path: Path, scenes: list[str], chunks) -> None:
    """Write a conf.tsv in the official make_links.py format (for cross-checking)."""
    lines = ["scenes = " + " ".join(scenes), ""]
    seen = set()
    for name, _, _ in chunks:
        pat = name
        for s in scenes:
            pat = pat.replace(f".{s}.", ".{scene}.")
        if pat not in seen:
            seen.add(pat)
            lines.append(f"\t{pat}")
    path.write_text("\n".join(lines) + "\n")


def head_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", "")) or None
    except Exception:
        return None


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=False,
                    help="scene names from scenes.py (e.g. dragon skate) or 'all'")
    ap.add_argument("--server", choices=SERVERS, default="storage.yandexcloud.net")
    ap.add_argument("--with-raw-depth", action="store_true", help="also raw (distorted) depth")
    ap.add_argument("--with-ir", action="store_true", help="also undistorted IR images")
    ap.add_argument("--with-sl-depth", action="store_true",
                    help="also SL reference depth reprojected to each depth sensor (addons)")
    ap.add_argument("--fetch-tooling", action="store_true",
                    help=f"shallow-clone {REPO_URL} into <root>/_tools/sk3d_data (needs network; honours dry-run)")
    ap.add_argument("--list-scenes", action="store_true", help="list scene names (needs tooling cloned)")
    ap.add_argument("--yes", action="store_true", help="actually download (default: dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="explicit dry-run (the default)")
    ap.add_argument("--accept-license", action="store_true", help="confirm you accept CC BY-NC 4.0")
    ap.add_argument("--check-remote", action="store_true",
                    help="HEAD each URL for Content-Length (small network use, no data)")
    ap.add_argument("--no-unzip", action="store_true")
    args = ap.parse_args()

    root = data_root()
    tool_dir = root / "_tools" / "sk3d_data"
    do_it = args.yes and not args.dry_run
    print(f"[skoltech3d] root={root}  mode={'DOWNLOAD' if do_it else 'DRY-RUN'}")
    print(f"[skoltech3d] LICENCE: {LICENSE_TXT}")
    if do_it and not args.accept_license:
        print("ERROR: --accept-license is required with --yes", file=sys.stderr)
        return 2

    if args.fetch_tooling:
        cmd = ["git", "clone", "--depth", "1", REPO_URL, str(tool_dir)]
        print("[tooling]", " ".join(cmd))
        if do_it and not tool_dir.exists():
            tool_dir.parent.mkdir(parents=True, exist_ok=True)
            if run(cmd):
                return 1
    names = known_scenes(tool_dir)
    if args.list_scenes:
        if names is None:
            print("scenes.py not found; run with --fetch-tooling --yes --accept-license first", file=sys.stderr)
            return 1
        print("\n".join(names))
        return 0
    if not args.scenes:
        print("ERROR: --scenes required (names or 'all')", file=sys.stderr)
        return 2
    scenes = args.scenes
    if names is not None:
        if scenes == ["all"]:
            scenes = sorted(names)
        bad = [s for s in scenes if s not in names]
        if bad:
            print(f"ERROR: unknown scene(s): {bad}", file=sys.stderr)
            return 2
    elif scenes == ["all"]:
        print("ERROR: 'all' needs scenes.py (use --fetch-tooling first)", file=sys.stderr)
        return 2
    else:
        print("[warn] scene names not validated (tooling not cloned); typos will 404")

    chunks = build_chunks(scenes, args.with_raw_depth, args.with_ir, args.with_sl_depth)
    base = SERVERS[args.server]
    total = 0.0
    unknown = 0
    print(f"\n{'chunk':72s} {'~GiB':>7s}  note")
    for name, gib, note in chunks:
        if gib is None:
            unknown += 1
        else:
            total += gib
        print(f"{name:72s} {('?' if gib is None else f'{gib:.2f}'):>7s}  {note}")
    print(f"\nEXPECTED SIZE ~ {total:.1f} GiB (+{unknown} chunk(s) of unknown size). Sizes from the public "
          f"'Skoltech3D data' sheet; '.all.' chunks cannot be subset by scene.")
    print("Checksums: none published upstream -> verifying HEAD Content-Length + `unzip -tq`.")

    zips = root / "_zips"
    conf = root / "_conf.tsv"
    print(f"\nofficial-equivalent: PYTHONPATH={tool_dir}/src python3 {tool_dir}/src/sk3d/data/make_links.py --conf {conf} --server {args.server}")
    if not do_it:
        write_note = "(would write)"
        print(f"conf.tsv {write_note}: {conf}")
        for name, _, _ in chunks:
            url = f"{base}/{name}"
            extra = ""
            if args.check_remote:
                sz = head_size(url)
                extra = f"  [remote {sz} B]" if sz else "  [remote size unavailable]"
            print(f"  curl -L -C - -o {zips/name} {url}{extra}")
        print("\nDry-run only. Re-run with --yes --accept-license to download.")
        return 0

    root.mkdir(parents=True, exist_ok=True)
    zips.mkdir(exist_ok=True)
    write_conf(conf, scenes, chunks)
    for name, _, _ in chunks:
        url = f"{base}/{name}"
        dest = zips / name
        want = head_size(url)
        # -C - resumes; --fail so HTTP errors are not saved as data.
        if run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url]):
            print(f"FAILED: {name}", file=sys.stderr)
            return 1
        if want and dest.stat().st_size != want:
            print(f"SIZE MISMATCH {name}: {dest.stat().st_size} != {want}", file=sys.stderr)
            return 1
        if shutil.which("unzip") and run(["unzip", "-tq", str(dest)]):
            print(f"ZIP INTEGRITY FAILED: {name}", file=sys.stderr)
            return 1
        if not args.no_unzip and run(["unzip", "-qn", str(dest), "-d", str(root)]):
            return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
