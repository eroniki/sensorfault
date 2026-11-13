#!/usr/bin/env bash
# Download Replica v1.0 (facebookresearch/Replica-Dataset) -- dry-run by default.
#
# Verified (2026-11-13) from the repo's download.sh and GitHub release v1.0 asset list:
#   * 17 parts replica_v1_0.tar.gz.parta{a..q}; 16 x 2,000,000,000 B + 1,859,047,808 B
#     = 33.86 GB compressed (extracted size: TODO(verify), expect ~2x).
#   * upstream download.sh resumes with `wget --continue`; no checksums are published
#     -> we verify exact part sizes (below) and tar integrity (gzip -t on the stream).
#   * extra habitat configs: http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip
# LICENCE (repo LICENSE, "Replica Dataset Research Terms"): use/modify/publish ONLY for
#   non-commercial research or education; may share with colleagues only if they agree to
#   the same terms; provided AS IS. => DO NOT REDISTRIBUTE the dataset or any renderings
#   derived from it. Release only generator code + seeds + manifests.
#
# Usage: replica.sh [--yes] --accept-license [--dest DIR] [--with-habitat-configs]
#   default dest: ${SENSORFAULT_DATA:-./data}/replica
# Layout produced: <dest>/<scene>/{mesh.ply,textures/,glass.sur,habitat/...} e.g. room_0, office_0, apartment_0 ...
set -euo pipefail

YES=0; ACCEPT=0; HAB=0
DEST="${SENSORFAULT_DATA:-./data}/replica"
BASE="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
PARTS=(a b c d e f g h i j k l m n o p q)
FULL=2000000000; LAST=1859047808

while [ $# -gt 0 ]; do
  case "$1" in
    --yes) YES=1 ;;
    --dry-run) YES=0 ;;
    --accept-license) ACCEPT=1 ;;
    --with-habitat-configs) HAB=1 ;;
    --dest) DEST="$2"; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

echo "[replica] dest=$DEST mode=$([ $YES = 1 ] && echo DOWNLOAD || echo DRY-RUN)"
echo "[replica] LICENCE: Replica Dataset Research Terms (non-commercial research/education only)."
echo "[replica]          Do NOT redistribute the dataset or renderings. https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE"
echo "[replica] EXPECTED SIZE: 33.86 GB compressed (17 parts); ~2x that extracted (TODO(verify)); need ~100 GB free to be safe."
for p in "${PARTS[@]}"; do echo "  $BASE/replica_v1_0.tar.gz.part a$p" | sed 's/part a/parta/'; done
[ $HAB = 1 ] && echo "  http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip"

if [ $YES = 0 ]; then echo "Dry-run only. Re-run with --yes --accept-license to download."; exit 0; fi
[ $ACCEPT = 1 ] || { echo "ERROR: --accept-license required with --yes" >&2; exit 2; }

mkdir -p "$DEST/_parts"
cd "$DEST/_parts"
for p in "${PARTS[@]}"; do
  f="replica_v1_0.tar.gz.parta$p"
  want=$FULL; [ "$p" = q ] && want=$LAST
  wget --continue --tries=5 "$BASE/$f"
  got=$(stat -c %s "$f")
  [ "$got" -eq "$want" ] || { echo "SIZE MISMATCH $f: $got != $want" >&2; exit 1; }
done
if command -v pigz >/dev/null; then GZ="pigz -dc"; else GZ="gzip -dc"; fi
cat replica_v1_0.tar.gz.part?? | $GZ | tar -x -C "$DEST"
if [ $HAB = 1 ]; then
  mkdir -p "$DEST/_assets"
  wget --continue -P "$DEST/_assets" http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip
  unzip -qn "$DEST/_assets/additional_habitat_configs.zip" -d "$DEST"
fi
echo "done; parts kept in $DEST/_parts (delete to free 34 GB)"
