#!/bin/bash
# Download MimicGen SOURCE human datasets (10 demos/task) straight from HuggingFace.
#
# We bypass mimicgen/scripts/download_datasets.py on purpose: importing `mimicgen` pulls in
# robosuite -> mujoco.egl, which has no EGL device on a login node and dies with
# "AttributeError: 'NoneType' object has no attribute 'eglQueryString'". The registry in
# mimicgen/__init__.py is just HF_REPO_ID + "source/<task>.hdf5", so curl reproduces it exactly.
#
# Already present (prepared, datagen_info attached): hammer_cleanup, square, threading.
#
# Usage:  ./mg_download_source_datasets.sh          # 3 parallel
#         JOBS=1 ./mg_download_source_datasets.sh   # sequential
set -e
REPO=amandlek/mimicgen_datasets
DEST=/scratch1/hyeonhoo/code/mimicgen/datasets/source
TASKS="${TASKS:-kitchen coffee coffee_preparation nut_assembly mug_cleanup pick_place stack stack_three three_piece_assembly}"
JOBS=${JOBS:-3}
mkdir -p "$DEST"

fetch() {
  # Runs under `bash -c` from xargs, so the top-level `set -e` does not apply: check every step.
  t="$1"
  url="https://huggingface.co/datasets/$REPO/resolve/main/source/$t.hdf5"
  dest="$DEST/$t.hdf5"
  want=$(curl -sIL "$url" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{n=$2}END{gsub(/\r/,"",n);print n}')
  [ -n "$want" ] || { echo "FAIL $t (no remote size)" >&2; return 1; }
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" = "$want" ]; then
    echo "SKIP $t (complete, $want bytes)"; return 0
  fi
  echo "DOWNLOADING $t ($want bytes)"
  curl -sL --fail -C - -o "$dest.part" "$url" || { echo "FAIL $t (curl $?)" >&2; return 1; }
  got=$(stat -c%s "$dest.part")
  [ "$got" = "$want" ] || { echo "FAIL $t (truncated $got/$want)" >&2; return 1; }
  mv "$dest.part" "$dest"
  echo "DONE $t ($got bytes)"
}
export -f fetch
export DEST REPO

if echo "$TASKS" | tr ' ' '\n' | xargs -I{} -P "$JOBS" bash -c 'fetch {}'; then
  echo "ALL DOWNLOADS COMPLETE"
else
  echo "SOME DOWNLOADS FAILED -- rerun to retry" >&2; exit 1
fi
