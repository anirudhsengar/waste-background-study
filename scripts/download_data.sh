#!/usr/bin/env bash
# Download every public input into data/raw/, verify SHA-256 against data/hashes_sha256.txt and extract into
# data/ext/ with the folder names src/common.py expects. Files already in data/raw/ are reused.
#   RAW_DIR=/path/with/downloads  copy files from there instead of downloading
#   SKIP_U2NET=1                  skip u2net.onnx (176 MB; only `python -m src.s04_masks u2net` needs it)
set -euo pipefail
cd "$(dirname "$0")/.."
RAW_DIR=${RAW_DIR:-}
SKIP_U2NET=${SKIP_U2NET:-0}
PRAYITNO=https://raw.githubusercontent.com/kang-prayit/n1-trustworthy-waste-classification/b0e94ae2dad5aafb17b4b4865430ff239c6e677d/splits/raw_artifacts/trashnet
TRASHNET_COMMIT=6fa2b878c6c1b4304b91109070ce0edf9279bb31   # files match data/hashes_sha256.txt
TRASHNET=https://raw.githubusercontent.com/garythung/trashnet/$TRASHNET_COMMIT/data
mkdir -p data/raw/prayitno data/ext/trashnet data/ext/realwaste data/ext/seg data/ext/dtd

get() {
  local f=$1 url=$2
  [ -f "data/raw/$f" ] && return
  if [ -n "$RAW_DIR" ] && [ -f "$RAW_DIR/$f" ]; then echo "copy $f"; cp "$RAW_DIR/$f" "data/raw/$f"; return; fi
  echo "download $f"
  curl -L --fail --retry 3 -o "data/raw/$f.part" "$url" && mv "data/raw/$f.part" "data/raw/$f"
}

get trashnet.zip "https://github.com/garythung/trashnet/raw/$TRASHNET_COMMIT/data/dataset-resized.zip"
for s in train val test; do get "official_$s.txt" "$TRASHNET/one-indexed-files-notrash_$s.txt"; done
get realwaste.zip https://archive.ics.uci.edu/static/public/908/realwaste.zip
get trashnet-seg.zip https://data.mendeley.com/public-files/datasets/f6y37w8hxb/files/1faa3743-0159-42bf-b14a-b3528465ad64/file_downloaded
get dtd-r1.0.1.tar.gz https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
get prayitno/tn_index.csv "$PRAYITNO/index.csv"
get prayitno/tn_groups.npy "$PRAYITNO/groups.npy"
[ "$SKIP_U2NET" = 1 ] || get u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

echo "checking SHA-256"
if [ "$SKIP_U2NET" = 1 ]; then pattern=' u2net\.onnx$'; else pattern='^$'; fi
if ! (cd data/raw && grep -v -E "$pattern" ../hashes_sha256.txt | sha256sum -c -); then
  echo "HASH MISMATCH: delete the listed file(s) in data/raw and run again (a download may have been cut off)" >&2
  exit 1
fi

[ -d data/ext/trashnet/dataset-resized ] || unzip -q data/raw/trashnet.zip -d data/ext/trashnet
[ -d data/ext/realwaste/realwaste-main ] || unzip -q data/raw/realwaste.zip -d data/ext/realwaste
[ -d "data/ext/seg/trashnet - Copy" ] || unzip -q data/raw/trashnet-seg.zip -d data/ext/seg
[ -d data/ext/dtd/dtd ] || tar xzf data/raw/dtd-r1.0.1.tar.gz -C data/ext/dtd
echo "data ready in data/raw and data/ext"
