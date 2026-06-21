#!/bin/bash
# download_datasets.sh — Download CD benchmark datasets for PTA.
#
# Usage:
#   bash scripts/download_datasets.sh [--data-root /path/to/data]
#
# Defaults to ./data (same default as runner.py --data-root).
# Skips any dataset whose expected files already exist, so it is safe to rerun.
#
# Requires:
#   wget, tar, unzip, pip (for gdown — auto-installed if missing)
#
# Datasets covered: caltech101, dtd, eurosat, fgvc, oxford_flowers, oxford_pets, ucf101
# NOT YET covered:  imagenet, stanford_cars, food101, sun397, imagenetv2,
#                   imagenet-sketch, imagenet-a, imagenet-r  (stubs below)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DATA="./data"
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root) DATA="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "Data root: $DATA"
mkdir -p "$DATA"

# ---------------------------------------------------------------------------
# Dependency: gdown (Google Drive downloads)
# ---------------------------------------------------------------------------
if ! command -v gdown &> /dev/null; then
  echo "[setup] Installing gdown..."
  pip install -q gdown
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
gdrive() {
  # gdrive <file_id> <output_path>
  gdown --id "$1" -O "$2"
}

# ---------------------------------------------------------------------------
# caltech101  →  $DATA/caltech-101/
#   images:  101_ObjectCategories/
#   split:   split_zhou_Caltech101.json
# ---------------------------------------------------------------------------
CALTECH_DIR="$DATA/caltech-101"
if [ -d "$CALTECH_DIR/101_ObjectCategories" ] && [ -f "$CALTECH_DIR/split_zhou_Caltech101.json" ]; then
  echo "[caltech101] Already present, skipping."
else
  echo "[caltech101] Downloading..."
  mkdir -p "$CALTECH_DIR"

  if [ ! -d "$CALTECH_DIR/101_ObjectCategories" ]; then
    # vision.caltech.edu is dead; official mirror is now data.caltech.edu
    wget -c -L \
      "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1" \
      -O "$DATA/caltech-101.zip"
    unzip -q "$DATA/caltech-101.zip" -d "$DATA"
    rm -f "$DATA/caltech-101.zip"
    # zip extracts as $DATA/caltech-101/101_ObjectCategories/
  fi

  if [ ! -f "$CALTECH_DIR/split_zhou_Caltech101.json" ]; then
    gdrive "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN" "$CALTECH_DIR/split_zhou_Caltech101.json"
  fi

  echo "[caltech101] Done."
fi

# ---------------------------------------------------------------------------
# dtd  →  $DATA/dtd/
#   images:  images/
#   split:   split_zhou_DescribableTextures.json
# ---------------------------------------------------------------------------
DTD_DIR="$DATA/dtd"
if [ -d "$DTD_DIR/images" ] && [ -f "$DTD_DIR/split_zhou_DescribableTextures.json" ]; then
  echo "[dtd] Already present, skipping."
else
  echo "[dtd] Downloading..."
  mkdir -p "$DTD_DIR"

  if [ ! -d "$DTD_DIR/images" ]; then
    wget -c -L -P "$DATA" \
      "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    tar -xzf "$DATA/dtd-r1.0.1.tar.gz" -C "$DATA"
    rm -f "$DATA/dtd-r1.0.1.tar.gz"
    # archive extracts as $DATA/dtd/ already
  fi

  if [ ! -f "$DTD_DIR/split_zhou_DescribableTextures.json" ]; then
    gdrive "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x" "$DTD_DIR/split_zhou_DescribableTextures.json"
  fi

  echo "[dtd] Done."
fi

# ---------------------------------------------------------------------------
# eurosat  →  $DATA/eurosat/
#   images:  2750/
#   split:   split_zhou_EuroSAT.json
#
# Uses the HuggingFace mirror (same source as torchvision); extracts as 2750/.
# ---------------------------------------------------------------------------
EUROSAT_DIR="$DATA/eurosat"
if [ -d "$EUROSAT_DIR/2750" ] && [ -f "$EUROSAT_DIR/split_zhou_EuroSAT.json" ]; then
  echo "[eurosat] Already present, skipping."
else
  echo "[eurosat] Downloading..."
  mkdir -p "$EUROSAT_DIR"

  if [ ! -d "$EUROSAT_DIR/2750" ]; then
    # HuggingFace mirror (used by torchvision); extracts directly as 2750/
    wget -c -L \
      "https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip" \
      -O "$DATA/EuroSAT.zip"
    unzip -q "$DATA/EuroSAT.zip" -d "$EUROSAT_DIR"
    rm -f "$DATA/EuroSAT.zip"
  fi

  if [ ! -f "$EUROSAT_DIR/split_zhou_EuroSAT.json" ]; then
    gdrive "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o" "$EUROSAT_DIR/split_zhou_EuroSAT.json"
  fi

  echo "[eurosat] Done."
fi

# ---------------------------------------------------------------------------
# fgvc  →  $DATA/fgvc_aircraft/
#   images:  data/images/      (code expects fgvc_aircraft/data/images/)
#
# The archive extracts as fgvc-aircraft-2013b/; we rename the top-level dir
# to fgvc_aircraft so the nested data/ subdir matches what fgvc.py expects.
# ---------------------------------------------------------------------------
FGVC_DIR="$DATA/fgvc_aircraft"
if [ -d "$FGVC_DIR/data/images" ]; then
  echo "[fgvc] Already present, skipping."
else
  echo "[fgvc] Downloading..."

  wget -c -L -P "$DATA" \
    "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
  tar -xzf "$DATA/fgvc-aircraft-2013b.tar.gz" -C "$DATA"
  rm -f "$DATA/fgvc-aircraft-2013b.tar.gz"
  mv "$DATA/fgvc-aircraft-2013b" "$FGVC_DIR"

  echo "[fgvc] Done."
fi

# ---------------------------------------------------------------------------
# oxford_flowers  →  $DATA/oxford_flowers/
#   images:  jpg/
#   labels:  imagelabels.mat, cat_to_name.json
#   split:   split_zhou_OxfordFlowers.json
# ---------------------------------------------------------------------------
FLOWERS_DIR="$DATA/oxford_flowers"
if [ -d "$FLOWERS_DIR/jpg" ] && \
   [ -f "$FLOWERS_DIR/split_zhou_OxfordFlowers.json" ] && \
   [ -f "$FLOWERS_DIR/cat_to_name.json" ]; then
  echo "[oxford_flowers] Already present, skipping."
else
  echo "[oxford_flowers] Downloading..."
  mkdir -p "$FLOWERS_DIR"

  if [ ! -d "$FLOWERS_DIR/jpg" ]; then
    wget -c -L -P "$FLOWERS_DIR" \
      "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    tar -xzf "$FLOWERS_DIR/102flowers.tgz" -C "$FLOWERS_DIR"
    rm -f "$FLOWERS_DIR/102flowers.tgz"
    # archive extracts as $FLOWERS_DIR/jpg/
  fi

  if [ ! -f "$FLOWERS_DIR/imagelabels.mat" ]; then
    wget -c -L -P "$FLOWERS_DIR" \
      "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
  fi

  if [ ! -f "$FLOWERS_DIR/cat_to_name.json" ]; then
    gdrive "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0" "$FLOWERS_DIR/cat_to_name.json"
  fi

  if [ ! -f "$FLOWERS_DIR/split_zhou_OxfordFlowers.json" ]; then
    gdrive "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT" "$FLOWERS_DIR/split_zhou_OxfordFlowers.json"
  fi

  echo "[oxford_flowers] Done."
fi

# ---------------------------------------------------------------------------
# oxford_pets  →  $DATA/oxford_pets/
#   images:      images/
#   annotations: annotations/
#   split:       split_zhou_OxfordPets.json
# ---------------------------------------------------------------------------
PETS_DIR="$DATA/oxford_pets"
if [ -d "$PETS_DIR/images" ] && \
   [ -d "$PETS_DIR/annotations" ] && \
   [ -f "$PETS_DIR/split_zhou_OxfordPets.json" ]; then
  echo "[oxford_pets] Already present, skipping."
else
  echo "[oxford_pets] Downloading..."
  mkdir -p "$PETS_DIR"

  if [ ! -d "$PETS_DIR/images" ]; then
    wget -c -L -P "$PETS_DIR" \
      "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    tar -xzf "$PETS_DIR/images.tar.gz" -C "$PETS_DIR"
    rm -f "$PETS_DIR/images.tar.gz"
  fi

  if [ ! -d "$PETS_DIR/annotations" ]; then
    wget -c -L -P "$PETS_DIR" \
      "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    tar -xzf "$PETS_DIR/annotations.tar.gz" -C "$PETS_DIR"
    rm -f "$PETS_DIR/annotations.tar.gz"
  fi

  if [ ! -f "$PETS_DIR/split_zhou_OxfordPets.json" ]; then
    gdrive "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs" "$PETS_DIR/split_zhou_OxfordPets.json"
  fi

  echo "[oxford_pets] Done."
fi

# ---------------------------------------------------------------------------
# ucf101  →  $DATA/ucf101/
#   images:  UCF-101-midframes/     (mid-frame images, not raw video)
#   split:   split_zhou_UCF101.json
#
# Both files are hosted on Google Drive.
# ---------------------------------------------------------------------------
UCF_DIR="$DATA/ucf101"
if [ -d "$UCF_DIR/UCF-101-midframes" ] && [ -f "$UCF_DIR/split_zhou_UCF101.json" ]; then
  echo "[ucf101] Already present, skipping."
else
  echo "[ucf101] Downloading..."
  mkdir -p "$UCF_DIR"

  if [ ! -d "$UCF_DIR/UCF-101-midframes" ]; then
    gdrive "10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O" "$UCF_DIR/UCF-101-midframes.zip"
    unzip -q "$UCF_DIR/UCF-101-midframes.zip" -d "$UCF_DIR"
    rm -f "$UCF_DIR/UCF-101-midframes.zip"
  fi

  if [ ! -f "$UCF_DIR/split_zhou_UCF101.json" ]; then
    gdrive "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y" "$UCF_DIR/split_zhou_UCF101.json"
  fi

  echo "[ucf101] Done."
fi

# ---------------------------------------------------------------------------
# NOT YET IMPLEMENTED — stubs for remaining datasets
# ---------------------------------------------------------------------------
# imagenet        Manual download required (https://image-net.org/index.php)
# stanford_cars   Primary URLs (ai.stanford.edu) are down; check Kaggle mirror
# food101         https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
# sun397          http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
# imagenetv2      https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
# imagenet-sketch https://github.com/HaohanWang/ImageNet-Sketch (HuggingFace mirror recommended)
# imagenet-a      https://github.com/hendrycks/natural-adv-examples
# imagenet-r      https://github.com/hendrycks/imagenet-r

echo ""
echo "All requested datasets downloaded to: $DATA"
