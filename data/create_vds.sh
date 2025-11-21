#!/bin/bash
# Script to run create_vds_splits.py with predefined paths
# Usage: ./create_vds.sh

PARQUET="/work/dlclarge2/faridk-diff_force/datasets/Covla/labels_with_splits.parquet"
H5_DIR="/work/dlclarge2/faridk-diff_force/datasets/Covla/multi_8"
OUTPUT="/work/dlclarge2/faridk-diff_force/datasets/Covla/h5/vds_latent_8_splits_minimal.h5"


python3 create_vds_splits_latent.py \
    --parquet "$PARQUET" \
    --h5-dir "$H5_DIR" \
    --output "$OUTPUT" \
    --subsplit-cols is_dark turn_dir3 \
    --latents-dir "$H5_DIR" \
    --require-latents

echo "[INFO] Finished. Output written to $OUTPUT"

