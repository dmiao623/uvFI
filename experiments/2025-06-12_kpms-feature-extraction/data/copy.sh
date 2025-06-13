#!/bin/bash

src_dir="/projects/kumar-lab/miaod/projects/uvFI/experiments/2025-06-10_kpms-inference/outputs"
dest_dir="/projects/kumar-lab/miaod/projects/uvFI/experiments/2025-06-12_kpms-feature-extraction/data/2025-06-10_kpms-inference_data"

file_count=$(ls "$src_dir"/out_*.h5 2>/dev/null | wc -l)

mkdir -p "$dest_dir"
for file in "$src_dir"/out_*.h5; do
    ln -s "$file" "$dest_dir"/
done

echo "$file_count files copied from $src_dir to $dest_dir"
