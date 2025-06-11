#!/bin/bash

EXP_DIR="/projects/kumar-lab/miaod/projects/uvFI/experiments/2025-06-10_kpms-inference"
SOURCE_DIR="$EXP_DIR/data/videos/poses_csv"
TARGET_DIR_PREFIX="$EXP_DIR/data/videos/poses_csv_"
files_per_folder=8

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

files=($(find "$SOURCE_DIR" -maxdepth 1 -type f | sort))
total_files=${#files[@]}

echo "Found $total_files files in $SOURCE_DIR"

total_folders=$(( (total_files + files_per_folder - 1) / files_per_folder ))

echo "Will create $total_folders folders with up to $files_per_folder files per folder"

for ((folder=1; folder<=total_folders; folder++)); do
    target_dir="${TARGET_DIR_PREFIX}${folder}"
    mkdir -p "$target_dir"
    
    start_idx=$(( (folder-1) * files_per_folder ))
    end_idx=$(( folder * files_per_folder - 1 ))
    
    if [ $end_idx -ge $total_files ]; then
        end_idx=$(( total_files - 1 ))
    fi
    
    files_to_copy=$(( end_idx - start_idx + 1 ))
    
    echo "Creating folder: $target_dir with $files_to_copy files"
    
    for ((i=start_idx; i<=end_idx; i++)); do
        filename=$(basename "${files[$i]}")
        cp "${files[$i]}" "$target_dir/$filename"
        echo "  Copied: $filename to $target_dir/"
    done
    
    if [ $folder -eq $total_folders ] && [ $files_to_copy -lt $files_per_folder ]; then
        echo "Last folder contains $files_to_copy files (less than the full $files_per_folder)"
    fi
done

echo "Done! Distributed $total_files files into $total_folders folders."
