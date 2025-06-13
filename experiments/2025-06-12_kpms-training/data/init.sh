#!/bin/bash

src_dir="/projects/kumar-lab/sabnig/EM/Videos"
dest_dir="/projects/kumar-lab/miaod/projects/uvFI/experiments/2025-06-12_kpms-training/data"
video_path_list="${dest_dir}/good_video_paths.txt"
pose_path_list="${dest_dir}/good_pose_paths.txt"

echo "src_dir: $src_dir"
echo "dest_dir: $dest_dir"
echo "video_path_list: $video_path_list"
echo "pose_path_list: $pose_path_list"

echo
echo "started copying files to videos/..."
mkdir -p "$dest_dir/videos"
num_videos=0
while IFS= read -r relpath || [-n "$relpath" ]; do
  [[ -z "$relpath" || "${relpath:0:1}" == "#" ]] && continue

  filepath="$src_dir/$relpath"
  new_filepath="${filepath/_trimmed_overlay.mp4/_trimmed.mp4}"
  filename=$(basename "$filepath")
  new_filename="${new_filename/_trimmed.mp4/.mp4}"

  if [[ -f "$new_filepath" ]]; then
    ln -s "$new_filepath" "$dest_dir/videos/$new_filename"
    (( num_videos++ ))
  else
    echo "video not found: $new_filepath"
  fi
done < "$video_path_list"
echo "finished copying $num_videos files to videos/"

echo
echo "started copying files to poses/..."
mkdir -p "$dest_dir/poses"
num_poses=0
while IFS= read -r relpath || [-n "$relpath" ]; do
  [[ -z "$relpath" || "${relpath:0:1}" == "#" ]] && continue

  filepath="$src_dir/$relpath"
  new_filepath="${filepath/__trimmed_overlay_pose_est_v6.h5/_trimmed_pose_est_v6.h5}"
  filename=$(basename "$new_filepath")
  new_filename="${filename/_trimmed_pose_est_v6.h5/_pose_est_v6.h5}"

  if [[ -f "$new_filepath" ]]; then
    ln -s "$new_filepath" "$dest_dir/poses/$new_filename"
    (( num_poses++ ))
  else
    echo "pose not found: $new_filepath"
  fi
done < "$pose_path_list"
echo "finished copying $num_poses files to poses/"
