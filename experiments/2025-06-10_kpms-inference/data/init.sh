#!/bin/bash  

src_dirs=(
    "/projects/kumar-lab/sabnig/EM/Videos/LL1-B2B"
    "/projects/kumar-lab/sabnig/EM/Videos/LL2-B2B"
    "/projects/kumar-lab/sabnig/EM/Videos/LL3-B2B"
    "/projects/kumar-lab/sabnig/EM/Videos/LL4-B2B"
    "/projects/kumar-lab/sabnig/EM/Videos/LL5-B2B"
    "/projects/kumar-lab/sabnig/EM/Videos/LL6-B2B"
)

for src_dir in "${src_dirs[@]}"; do
	name=$(basename "$src_dir")

	dest_dir="/projects/kumar-lab/miaod/experiments/2025-06-10_kpms-inference/data"
	num_videos=$(find "$src_dir" -type f -iname "*.mp4" | wc -l)
	num_poses=$(find "$src_dir" -type f -iname "*_trimmed_pose_est_v6.h5" | wc -l)
	echo "src_dir: $src_dir"
	echo "dest_dir: $dest_dir"
	echo "num_videos: $num_videos"
	echo "num_poses: $num_poses"

	echo
	echo "started copying *.mp4 files to videos/..."
	mkdir -p "$dest_dir/videos"
	find "$src_dir" -type f -iname "*.mp4" | while read -r filepath; do
	    filename=$(basename "$filepath")
	    new_filename="${name}_${filename/_trimmed.mp4/.mp4}"
	    ln -s "$filepath" "$dest_dir/videos/$new_filename"
	done
	echo "finished copying *.mp4 files to videos/"

	echo
	echo "started copying *_trimmed_pose_est_v6.h5 files to poses/..."
	mkdir -p "$dest_dir/poses"
	find "$src_dir" -type f -iname "*_trimmed_pose_est_v6.h5" | while read -r filepath; do
	    filename=$(basename "$filepath")
	    new_filename="${name}_${filename/_trimmed_pose_est_v6.h5/_pose_est_v6.h5}"
	    ln -s "$filepath" "$dest_dir/poses/$new_filename"
	done
	echo "finished copying *_trimmed_pose_est_v6.h5 files to poses/"
done
