import h5py
import keypoint_moseq as kpms
import re
from pathlib import Path
from tqdm import tqdm

from utils import _read_project_info

def create_groups_csv():
    """Creates a <project_dir>/index.csv.

    KPMS expects a file that maps recording names to their corresponding groups,
    used to perform group-wise comparisons. This function creates that file in
    <project_dir>/index.csv and assigns each recording its own group.
    """

    project_dir, pose_dir = _read_project_info("project_dir", "pose_dir")
    index_path = Path(project_dir) / "index.csv"

    if index_path.is_file():
        print(f"File at {index_path} already exists, skipping this step")
        return

    csv_header = "name,group"
    csv_rows = "\n".join([
        f"{file.name},{file.stem}" for file in Path(pose_dir).iterdir()
    ])
    csv_text = csv_header + "\n" + csv_rows

    index_path.write_text(csv_text)

def combine_inference_results(batched_result_file_pat: str):

    project_dir, model_name = _read_project_info("project_dir", "model_name")
    result_dir = Path(project_dir) / model_name
    result_path = result_dir / "results.h5"

    if result_path.is_file():
        print(f"File at {result_path} already exists, skipping this step")
        return

    batched_result_file_pat = re.compile(batched_result_file_pat)
    matches = [
        p for p in result_dir.rglob("*")
        if p.is_file() and batched_result_file_pat.fullmatch(p.name)
    ]

    with h5py.File(result_path, "w") as h5out:
        for in_file in tqdm(matches):
            with h5py.File(in_file, "r") as h5in:
                for group_name in h5in:
                    h5in.copy(group_name, h5out)
        print(f"Successfully combined {len(matches)} files")

def extract_results_csv():
    """Converts the results file (located in <project_dir>/<model_name>/
    results.h5) to a .csv file"""

    project_dir, model_name = _read_project_info("project_dir", "model_name")
    results = kpms.load_results(project_dir, model_name)
    kpms.load_results(results, project_dir, project_name)
