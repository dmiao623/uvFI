import os 
import jax
import sys

#from jax_moseq.utils import set_mixed_map_iters
import keypoint_moseq as kpms
import pathlib
from src.methods import load_and_format_data

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

print(jax.devices()[0].platform)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

G_BASE_PATH = "/projects/kumar-lab/chouda/unsupervised/KPMS/kp-moseq/" 
G_PROJ_NAME = "12k_jabs250_Jan7"
G_PROJ_PATH = G_BASE_PATH + G_PROJ_NAME + "/"  
MODEL_NAME = "2025_01_07-16_11_15"

# Load model checkpoint
model = kpms.load_checkpoint(G_PROJ_PATH, MODEL_NAME)[0]

G_BASE_PATH = "/projects/kumar-lab/miaod/experiments/2025-06-10_kpms-inference/data/videos"
new_data_dir = "/projects/kumar-lab/miaod/experiments/2025-06-10_kpms-inference/data/poses_csv"
project_path = pathlib.Path(G_PROJ_PATH)
data, metadata, coordinates = load_and_format_data(new_data_dir, project_path)

def config_func(): return kpms.load_config(project_path)

results = kpms.apply_model(model, data, metadata, project_path, MODEL_NAME, **config_func(), results_path = "/projects/kumar-lab/miaod/experiments/2025-06-10_kpms-inference/out" + sys.argv[1] + ".h5")

