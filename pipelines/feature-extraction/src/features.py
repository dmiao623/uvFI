import numpy as np
import keypoint_moseq as kpms

from config import FIGURE_DIR
from utils import _read_project_info

def get_latent_embedding_statistics():
    project_dir, model_name = _read_project_info("project_dir", "model_name")
    results = kpms.load_results(project_dir, model_name)

    ret = {}
    for name, info in results.items():
        latent_embeddings = info["latent_state"]

        means   = latent_embeddings.mean(axis=0)
        medians = np.median(latent_embeddings, axis=0)
        stds    = latent_embeddings.std(axis=0, ddof=0)

        ret[name] = np.vstack([means, medians, stds])
    return ret

def get_syllable_frequencies(min_frequency: float=0.005, fps: int=30):
    project_dir, model_name = _read_project_info("project_dir", "model_name")

    moseq_df = kpms.compute_moseq_df(project_dir, model_name, smooth_heading=True)
    stats_df = kpms.compute_stats_df(project_dir, model_name, moseq_df,
                                     min_frequency=min_frequency, groupby=["name"], fps=fps)
    freq_wide = stats_df.pivot(index="name", columns="syllable", values="frequency").fillna(0)
    freq_array = freq_wide.to_numpy()
    return dict(zip(freq_wide.index, freq_array))

def get_transition_mats(min_frequency: float=0.005, normalize="bigram", enable_visualization=False):
    project_dir, model_name = _read_project_info("project_dir", "model_name")

    trans_mats, usages, groups, syll_include = kpms.generate_transition_matrices(
        project_dir, model_name, normalize=normalize, min_frequency=min_frequency
    )

    if enable_visualization:
        kpms.visualize_transition_bigram(
            project_dir,
            model_name,
            groups,
            trans_mats,
            syll_include,
            normalize=normalize,
            show_syllable_names=False,
            save_dir=FIGURE_DIR
        )

    results = kpms.load_results(project_dir, model_name)
    names = list(dict.fromkeys(results))
    assert len(names) == len(trans_mats)

    return dict(zip(names, trans_mats))



def get_all_features(min_frequency: float=0.005, fps: int=30, normalize="bigram"):
    labels = ["embedding_stats", "syllable_freqs", "transition_mats"]
    features = [
        get_latent_embedding_statistics(),
        get_syllable_frequencies(min_frequency, fps),
        get_transition_mats(min_frequency, normalize, False)
    ]

    return {
        k: {lbl: d[k] for lbl, d in zip(labels, features)}
        for k in features[0]
    }

def flatten_features(features_dict):
    row_labels = list(features_dict.keys())
    example_sample = features_dict[row_labels[0]]
    column_labels = []

    for feature_type, array in example_sample.items():
        flat_len = array.size
        column_labels.extend([f"{feature_type}_{i}" for i in range(flat_len)])

    matrix = []
    for sample_id in row_labels:
        flat_features = []
        for feature_type in example_sample:
            flat_array = features_dict[sample_id][feature_type].flatten()
            flat_features.append(flat_array)
        matrix.append(np.concatenate(flat_features))

    matrix = np.vstack(matrix)
    return matrix, row_labels, column_labels
