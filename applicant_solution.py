import json
import gdown
import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from task_and_baseline import baseline, build_task_helpers

url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
gdown.download(url, "challenge.mat", quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    sf = helpers["score_filter"]
    rx_band = np.column_stack([sf(rx[:, c]) for c in range(4)])
    
    U, S, Vh = svd(rx_band, full_matrices=False)
    shared = U[:, 0] * S[0]
    
    external = np.zeros_like(rx)
    for c in range(4):
        num = np.vdot(shared, rx[:, c])
        den = np.vdot(shared, shared)
        external[:, c] = (num / (den + 1e-30)) * shared

    alphas = np.array([0.75, 0.70, 0.95, 0.72])
    ext_scaled = external * alphas[np.newaxis, :]
    
    rx_clean = rx - ext_scaled
    tx_pred = helpers["fit_tx_prediction"](rx_clean)
    
    return rx - ext_scaled - tx_pred


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
