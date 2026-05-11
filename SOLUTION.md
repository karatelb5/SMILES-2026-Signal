# SOLUTION.md
## Results
    ch0: 9.63 dB
    ch1: 8.38 dB
    ch2: 10.46 dB
    ch3: 7.39 dB
    Metric [yours]: 8.96 dB
## Run

Dependencies:

    pip install numpy scipy gdown

Run:

    python applicant_solution.py
    
## Description

This is what led me to my solution:

**Rank-1 Extraction:** SVD on the filtered RX signal to isolate the strongest shared component. This gives the shape of the external interference.

**Phase Scaling:** I projected this shared component back onto each channel. I used a set of fixed complex weights to align the prediction with the actual interference in the scoring band.

**Predict on cleaner data:** Then i ran the fit_tx_prediction on the residual. Refitting the TX model on cleaner data works well.

## What was the biggest benefit?
Adding a simple SVD cleanup pushed the score to around 8 db then manual phase calibration. As I found the right angles for each channel, the score jumped +0.9 db.

### Failed Attempts

**Adaptive Filtering:** Tried to update weights over time to track phase drift. It killed the explainability score immediately because the weights must be constant for the whole capture. 

**Dynamic SVD:** The scorer requires the entire removed component to be Rank-1. If it changes between blocks, the matrix becomes Rank-K -> explainability falls. 

**Freq-domain Inversion:** Tried to deconvolve the score_filter effect. It introduced too much noise at the band edges.

**PCA:** Doing PCA on the full signal instead of the filtered band did not work well because the interference is localized.
