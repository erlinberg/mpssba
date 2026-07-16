# MPSSBA (Matrix Product State Spectral Bias Analysis)

MPSSBA is a research pipeline designed to evaluate the learning dynamics of Matrix Product States (MPS) on Fourier coefficients. 

## Installation

Ensure you have a Python 3.10+ virtual environment.

```bash
# Clone the repository and install in editable mode
pip install -e .

```

*Note: The pipeline relies on `qfso`, ensure it is installed separately*

## Configuration

Experiments are driven by JSON configurations.

**Example (`configs/config.example.json`)**:

```json
{
    "model": {
        "n_qubits": 12,
        "bond_dim": 64
    },
    "training": {
        "masking": {
            "type": "corr_2body", // Can be hamming, correlation, corr_2body or corr_2body_lower
            "value": 12 // Limits the corresponding parameter (Hamming Weight for hamming and Correlation Length for correlation, corr_2body and corr_2body_lower which denotes the limit from below)
        },
        "epochs": 1000,
        "tolerance": 1e-17,
        "record_every": 10 // Step of recording MPS for further postprocessing during training
    },
    "dist": {
        "method": "boltzmann", // Currently only boltzman distribution is allowed
        "kwargs": {
            "target_entropy": 3.4
        }
    },
    "init": {
        "method": "rnd_full", // Can be rnd_full, rnd_fact, uniform and from_pickle
        "kwargs": {} // Currently only used by from_pickle - parameter file_path should be specified as the path of a packle generated before
    },
    "batch": {
        "runs": 10, // # of runs which share same parameters except the random seed
        "postfix": "_run{run_idx}", // Prefix of the runs generated
        "run_continue": true, // If the run will be a continuation of a previous one
        "previous_run_path": "results/runs/run1" // If the run will be a continuation the absolute path of the previous run
    }
}
```

## Usage

### 1. Run the Training Pipeline

You can trigger a batched run using the main pipeline script. The script will automatically handle multiple seeds and save the raw state histories.

```bash
python scripts/run_pipeline.py --config configs/config.example.json
```

Output data will be stored under `results/runs/<timestamp>_<run_signature>`.

### 2. Manual Post-Processing (Optional)

If you wish to recalculate the metric distributions (Hamming mass, etc.) without retraining:

```bash
python scripts/postprocess.py --all
```

### 3. Generate Visualizations

```bash
python scripts/plot.py --runs results/runs/<timestamp>
```
