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
            "type": "corr_2body", // Can be hamming, correlation and corr_2body
            "value": 12 // Limits the corresponding parameter (Hamming Weight for hamming and Correlation Length for correlation and corr_2body)
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
    "batch": {
        "runs": 10 // # of runs which share same parameters except the random seed
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
