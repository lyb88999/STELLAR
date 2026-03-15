# STELLAR: Similarity-Based Satellite Federated Learning for Malicious Traffic Recognition

> **Published in:** IEEE Transactions on Information Forensics and Security, vol. 21, pp. 1766–1780, 2026.
> DOI: [10.1109/TIFS.2026.3659044](https://doi.org/10.1109/TIFS.2026.3659044)

[[中文文档]](README_zh.md)

STELLAR is a federated learning framework for LEO satellite constellation networks, targeting **malicious traffic recognition (intrusion detection)**. It proposes a **multi-dimensional similarity-based grouping** strategy that clusters satellites by model parameter similarity, loss landscape similarity, and prediction distribution similarity, enabling efficient intra-orbit collaborative training while adapting to the highly dynamic and resource-constrained satellite environment.

## System Architecture

```mermaid
flowchart TD
    subgraph CFG["Configuration  (YAML)"]
        C1["Constellation\nIridium-like: 6 orbits × 11 sats = 66\nOneWeb: 18 orbits × ~36 sats = 651"]
        C2["Dataset\nHe et al. TIFS 2025 (primary) / CICIDS-2017"]
        C3["FL Hyperparameters\nrounds · lr · batch_size · α · μ"]
        C1 --- C2 --- C3
    end

    subgraph DATA["Data Layer"]
        DG["Data Generator\nRealTrafficGenerator (He et al. 2025) / CICIDS2017Generator"]
        NI["Dirichlet Non-IID Partitioning\nalpha · region similarity"]
        SD["Per-Satellite Dataset Shards\n(orbit-correlated distribution)"]
        DG --> NI --> SD
    end

    subgraph NET["Satellite Network Simulation"]
        TLE["TLE Orbital Mechanics\nSkyfield · position(t)"]
        ISL["Dynamic ISL Topology\nconfigurable max_distance"]
        EM["Solar Energy Model\nbattery_level(t)"]
        TLE --> ISL --> EM
    end

    subgraph FL["Federated Learning Core  —  One Round"]
        subgraph CLI["Satellite Clients  ×N"]
            LT["Local Training\nMLP · SGD · local_epochs"]
            EC["Energy-Aware Scheduler\nbattery check per batch"]
            LT --- EC
        end

        subgraph SG["STELLAR Similarity Grouping  (every K rounds)"]
            SP["Parameter Similarity\ncosine distance · α = 0.4"]
            SL["Loss Similarity\ninverse loss diff · β = 0.3"]
            SH["Prediction Similarity\nHellinger distance · γ = 0.3"]
            CS["Composite Score\nS = α·S_param + β·S_loss + γ·S_pred"]
            GF["Greedy Group Formation\nhop-radius · max_group_size · threshold"]
            SP & SL & SH --> CS --> GF
        end

        subgraph AGG["Propagation-Aware Aggregation"]
            IA["Intra-Orbit FedAvg\ngroup representatives only"]
            GS["Ground Station Aggregation\nISL relay · visibility scheduling"]
            GA["Global FedAvg\nweighted average across ground stations"]
            IA --> GS --> GA
        end

        CLI --> SG
        SG --> AGG
        GA -->|"broadcast global model"| CLI
    end

    subgraph OUT["Evaluation & Output"]
        MT["Accuracy · F1 (macro/weighted)\nPrecision · Recall  per round"]
        EN["Training Energy · Comm Energy\nper-orbit breakdown"]
        MT --- EN
    end

    CFG -->|"num_satellites\ntle_file"| NET
    CFG -->|"csv_path\nalpha"| DATA
    CFG -->|"rounds · lr · mu"| FL
    SD --> CLI
    NET --> FL
    AGG --> OUT
```

## Overview

Traditional federated learning algorithms (FedAvg, FedProx) are designed for terrestrial networks and do not account for the unique characteristics of satellite networks:

- Intermittent and topology-changing inter-satellite links (ISLs)
- Heterogeneous non-IID data distributions across orbital planes
- Strict energy budgets driven by solar harvesting cycles
- High communication latency between satellites and ground stations

STELLAR addresses these challenges through:

1. **Similarity-based satellite grouping** — satellites are clustered by a composite similarity score combining parameter distance, loss divergence, and prediction agreement
2. **Propagation-constrained aggregation** — model updates propagate only within reachable hops, respecting link availability
3. **Energy-aware scheduling** — training and transmission are gated by real-time battery level estimates based on orbital solar exposure

## Repository Structure

```
stellar/
├── configs/                    # Experiment configuration files (YAML)
│   ├── baseline_config.yaml         # Base FL configuration (Iridium-like, 66 sats)
│   ├── similarity_grouping_config.yaml  # STELLAR algorithm config
│   ├── fedavg_config.yaml           # FedAvg baseline config
│   ├── fedprox_config.yaml          # FedProx baseline config
│   ├── propagation_fedavg_config.yaml   # Propagation-constrained FedAvg
│   ├── propagation_fedprox_config.yaml  # Propagation-constrained FedProx
│   ├── sda_fl_config.yaml           # SDA-FL baseline config
│   ├── Iridium_TLEs.txt             # Iridium NEXT TLE orbital elements
│   └── energy_config.yaml           # Satellite energy model parameters
│
├── data_simulator/             # Dataset loading and non-IID data partitioning
│   ├── real_traffic_generator.py    # CSV traffic dataset loader (He et al. TIFS 2025 primary dataset)
│   ├── cicids2017_generator.py      # CICIDS-2017 dataset loader
│   ├── non_iid_generator.py         # Dirichlet non-IID data partitioning
│   └── network_traffic_generator.py # Synthetic traffic generator
│
├── fl_core/                    # Core federated learning components
│   ├── client/
│   │   ├── satellite_client.py      # Satellite node training client
│   │   └── fedprox_client.py        # FedProx proximal-term client
│   ├── aggregation/
│   │   ├── intra_orbit.py           # Intra-orbit aggregation logic
│   │   ├── ground_station.py        # Ground station aggregation
│   │   └── global_aggregator.py     # Global model aggregation
│   └── models/
│       ├── real_traffic_model.py    # MLP classifier for traffic data
│       └── hybrid_traffic_model.py  # Hybrid autoencoder+classifier model
│
├── simulation/                 # Satellite network simulation
│   ├── network_model.py             # TLE-based orbital mechanics & ISL model
│   ├── topology_manager.py          # Dynamic topology and spectral grouping
│   ├── energy_model.py              # Solar power and battery simulation
│   ├── comm_scheduler.py            # Communication scheduling
│   └── network_manager.py           # Network state management
│
├── experiments/                # Experiment runners
│   ├── baseline_experiment.py       # Base experiment class
│   ├── grouping_experiment.py       # STELLAR (similarity grouping)
│   ├── fedavg_experiment.py         # Standard FedAvg
│   ├── fedprox_experiment.py        # Standard FedProx
│   ├── propagation_fedavg_experiment.py  # Propagation-limited FedAvg
│   ├── propagation_fedprox_experiment.py # Propagation-limited FedProx
│   ├── sda_fl_experiment.py         # SDA-FL (GAN-based data augmentation FL)
│   ├── async_experiment.py          # Asynchronous FL variant
│   └── run_fair_comparison_satfl.py # Main comparison script
│
├── visualization/              # Plotting utilities
│   ├── visualization.py             # Training curve visualization
│   └── comparison_visualization.py  # Multi-algorithm comparison plots
│
├── tests/                      # Unit tests
├── requirements.txt
└── setup.py
```

## Installation

**Requirements:** Python 3.9+, PyTorch 1.13+

```bash
git clone https://github.com/your-username/stellar.git
cd stellar
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Dataset

STELLAR uses network traffic classification as the FL task and supports two datasets. If you do not have these real datasets on hand, we provide a script to generate a dummy dataset so you can run the code out-of-the-box.

### 🚀 (Recommended) Generate Dummy Dataset

After cloning the repository, it is highly recommended to run this script to generate `data/STIN.csv`. This ensures all subsequent commands will run smoothly without raising `FileNotFoundError`:

```bash
python scripts/generate_dummy_data.py
# Generates a dummy dataset with 5000 samples and 10 features in data/STIN.csv
```

### Primary dataset — Satellite-terrestrial network traffic (He et al., TIFS 2025)

The main experiments in our paper use the network traffic dataset released alongside:

> J. He, X. Li, X. Zhang, W. Niu and F. Li, "A Synthetic Data-Assisted Satellite Terrestrial Integrated Network Intrusion Detection Framework," *IEEE Transactions on Information Forensics and Security*, vol. 20, pp. 1739–1754, 2025. DOI: [10.1109/TIFS.2025.3530676](https://doi.org/10.1109/TIFS.2025.3530676)

We downloaded the raw traffic files from that work, performed preprocessing (feature selection, label normalization, class merging), and concatenated them into a single CSV file. The merged file is used directly by `RealTrafficGenerator` via the `csv_path` config key.

```yaml
data:
  dataset: "real_traffic"
  csv_path: "data/satellite_traffic.csv"   # merged CSV from He et al. (2025)
```

The CSV must contain numerical feature columns plus a `Label` column (string class names or binary 0/1 are both supported).

### Secondary dataset — CICIDS-2017

As an alternative benchmark, STELLAR also supports the publicly available CICIDS-2017 intrusion detection dataset from the University of New Brunswick:

```
https://www.unb.ca/cic/datasets/ids-2017.html
```

Download the `MachineLearningCVE` folder and use `configs/cicids2017_config.yaml`.

**Supported datasets summary:**

| Dataset | Description | Config |
|---|---|---|
| He et al. TIFS 2025 (primary) | Satellite-terrestrial network traffic, merged CSV | `real_traffic` + `csv_path` |
| CICIDS-2017 | General intrusion detection benchmark | `cicids2017` |

## Quick Start

### Run the full comparison (STELLAR vs. baselines)

> **Note:** Training rounds and other hyperparameters are set in the YAML config file (e.g., `configs/similarity_grouping_config.yaml`) via the `fl.num_rounds` field. **They are not passed as command-line arguments.**

```bash
# Run directly (rounds and other params are configured in YAML)
python -m experiments.run_fair_comparison_satfl
```

Results are automatically saved to `comparison_results/with_satfl_<timestamp>/`.

**Available command-line arguments:**

| Argument | Default | Description |
|---|---|---|
| `--target-sats` | `0` | Target satellite count (0 = use STELLAR's average) |
| `--fedprox-mu` | `0.01` | FedProx proximal term coefficient μ |
| `--config-dir` | `configs` | Directory containing config YAML files |
| `--satfl-noise-dim` | `100` | SDA-FL noise dimension |
| `--satfl-samples` | `1000` | Number of synthetic samples generated by SDA-FL |
| `--replot` | — | Replot mode: skip training, regenerate plots from saved data |
| `--data-dir` | — | Directory of saved experiment data (used with `--replot`) |
| `--output-dir` | — | Directory for output plots (used with `--replot`) |
| `--format` | `png` | Plot format: `png` / `pdf` / `svg` / `jpg` |
| `--dpi` | `150` | Plot DPI |
| `--no-grid` | — | Disable grid lines in plots |

**Examples:**

```bash
# Run with a custom FedProx μ value
python -m experiments.run_fair_comparison_satfl --fedprox-mu 0.001

# Regenerate plots from a previous run (no retraining needed)
python -m experiments.run_fair_comparison_satfl \
    --replot \
    --data-dir comparison_results/with_satfl_20260315_153000 \
    --output-dir my_plots/ \
    --format pdf
```

### Run individual algorithms

```bash
# STELLAR (proposed method)
python -c "
from experiments.grouping_experiment import SimilarityGroupingExperiment
exp = SimilarityGroupingExperiment('configs/similarity_grouping_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"

# FedAvg baseline
python -c "
from experiments.fedavg_experiment import FedAvgExperiment
exp = FedAvgExperiment('configs/fedavg_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"

# FedProx baseline
python -c "
from experiments.fedprox_experiment import FedProxExperiment
exp = FedProxExperiment('configs/fedprox_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"

# SDA-FL baseline
python -c "
from experiments.sda_fl_experiment import SDAFLExperiment
exp = SDAFLExperiment('configs/sda_fl_config.yaml')
exp.prepare_data()
exp.setup_clients()
stats = exp.train()
"
```

## Configuration

> **Note:** The framework will automatically detect the actual number of satellites, orbits, and satellites per orbit based on the TLE file. This overrides the values of `fl.num_satellites`, `fl.num_orbits`, and `fl.satellites_per_orbit` in the configuration.

### General Parameters (Shared by all experiments)

```yaml
fl:
  num_satellites: 66          # Total number of satellites (Overridden by auto-detection)
  num_orbits: 6               # Number of orbital planes (Overridden by auto-detection)
  satellites_per_orbit: 11    # Satellites per orbital plane (Overridden by auto-detection)
  num_rounds: 20              # Total number of FL communication rounds
  round_interval: 600         # Simulated seconds between rounds

network:
  tle_file: "configs/Iridium_TLEs.txt"  # TLE orbital elements file (determines constellation)
  max_distance: 4000.0                   # Maximum inter-satellite communication distance (km)

data:
  dataset: "real_traffic"     # Dataset type: real_traffic | cicids2017
  csv_path: "data/STIN.csv"   # Path to the CSV data file
  iid: false                  # true = IID distribution; false = Non-IID (Dirichlet partition)
  alpha: 0.5                  # Dirichlet parameter (only active if non-IID)
  test_size: 0.2              # Test set ratio
  region_similarity: false    # Enable orbital region similarity data partitioning
  overlap_ratio: 0.5          # Region overlap ratio (only active if region_similarity: true)

model:
  type: "traffic_classifier"  # Model type: traffic_classifier | hybrid_traffic_classifier
  hidden_dim: 64              # Hidden layer dimension

client:
  batch_size: 32              # Local training batch size
  local_epochs: 5             # Local training epochs per round
  learning_rate: 0.01         # Local training learning rate
  momentum: 0.9               # SGD momentum coefficient
  compute_capacity: 1.0       # Computing capacity coefficient (placeholder, currently not affecting training)
  storage_capacity: 1000.0    # Storage capacity limit (MB, placeholder)

aggregation:
  min_updates: 2              # Minimum number of satellite updates required to trigger aggregation
  max_staleness: 300.0        # Maximum allowed model staleness (seconds)
  timeout: 600.0              # Aggregation wait timeout (seconds)
  weighted_average: true      # Weight aggregation by sample count

energy:
  config_file: "configs/energy_config.yaml"  # Path to energy model configuration file
```

### STELLAR-specific Parameters

```yaml
group:
  max_distance: 2             # Hop radius for similarity search (hops for Iridium; km for OneWeb)
  max_group_size: 5           # Maximum satellites per similarity group
  max_group_size_threshold: 4 # Triggers similarity threshold adjustment when exceeded
  similarity_threshold: 0.5   # Minimum composite similarity score to form a group
  similarity_refresh_rounds: 5 # Re-compute grouping every N rounds
  initial_group_size: 1       # Initial satellites per group (cold start)
  weights:
    alpha: 0.4                # Weight for model parameter cosine similarity
    beta: 0.3                 # Weight for loss curve similarity
    gamma: 0.3                # Weight for prediction distribution Hellinger distance
```

### FedProx-specific Parameters

```yaml
fedprox:
  mu: 0.01    # Proximal term coefficient, controls the penalty for local model diverging from global model
```

### Propagation-constrained Aggregation Parameters (Prop-FedAvg / Prop-FedProx / SDA-FL)

```yaml
propagation:
  hops: 2                    # Maximum propagation hops for model updates
  max_satellites: 648        # Maximum number of participating satellites in propagation range
  intra_orbit_links: true    # Allow ISL within the same orbit
  inter_orbit_links: true    # Allow ISL across different orbits
  link_reliability: 0.95     # ISL link reliability probability
  energy_per_hop: 0.05       # Communication energy cost per hop (Wh)
  # propagation_delay: 10    # Propagation delay per hop (ms) - Present in config but unread by code
```

### SDA-FL-specific Parameters

```yaml
sda_fl:
  noise_dim: 100             # GAN generator input noise vector dimension
  num_synthetic_samples: 1000  # Total synthetic samples generated per round
  gan_samples_per_client: 100  # Synthetic samples distributed per client
  gan_epochs: 50             # GAN training iterations per round
  initial_rounds: 3          # Warm-up rounds before GAN training starts (pure FL phase)
  regenerate_interval: 5     # Retrain GAN every N rounds
  pseudo_threshold: 0.8      # Pseudo-label confidence threshold
  dp_epsilon: 1.0            # Differential privacy ε parameter (smaller means stronger privacy)
  dp_delta: 1.0e-05          # Differential privacy δ parameter
```

### Execution Parameters

```yaml
execution:
  max_workers: 8             # Number of parallel threads (only read by SDA-FL)
  random_seed: 42            # Random seed (Present in config but unread by code)
  log_level: "INFO"          # Logging level (Present in config but unread by code)
```

### Robustness Testing Parameters

```yaml
robustness:
  parameter_noise_level: 0.1  # Parameter perturbation noise intensity (0.0 means disabled)
  noise_start_round: 5        # Round number to start injecting noise
```

### ⚠️ Parameters Present in Configs but Ignored by Current Code

| Parameter | Block | Description |
|---|---|---|
| `ground_station.*` | `ground_station` | Ground station bandwidth and storage limits are hardcoded in the code, config values are ignored |
| `early_stopping.*` | `early_stopping` | Early stopping logic is not implemented in current experiments |
| `fedavg.participation_rate` | `fedavg` | Full participation is used, this parameter has no effect |
| `execution.random_seed` | `execution` | Global random seed is not applied |
| `execution.log_level` | `execution` | Logging level is hardcoded to INFO |
| `propagation.propagation_delay` | `propagation` | Field exists in config but is unread by code |

## Algorithms

| Algorithm | Class | Config | Description |
|---|---|---|---|
| **STELLAR** | `SimilarityGroupingExperiment` | `similarity_grouping_config.yaml` | Proposed method: similarity-based grouping |
| FedAvg | `FedAvgExperiment` | `fedavg_config.yaml` | McMahan et al. (2017) |
| FedProx | `FedProxExperiment` | `fedprox_config.yaml` | Li et al. (2020) |
| Prop-FedAvg | `LimitedPropagationFedAvg` | `propagation_fedavg_config.yaml` | FedAvg with hop-limited propagation |
| Prop-FedProx | `LimitedPropagationFedProx` | `propagation_fedprox_config.yaml` | FedProx with hop-limited propagation |
| SDA-FL | `SDAFLExperiment` | `sda_fl_config.yaml` | GAN-based synthetic data augmentation FL |

## Tests

```bash
pytest tests/ -v
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{li2026stellar,
  author  = {Li, Yubo and Zhang, Li and Li, Kai and Su, Haoru},
  journal = {IEEE Transactions on Information Forensics and Security},
  title   = {STELLAR: Similarity-Based Satellite Federated Learning for Malicious Traffic Recognition},
  year    = {2026},
  volume  = {21},
  pages   = {1766--1780},
  doi     = {10.1109/TIFS.2026.3659044}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
