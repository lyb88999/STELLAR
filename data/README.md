# Data Directory

Place your dataset files here before running experiments.

## Required files

### Option 1: Custom traffic dataset (default)

Provide a CSV file with numerical network flow features and a `Label` column:

```
data/traffic_data.csv
```

The `Label` column should contain binary labels (e.g., `0` = benign, `1` = malicious), or
multi-class string labels that will be binarized automatically.

### Option 2: CICIDS-2017

Download the CICIDS-2017 dataset from the University of New Brunswick:
https://www.unb.ca/cic/datasets/ids-2017.html

Extract the `MachineLearningCVE` folder (containing 8 CSV files) into this directory:

```
data/MachineLearningCVE/
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

Then use `configs/cicids2017_config.yaml` as your configuration file.

## Updating the config

After placing your data, update `csv_path` in the relevant config file:

```yaml
data:
  csv_path: "data/your_filename.csv"
```
