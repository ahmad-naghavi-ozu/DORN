# DORN Dataset Processing Framework

This is an organized framework for running DORN on various RGB-DSM datasets.

## Directory Structure

```
DORN/
├── dataset_adapters/          # Dataset-specific adapters
│   ├── base_dataset.py       # Base adapter class
│   ├── dfc2023s_adapter.py   # DFC2023S specific adapter
│   └── ...                   # Other dataset adapters
├── scripts/                  # Main processing scripts
│   └── run_dorn.py          # Universal DORN runner
├── configs/                  # Configuration files
│   ├── dataset_configs.py   # Configuration templates
│   ├── dfc2023s.yaml        # DFC2023S config
│   └── ...                  # Other configs
├── results/                  # Output results
├── examples/                 # Example usage scripts
└── models/                   # Pre-trained models
    ├── KITTI/
    └── NYUV2/
```

## Quick Start

1. **Test dataset setup:**
   ```bash
   python scripts/run_dorn.py --dataset_path /path/to/dataset --test_setup
   ```

2. **Process a few samples:**
   ```bash
   python scripts/run_dorn.py --dataset_path /path/to/dataset --split test --limit 5
   ```

3. **Full processing:**
   ```bash
   python scripts/run_dorn.py --dataset_path /path/to/dataset --split test
   ```

## Supported Dataset Structure

```
dataset/
├── train/
│   ├── rgb/
│   └── dsm/
├── valid/
│   ├── rgb/
│   └── dsm/
└── test/
    ├── rgb/
    └── dsm/
```

## Configuration

Create custom configurations by copying and modifying templates in `configs/`.

## Adding New Datasets

1. Create a new adapter in `dataset_adapters/` inheriting from `BaseDatasetAdapter`
2. Add dataset-specific configuration in `configs/`
3. The framework will automatically handle the rest!

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Caffe (optional, will use mock inference if not available)

## Notes

- If Caffe is not available, the system will use mock inference for testing
- Results are saved in both PNG format and JSON summary
- Metrics are calculated automatically if ground truth DSM is available
