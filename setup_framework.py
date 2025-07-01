#!/usr/bin/env python3
"""
Setup script for DORN dataset processing
"""

import os
import sys
from pathlib import Path
import argparse

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from configs.dataset_configs import create_default_configs


def setup_directory_structure():
    """Create the organized directory structure"""
    base_dirs = [
        'dataset_adapters',
        'scripts', 
        'configs',
        'results',
        'models/KITTI',
        'models/NYUV2'
    ]
    
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")


def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        'dataset_adapters/__init__.py',
        'configs/__init__.py',
        'scripts/__init__.py'
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.touch()
    
    print("Python package files created!")


def create_example_usage():
    """Create example usage scripts"""
    
    example_script = """#!/usr/bin/env python3
'''
Example usage of DORN with DFC2023S dataset
'''

import sys
from pathlib import Path

# Add DORN root to path
dorn_root = Path(__file__).parent.parent
sys.path.append(str(dorn_root))

from scripts.run_dorn import DORNRunner

def main():
    # Example 1: Test dataset setup
    print("=== Testing Dataset Setup ===")
    runner = DORNRunner()
    
    dataset_path = "/home/asfand/Ahmad/datasets/DFC2023S"
    runner.setup_dataset(dataset_path, 'dfc2023s')
    
    # Get dataset statistics
    stats = runner.adapter.get_dataset_stats()
    print(f"Dataset: {stats['dataset_name']}")
    for split, info in stats['splits'].items():
        print(f"  {split}: {info['num_pairs']} pairs")
    
    # Example 2: Process a few samples
    print("\\n=== Processing Sample Images ===")
    results = runner.process_dataset(
        dataset_path, 
        split='test', 
        limit=3,  # Process only 3 images for testing
        output_dir='./test_results'
    )
    
    print(f"Processed {len(results)} images")
    print("Results saved to ./test_results/")

if __name__ == "__main__":
    main()
"""
    
    example_path = Path('examples/run_dfc2023s_example.py')
    example_path.parent.mkdir(exist_ok=True)
    
    with open(example_path, 'w') as f:
        f.write(example_script)
    
    print(f"Example script created: {example_path}")


def create_readme():
    """Create README for the organized structure"""
    
    readme_content = """# DORN Dataset Processing Framework

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
"""
    
    with open('README_FRAMEWORK.md', 'w') as f:
        f.write(readme_content)
    
    print("README created: README_FRAMEWORK.md")


def main():
    parser = argparse.ArgumentParser(description='Setup DORN processing framework')
    parser.add_argument('--all', action='store_true', help='Run all setup steps')
    parser.add_argument('--dirs', action='store_true', help='Create directory structure')
    parser.add_argument('--configs', action='store_true', help='Create configuration files')
    parser.add_argument('--examples', action='store_true', help='Create example scripts')
    parser.add_argument('--readme', action='store_true', help='Create README')
    
    args = parser.parse_args()
    
    if args.all or args.dirs:
        setup_directory_structure()
        create_init_files()
    
    if args.all or args.configs:
        create_default_configs('./configs')
    
    if args.all or args.examples:
        create_example_usage()
    
    if args.all or args.readme:
        create_readme()
    
    if not any(vars(args).values()):
        print("No setup options specified. Use --help for options or --all for complete setup.")
        return
    
    print("\n=== Setup Complete! ===")
    print("You can now:")
    print("1. Test the framework: python scripts/run_dorn.py --dataset_path /path/to/dataset --test_setup")
    print("2. Run example: python examples/run_dfc2023s_example.py")
    print("3. Check README_FRAMEWORK.md for detailed instructions")


if __name__ == "__main__":
    main()
