#!/usr/bin/env python3
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
    print("\n=== Processing Sample Images ===")
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
