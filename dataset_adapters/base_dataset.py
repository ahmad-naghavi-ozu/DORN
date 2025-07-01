#!/usr/bin/env python3
"""
Base Dataset Adapter for DORN
Handles RGB-DSM paired datasets with train/valid/test structure
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
import argparse
from abc import ABC, abstractmethod


class BaseDatasetAdapter(ABC):
    """Base class for dataset adapters that work with RGB-DSM pairs"""
    
    def __init__(self, dataset_path, dataset_name):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.splits = ['train', 'valid', 'test']
        self.modalities = ['rgb', 'dsm']
        
        # Validate dataset structure
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate that the dataset follows the expected structure"""
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        
        for split in self.splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                print(f"Warning: Split '{split}' not found in dataset")
                continue
                
            for modality in self.modalities:
                modality_path = split_path / modality
                if not modality_path.exists():
                    print(f"Warning: Modality '{modality}' not found in split '{split}'")
    
    def get_image_pairs(self, split='train', limit=None):
        """Get RGB-DSM image pairs for a given split"""
        rgb_path = self.dataset_path / split / 'rgb'
        dsm_path = self.dataset_path / split / 'dsm'
        
        if not rgb_path.exists() or not dsm_path.exists():
            print(f"Warning: RGB or DSM path not found for split '{split}'")
            return []
        
        # Get all RGB images
        rgb_files = sorted(glob.glob(str(rgb_path / "*.png")) + 
                          glob.glob(str(rgb_path / "*.jpg")) + 
                          glob.glob(str(rgb_path / "*.tif")) +
                          glob.glob(str(rgb_path / "*.tiff")))
        
        pairs = []
        for rgb_file in rgb_files:
            rgb_name = Path(rgb_file).stem
            
            # Look for corresponding DSM file
            dsm_candidates = [
                dsm_path / f"{rgb_name}.png",
                dsm_path / f"{rgb_name}.jpg", 
                dsm_path / f"{rgb_name}.tif",
                dsm_path / f"{rgb_name}.tiff"
            ]
            
            dsm_file = None
            for candidate in dsm_candidates:
                if candidate.exists():
                    dsm_file = str(candidate)
                    break
            
            if dsm_file:
                pairs.append({
                    'rgb': rgb_file,
                    'dsm': dsm_file,
                    'id': rgb_name,
                    'split': split
                })
        
        if limit:
            pairs = pairs[:limit]
            
        return pairs
    
    def get_dataset_stats(self):
        """Get statistics about the dataset"""
        stats = {
            'dataset_name': self.dataset_name,
            'dataset_path': str(self.dataset_path),
            'splits': {}
        }
        
        for split in self.splits:
            pairs = self.get_image_pairs(split)
            stats['splits'][split] = {
                'num_pairs': len(pairs),
                'pairs_found': len(pairs) > 0
            }
        
        return stats
    
    @abstractmethod
    def preprocess_for_dorn(self, rgb_image, dsm_image=None):
        """Preprocess images for DORN inference"""
        pass
    
    def load_image_pair(self, pair_info):
        """Load RGB and DSM images from pair info"""
        rgb_img = cv2.imread(pair_info['rgb'])
        dsm_img = cv2.imread(pair_info['dsm'], cv2.IMREAD_UNCHANGED) if pair_info['dsm'] else None
        
        return rgb_img, dsm_img
    
    def save_results(self, results, output_dir, split='test'):
        """Save DORN results"""
        output_path = Path(output_dir) / self.dataset_name / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            output_file = output_path / f"{result['id']}_dorn_depth.png"
            cv2.imwrite(str(output_file), result['depth_pred'])
            
            # Also save metadata
            metadata_file = output_path / f"{result['id']}_metadata.txt"
            with open(metadata_file, 'w') as f:
                f.write(f"Image ID: {result['id']}\n")
                f.write(f"Original RGB: {result.get('rgb_path', 'N/A')}\n")
                f.write(f"Original DSM: {result.get('dsm_path', 'N/A')}\n")
                f.write(f"Processing time: {result.get('processing_time', 'N/A')}\n")


class GenericDatasetAdapter(BaseDatasetAdapter):
    """Generic adapter for any RGB-DSM dataset following the standard structure"""
    
    def __init__(self, dataset_path, dataset_name=None):
        if dataset_name is None:
            dataset_name = Path(dataset_path).name
        super().__init__(dataset_path, dataset_name)
    
    def preprocess_for_dorn(self, rgb_image, dsm_image=None):
        """
        Preprocess RGB image for DORN
        Based on the original DORN preprocessing
        """
        if rgb_image is None:
            return None
            
        # Convert to float32
        img = rgb_image.astype(np.float32)
        
        # Apply DORN's pixel means (BGR format)
        pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])
        img -= pixel_means
        
        return img


def main():
    parser = argparse.ArgumentParser(description='Dataset Adapter for DORN')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, 
                       help='Name of the dataset (defaults to directory name)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to analyze')
    parser.add_argument('--limit', type=int,
                       help='Limit number of pairs to process')
    
    args = parser.parse_args()
    
    # Create adapter
    adapter = GenericDatasetAdapter(args.dataset_path, args.dataset_name)
    
    # Print dataset statistics
    stats = adapter.get_dataset_stats()
    print("Dataset Statistics:")
    print(f"Name: {stats['dataset_name']}")
    print(f"Path: {stats['dataset_path']}")
    print("\nSplits:")
    for split, info in stats['splits'].items():
        print(f"  {split}: {info['num_pairs']} pairs")
    
    # Get pairs for specified split
    pairs = adapter.get_image_pairs(args.split, args.limit)
    print(f"\nFound {len(pairs)} pairs in '{args.split}' split")
    
    if pairs:
        print("\nFirst few pairs:")
        for i, pair in enumerate(pairs[:3]):
            print(f"  {i+1}. ID: {pair['id']}")
            print(f"     RGB: {pair['rgb']}")
            print(f"     DSM: {pair['dsm']}")


if __name__ == "__main__":
    main()
