#!/usr/bin/env python3
"""
DFC2023S Dataset Adapter for DORN
Specialized adapter for DFC2023S and similar urban planning datasets
"""

import cv2
import numpy as np
from pathlib import Path
from .base_dataset import BaseDatasetAdapter


class DFC2023SAdapter(BaseDatasetAdapter):
    """Adapter specifically for DFC2023S dataset and similar urban datasets"""
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path, "DFC2023S")
        
        # DFC2023S specific parameters
        self.target_size = (513, 385)  # DORN KITTI input size
        self.depth_scale = 1.0  # Adjust based on DSM units
        
    def preprocess_for_dorn(self, rgb_image, dsm_image=None):
        """
        Preprocess RGB image for DORN (KITTI model)
        """
        if rgb_image is None:
            return None
            
        # Convert to float32
        img = rgb_image.astype(np.float32)
        H, W = img.shape[:2]
        
        # Apply DORN's pixel means (BGR format)
        pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])
        img -= pixel_means
        
        # Resize to DORN input size while maintaining aspect ratio
        img_resized = cv2.resize(img, (W, 385), interpolation=cv2.INTER_LINEAR)
        
        return img_resized, H, W
    
    def preprocess_dsm(self, dsm_image):
        """
        Preprocess DSM for evaluation/comparison
        """
        if dsm_image is None:
            return None
            
        # Convert to meters if needed (depends on DSM format)
        # Assuming DSM is already in appropriate units
        dsm = dsm_image.astype(np.float32)
        
        # Handle potential invalid values
        dsm[dsm <= 0] = np.nan
        
        return dsm
    
    def postprocess_dorn_output(self, dorn_output, original_height, original_width):
        """
        Postprocess DORN output to match original image size
        """
        # DORN KITTI output processing (from demo_kitti.py)
        ord_score = dorn_output - 1.0
        ord_score = (ord_score + 40.0) / 25.0
        ord_score = np.exp(ord_score)
        
        # Resize back to original size
        depth_pred = cv2.resize(ord_score, (original_width, original_height), 
                               interpolation=cv2.INTER_LINEAR)
        
        # Convert to appropriate depth units (meters)
        depth_pred = depth_pred * self.depth_scale
        
        return depth_pred
    
    def calculate_metrics(self, pred_depth, gt_dsm, mask=None):
        """
        Calculate depth estimation metrics
        """
        if mask is None:
            mask = ~np.isnan(gt_dsm) & ~np.isnan(pred_depth) & (gt_dsm > 0)
        
        pred_valid = pred_depth[mask]
        gt_valid = gt_dsm[mask]
        
        if len(pred_valid) == 0:
            return {}
        
        # Common depth estimation metrics
        abs_diff = np.abs(pred_valid - gt_valid)
        rel_diff = abs_diff / gt_valid
        
        metrics = {
            'mae': np.mean(abs_diff),
            'rmse': np.sqrt(np.mean((pred_valid - gt_valid) ** 2)),
            'abs_rel': np.mean(rel_diff),
            'sq_rel': np.mean(((pred_valid - gt_valid) ** 2) / gt_valid),
        }
        
        # Threshold accuracies
        max_ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        for thresh in [1.25, 1.25**2, 1.25**3]:
            metrics[f'delta_{thresh:.3f}'] = np.mean(max_ratio < thresh)
        
        return metrics


def create_dfc2023s_config():
    """Create configuration for DFC2023S dataset"""
    config = {
        'dataset_name': 'DFC2023S',
        'input_size': (513, 385),
        'pixel_means': [103.0626, 115.9029, 123.1516],
        'depth_range': [0, 100],  # meters, adjust based on your data
        'model_type': 'KITTI',  # Use KITTI model for outdoor scenes
        'batch_size': 1,
        'preprocessing': {
            'resize_method': 'linear',
            'normalize': True,
            'subtract_mean': True
        },
        'evaluation': {
            'metrics': ['mae', 'rmse', 'abs_rel', 'sq_rel', 'delta_1.25'],
            'mask_invalid': True,
            'depth_cap': 80.0  # Cap depth at 80m for evaluation
        }
    }
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DFC2023S Dataset Adapter Test')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/asfand/Ahmad/datasets/DFC2023S',
                       help='Path to DFC2023S dataset')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'])
    
    args = parser.parse_args()
    
    # Test the adapter
    adapter = DFC2023SAdapter(args.dataset_path)
    
    # Print statistics
    stats = adapter.get_dataset_stats()
    print("DFC2023S Dataset Statistics:")
    for split, info in stats['splits'].items():
        print(f"  {split}: {info['num_pairs']} RGB-DSM pairs")
    
    # Test preprocessing on a sample
    pairs = adapter.get_image_pairs(args.split, limit=1)
    if pairs:
        print(f"\nTesting preprocessing on sample from {args.split}:")
        pair = pairs[0]
        rgb_img, dsm_img = adapter.load_image_pair(pair)
        
        if rgb_img is not None:
            print(f"Original RGB shape: {rgb_img.shape}")
            processed_rgb, H, W = adapter.preprocess_for_dorn(rgb_img)
            print(f"Processed RGB shape: {processed_rgb.shape}")
            print(f"Original dimensions: {H}x{W}")
            
        if dsm_img is not None:
            print(f"Original DSM shape: {dsm_img.shape}")
            processed_dsm = adapter.preprocess_dsm(dsm_img)
            print(f"DSM value range: {np.nanmin(processed_dsm):.2f} - {np.nanmax(processed_dsm):.2f}")
