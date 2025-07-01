#!/usr/bin/env python3
"""
Configuration templates for different datasets
"""

import yaml
from pathlib import Path


def get_dfc2023s_config():
    """Configuration for DFC2023S dataset"""
    return {
        'dataset': {
            'name': 'DFC2023S',
            'type': 'urban_planning',
            'modalities': ['rgb', 'dsm'],
            'splits': ['train', 'valid', 'test']
        },
        'preprocessing': {
            'target_size': [513, 385],  # Width, Height for DORN KITTI
            'pixel_means': [103.0626, 115.9029, 123.1516],  # BGR
            'resize_method': 'linear',
            'crop_method': 'sliding_window',
            'crop_size': [513, 385],
            'crop_stride': 256
        },
        'model': {
            'type': 'DORN_KITTI',
            'architecture': 'ResNet101',
            'pretrained': True,
            'model_path': 'models/KITTI/cvpr_kitti.caffemodel',
            'deploy_path': 'models/KITTI/deploy.prototxt'
        },
        'inference': {
            'batch_size': 1,
            'overlap_handling': 'average',
            'output_format': 'png',
            'depth_scale': 256.0,  # For 16-bit PNG output
            'depth_range': [0, 80]  # meters
        },
        'evaluation': {
            'metrics': ['mae', 'rmse', 'abs_rel', 'sq_rel', 'delta_1.25', 'delta_1.25^2', 'delta_1.25^3'],
            'depth_cap': 80.0,
            'min_depth': 0.1,
            'mask_invalid': True
        }
    }


def get_generic_outdoor_config():
    """Generic configuration for outdoor RGB-DSM datasets"""
    return {
        'dataset': {
            'name': 'Generic_Outdoor',
            'type': 'outdoor_depth',
            'modalities': ['rgb', 'dsm'],
            'splits': ['train', 'valid', 'test']
        },
        'preprocessing': {
            'target_size': [513, 385],
            'pixel_means': [103.0626, 115.9029, 123.1516],
            'resize_method': 'linear',
            'crop_method': 'center',
            'crop_size': [513, 385]
        },
        'model': {
            'type': 'DORN_KITTI',
            'architecture': 'ResNet101',
            'pretrained': True,
            'model_path': 'models/KITTI/cvpr_kitti.caffemodel',
            'deploy_path': 'models/KITTI/deploy.prototxt'
        },
        'inference': {
            'batch_size': 1,
            'output_format': 'png',
            'depth_scale': 256.0,
            'depth_range': [0, 100]
        },
        'evaluation': {
            'metrics': ['mae', 'rmse', 'abs_rel', 'sq_rel'],
            'depth_cap': 100.0,
            'min_depth': 0.1
        }
    }


def get_indoor_config():
    """Configuration for indoor RGB-D datasets (using NYU model)"""
    return {
        'dataset': {
            'name': 'Generic_Indoor',
            'type': 'indoor_depth',
            'modalities': ['rgb', 'depth'],
            'splits': ['train', 'valid', 'test']
        },
        'preprocessing': {
            'target_size': [353, 257],  # NYU input size
            'pixel_means': [103.0626, 115.9029, 123.1516],
            'resize_method': 'linear'
        },
        'model': {
            'type': 'DORN_NYUV2',
            'architecture': 'VGG16',
            'pretrained': True,
            'model_path': 'models/NYUV2/cvpr_nyuv2.caffemodel',
            'deploy_path': 'models/NYUV2/deploy.prototxt'
        },
        'inference': {
            'batch_size': 1,
            'output_format': 'png',
            'depth_scale': 255.0,
            'depth_range': [0, 10]  # meters
        },
        'evaluation': {
            'metrics': ['mae', 'rmse', 'abs_rel', 'sq_rel'],
            'depth_cap': 10.0,
            'min_depth': 0.1
        }
    }


def save_config(config, config_path):
    """Save configuration to YAML file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_configs(configs_dir):
    """Create default configuration files"""
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save default configurations
    save_config(get_dfc2023s_config(), configs_dir / 'dfc2023s.yaml')
    save_config(get_generic_outdoor_config(), configs_dir / 'generic_outdoor.yaml')
    save_config(get_indoor_config(), configs_dir / 'indoor.yaml')
    
    print(f"Default configurations saved to {configs_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Manager')
    parser.add_argument('--create_defaults', action='store_true',
                       help='Create default configuration files')
    parser.add_argument('--configs_dir', type=str, default='./configs',
                       help='Directory to save configurations')
    
    args = parser.parse_args()
    
    if args.create_defaults:
        create_default_configs(args.configs_dir)
