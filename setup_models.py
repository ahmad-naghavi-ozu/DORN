#!/usr/bin/env python3
"""
DORN Model Setup and Download Script
"""

import os
import sys
import urllib.request
import subprocess
from pathlib import Path
import argparse


def check_system_dependencies():
    """Check if required system dependencies are installed"""
    print("Checking system dependencies...")
    
    dependencies = {
        'git': 'git --version',
        'cmake': 'cmake --version', 
        'make': 'make --version',
        'gcc': 'gcc --version',
        'g++': 'g++ --version'
    }
    
    missing = []
    for dep, cmd in dependencies.items():
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {dep} found")
            else:
                missing.append(dep)
        except FileNotFoundError:
            missing.append(dep)
            print(f"  ✗ {dep} not found")
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please install them using:")
        print("  Ubuntu/Debian: sudo apt-get install build-essential cmake git")
        print("  CentOS/RHEL: sudo yum groupinstall 'Development Tools' && sudo yum install cmake git")
        return False
    
    print("All system dependencies found!")
    return True


def install_caffe_dependencies():
    """Install Python dependencies for Caffe"""
    print("Installing Caffe Python dependencies...")
    
    caffe_deps = [
        'protobuf>=3.6.0',
        'leveldb>=0.191',  
        'python-gflags>=2.0',
        'scikit-image>=0.15.0',
        'python-dateutil>=1.4,<2'
    ]
    
    try:
        # Install using pip since some packages might not be in conda
        for dep in caffe_deps:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True)
        
        print("Caffe dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing Caffe dependencies: {e}")
        return False


def try_download_models():
    """Try to download pre-trained models from various sources"""
    print("Attempting to download pre-trained DORN models...")
    
    models_info = {
        'KITTI': {
            'deploy': 'models/KITTI/deploy.prototxt',
            'model': 'models/KITTI/cvpr_kitti.caffemodel',
            'urls': [
                # Add potential URLs here when found
                'https://github.com/hufu6371/DORN/releases/download/v1.0/cvpr_kitti.caffemodel',
                # Google Drive links would need special handling
            ]
        },
        'NYUV2': {
            'deploy': 'models/NYUV2/deploy.prototxt', 
            'model': 'models/NYUV2/cvpr_nyuv2.caffemodel',
            'urls': [
                'https://github.com/hufu6371/DORN/releases/download/v1.0/cvpr_nyuv2.caffemodel',
            ]
        }
    }
    
    downloaded = []
    
    for model_name, info in models_info.items():
        model_path = Path(info['model'])
        
        if model_path.exists():
            print(f"  ✓ {model_name} model already exists")
            downloaded.append(model_name)
            continue
            
        print(f"  Trying to download {model_name} model...")
        
        success = False
        for url in info['urls']:
            try:
                print(f"    Trying URL: {url}")
                urllib.request.urlretrieve(url, model_path)
                if model_path.exists() and model_path.stat().st_size > 1000:  # Basic check
                    print(f"    ✓ Downloaded {model_name} model")
                    downloaded.append(model_name)
                    success = True
                    break
                else:
                    model_path.unlink(missing_ok=True)
                    
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                model_path.unlink(missing_ok=True)
        
        if not success:
            print(f"    ✗ Could not download {model_name} model")
    
    return downloaded


def check_caffe_build():
    """Check if Caffe is built and working"""
    print("Checking Caffe build...")
    
    caffe_path = Path('caffe/python/caffe')
    if not caffe_path.exists():
        print("  ✗ Caffe Python interface not built")
        return False
    
    try:
        sys.path.insert(0, str(Path('caffe/python').absolute()))
        import caffe
        print("  ✓ Caffe Python interface available")
        return True
    except ImportError as e:
        print(f"  ✗ Caffe import failed: {e}")
        return False


def build_caffe():
    """Build Caffe from source"""
    print("Building Caffe from source...")
    
    caffe_dir = Path('caffe')
    if not caffe_dir.exists():
        print("  ✗ Caffe directory not found")
        return False
    
    # Check for Makefile.config
    makefile_config = caffe_dir / 'Makefile.config'
    if not makefile_config.exists():
        print("  Creating basic Makefile.config...")
        create_basic_makefile_config(makefile_config)
    
    try:
        os.chdir(caffe_dir)
        
        # Build Caffe
        print("  Running make...")
        subprocess.run(['make', 'all', '-j4'], check=True)
        
        print("  Building Python interface...")
        subprocess.run(['make', 'pycaffe', '-j4'], check=True)
        
        os.chdir('..')
        print("  ✓ Caffe built successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        os.chdir('..')
        print(f"  ✗ Caffe build failed: {e}")
        return False


def create_basic_makefile_config(config_path):
    """Create a basic Makefile.config for CPU-only build"""
    config_content = """# Basic CPU-only configuration for DORN
CPU_ONLY := 1

# Python configuration
PYTHON_INCLUDE := /usr/include/python3.8 \\
                 /usr/lib/python3.8/dist-packages/numpy/core/include

# We need to be able to find Python.h and numpy/arrayobject.h
PYTHON_LIB := /usr/lib

# Whatever else you find you need goes here
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial

# Build configuration
WITH_PYTHON_LAYER := 1
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)


def create_alternative_inference():
    """Create an alternative inference script that doesn't require Caffe"""
    print("Creating alternative inference method...")
    
    alt_script = """#!/usr/bin/env python3
'''
Alternative DORN inference using PyTorch/ONNX conversion
This is a fallback when Caffe is not available
'''

import torch
import torch.nn as nn
import numpy as np
import cv2

class SimpleDORN(nn.Module):
    '''Simplified DORN-like network for demonstration'''
    
    def __init__(self):
        super().__init__()
        # This is a very simplified version - not the actual DORN
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # This is just a placeholder - not actual DORN inference
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        # ... simplified processing
        depth = torch.rand(x.shape[0], 1, x.shape[2]//4, x.shape[3]//4)  # Mock output
        return depth * 80.0  # Scale to 80m max depth

def pytorch_dorn_inference(rgb_image):
    '''Mock DORN inference using PyTorch'''
    # This is just for demonstration - not real DORN
    print("Using PyTorch mock inference (not actual DORN)")
    
    # Simple depth estimation based on image processing
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    depth = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32)
    depth = (depth / 255.0) * 50.0  # Scale to 0-50m
    
    return depth

if __name__ == "__main__":
    print("Alternative DORN inference module created")
"""
    
    alt_path = Path('scripts/alternative_dorn.py')
    with open(alt_path, 'w') as f:
        f.write(alt_script)
    
    print(f"  ✓ Alternative inference created: {alt_path}")


def main():
    parser = argparse.ArgumentParser(description='DORN Setup Script')
    parser.add_argument('--check-deps', action='store_true', help='Check system dependencies')
    parser.add_argument('--install-deps', action='store_true', help='Install Python dependencies')
    parser.add_argument('--download-models', action='store_true', help='Download pre-trained models')
    parser.add_argument('--build-caffe', action='store_true', help='Build Caffe from source')
    parser.add_argument('--all', action='store_true', help='Run all setup steps')
    
    args = parser.parse_args()
    
    if args.all or args.check_deps:
        if not check_system_dependencies():
            print("Please install missing system dependencies first.")
            if not args.all:
                return
    
    if args.all or args.install_deps:
        install_caffe_dependencies()
    
    if args.all or args.download_models:
        downloaded = try_download_models()
        if not downloaded:
            print("\\nNo models downloaded. You may need to:")
            print("1. Check the original DORN repository for model links")
            print("2. Download models manually to models/KITTI/ and models/NYUV2/")
    
    if args.all or args.build_caffe:
        if not build_caffe():
            print("\\nCaffe build failed. You can still use the framework with mock inference.")
            create_alternative_inference()
    
    # Final check
    print("\\n=== Final Status ===")
    caffe_works = check_caffe_build()
    
    if not caffe_works:
        print("Caffe not available - using mock inference")
        create_alternative_inference()
    
    print("\\nSetup complete! You can now:")
    print("1. Test framework: python scripts/run_dorn.py --dataset_path /path/to/dataset --test_setup")
    print("2. Run inference: python scripts/run_dorn.py --dataset_path /path/to/dataset --split test --limit 5")


if __name__ == "__main__":
    main()
