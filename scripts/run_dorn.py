#!/usr/bin/env python3
"""
Universal DORN Runner for RGB-DSM Datasets
Supports multiple datasets with RGB-DSM structure
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
import yaml
import json

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

try:
    from dataset_adapters.base_dataset import GenericDatasetAdapter
    from dataset_adapters.dfc2023s_adapter import DFC2023SAdapter
    from configs.dataset_configs import load_config, get_dfc2023s_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the DORN root directory")
    sys.exit(1)


class DORNRunner:
    """Universal DORN runner for different datasets"""
    
    def __init__(self, config_path=None, config_dict=None):
        if config_path:
            self.config = load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            # Use default DFC2023S config
            self.config = get_dfc2023s_config()
        
        self.caffe_available = self._check_caffe()
        self.adapter = None
        
    def _check_caffe(self):
        """Check if Caffe is available"""
        try:
            import caffe
            return True
        except ImportError:
            print("Warning: Caffe not available. Using mock inference for testing.")
            return False
    
    def setup_dataset(self, dataset_path, dataset_type='auto'):
        """Setup dataset adapter"""
        if dataset_type == 'dfc2023s' or (dataset_type == 'auto' and 'DFC2023S' in str(dataset_path)):
            self.adapter = DFC2023SAdapter(dataset_path)
        else:
            # Use generic adapter
            dataset_name = Path(dataset_path).name
            self.adapter = GenericDatasetAdapter(dataset_path, dataset_name)
        
        print(f"Dataset adapter setup: {self.adapter.dataset_name}")
        return self.adapter
    
    def setup_caffe_model(self):
        """Setup Caffe model if available"""
        if not self.caffe_available:
            return None
            
        try:
            import caffe
            
            model_config = self.config['model']
            deploy_path = model_config['deploy_path']
            model_path = model_config['model_path']
            
            # Check if model files exist
            if not os.path.exists(deploy_path):
                print(f"Deploy file not found: {deploy_path}")
                return None
                
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                print("You need to download the pre-trained model.")
                return None
            
            # Setup Caffe
            caffe.set_mode_gpu()
            caffe.set_device(0)
            
            net = caffe.Net(deploy_path, model_path, caffe.TEST)
            return net
            
        except Exception as e:
            print(f"Error setting up Caffe model: {e}")
            return None
    
    def mock_dorn_inference(self, rgb_image):
        """
        Mock DORN inference for testing when Caffe is not available
        Generates a simple depth map based on image gradients
        """
        if rgb_image is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Simple depth estimation based on gradients and intensity
        # This is just for testing - not actual depth estimation
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Combine intensity and gradients for mock depth
        depth = gray.astype(np.float32) / 255.0 * 50.0  # 0-50m range
        depth += (gradient_mag / gradient_mag.max()) * 10.0  # Add gradient contribution
        
        # Add some noise for realism
        noise = np.random.normal(0, 2, depth.shape)
        depth += noise
        depth = np.clip(depth, 0, 80)
        
        return depth
    
    def run_dorn_inference(self, rgb_image, net=None):
        """Run DORN inference on RGB image"""
        if net is None:
            # Use mock inference
            return self.mock_dorn_inference(rgb_image)
            
        try:
            # Real DORN inference (based on demo_kitti.py)
            img = rgb_image.astype(np.float32)
            H, W = img.shape[:2]
            
            pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])
            img -= pixel_means
            img = cv2.resize(img, (W, 385), interpolation=cv2.INTER_LINEAR)
            
            ord_score = np.zeros((385, W), dtype=np.float32)
            counts = np.zeros((385, W), dtype=np.float32)
            
            # Sliding window inference
            for i in range(4):
                h0 = 0
                h1 = 385
                w0 = int(0 + i*256)
                w1 = w0 + 513
                if w1 > W:
                    w0 = W - 513
                    w1 = W
                
                data = img[h0:h1, w0:w1, :]
                data = data[None, :]
                data = data.transpose(0,3,1,2)
                
                # Forward pass
                net.blobs['data'].reshape(*(data.shape))
                net.blobs['data'].data[...] = data.astype(np.float32, copy=False)
                net.forward()
                
                pred = net.blobs['decode_ord'].data.copy()
                pred = pred[0,0,:,:]
                ord_score[h0:h1,w0:w1] += pred
                counts[h0:h1,w0:w1] += 1.0
            
            ord_score = ord_score/counts - 1.0
            ord_score = (ord_score + 40.0)/25.0
            ord_score = np.exp(ord_score)
            ord_score = cv2.resize(ord_score, (W, H), interpolation=cv2.INTER_LINEAR)
            
            return ord_score
            
        except Exception as e:
            print(f"Error in DORN inference: {e}")
            return self.mock_dorn_inference(rgb_image)
    
    def process_dataset(self, dataset_path, split='test', limit=None, output_dir='./results'):
        """Process entire dataset split"""
        # Setup dataset
        self.setup_dataset(dataset_path)
        
        # Setup model
        net = self.setup_caffe_model()
        if net is None:
            print("Using mock inference (Caffe model not available)")
        
        # Get image pairs
        pairs = self.adapter.get_image_pairs(split, limit)
        print(f"Processing {len(pairs)} image pairs from {split} split")
        
        results = []
        
        for i, pair in enumerate(pairs):
            print(f"Processing {i+1}/{len(pairs)}: {pair['id']}")
            
            start_time = time.time()
            
            # Load images
            rgb_img, dsm_img = self.adapter.load_image_pair(pair)
            
            if rgb_img is None:
                print(f"  Skipping: Could not load RGB image")
                continue
            
            # Run inference
            depth_pred = self.run_dorn_inference(rgb_img, net)
            
            if depth_pred is None:
                print(f"  Skipping: Inference failed")
                continue
            
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                'id': pair['id'],
                'rgb_path': pair['rgb'],
                'dsm_path': pair['dsm'],
                'depth_pred': (depth_pred * 256.0).astype(np.uint16),  # For 16-bit PNG
                'processing_time': processing_time
            }
            
            # Calculate metrics if DSM is available
            if dsm_img is not None and hasattr(self.adapter, 'calculate_metrics'):
                dsm_processed = self.adapter.preprocess_dsm(dsm_img)
                if dsm_processed is not None:
                    depth_resized = cv2.resize(depth_pred, (dsm_processed.shape[1], dsm_processed.shape[0]))
                    metrics = self.adapter.calculate_metrics(depth_resized, dsm_processed)
                    result['metrics'] = metrics
            
            results.append(result)
            print(f"  Completed in {processing_time:.2f}s")
        
        # Save results
        self.adapter.save_results(results, output_dir, split)
        
        # Save summary
        self._save_summary(results, output_dir, split)
        
        return results
    
    def _save_summary(self, results, output_dir, split):
        """Save processing summary"""
        output_path = Path(output_dir) / self.adapter.dataset_name / split
        summary_file = output_path / 'processing_summary.json'
        
        summary = {
            'dataset': self.adapter.dataset_name,
            'split': split,
            'total_processed': len(results),
            'config': self.config,
            'results': []
        }
        
        # Aggregate metrics
        all_metrics = {}
        for result in results:
            summary['results'].append({
                'id': result['id'],
                'processing_time': float(result['processing_time']),
                'metrics': {k: float(v) for k, v in result.get('metrics', {}).items()}
            })
            
            # Collect metrics for averaging
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(float(value))
        
        # Calculate average metrics
        if all_metrics:
            avg_metrics = {metric: float(np.mean(values)) for metric, values in all_metrics.items()}
            summary['average_metrics'] = avg_metrics
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Universal DORN Runner')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--dataset_type', type=str, default='auto',
                       choices=['auto', 'dfc2023s', 'generic'],
                       help='Type of dataset adapter to use')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--limit', type=int,
                       help='Limit number of images to process')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--test_setup', action='store_true',
                       help='Just test the setup without running inference')
    
    args = parser.parse_args()
    
    # Create runner
    runner = DORNRunner(config_path=args.config)
    
    if args.test_setup:
        # Test setup
        print("Testing setup...")
        runner.setup_dataset(args.dataset_path, args.dataset_type)
        stats = runner.adapter.get_dataset_stats()
        print("Dataset statistics:")
        for split, info in stats['splits'].items():
            print(f"  {split}: {info['num_pairs']} pairs")
        return
    
    # Run processing
    results = runner.process_dataset(
        args.dataset_path, 
        split=args.split, 
        limit=args.limit,
        output_dir=args.output_dir
    )
    
    print(f"\nCompleted processing {len(results)} images")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
