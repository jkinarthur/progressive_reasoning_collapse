"""
Main Runner Script for Progressive Reasoning Collapse Experiments

This script orchestrates all experiments for validating the PRC framework
and evaluating the CARR (Collapse-Aware Register Recommendation) method.

Usage:
    python main.py --experiments all --datasets ml-1m amazon-beauty
    python main.py --experiments 1 2 3 --datasets ml-1m
    python main.py --experiment 4 --ablation-only
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    DEFAULT_DATASETS, ALL_EXPERIMENTS
)


def setup_environment():
    """Setup the experimental environment"""
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
        warnings.warn("Running on CPU may be very slow for LLM-based experiments")
    
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment_1(datasets: list, device: str):
    """Run Experiment 1: Progressive Collapse Validation"""
    from exp1_progressive_collapse import run_experiment_1 as exp1_runner
    print("\n" + "="*70)
    print("EXPERIMENT 1: Progressive Collapse Validation (Theorems 1 & 2)")
    print("="*70)
    return exp1_runner(datasets=datasets, device=device)


def run_experiment_2(datasets: list, device: str):
    """Run Experiment 2: Critical Depth Analysis"""
    from exp2_critical_depth import run_experiment_2 as exp2_runner
    print("\n" + "="*70)
    print("EXPERIMENT 2: Critical Depth Analysis (Theorem 3)")
    print("="*70)
    return exp2_runner(datasets=datasets, device=device)


def run_experiment_3(datasets: list, device: str):
    """Run Experiment 3: CARR Comparative Evaluation"""
    from exp3_carr_comparison import run_experiment_3 as exp3_runner
    print("\n" + "="*70)
    print("EXPERIMENT 3: CARR Comparative Evaluation")
    print("="*70)
    return exp3_runner(datasets=datasets, device=device)


def run_experiment_4(datasets: list, device: str):
    """Run Experiment 4: Ablation Studies"""
    from exp4_ablation import run_experiment_4 as exp4_runner
    print("\n" + "="*70)
    print("EXPERIMENT 4: Ablation Studies")
    print("="*70)
    return exp4_runner(datasets=datasets, device=device)


def run_experiment_5(datasets: list, device: str):
    """Run Experiment 5: Minority-Intent Preservation"""
    from exp5_minority_intent import run_experiment_5 as exp5_runner
    print("\n" + "="*70)
    print("EXPERIMENT 5: Minority-Intent Preservation Analysis")
    print("="*70)
    return exp5_runner(datasets=datasets, device=device)


def run_experiment_6(datasets: list, device: str):
    """Run Experiment 6: Visualization of Layerwise Collapse"""
    from exp6_visualization import run_experiment_6 as exp6_runner
    print("\n" + "="*70)
    print("EXPERIMENT 6: Visualization of Layerwise Collapse")
    print("="*70)
    return exp6_runner(datasets=datasets, device=device)


EXPERIMENT_RUNNERS = {
    1: run_experiment_1,
    2: run_experiment_2,
    3: run_experiment_3,
    4: run_experiment_4,
    5: run_experiment_5,
    6: run_experiment_6,
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Progressive Reasoning Collapse experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --experiments all
  python main.py --experiments 1 2 3 --datasets ml-1m amazon-beauty
  python main.py --experiment 4 --num-runs 5
  python main.py --experiment 6 --num-samples 500
        """
    )
    
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        default=['all'],
        help='Experiments to run: "all" or space-separated numbers (1-6)'
    )
    
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['ml-1m'],
        choices=['ml-1m', 'amazon-beauty', 'amazon-toys', 'yelp'],
        help='Datasets to use'
    )
    
    parser.add_argument(
        '--device',
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of runs for statistical significance (Exp 4)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=500,
        help='Number of samples for visualization (Exp 6)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for results'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick run with reduced samples for testing'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup
    print("="*70)
    print("Progressive Reasoning Collapse (PRC) Experiment Suite")
    print("CARR: Collapse-Aware Register Recommendation")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = args.device or setup_environment()
    
    # Determine experiments to run
    if 'all' in args.experiments:
        experiments_to_run = list(range(1, 7))
    else:
        experiments_to_run = [int(e) for e in args.experiments]
    
    print(f"\nExperiments to run: {experiments_to_run}")
    print(f"Datasets: {args.datasets}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    
    if args.quick:
        print("\n*** QUICK MODE: Using reduced samples ***")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Run experiments
    all_results = {}
    
    for exp_num in experiments_to_run:
        if exp_num not in EXPERIMENT_RUNNERS:
            print(f"Warning: Experiment {exp_num} not found, skipping")
            continue
        
        try:
            results = EXPERIMENT_RUNNERS[exp_num](args.datasets, device)
            all_results[f'experiment_{exp_num}'] = results
            print(f"\nExperiment {exp_num} completed successfully")
        except Exception as e:
            print(f"\nError in Experiment {exp_num}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[f'experiment_{exp_num}'] = {'error': str(e)}
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'combined_results_{timestamp}.json'
    
    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENT SUITE COMPLETED")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_file}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    
    return all_results


if __name__ == "__main__":
    main()
