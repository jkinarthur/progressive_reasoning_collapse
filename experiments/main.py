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


def run_experiment_1b(datasets: list, device: str):
    """Run Experiment 1b: Intent Validation + Jacobian Spectral Norm (R1, R2)"""
    from exp1b_intent_validation import run_experiment_1b as exp1b_runner
    print("\n" + "="*70)
    print("EXPERIMENT 1b: Latent Intent Validation + Jacobian Spectral Norm (R1+R2)")
    print("="*70)
    return exp1b_runner(datasets=datasets, device=device)


def run_experiment_7(datasets: list, device: str):
    """Run Experiment 7: T5-Large LLM Experiments (R3)"""
    from exp7_t5_large import run_experiment_7 as exp7_runner
    print("\n" + "="*70)
    print("EXPERIMENT 7: T5-Large LLM Experiments (R3)")
    print("="*70)
    return exp7_runner(datasets=datasets, device=device)


def run_experiment_8(datasets: list, device: str):
    """Run Experiment 8: Robustness Analysis (R9)"""
    from exp8_robustness import run_experiment_8 as exp8_runner
    print("\n" + "="*70)
    print("EXPERIMENT 8: Robustness Analysis (R9)")
    print("="*70)
    return exp8_runner(datasets=datasets, device=device)


def run_experiment_9(datasets: list, device: str):
    """Run Experiment 9: Failure Mode Analysis (R10)"""
    from exp9_failure_modes import run_experiment_9 as exp9_runner
    print("\n" + "="*70)
    print("EXPERIMENT 9: Failure Mode Analysis (R10)")
    print("="*70)
    return exp9_runner(datasets=datasets, device=device)


EXPERIMENT_RUNNERS = {
    1:    run_experiment_1,
    "1b": run_experiment_1b,
    2:    run_experiment_2,
    3:    run_experiment_3,
    4:    run_experiment_4,
    5:    run_experiment_5,
    6:    run_experiment_6,
    7:    run_experiment_7,
    8:    run_experiment_8,
    9:    run_experiment_9,
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
        help=(
            'Experiments to run: "all" or space-separated IDs. '
            'Original: 1 2 3 4 5 6. '
            'New (reviewer): 1b 7 8 9'
        )
    )

    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=['ml-1m'],
        choices=['ml-1m', 'amazon-beauty', 'amazon-toys', 'yelp', 'mind'],
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
        experiments_to_run = list(ALL_EXPERIMENTS)
    else:
        # Parse IDs: '1b' stays as string, numeric IDs become ints
        experiments_to_run = [
            e if e == '1b' else int(e)
            for e in args.experiments
        ]

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
            print(f"Warning: Experiment {exp_num!r} not found, skipping")
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
    
    # Convert numpy arrays/torch tensors/DataFrames for JSON serialization
    def convert_for_json(obj, _seen=None):
        import numpy as np
        import torch
        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            if isinstance(obj, pd.Series):
                return obj.tolist()
        except ImportError:
            pass
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen:
            return None  # break circular reference
        if isinstance(obj, (dict, list)):
            _seen.add(obj_id)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v, _seen) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v, _seen) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)  # fallback: stringify unknown types
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2, default=str)
    
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
