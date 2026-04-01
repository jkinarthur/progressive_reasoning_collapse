"""
Data loading and preprocessing for PRC Experiments
Supports: MovieLens-1M, Amazon Beauty, Amazon Toys, Yelp

Handles:
- Data downloading and extraction
- Sequential interaction preprocessing
- Text prompt generation for LLM-based recommendation
- Informative subsequence (τ) extraction strategies
"""

import os
import json
import pickle
import random
import gzip
import requests
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import DatasetConfig, DATA_DIR


# =============================================================================
# Dataset Download URLs
# =============================================================================
DATASET_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "amazon-beauty": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
    "amazon-toys": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
}


# =============================================================================
# Data Download and Extraction
# =============================================================================
def download_dataset(dataset_name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download dataset if not exists"""
    
    dataset_dir = data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name not in DATASET_URLS:
        print(f"Dataset {dataset_name} URL not available. Please download manually.")
        return dataset_dir
    
    url = DATASET_URLS[dataset_name]
    
    if dataset_name == "ml-1m":
        zip_path = dataset_dir / "ml-1m.zip"
        if not (dataset_dir / "ratings.dat").exists():
            print(f"Downloading {dataset_name}...")
            response = requests.get(url, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            # Move files up one level
            for f in (dataset_dir / "ml-1m").iterdir():
                f.rename(dataset_dir / f.name)
                
    elif dataset_name.startswith("amazon"):
        gz_path = dataset_dir / f"{dataset_name}.json.gz"
        if not (dataset_dir / "interactions.json").exists():
            print(f"Downloading {dataset_name}...")
            response = requests.get(url, stream=True)
            with open(gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # Extract
            with gzip.open(gz_path, 'rt', encoding='utf-8') as f_in:
                data = [json.loads(line) for line in f_in]
            with open(dataset_dir / "interactions.json", 'w') as f_out:
                json.dump(data, f_out)
    
    return dataset_dir


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_movielens(data_path: Path, min_interactions: int = 5) -> Tuple[Dict, Dict, Dict]:
    """Load MovieLens-1M dataset"""
    
    ratings_file = data_path / "ratings.dat"
    movies_file = data_path / "movies.dat"
    
    # Load ratings
    ratings = []
    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line in f:
            user, item, rating, timestamp = line.strip().split('::')
            ratings.append({
                'user_id': int(user),
                'item_id': int(item),
                'rating': float(rating),
                'timestamp': int(timestamp)
            })
    
    # Load movie metadata
    item_meta = {}
    with open(movies_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                item_id = int(parts[0])
                title = parts[1]
                genres = parts[2].split('|')
                item_meta[item_id] = {'title': title, 'genres': genres}
    
    # Build user interaction sequences
    user_sequences = defaultdict(list)
    for r in ratings:
        user_sequences[r['user_id']].append((r['item_id'], r['timestamp'], r['rating']))
    
    # Sort by timestamp and filter
    filtered_sequences = {}
    for user_id, interactions in user_sequences.items():
        interactions.sort(key=lambda x: x[1])
        if len(interactions) >= min_interactions:
            filtered_sequences[user_id] = [item_id for item_id, _, _ in interactions]
    
    # Create item ID mapping
    all_items = set()
    for seq in filtered_sequences.values():
        all_items.update(seq)
    item2idx = {item: idx for idx, item in enumerate(sorted(all_items), start=1)}
    item2idx[0] = 0  # Padding
    
    # Remap sequences
    for user_id in filtered_sequences:
        filtered_sequences[user_id] = [item2idx[item] for item in filtered_sequences[user_id]]
    
    return filtered_sequences, item2idx, item_meta


def load_amazon(data_path: Path, min_interactions: int = 5) -> Tuple[Dict, Dict, Dict]:
    """Load Amazon dataset (Beauty or Toys)"""
    
    interactions_file = data_path / "interactions.json"
    
    with open(interactions_file, 'r') as f:
        data = json.load(f)
    
    # Build user sequences
    user_sequences = defaultdict(list)
    item_meta = {}
    
    for review in data:
        user_id = review.get('reviewerID', review.get('user_id'))
        item_id = review.get('asin', review.get('item_id'))
        timestamp = review.get('unixReviewTime', review.get('timestamp', 0))
        
        user_sequences[user_id].append((item_id, timestamp))
        
        if item_id not in item_meta:
            item_meta[item_id] = {
                'title': review.get('summary', ''),
                'category': review.get('category', [])
            }
    
    # Sort and filter
    filtered_sequences = {}
    for user_id, interactions in user_sequences.items():
        interactions.sort(key=lambda x: x[1])
        if len(interactions) >= min_interactions:
            filtered_sequences[user_id] = [item_id for item_id, _ in interactions]
    
    # Create item ID mapping
    all_items = set()
    for seq in filtered_sequences.values():
        all_items.update(seq)
    item2idx = {item: idx for idx, item in enumerate(sorted(all_items), start=1)}
    item2idx['<PAD>'] = 0
    
    # Remap
    for user_id in filtered_sequences:
        filtered_sequences[user_id] = [item2idx.get(item, 0) for item in filtered_sequences[user_id]]
    
    return filtered_sequences, item2idx, item_meta


def load_dataset(config: DatasetConfig) -> Tuple[Dict, Dict, Dict]:
    """Load dataset based on configuration"""
    
    data_path = Path(config.path)
    
    if "ml" in config.name.lower() or "movielens" in config.name.lower():
        return load_movielens(data_path, config.min_interactions)
    elif "amazon" in config.name.lower():
        return load_amazon(data_path, config.min_interactions)
    else:
        raise ValueError(f"Unknown dataset: {config.name}")


# =============================================================================
# Train/Val/Test Split
# =============================================================================
def split_sequences(
    sequences: Dict[int, List[int]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[Dict, Dict, Dict]:
    """Split sequences chronologically (leave-one-out style)"""
    
    train_seqs = {}
    val_seqs = {}
    test_seqs = {}
    
    for user_id, seq in sequences.items():
        if len(seq) < 3:
            continue
            
        # Leave-last-two-out for validation and test
        train_seqs[user_id] = seq[:-2]
        val_seqs[user_id] = seq[:-1]  # All but last for input, last for target
        test_seqs[user_id] = seq      # Full sequence, last is target
    
    return train_seqs, val_seqs, test_seqs


# =============================================================================
# Informative Subsequence (τ) Extraction
# =============================================================================
class SubsequenceExtractor:
    """Extract informative subsequences for evidence survival analysis"""
    
    def __init__(self, strategy: str = "recency"):
        self.strategy = strategy
    
    def extract(
        self,
        sequence: List[int],
        attention_weights: Optional[np.ndarray] = None,
        item_frequency: Optional[Dict[int, int]] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Extract informative subsequence τ from sequence H
        
        Returns:
            tau: The informative subsequence
            H_minus_tau: Sequence with τ removed
        """
        
        if len(sequence) < 3:
            return [], sequence
        
        if self.strategy == "recency":
            # Remove recent items (last 20%)
            n_remove = max(1, len(sequence) // 5)
            tau = sequence[-n_remove:]
            H_minus_tau = sequence[:-n_remove]
            
        elif self.strategy == "minority":
            # Remove items that are less frequent (minority interests)
            if item_frequency is None:
                # Use position-based heuristic
                tau = [sequence[i] for i in range(0, len(sequence), 3)]
            else:
                # Remove items with low global frequency
                sorted_items = sorted(
                    [(i, item) for i, item in enumerate(sequence)],
                    key=lambda x: item_frequency.get(x[1], 0)
                )
                n_remove = max(1, len(sequence) // 5)
                remove_indices = set([idx for idx, _ in sorted_items[:n_remove]])
                tau = [sequence[i] for i in range(len(sequence)) if i in remove_indices]
                H_minus_tau = [sequence[i] for i in range(len(sequence)) if i not in remove_indices]
                return tau, H_minus_tau
                
        elif self.strategy == "attention":
            # Remove high-attention items
            if attention_weights is None:
                # Fallback to random
                return self._random_extraction(sequence)
            
            n_remove = max(1, len(sequence) // 5)
            top_indices = np.argsort(attention_weights)[-n_remove:]
            tau = [sequence[i] for i in top_indices]
            H_minus_tau = [sequence[i] for i in range(len(sequence)) if i not in top_indices]
            
        elif self.strategy == "random":
            return self._random_extraction(sequence)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return tau, H_minus_tau
    
    def _random_extraction(self, sequence: List[int]) -> Tuple[List[int], List[int]]:
        """Random ablation (control condition)"""
        n_remove = max(1, len(sequence) // 5)
        remove_indices = set(random.sample(range(len(sequence)), n_remove))
        tau = [sequence[i] for i in range(len(sequence)) if i in remove_indices]
        H_minus_tau = [sequence[i] for i in range(len(sequence)) if i not in remove_indices]
        return tau, H_minus_tau


# =============================================================================
# PyTorch Dataset
# =============================================================================
class SequentialRecDataset(Dataset):
    """Dataset for sequential recommendation"""
    
    def __init__(
        self,
        sequences: Dict[int, List[int]],
        max_seq_len: int = 50,
        item2idx: Optional[Dict] = None,
        mode: str = "train",
        tau_strategy: Optional[str] = None
    ):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.item2idx = item2idx
        self.mode = mode
        self.tau_strategy = tau_strategy
        
        self.user_ids = list(sequences.keys())
        self.num_items = max(
            max(seq) for seq in sequences.values() if len(seq) > 0
        ) + 1
        
        if tau_strategy:
            self.subsequence_extractor = SubsequenceExtractor(tau_strategy)
        else:
            self.subsequence_extractor = None
        
        # Compute item frequencies for minority-intent extraction
        self.item_frequency = defaultdict(int)
        for seq in sequences.values():
            for item in seq:
                self.item_frequency[item] += 1
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id = self.user_ids[idx]
        sequence = self.sequences[user_id]
        
        if self.mode == "train":
            # Random subsequence for training
            if len(sequence) > self.max_seq_len + 1:
                start = random.randint(0, len(sequence) - self.max_seq_len - 1)
                sequence = sequence[start:start + self.max_seq_len + 1]
            
            input_seq = sequence[:-1]
            target = sequence[-1]
        else:
            # For val/test, use all history to predict last item
            input_seq = sequence[:-1]
            target = sequence[-1]
        
        # Padding
        seq_len = len(input_seq)
        if seq_len > self.max_seq_len:
            input_seq = input_seq[-self.max_seq_len:]
            seq_len = self.max_seq_len
        
        padded_seq = [0] * (self.max_seq_len - seq_len) + input_seq
        attention_mask = [0] * (self.max_seq_len - seq_len) + [1] * seq_len
        
        result = {
            'user_id': user_id,
            'input_ids': torch.tensor(padded_seq, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'seq_len': seq_len
        }
        
        # Add ablated sequence for evidence survival computation
        if self.subsequence_extractor is not None:
            tau, H_minus_tau = self.subsequence_extractor.extract(
                input_seq,
                item_frequency=dict(self.item_frequency)
            )
            
            # Pad ablated sequence
            ablated_len = len(H_minus_tau)
            if ablated_len > self.max_seq_len:
                H_minus_tau = H_minus_tau[-self.max_seq_len:]
                ablated_len = self.max_seq_len
            
            padded_ablated = [0] * (self.max_seq_len - ablated_len) + H_minus_tau
            ablated_mask = [0] * (self.max_seq_len - ablated_len) + [1] * ablated_len
            
            result['ablated_input_ids'] = torch.tensor(padded_ablated, dtype=torch.long)
            result['ablated_attention_mask'] = torch.tensor(ablated_mask, dtype=torch.long)
            result['tau_indices'] = torch.tensor(tau[:10] if len(tau) > 10 else tau + [0]*(10-len(tau)), dtype=torch.long)
        
        return result


# =============================================================================
# Data Module
# =============================================================================
class RecommendationDataModule:
    """Data module for managing datasets and dataloaders"""
    
    def __init__(
        self,
        config: DatasetConfig,
        batch_size: int = 64,
        num_workers: int = 4,
        tau_strategy: Optional[str] = None
    ):
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tau_strategy = tau_strategy
        
        self._setup()
    
    def _setup(self):
        """Load and prepare data"""
        
        # Download if needed
        # Map config name to dataset key used in DATASET_URLS
        name_lower = self.config.name.lower()
        if "movielens" in name_lower or "ml-1m" in name_lower:
            dataset_key = "ml-1m"
        elif "beauty" in name_lower:
            dataset_key = "amazon-beauty"
        elif "toys" in name_lower:
            dataset_key = "amazon-toys"
        else:
            dataset_key = name_lower.replace(" ", "-")
        download_dataset(dataset_key, DATA_DIR)
        
        # Load data
        self.sequences, self.item2idx, self.item_meta = load_dataset(self.config)
        
        # Split
        self.train_seqs, self.val_seqs, self.test_seqs = split_sequences(
            self.sequences,
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio
        )
        
        self.num_items = len(self.item2idx)
        self.num_users = len(self.sequences)
        
        print(f"Dataset: {self.config.name}")
        print(f"  Users: {self.num_users}")
        print(f"  Items: {self.num_items}")
        print(f"  Train sequences: {len(self.train_seqs)}")
        print(f"  Val sequences: {len(self.val_seqs)}")
        print(f"  Test sequences: {len(self.test_seqs)}")
    
    def train_dataloader(self) -> DataLoader:
        dataset = SequentialRecDataset(
            self.train_seqs,
            max_seq_len=self.config.max_seq_len,
            item2idx=self.item2idx,
            mode="train",
            tau_strategy=self.tau_strategy
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        dataset = SequentialRecDataset(
            self.val_seqs,
            max_seq_len=self.config.max_seq_len,
            item2idx=self.item2idx,
            mode="val",
            tau_strategy=self.tau_strategy
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        dataset = SequentialRecDataset(
            self.test_seqs,
            max_seq_len=self.config.max_seq_len,
            item2idx=self.item2idx,
            mode="test",
            tau_strategy=self.tau_strategy
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# =============================================================================
# Utility Functions
# =============================================================================
def compute_item_frequency(sequences: Dict[int, List[int]]) -> Dict[int, int]:
    """Compute item frequency across all sequences"""
    freq = defaultdict(int)
    for seq in sequences.values():
        for item in seq:
            freq[item] += 1
    return dict(freq)


def identify_minority_items(
    item_frequency: Dict[int, int],
    percentile: float = 20
) -> Set[int]:
    """Identify minority (tail) items by frequency"""
    freqs = list(item_frequency.values())
    threshold = np.percentile(freqs, percentile)
    return {item for item, freq in item_frequency.items() if freq <= threshold}


if __name__ == "__main__":
    # Test data loading
    from config import DATASET_CONFIGS
    
    config = DATASET_CONFIGS["ml-1m"]
    data_module = RecommendationDataModule(config, batch_size=32, tau_strategy="recency")
    
    # Test batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print("\nBatch keys:", batch.keys())
    print("Input shape:", batch['input_ids'].shape)
    print("Target shape:", batch['target'].shape)
    if 'ablated_input_ids' in batch:
        print("Ablated input shape:", batch['ablated_input_ids'].shape)
