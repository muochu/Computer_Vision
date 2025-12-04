"""Main training and evaluation pipeline."""
import os
import sys
import numpy as np
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_all_sequences
from data.data_loader import load_sequence
from core import create_mhi, extract_features
from ml import train_classifier, evaluate_classifier, save_classifier
from utils import plot_confusion_matrix

ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


def extract_features_from_sequences(sequences_info, data_dir, split_filter, tau=15, theta=30, use_scaled=True):
    """Extract features by processing sequences one at a time (memory-efficient)."""
    X, y = [], []
    
    # Filter sequences by split
    filtered_seqs = [s for s in sequences_info if s['split'] == split_filter]
    total = len(filtered_seqs)
    
    print(f"  Processing {total} sequences for {split_filter} split...")
    start_time = time.time()
    
    for idx, seq_info in enumerate(filtered_seqs):
        seq_start = time.time()
        
        # Load only this sequence (memory-efficient)
        sequences = load_sequence(data_dir, seq_info['action'], seq_info['video_name'],
                                 seq_info['frame_ranges'])
        
        # Process each sequence in this video
        sequences_processed = 0
        for seq in sequences:
            if len(seq) < 2:
                continue
                
            mhi = create_mhi(seq, tau=tau, theta=theta)
            features = extract_features(mhi, use_scaled=use_scaled)
            X.append(features)
            y.append(seq_info['action'])
            sequences_processed += 1
        
        # Progress update with time estimates - update every sequence
        elapsed = time.time() - start_time
        progress = (idx + 1) / total
        if progress > 0:
            eta = elapsed / progress * (1 - progress)
            bar_length = 40
            filled = int(bar_length * progress)
            bar = '=' * filled + '-' * (bar_length - filled)
            seq_time = time.time() - seq_start
            # Show action name and current video being processed
            action_name = seq_info['action'][:8].ljust(8)  # Truncate for display
            print(f"  [{bar}] {progress*100:.1f}% ({idx+1}/{total}) | "
                  f"{action_name} | {sequences_processed} seqs | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m    ", end='\r', flush=True)
    
    total_time = time.time() - start_time
    print(f"\n  ✓ Completed {len(X)} feature vectors extracted in {total_time:.1f}s ({total_time/60:.1f} min)")
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description='MHI Activity Classification')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--seq_file', type=str, default='00sequence.txt')
    parser.add_argument('--tau', type=int, default=15)
    parser.add_argument('--theta', type=int, default=30)
    parser.add_argument('--method', type=str, default='knn', choices=['knn', 'svm'])
    parser.add_argument('--use_scaled', action='store_true')
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Use relative paths only (requirement: no absolute paths)
    # All paths are relative to the project root directory
    
    print("Loading sequence metadata...")
    sequences_info = get_all_sequences(args.seq_file)
    print(f"Found {len(sequences_info)} sequences")
    
    print("\nExtracting features (memory-efficient, one sequence at a time)...")
    print("  This may take 10-20 minutes but uses minimal memory...")
    
    print("\n[1/3] Processing training set...")
    X_train, y_train = extract_features_from_sequences(
        sequences_info, args.data_dir, 'train',
        tau=args.tau, theta=args.theta, use_scaled=args.use_scaled
    )
    
    print("\n[2/3] Processing validation set...")
    X_val, y_val = extract_features_from_sequences(
        sequences_info, args.data_dir, 'val',
        tau=args.tau, theta=args.theta, use_scaled=args.use_scaled
    )
    
    print("\n[3/3] Processing test set...")
    X_test, y_test = extract_features_from_sequences(
        sequences_info, args.data_dir, 'test',
        tau=args.tau, theta=args.theta, use_scaled=args.use_scaled
    )
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:      {len(X_test)} samples")
    print(f"{'='*60}\n")
    
    print("Training classifier...")
    train_start = time.time()
    clf = train_classifier(X_train, y_train, method=args.method)
    train_time = time.time() - train_start
    print(f"  ✓ Classifier trained in {train_time:.2f}s\n")
    
    print("Evaluating on validation set...")
    val_start = time.time()
    val_acc, _, _ = evaluate_classifier(clf, X_val, y_val)
    val_time = time.time() - val_start
    print(f"  ✓ Validation accuracy: {val_acc:.4f} ({val_time:.2f}s)\n")
    
    print("Evaluating on test set...")
    test_start = time.time()
    test_acc, _, y_pred = evaluate_classifier(clf, X_test, y_test)
    test_time = time.time() - test_start
    print(f"  ✓ Test accuracy: {test_acc:.4f} ({test_time:.2f}s)\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_classifier(clf, os.path.join(args.output_dir, 'classifier.pkl'))
    plot_confusion_matrix(y_test, y_pred, ACTIONS,
                         save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
