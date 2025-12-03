"""Main training and evaluation pipeline."""
import os
import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_all_sequences, load_all_data
from core import create_mhi, extract_features
from ml import train_classifier, evaluate_classifier, save_classifier
from utils import plot_confusion_matrix

ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


def extract_features_from_data(data, tau=15, theta=30, use_scaled=True):
    """Extract features from all sequences."""
    X, y = [], []
    
    for split in data.keys():
        for action in ACTIONS:
            for sequence in data[split][action]:
                mhi = create_mhi(sequence, tau=tau, theta=theta)
                features = extract_features(mhi, use_scaled=use_scaled)
                X.append(features)
                y.append(action)
    
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
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(project_root, args.data_dir)
    if not os.path.isabs(args.seq_file):
        args.seq_file = os.path.join(project_root, args.seq_file)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    print("Loading sequences...")
    sequences_info = get_all_sequences(args.seq_file)
    print(f"Found {len(sequences_info)} sequences")
    
    print("Loading video data...")
    data, _ = load_all_data(args.data_dir, sequences_info)
    
    print("Extracting features...")
    X_train, y_train = extract_features_from_data(
        {'train': data['train']}, tau=args.tau, theta=args.theta, 
        use_scaled=args.use_scaled
    )
    X_val, y_val = extract_features_from_data(
        {'val': data['val']}, tau=args.tau, theta=args.theta,
        use_scaled=args.use_scaled
    )
    X_test, y_test = extract_features_from_data(
        {'test': data['test']}, tau=args.tau, theta=args.theta,
        use_scaled=args.use_scaled
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("Training classifier...")
    clf = train_classifier(X_train, y_train, method=args.method)
    
    print("Evaluating...")
    val_acc, _, _ = evaluate_classifier(clf, X_val, y_val)
    test_acc, _, y_pred = evaluate_classifier(clf, X_test, y_test)
    
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_classifier(clf, os.path.join(args.output_dir, 'classifier.pkl'))
    plot_confusion_matrix(y_test, y_pred, ACTIONS,
                         save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
