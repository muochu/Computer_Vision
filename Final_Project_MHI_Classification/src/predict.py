"""Predict action on arbitrary video."""
import os
import sys
import cv2
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import create_mhi, extract_features
from ml import load_classifier


def predict_video(video_path, classifier_path, tau=15, theta=30, use_scaled=True):
    """Predict action for video file."""
    clf = load_classifier(classifier_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    
    if len(frames) < 2:
        raise ValueError("Video must have at least 2 frames")
    
    mhi = create_mhi(frames, tau=tau, theta=theta)
    features = extract_features(mhi, use_scaled=use_scaled).reshape(1, -1)
    
    prediction = clf.predict(features)[0]
    probability = clf.predict_proba(features)[0]
    
    return prediction, probability, mhi


def main():
    parser = argparse.ArgumentParser(description='Predict action from video')
    parser.add_argument('video_path', type=str)
    parser.add_argument('--classifier', type=str, default='results/classifier.pkl')
    parser.add_argument('--tau', type=int, default=15)
    parser.add_argument('--theta', type=int, default=30)
    parser.add_argument('--use_scaled', action='store_true')
    parser.add_argument('--output_mhi', type=str, default=None)
    
    args = parser.parse_args()
    
    # Use relative paths only (requirement: no absolute paths)
    # All paths are relative to the project root directory
    
    print(f"Processing: {args.video_path}")
    prediction, probabilities, mhi = predict_video(
        args.video_path, args.classifier, args.tau, args.theta, args.use_scaled
    )
    
    print(f"\nPredicted: {prediction}")
    print("\nProbabilities:")
    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    for cls, prob in zip(classes, probabilities):
        print(f"  {cls}: {prob:.4f}")
    
    if args.output_mhi:
        cv2.imwrite(args.output_mhi, mhi)
        print(f"\nMHI saved: {args.output_mhi}")


if __name__ == '__main__':
    main()
