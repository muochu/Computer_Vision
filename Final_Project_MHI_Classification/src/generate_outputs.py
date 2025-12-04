"""Generate output videos and visualizations for report."""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import get_all_sequences
from data.data_loader import load_sequence
from core import create_mhi, extract_features
from ml import load_classifier
from utils import (visualize_mhi, create_output_video, visualize_frame_differencing_comparison,
                   visualize_pipeline_diagram, visualize_success_case, visualize_failure_case)

ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


def generate_sample_mhis(data_dir, seq_file, output_dir, tau=15, theta=30, num_samples=2):
    """Generate sample MHI images for each action."""
    sequences_info = get_all_sequences(seq_file)
    os.makedirs(output_dir, exist_ok=True)
    
    samples_generated = {action: 0 for action in ACTIONS}
    
    for seq_info in sequences_info:
        action = seq_info['action']
        if samples_generated[action] >= num_samples:
            continue
        
        sequences = load_sequence(data_dir, action, seq_info['video_name'], 
                                 [seq_info['frame_ranges'][0]])
        
        if sequences and len(sequences[0]) > 1:
            mhi = create_mhi(sequences[0], tau=tau, theta=theta)
            output_path = os.path.join(output_dir, f'mhi_{action}_{samples_generated[action]}.png')
            visualize_mhi(mhi, title=f'MHI: {action}', save_path=output_path)
            samples_generated[action] += 1
            print(f"Generated MHI: {action}")


def generate_figure2_frame_differencing(data_dir, seq_file, output_path, theta=30):
    """Generate Figure 2: Frame differencing before/after morphological opening."""
    sequences_info = get_all_sequences(seq_file)
    
    # Find a good sequence with visible motion
    for seq_info in sequences_info:
        sequences = load_sequence(data_dir, seq_info['action'], seq_info['video_name'],
                                 [seq_info['frame_ranges'][0]])
        
        if sequences and len(sequences[0]) > 1:
            frames = sequences[0]
            visualize_frame_differencing_comparison(frames, theta=theta, save_path=output_path)
            print(f"Generated Figure 2: Frame differencing comparison")
            return
    print("Warning: Could not generate Figure 2 - no suitable sequence found")


def generate_figure3_pipeline_diagram(output_path):
    """Generate Figure 3: Processing pipeline diagram."""
    visualize_pipeline_diagram(save_path=output_path)
    print("Generated Figure 3: Pipeline diagram")


def generate_figure5_success_case(data_dir, seq_file, classifier_path, output_path, 
                                  tau=15, theta=30):
    """Generate Figure 5: Success case (boxing correctly classified)."""
    sequences_info = get_all_sequences(seq_file)
    clf = load_classifier(classifier_path)
    
    # Find a boxing sequence that gets correctly classified
    for seq_info in sequences_info:
        if seq_info['action'] != 'boxing':
            continue
        
        sequences = load_sequence(data_dir, seq_info['action'], seq_info['video_name'],
                                 [seq_info['frame_ranges'][0]])
        
        if sequences and len(sequences[0]) > 1:
            frames = sequences[0]
            mhi = create_mhi(frames, tau=tau, theta=theta)
            features = extract_features(mhi, use_scaled=True)
            prediction = clf.predict(features.reshape(1, -1))[0]
            
            if prediction == 'boxing':  # Only use if correctly classified
                visualize_success_case(frames, mhi, 'boxing', prediction, save_path=output_path)
                print(f"Generated Figure 5: Success case (boxing -> {prediction})")
                return
    
    print("Warning: Could not find correctly classified boxing sequence for Figure 5")


def generate_figure6_failure_case(data_dir, seq_file, classifier_path, output_path,
                                  tau=15, theta=30):
    """Generate Figure 6: Failure case (jogging vs running confusion)."""
    sequences_info = get_all_sequences(seq_file)
    clf = load_classifier(classifier_path)
    
    jogging_frames = None
    jogging_mhi = None
    jogging_pred = None
    running_frames = None
    running_mhi = None
    running_pred = None
    
    # Find jogging and running sequences
    for seq_info in sequences_info:
        if seq_info['action'] == 'jogging' and jogging_frames is None:
            sequences = load_sequence(data_dir, seq_info['action'], seq_info['video_name'],
                                     [seq_info['frame_ranges'][0]])
            if sequences and len(sequences[0]) > 1:
                jogging_frames = sequences[0]
                jogging_mhi = create_mhi(jogging_frames, tau=tau, theta=theta)
                features = extract_features(jogging_mhi, use_scaled=True)
                jogging_pred = clf.predict(features.reshape(1, -1))[0]
        
        elif seq_info['action'] == 'running' and running_frames is None:
            sequences = load_sequence(data_dir, seq_info['action'], seq_info['video_name'],
                                     [seq_info['frame_ranges'][0]])
            if sequences and len(sequences[0]) > 1:
                running_frames = sequences[0]
                running_mhi = create_mhi(running_frames, tau=tau, theta=theta)
                features = extract_features(running_mhi, use_scaled=True)
                running_pred = clf.predict(features.reshape(1, -1))[0]
        
        if jogging_frames is not None and running_frames is not None:
            break
    
    if jogging_frames is not None and running_frames is not None:
        visualize_failure_case(jogging_frames, jogging_mhi, 'jogging',
                              running_frames, running_mhi, 'running',
                              jogging_pred, running_pred, save_path=output_path)
        print(f"Generated Figure 6: Failure case (jogging->{jogging_pred}, running->{running_pred})")
    else:
        print("Warning: Could not generate Figure 6 - missing jogging or running sequences")


def generate_output_video(data_dir, seq_file, classifier_path, output_dir, 
                         tau=15, theta=30, num_videos=1):
    """Generate output videos with predictions."""
    sequences_info = get_all_sequences(seq_file)
    clf = load_classifier(classifier_path)
    os.makedirs(output_dir, exist_ok=True)
    
    videos_generated = {action: 0 for action in ACTIONS}
    
    for seq_info in sequences_info:
        action = seq_info['action']
        if videos_generated[action] >= num_videos:
            continue
        
        sequences = load_sequence(data_dir, action, seq_info['video_name'], 
                                 [seq_info['frame_ranges'][0]])
        
        if sequences and len(sequences[0]) > 1:
            frames = sequences[0]
            mhi = create_mhi(frames, tau=tau, theta=theta)
            features = extract_features(mhi, use_scaled=True)
            prediction = clf.predict(features.reshape(1, -1))[0]
            
            output_path = os.path.join(output_dir, f'output_{action}_{videos_generated[action]}.mp4')
            create_output_video(frames, [prediction], output_path, fps=25)
            
            videos_generated[action] += 1
            print(f"Generated video: {action} -> {prediction}")


def main():
    parser = argparse.ArgumentParser(description='Generate outputs for report')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--seq_file', type=str, default='00sequence.txt')
    parser.add_argument('--classifier', type=str, default='results/classifier.pkl')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--tau', type=int, default=15)
    parser.add_argument('--theta', type=int, default=30)
    parser.add_argument('--mhi_samples', type=int, default=2)
    parser.add_argument('--video_samples', type=int, default=1)
    
    args = parser.parse_args()
    
    # Use relative paths only (requirement: no absolute paths)
    # All paths are relative to the project root directory
    
    print("Generating sample MHIs...")
    generate_sample_mhis(args.data_dir, args.seq_file, 
                        os.path.join(args.output_dir, 'mhis'),
                        args.tau, args.theta, args.mhi_samples)
    
    print("\nGenerating output videos...")
    generate_output_video(args.data_dir, args.seq_file, args.classifier,
                         os.path.join(args.output_dir, 'videos'),
                         args.tau, args.theta, args.video_samples)
    
    print("\nGenerating report figures...")
    figures_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Figure 2: Frame differencing comparison
    generate_figure2_frame_differencing(args.data_dir, args.seq_file,
                                       os.path.join(figures_dir, 'figure2_frame_differencing.png'),
                                       args.theta)
    
    # Figure 3: Pipeline diagram
    generate_figure3_pipeline_diagram(os.path.join(figures_dir, 'figure3_pipeline.png'))
    
    # Figure 5: Success case
    generate_figure5_success_case(args.data_dir, args.seq_file, args.classifier,
                                   os.path.join(figures_dir, 'figure5_success_case.png'),
                                   args.tau, args.theta)
    
    # Figure 6: Failure case
    generate_figure6_failure_case(args.data_dir, args.seq_file, args.classifier,
                                  os.path.join(figures_dir, 'figure6_failure_case.png'),
                                  args.tau, args.theta)
    
    print(f"\nAll outputs saved to {args.output_dir}/")
    print(f"  - MHIs: {args.output_dir}/mhis/")
    print(f"  - Videos: {args.output_dir}/videos/")
    print(f"  - Report figures: {args.output_dir}/figures/")


if __name__ == '__main__':
    main()
