"""Video sequence loading utilities."""
import os
import cv2
import numpy as np


def load_video_frames(video_path, start_frame, end_frame):
    """Extract frames from video within specified range."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if start_frame <= frame_idx <= end_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        
        if frame_idx > end_frame:
            break
        
        frame_idx += 1
    
    cap.release()
    return frames


def load_sequence(data_dir, action, video_name, frame_ranges):
    """Load sequences from video file."""
    video_path = os.path.join(data_dir, action, video_name)
    
    if not os.path.exists(video_path):
        return []
    
    sequences = []
    for start, end in frame_ranges:
        frames = load_video_frames(video_path, start, end)
        if frames:
            sequences.append(frames)
    
    return sequences


def load_all_data(data_dir, sequences_info):
    """Load all sequences from dataset."""
    actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    data = {split: {action: [] for action in actions} for split in ['train', 'val', 'test']}
    labels = {split: [] for split in ['train', 'val', 'test']}
    
    for seq_info in sequences_info:
        split = seq_info['split']
        action = seq_info['action']
        sequences = load_sequence(data_dir, action, seq_info['video_name'], 
                                seq_info['frame_ranges'])
        
        for seq in sequences:
            data[split][action].append(seq)
            labels[split].append(action)
    
    return data, labels


