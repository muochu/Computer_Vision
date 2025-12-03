"""Parse sequence file to extract frame ranges."""
import re
from collections import defaultdict

TRAIN_SUBJECTS = {11, 12, 13, 14, 15, 16, 17, 18}
VAL_SUBJECTS = {19, 20, 21, 23, 24, 25, 1, 4}
TEST_SUBJECTS = {22, 2, 3, 5, 6, 7, 8, 9, 10}
ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

SKIP_PATTERNS = {'This', 'd1', 'In our', 'Training', 'Validation', 'Test', '*missing*'}


def parse_sequence_file(seq_file_path):
    """Parse sequence file and return organized sequences by split and action."""
    sequences = defaultdict(lambda: defaultdict(list))
    
    with open(seq_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or any(line.startswith(p) for p in SKIP_PATTERNS):
                continue
            
            match = re.match(r'person(\d+)_(\w+)_d(\d+)\s+frames\s+(.+)', line)
            if not match:
                continue
            
            person_id = int(match.group(1))
            action = match.group(2)
            scenario = int(match.group(3))
            frame_ranges_str = match.group(4)
            
            frame_ranges = []
            for range_str in frame_ranges_str.split(','):
                range_str = range_str.strip()
                if '-' in range_str:
                    start, end = map(int, range_str.split('-'))
                    frame_ranges.append((start - 1, end - 1))
            
            if person_id in TRAIN_SUBJECTS:
                split = 'train'
            elif person_id in VAL_SUBJECTS:
                split = 'val'
            elif person_id in TEST_SUBJECTS:
                split = 'test'
            else:
                continue
            
            video_name = f"person{person_id:02d}_{action}_d{scenario}_uncomp.avi"
            
            sequences[split][action].append({
                'person_id': person_id,
                'scenario': scenario,
                'frame_ranges': frame_ranges,
                'video_name': video_name,
                'action': action
            })
    
    return sequences


def get_all_sequences(seq_file_path):
    """Return flat list of all sequences with split information."""
    sequences_dict = parse_sequence_file(seq_file_path)
    all_sequences = []
    
    for split in ['train', 'val', 'test']:
        for action in ACTIONS:
            for seq_info in sequences_dict[split][action]:
                seq_info['split'] = split
                all_sequences.append(seq_info)
    
    return all_sequences

