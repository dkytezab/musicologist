import os
import random
from typing import List
import subprocess
import torchaudio
import torch

AUDIO_DIR = 'data/tempo/audio'
ANNOTATION_DIR = 'data/tempo/annotations_v2/tempo'
SPLIT_DIR = 'data/tempo/audio_splits'
SPLIT_DURATION = 10
SPLITS_PER_AUDIO = 5

BUCKET_SIZE = 10
PADDING_BPM = 5
BUCKET_STARTS = [80, 100, 160]
MAX_BUCKET_CONTENTS = 300
BUCKET_DIR = 'data/tempo/buckets'


def convert_mp3_to_wav(mp3_path, wav_path):
    subprocess.call(['ffmpeg', '-i', mp3_path, wav_path, '-y'])

def split_audio(audio: torch.Tensor,
                sample_rate: int,
                split_duration: int,
                n_splits: int,
                seed: int = 42) -> list:
    """
    Split an audio file into smaller segments, randomly sampled.
    Args:
        audio_path (str): Path to the audio file.
        split_duration (int): Duration of each split in seconds.
        n_splits (int): Number of splits to create.
    """
    torch.manual_seed(seed)
    audio_length = audio.size(1)
    split_length = split_duration * sample_rate
    if audio_length < split_length:
        print(f"Audio length {audio_length} is shorter than split length {split_length}.")
        return []
    start_indices = torch.randint(0, audio_length - split_length, (n_splits,))
    splits = [audio[:, start:start + split_length] for start in start_indices]
    return splits

def split_dataset(audio_dir: str,
                  annotation_dir: str,
                  split_duration: int,
                  splits_per_audio: int,
                  output_dir: str,
                  csv_path: str) -> None:
    """
    Split the dataset into smaller segments.
    Args:
        audio_dir (str): Directory containing the audio files.
        annotation_dir (str): Directory containing the annotations.
        split_duration (int): Duration of each split in seconds.
        splits_per_audio (int): Number of splits to create for each audio file.
        output_dir (str): Directory to save the split audio files.
        csv_path (str): Path to the CSV file containing the annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for audio_file in os.listdir(audio_dir):
        if not audio_file.endswith('.mp3'):
            continue
        audio_index = audio_file.split('.')[0]
        mp3_path = os.path.join(audio_dir, audio_file)
        wav_path = os.path.join('/tmp', f"{audio_index}.wav")
        convert_mp3_to_wav(mp3_path, wav_path)
        annotation_file = os.path.join(annotation_dir, f"{audio_index}.LOFI.bpm")
        if not os.path.exists(annotation_file):
            print(f"Annotation file {annotation_file} does not exist.")
            continue
        waveform, sample_rate = torchaudio.load(wav_path)
        
        splits = split_audio(waveform, sample_rate, split_duration, splits_per_audio)
        split_filenames = []

        for i, split in enumerate(splits):
            split_filename = f"{os.path.splitext(audio_file)[0]}_split_{i}.wav"
            split_filenames.append(split_filename)
            split_path = os.path.join(output_dir, split_filename)
            torchaudio.save(split_path, split, sample_rate=sample_rate)
            print(f"Saved split audio {i} to {split_path}")
        
        # get annotation
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
        annotation = annotations[0].strip()
        # save annotation to csv
        with open(csv_path, 'a') as f:
            for split_filename in split_filenames:
                f.write(f"{split_filename},{annotation}\n")
            print(f"Saved annotation for {audio_file} to {csv_path}")

def make_buckets(audio_split_dir: str,
                 bucket_size: int,
                 padding_bpm: int,
                 bucket_starts: List[int],
                 max_bucket_contents: int,
                 output_dir: str,
                 seed: int = 42) -> None:
    """
    Create buckets of audio files based on annotated bpms.
    Args:
        audio_split_dir (str): Directory containing the split audio files.
        bucket_size (int): Size of each bucket in seconds.
        padding_bpm (int): Padding BPM for negative samples.
        bucket_starts (List[int]): List of start times for each bucket.
        max_bucket_contents (int): Maximum number of samples in each bucket.
        output_dir (str): Directory to save the buckets.
        seed (int): Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    with open(os.path.join(audio_split_dir, 'annotations.csv'), 'r') as f:
        annotations = f.readlines()
    annotations = [line.strip().split(',') for line in annotations[1:]]
    annotations = [(line[0], round(float(line[1]))) for line in annotations]

    for start in bucket_starts:
        bucket_dir = os.path.join(output_dir, f"bucket_{start}_{start + bucket_size}")
        pos_dir = os.path.join(bucket_dir, 'pos')
        neg_dir = os.path.join(bucket_dir, 'neg')
        os.makedirs(bucket_dir, exist_ok=True)
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        # make positive bucket
        matching_annotations = [line for line in annotations if start <= line[1] < start + bucket_size]
        if len(matching_annotations) > max_bucket_contents:
            matching_annotations = random.sample(matching_annotations, max_bucket_contents)
        for annotation in matching_annotations:
            audio_file = annotation[0]
            audio_path = os.path.join(audio_split_dir, f"{audio_file}")
            if os.path.exists(audio_path):
                bucket_audio_path = os.path.join(pos_dir, f"{audio_file}")
                subprocess.call(['cp', audio_path, bucket_audio_path])
                print(f"Copied {audio_path} to {bucket_audio_path}")
            else:
                print(f"Audio file {audio_path} does not exist.")
        # make negative bucket
        non_matching_annotations = [line for line in annotations if not (start <= (line[1] - padding_bpm) < (start + bucket_size + padding_bpm))]
        if len(non_matching_annotations) > max_bucket_contents:
            non_matching_annotations = random.sample(non_matching_annotations, max_bucket_contents)
        for annotation in non_matching_annotations:
            audio_file = annotation[0]
            audio_path = os.path.join(audio_split_dir, f"{audio_file}")
            if os.path.exists(audio_path):
                bucket_audio_path = os.path.join(neg_dir, f"{audio_file}")
                subprocess.call(['cp', audio_path, bucket_audio_path])
                print(f"Copied {audio_path} to {bucket_audio_path}")
            else:
                print(f"Audio file {audio_path} does not exist.")

if __name__ == "__main__":
    # csv_path = os.path.join(SPLIT_DIR, 'annotations.csv')
    # with open(csv_path, 'w') as f:
    #     f.write("filename,annotation\n")
    
    # split_dataset(AUDIO_DIR, ANNOTATION_DIR, SPLIT_DURATION, SPLITS_PER_AUDIO, SPLIT_DIR, csv_path)

    make_buckets(SPLIT_DIR, BUCKET_SIZE, PADDING_BPM, BUCKET_STARTS, MAX_BUCKET_CONTENTS, BUCKET_DIR)
