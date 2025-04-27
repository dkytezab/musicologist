import matplotlib.pyplot as plt
import imageio
import torch
import torchaudio
import sys
import os
from glob import glob

def preprocess_audio(audio_path):
    audio_input, sr = torchaudio.load(audio_path)
    if audio_input.shape[0] > 1:
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)
    # Resample to 48kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
    audio_input = resampler(audio_input)
    # Truncate to the first 10 seconds (480,000 samples)
    if audio_input.shape[-1] > 480000:
        audio_input = audio_input[..., :480000]
    return audio_input

def create_tensors(audio_base,
                   prompt_index):
    audio_files = []
    for i in range(1, 100):
        pattern = f"{audio_base}/diff_step_{i}/steps_*_prompt_*_sample_*.wav"
        matches = glob(pattern)
        audio_files.append(matches[0])
    print(f"Found {len(audio_files)} audio files")
    audio_tensors = [preprocess_audio(audio_path) for audio_path in audio_files]
    return audio_tensors

def plot_spectrogram(audio_tensor, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.specgram(audio_tensor.numpy()[0], Fs=48000, NFFT=1024, noverlap=512)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.show()
    plt.savefig(f"viz/outputs/{title}.png", dpi=300)

def create_frames(audio_tensors, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, audio_tensor in enumerate(audio_tensors):
        plot_spectrogram(audio_tensor, title=f"frame_{i+1}")
        plt.savefig(f"{output_dir}/frame_{i+1}.png", dpi=300)
        plt.close()

def create_gif(frames_dir, output_path):
    frame_files = glob(os.path.join(frames_dir, "Spectrogram Frame *.png"))
    frame_files.sort(key=lambda x: int(x.split("Spectrogram Frame ")[1].split(".")[0]))
    with imageio.get_writer(output_path, mode='I', duration=0.2) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)
            del image

# audio_tensors = create_tensors(
#     audio_base='data/generated',
#     prompt_index=0
# )

# create_frames(
#     audio_tensors=audio_tensors,
#     output_dir='viz/outputs'
# )

create_gif(
    frames_dir='viz/outputs',
    output_path='viz/outputs/spectrogram_algerian.gif'
)
