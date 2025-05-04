from utils import preprocess_audio, get_embedding, save_embeddings
from load_model import get_embed_model

import os
import torch

AUDIO_DIR = 'data/tempo/buckets'
MODEL_NAME = 'laion-clap'
BUCKET_TYPES = ['pos', 'neg']

def make_embeds(audio_dir: str, model_name: str, bucket_types: list):
    """
    Generate embeddings for audio files in specified directories.
    
    Args:
        audio_dir (str): Directory containing audio files.
        model_name (str): Name of the model to use for generating embeddings.
        bucket_types (list): List of bucket types to process.
    """
    # Load the model
    model, processor = get_embed_model(model_name=model_name)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == "laion-clap":
        embed_dim = 512
    elif model_name == "muq":
        embed_dim = 512

    # Iterate through each bucket and type
    for bucket in os.listdir(AUDIO_DIR):
        bucket_dir = os.path.join(AUDIO_DIR, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        
        for bucket_type in BUCKET_TYPES:
            bucket_path = os.path.join(bucket_dir, bucket_type)
            if not os.path.isdir(bucket_path):
                continue
            print(f"Processing {bucket_type} in {bucket}")
            embed_tensor = torch.zeros((len(os.listdir(bucket_path)), embed_dim))
            audio_files = [f for f in os.listdir(bucket_path) if f.endswith('.wav')]
            if not audio_files:
                print(f"No audio files found in {bucket_path}")
                continue
            for i, audio_file in enumerate(audio_files):
                audio_path = os.path.join(bucket_path, audio_file)
                print(f"Processing {audio_path}")
                embed_tensor[i] = get_embedding(audio_path, model, processor)
            # Save the embeddings
            save_path = os.path.join(bucket_path, f"{bucket_type}_embeddings.pt")
            save_embeddings(embed_tensor, 0, model_name=model_name, out_dir=bucket_path)
            print(f"Saved embeddings to {save_path}")
            print(f"Processed {len(audio_files)} files in {bucket_type} of {bucket}")

if __name__ == "__main__":
    make_embeds(AUDIO_DIR, MODEL_NAME, BUCKET_TYPES)
    print("Embeddings generation complete.")
