from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import importlib.util
from json import loads
from pathlib import Path
import pandas as pd
import os
import torch
from huggingface_hub import snapshot_download
from transformers import ClapModel, ClapProcessor, AutoProcessor, MusicgenForConditionalGeneration
from muq import MuQMuLan
import torchaudio
import numpy as np
import warnings

from concept_filters import NSYNTH_FILTER_DICT, NSYNTH_PADDER_DICT, create_concept_filter

class NSynthDataset(object):
    def __init__(self, split: str) -> None:
        self.split = split
        self.path = Path(f"data/nsynth/nsynth-{split}")
        self.audio_path = self.path / "audio"
        self.json_path = self.path / "examples.json"

    def _dataset(self) -> Dict:
        return loads(self.json_path.read_text())
    
    def _get_new_json(self, 
                      concept_filters: Tuple[Callable[[dict], bool], ...],
                      concept_filter_names: Tuple[str, ...],) -> Dict:
        dataset = self._dataset()
        new_dataset = {}
        items = iter(dataset.items())

        self.total_samples = len(dataset)

        for idx in range(self.total_samples):
            fpath, sample_info = next(items)
            sample = {                    
                **sample_info,
                 "audio_fpath": str(self.audio_path / Path(fpath + ".wav"))
            }

            df_row = pd.DataFrame([sample_info])

            for func, name in zip(concept_filters, concept_filter_names):
                filtered = func(df_row)
                sample[name] = not filtered.empty
            
            new_dataset[fpath] = sample

        return new_dataset
    
class ConceptDataset(NSynthDataset):
    def __init__(self,
                split: str,
                concept_filter: str,
                concept_filter_path: Optional[str] = None,
                csv_path: Optional[str] = None,
                dir_path: Optional[str] = None,
                get_true: bool = True,
                pos_limit: Optional[int] = None,
                neg_limit: Optional[int] = None,
                overwrite: bool = False,
                ) -> None:
          
        super().__init__(split=split)

        self.concept_filter = concept_filter
        self.concept_padder = f"{concept_filter}_like" 

        self.dir_path = dir_path if dir_path is not None else f"data/concepts/{concept_filter}"
        self.csv_path = csv_path if csv_path is not None else f"data/concepts/{concept_filter}/audio_info.csv"
        self.concept_filter_path = concept_filter_path if concept_filter_path is not None else f"interp/concept_filters.py"

        self.overwrite = overwrite
        self.get_true = get_true

        # Loading in filters
        print("Getting concept filter, padder...")
        self._get_filters()
        
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        # Loading in the csv if present
        if not self.overwrite and os.path.exists(self.csv_path):
            print(f"Loading from csv at {self.csv_path}...")
            self._load_from_csv()
        
        # Otherwise, create from scratch
        else:
            print("Getting new json...")
            new_json = self._get_new_json(concept_filters=[self.concept_filter_func, self.concept_padder_func],
                                          concept_filter_names=[self.concept_filter, self.concept_padder])
            if len(new_json) < 1:
                raise ValueError("Produced json is empty")
            print("Getting positive, negative limits...")
            self._get_pos_neg_limits(new_json, pos_limit, neg_limit)
            print("Writing csv...")
            self._write_csv(new_json)
    
    def _get_filters(self):
        col_dict, get_true, logic = NSYNTH_FILTER_DICT[self.concept_filter]
        self.concept_filter_func = create_concept_filter(col_dict, get_true, logic)

        col_dict, get_true, logic = NSYNTH_PADDER_DICT[self.concept_padder]
        self.concept_padder_func = create_concept_filter(col_dict, get_true, logic)

    def _load_from_csv(self):
        df = pd.read_csv(self.csv_path)
        
        self.samples = df
        self.pos_samples = df[df["binary_class"] == 1]
        self.neg_samples = df[df["binary_class"] == 0]

        self.num_pos_samples = self.pos_limit = len(self.pos_samples)
        self.num_neg_samples = self.neg_limit = len(self.neg_samples)
        self.num_samples = len(self.samples)

        self.possible_pos_samples = None
        self.possible_neg_samples = None

    def _get_pos_neg_limits(self, new_json, pos_limit, neg_limit) -> None:
        pos_df, neg_df = self._get_pos_neg_df(new_json)
        
        self.possible_pos_samples = pos_df.shape[0]
        self.possible_neg_samples = neg_df.shape[0]

        pos_limit = pos_limit if pos_limit is not None else self.possible_pos_samples
        neg_limit = neg_limit if neg_limit is not None else self.possible_neg_samples

        balanced_lim = min(pos_limit // 2, neg_limit // 2, 15000)

        self.pos_limit, self.neg_limit = balanced_lim, balanced_lim
    
    def _write_csv(self, new_json: Dict) -> None:
        pos_df, neg_df = self._get_pos_neg_df(new_json)

        if self.pos_limit is not None:
            pos_df = pos_df.copy().sample(n=self.pos_limit, random_state=None)
        if self.neg_limit is not None:
            neg_df = neg_df.copy().sample(n=self.neg_limit, random_state=None)

        pos_df["binary_class"] = 1
        neg_df["binary_class"] = 0

        pos_df["binary_class_str"] = "positive"
        neg_df["binary_class_str"] = "negative"

        self.pos_samples = pos_df
        self.neg_samples = neg_df
        self.samples = pd.concat([pos_df, neg_df])
        
        self.num_pos_samples = len(self.pos_samples)
        self.num_neg_samples = len(self.neg_samples)
        self.num_samples = len(self.samples)
    
        self.samples.to_csv(self.csv_path, index=False)

    def _get_pos_neg_df(self, new_json) -> None:
        df = pd.DataFrame.from_dict(new_json, orient="index")

        pos_paths = [fpath for fpath, sample in new_json.items()
                 if sample.get(self.concept_filter, True) and sample.get(self.concept_padder, True)]
        neg_paths = [fpath for fpath, sample in new_json.items()
                 if not sample.get(self.concept_filter, False) and not sample.get(self.concept_padder, False)]
        
        pos_df = df.loc[pos_paths].copy()
        neg_df = df.loc[neg_paths].copy()

        return pos_df, neg_df
    
    def get_embeds(self, 
                   model_name: str,
                   save: bool = False,
                   cache = None,
                   save_path: Optional[str] = None,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu",
                   ):
        
        # Make sure save_path is none for ease
        save_path = save_path if save_path is not None else f"{self.dir_path}/{model_name}_embeddings.pt"
        
        if os.path.exists(save_path) and not self.overwrite:
            print(f"Embeddings exist at {save_path}")
            embeds = torch.load(save_path)
        
        else:
            # Loading the model and processor
            # We rip this code from embeddings.load_model
            cache = cache if cache is not None else cache_model(model_name=model_name)

            if model_name == "laion-clap":
                model = ClapModel.from_pretrained(cache, local_files_only=True).to(device).eval()
                processor = AutoProcessor.from_pretrained(cache, local_files_only=True)
                new_sr = 48000
                embed_dim = 512
                num_seconds = 10
            
            elif model_name == "muq":
                model = MuQMuLan.from_pretrained(cache, local_files_only=True).to(device).eval()
                processor = None
                new_sr = 24000
                embed_dim = None
                num_seconds = None

            old_sr, audio_tensor = self._load_audios_to_batch(self.samples["audio_fpath"].tolist())
            processed_audio_tensor = self._preprocess_audio_tensor(audio_tensor=audio_tensor,
                                                                old_sr=old_sr, new_sr=new_sr,
                                                                num_seconds=num_seconds)
            
            embeds = self._get_embeds_from_model(model=model, processor=processor,
                                                device=device,
                                                new_sr=new_sr,
                                                num_seconds=num_seconds,
                                                audio_tensor=processed_audio_tensor,
                                                embed_dim=embed_dim,
                                                )
            if save:
                self._save_embeds(embeds, model_name=model_name, save_path=save_path)

        return embeds

    def _load_audios_to_batch(self, audio_paths: List[str]) -> torch.Tensor:
        # Assumes all audio files are the same length and 2D
        channels, frames = torchaudio.load(audio_paths[0])[0].shape
        audio_tensor = torch.zeros((self.num_samples, channels, frames))
        for idx in range(self.num_samples):
            audio_path = audio_paths[idx]
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Cannot find audio file at {audio_path}")
            audio, old_sr = torchaudio.load(audio_path)
            audio_tensor[idx, :, :] = audio
        return old_sr, audio_tensor
    

    def _preprocess_audio_tensor(self, audio_tensor: torch.Tensor, old_sr: int, new_sr: int, num_seconds: int = 10) -> torch.Tensor:
        resampler = torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=new_sr)
        # audio_tensor shape is (batch_size, channels, frames)
        # Re-sampling to mono
        if audio_tensor.shape[1] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=1, keepdim=False)
        # Resampling to new sampling rate
        new_frames = int(new_sr * num_seconds)
        new_tensor = torch.zeros((self.num_samples, new_frames))
        # Iterating over audio
        for idx in range(self.num_samples):
            resampled = resampler(audio_tensor[idx, :]).squeeze(0)
            # If audio is too long
            if resampled.shape[-1] > new_sr * num_seconds:
                resampled = resampled[:new_sr * num_seconds]
            # If audio is too short
            elif resampled.shape[-1] < new_sr * num_seconds:
                pad_len = new_frames - resampled.shape[-1]
                resampled = torch.nn.functional.pad(resampled, (0, pad_len), 'constant', 0)

            new_tensor[idx, :] = resampled
        return new_tensor
    

    def _get_embeds_from_model(self, model, processor, 
                               audio_tensor: torch.Tensor, 
                               device, new_sr: int,
                               num_seconds: int,
                               embed_dim: int,
                               ) -> torch.Tensor:
        batch_size, frames = audio_tensor.shape
        new_tens = torch.zeros([batch_size, embed_dim])

        audio_np = audio_tensor.cpu().numpy().astype(np.float32)

        if isinstance(model, ClapModel):
            for idx in range(batch_size):
                input_audio = audio_np[idx, :]
                encoding = processor(audios=input_audio, sampling_rate=new_sr, return_tensors="pt")
                encoding_items = {k: v.to(device) for k, v in encoding.items()}
                with torch.no_grad():
                    audio_embed = model.get_audio_features(**encoding_items).squeeze(0).cpu()
                new_tens[idx, :] = audio_embed
                print(f"Produced embeddings from index {idx}")
        
        elif isinstance(model, MuQMuLan):
            raise NotImplementedError("MuQMuLan model not supported yet")
        
        else:
            raise ValueError("Specified model not presently supported")
        
        return new_tens


    def _save_embeds(self, embeds: torch.Tensor, save_path: Optional[None], model_name: str,) -> None:

        if save_path is None:
            save_path = f"{self.dir_path}/{model_name}_embeddings.pt"

        torch.save(embeds, save_path)
        print(f"Embeddings saved to {save_path}")

def cache_model(model_name: str) -> str:
        cache_dir = f"models/{model_name}"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if model_name == "laion-clap":
            repo_id = "laion/larger_clap_general"
            downloaded_model = snapshot_download(repo_id, cache_dir=cache_dir)
        elif model_name == "muq":
            repo_id = "OpenMuQ/MuQ-MuLan-large"
            downloaded_model = snapshot_download(repo_id, cache_dir=cache_dir)
        else:
            raise NotImplemented("Specified embeddings model not supported. Try laion-clap or muq")
        return downloaded_model