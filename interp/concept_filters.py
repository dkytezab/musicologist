from typing import List, Callable, Any, Dict, Optional
import pandas as pd
import importlib.util
from pathlib import Path
import re

# Gets a filter function
def create_concept_filter(cols_aspects_dict: Dict[str, str],
                          get_true: bool = True, 
                          logic: str = "or") -> Callable[[pd.DataFrame], pd.DataFrame]:
   
    def filter_function(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        mask = pd.Series(False, index=idx) if logic=="or" else pd.Series(True, index=idx)

        for col, aspects in cols_aspects_dict.items():
            # build a single regex to test any aspect
            series = df[col]
            is_list_col = (
                series.dtype == object 
                and series.apply(lambda cell: isinstance(cell, list)).all()
            )
            if is_list_col:
                # build a row‑wise Boolean Series
                col_mask = series.apply(
                    lambda lst: any(asp in (lst or []) for asp in aspects)
                )
            else:
                pattern = "|".join(re.escape(a) for a in aspects)
                col_mask = df[col].str.contains(pattern, na=False)  # element-wise test
            if logic == "or":
                mask |= col_mask
            else:  # "and"
                mask &= col_mask

        # 2) apply inversion if needed
        return df[mask] if get_true else df[~mask]

    return filter_function
            

def get_all_concepts():
    return list(NSYNTH_FILTER_DICT.keys())

NSYNTH_FILTER_DICT = {
    # Individual instrument families - 11 total
    "is_brass":         [{"instrument_family_str": ["brass"]}, True, "or"],
    "is_bass":          [{"instrument_family_str": ["bass"]}, True, "or"],
    "is_flute":         [{"instrument_family_str": ["flute"]}, True, "or"],
    "is_guitar":        [{"instrument_family_str": ["guitar"]}, True, "or"],
    "is_mallet":        [{"instrument_family_str": ["mallet"]}, True, "or"],
    "is_organ":         [{"instrument_family_str": ["organ"]}, True, "or"],
    "is_reed":          [{"instrument_family_str": ["reed"]}, True, "or"],
    "is_string":        [{"instrument_family_str": ["string"]}, True, "or"],
    "is_synth_lead":    [{"instrument_family_str": ["synth_lead"]}, True, "or"],
    "is_keyboard":      [{"instrument_family_str": ["keyboard"]}, True, "or"],
    "is_vocal":         [{"instrument_family_str": ["vocal"]}, True, "or"],
    
    # Select note qualities - 6 total
    "is_long":          [{"qualities_str": ["long_release", "reverb"]}, True, "or"],
    "is_harmonic":      [{"qualities_str": ["bright", "multiphonic"]}, True, "or"],
    "is_rhythmic":      [{"qualities_str": ["tempo-synced"]}, True, "or"],
    "is_staccato":      [{"qualities_str": ["percussive", "fast_decay"]}, True, "or"],
    "is_distorted":     [{"qualities_str": ["distortion"]}, True, "or"],
    "is_warm":          [{"qualities_str": ["dark"]}, True, "or"],

    # Source qualities - 3 total
    "is_acoustic":      [{"instrument_source_str": ["acoustic"]}, True, "or"],
    "is_electronic":    [{"instrument_source_str": ["electronic"]}, True, "or"],
    "is_synthetic":     [{"instrument_source_str": ["synthetic"]}, True, "or"],

    # Higher-level concepts - 8 total
    "is_orchestral":    [{"instrument_family_str": ["bass", "string", "brass", "reed"],
                          "instrument_source_str": ["acoustic"]}, True, "and"],
    "is_acoustic_band": [{"instrument_family_str": ["bass", "brass", "reed", "keyboard", "guitar"],
                          "instrument_source_str": ["acoustic"]}, True, "and"],
    "is_percussion":    [{"instrument_family_str": ["mallet"],
                          "qualities_str": ["percussive"]}, True, "or"],
    "is_techno":        [{"instrument_family_str": ["synth_lead", "mallet", "brass",],
                          "instrument_source_str": ["synthetic"]}, True, "and",],
    "is_plucked":       [{"instrument_family_str": ["string", "bass", "guitar",]}, True, "or"],
    "is_blown":         [{"instrument_family_str": ["brass", "flute", "reed"],}, True, "or"],
    "is_hit":           [{"instrument_family_str": ["mallet", "keyboard", "synth_lead"]}, True, "or"],
    "is_atmospheric":   [{"instrument_family_str": ["organ", "synth_lead", "vocal"]},
                         {"qualities_str": ["long_release", "reverb", "nonlinear_env"]}, True, "and"],
}

NSYNTH_PADDER_DICT = {
    # Individual instrument families - 11 total
    "is_brass_like":        [{"instrument_family_str": ["brass", "reed"]}, True, "or"],
    "is_bass_like":         [{"instrument_family_str": ["bass", "string", "guitar"]}, True, "or"],
    "is_flute_like":        [{"instrument_family_str": ["flute", "reed"]}, True, "or"],
    "is_guitar_like":       [{"instrument_family_str": ["guitar", "bass", "string"]}, True, "or"],
    "is_mallet_like":       [{"instrument_family_str": ["mallet", "keyboard"]}, True, "or"],
    "is_organ_like":        [{"instrument_family_str": ["organ", "keyboard"]}, True, "or"],
    "is_reed_like":         [{"instrument_family_str": ["reed", "brass", "flute"]}, True, "or"],
    "is_string_like":       [{"instrument_family_str": ["string", "bass", "guitar"]}, True, "or"],
    "is_synth_lead_like":   [{"instrument_family_str": ["synth_lead", "organ"]}, True, "or"],
    "is_keyboard_like":     [{"instrument_family_str": ["keyboard", "organ"]}, True, "or"],
    "is_vocal_like":        [{"instrument_family_str": ["vocal"]}, True, "or"],

    # Select note qualities - 6 total
    "is_long_like":          [{"qualities_str": ["long_release", "reverb", "nonlinear_env"]}, True, "or"],
    "is_harmonic_like":      [{"qualities_str": ["bright", "multiphonic"]}, True, "or"],
    "is_rhythmic_like":      [{"qualities_str": ["tempo-synced", "percussive"]}, True, "or"],
    "is_staccato_like":      [{"qualities_str": ["percussive", "fast_decay", "tempo-synced"]}, True, "or"],
    "is_distorted_like":     [{"qualities_str": ["distortion", "nonlinear_env", "reverb"]}, True, "or"],
    "is_warm_like":          [{"qualities_str": ["dark", "reverb"]}, True, "or"],

    # Source qualities - 3 total
    "is_acoustic_like":      [{"instrument_source_str": ["acoustic"]}, True, "or"],
    "is_electronic_like":    [{"instrument_source_str": ["electronic", "synthetic"]}, True, "or"],
    "is_synthetic_like":     [{"instrument_source_str": ["synthetic", "electronic"]}, True, "or"],

    # Higher-level concepts - 8 total
    "is_orchestral_like":    [{"instrument_family_str": ["bass", "string", "brass", "reed", "organ", "keyboard",],
                               "instrument_source_str": ["acoustic"]}, True, "and"],
    "is_acoustic_band_like": [{"instrument_family_str": ["bass", "brass", "reed", "keyboard", "guitar", "string"],
                               "instrument_source_str": ["acoustic"]}, True, "and"],
    "is_percussion_like":    [{"instrument_family_str": ["mallet", "keyboard"],
                               "qualities_str": ["percussive", "fast_decay"]}, True, "or"],
    "is_techno_like":        [{"instrument_family_str": ["synth_lead", "mallet", "brass",],
                                "instrument_source_str": ["synthetic", "electronic"]}, True, "and",],
    "is_plucked_like":       [{"instrument_family_str": ["string", "bass", "guitar", "keyboard"]}, True, "or"],
    "is_blown_like":         [{"instrument_family_str": ["brass", "flute", "reed", "organ", "vocal"],}, True, "or"],
    "is_hit_like":           [{"instrument_family_str": ["mallet", "keyboard", "synth_lead", "organ",]}, True, "or"],
    "is_atmospheric_like":   [{"instrument_family_str": ["organ", "synth_lead", "vocal"]},
                              {"qualities_str": ["long_release", "reverb", "nonlinear_env"]}, True, "and"],
}

GEN_AUDIO_FILTER_DICT = {
    # Individual instrument families - 11 total
    "is_brass":         [{"tag.aspects": ["brass"]}, True, "or"],
    "is_bass":          [{"tag.aspects": ["bass"]}, True, "or"],
    "is_flute":         [{"tag.aspects": ["flute"]}, True, "or"],
    "is_guitar":        [{"tag.aspects": ["guitar"]}, True, "or"],
    "is_mallet":        [{"tag.aspects": ["mallet"]}, True, "or"],
    "is_organ":         [{"tag.aspects": ["organ"]}, True, "or"],
    "is_reed":          [{"tag.aspects": ["reed"]}, True, "or"],
    "is_string":        [{"tag.aspects": ["string"]}, True, "or"],
    "is_synth_lead":    [{"tag.aspects": ["synth_lead"]}, True, "or"],
    "is_keyboard":      [{"tag.aspects": ["keyboard"]}, True, "or"],
    "is_vocal":         [{"tag.aspects": ["vocal"]}, True, "or"],
    
    # Select note qualities - 6 total
    "is_long":          [{"tag.aspects": ["long_release", "reverb"]}, True, "or"],
    "is_harmonic":      [{"tag.aspects": ["bright", "multiphonic"]}, True, "or"],
    "is_rhythmic":      [{"tag.aspects": ["tempo-synced"]}, True, "or"],
    "is_staccato":      [{"tag.aspects": ["percussive", "fast_decay"]}, True, "or"],
    "is_distorted":     [{"tag.aspects": ["distortion"]}, True, "or"],
    "is_warm":          [{"tag.aspects": ["dark"]}, True, "or"],

    # Source qualities - 3 total
    "is_acoustic":      [{"tag.aspects": ["acoustic"]}, True, "or"],
    "is_electronic":    [{"tag.aspects": ["electronic"]}, True, "or"],
    "is_synthetic":     [{"tag.aspects": ["synthetic"]}, True, "or"],

    # Higher-level concepts - 8 total
    "is_orchestral":    [{"tag.aspects": ["bass", "string", "brass", "reed",]}, True, "or"],
    "is_acoustic_band": [{"instrument_family_str": ["bass", "brass", "reed", "keyboard", "guitar",]}, True, "or"],
    "is_percussion":    [{"instrument_family_str": ["mallet", "percussive", "drum", "drums"]}, True, "or"],
    "is_techno":        [{"instrument_family_str": ["synth_lead", "mallet", "drum", "drums", "brass", "synthetic"]}, True, "or",],
    "is_plucked":       [{"tag.aspects": ["string", "bass", "guitar",]}, True, "or"],
    "is_blown":         [{"tag.aspects": ["brass", "flute", "reed"],}, True, "or"],
    "is_hit":           [{"tag.aspects": ["mallet", "keyboard", "synth_lead"]}, True, "or"],
    "is_atmospheric":   [{"tag.aspects": ["organ", "synth_lead", "vocal", "long_release", "reverb", "nonlinear_env"]}, True, "or"],
}

GEN_AUDIO_PADDER_DICT = {
    # Individual instrument families - 11 total
    "is_brass_like":        [{"tag.aspects": ["brass", "reed"]}, True, "or"],
    "is_bass_like":         [{"tag.aspects": ["bass", "string", "guitar"]}, True, "or"],
    "is_flute_like":        [{"tag.aspects": ["flute", "reed"]}, True, "or"],
    "is_guitar_like":       [{"tag.aspects": ["guitar", "bass", "string"]}, True, "or"],
    "is_mallet_like":       [{"tag.aspects": ["mallet", "keyboard"]}, True, "or"],
    "is_organ_like":        [{"tag.aspects": ["organ", "keyboard"]}, True, "or"],
    "is_reed_like":         [{"tag.aspects": ["reed", "brass", "flute"]}, True, "or"],
    "is_string_like":       [{"tag.aspects": ["string", "bass", "guitar"]}, True, "or"],
    "is_synth_lead_like":   [{"tag.aspects": ["synth_lead", "organ"]}, True, "or"],
    "is_keyboard_like":     [{"tag.aspects": ["keyboard", "organ"]}, True, "or"],
    "is_vocal_like":        [{"tag.aspects": ["vocal"]}, True, "or"],

    # Select note qualities - 6 total
    "is_long_like":          [{"tag.aspects": ["long_release", "reverb", "nonlinear_env"]}, True, "or"],
    "is_harmonic_like":      [{"tag.aspects": ["bright", "multiphonic"]}, True, "or"],
    "is_rhythmic_like":      [{"tag.aspects": ["tempo-synced", "percussive"]}, True, "or"],
    "is_staccato_like":      [{"tag.aspects": ["percussive", "fast_decay", "tempo-synced"]}, True, "or"],
    "is_distorted_like":     [{"tag.aspects": ["distortion", "nonlinear_env", "reverb"]}, True, "or"],
    "is_warm_like":          [{"tag.aspects": ["dark", "reverb"]}, True, "or"],

    # Source qualities - 3 total
    "is_acoustic_like":      [{"tag.aspects": ["acoustic"]}, True, "or"],
    "is_electronic_like":    [{"tag.aspects": ["electronic", "synthetic"]}, True, "or"],
    "is_synthetic_like":     [{"tag.aspects": ["synthetic", "electronic"]}, True, "or"],

    # Higher-level concepts - 8 total
    "is_orchestral_like":    [{"tag.aspects": ["bass", "string", "brass", "reed", "organ", "keyboard",]}, True, "or"],
    "is_acoustic_band_like": [{"tag.aspects": ["bass", "brass", "reed", "keyboard", "guitar", "string",]}, True, "or"],
    "is_percussion_like":    [{"tag.aspects": ["mallet", "keyboard", "percussive", "fast_decay"]}, True, "or"],
    "is_techno_like":        [{"tag.aspects": ["synth_lead", "mallet", "brass", "synthetic", "electronic"]}, True, "or",],
    "is_plucked_like":       [{"tag.aspects": ["string", "bass", "guitar", "keyboard"]}, True, "or"],
    "is_blown_like":         [{"tag.aspects": ["brass", "flute", "reed", "organ", "vocal"],}, True, "or"],
    "is_hit_like":           [{"tag.aspects": ["mallet", "keyboard", "synth_lead", "organ",]}, True, "or"],
    "is_atmospheric_like":   [{"tag.aspects": ["organ", "synth_lead", "vocal", "long_release", "reverb", "nonlinear_env"]}, True, "or"],
}























# # Loading in concept filters
# def load_concept_filter(func_name: str, concept_filter_path: Optional[str] = None,) -> Callable:
#         concept_filter_path = concept_filter_path if concept_filter_path is not None else "interp/concept_filters.py"
        
#         spec = importlib.util.spec_from_file_location(func_name, concept_filter_path)
#         func_file = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(func_file)
#         func = getattr(func_file, func_name, None)
#         if func == None:
#             raise AttributeError("Specified concept filter/padder not found in file")
#         else:
#             return func

# # Functions for processing NSynth
# def is_in_instrument_family(families: List[str], example_audio: Dict[str, Any]) -> bool:
#     return example_audio["instrument_family_str"] in families

# def has_quality(qualities: List[str], example_audio: Dict[str, Any]) -> bool:
#     for quality in qualities:
#         if quality in example_audio["qualities_str"]:
#             return True

# def has_source(sources: List[str], example_audio: Dict[str, Any]) -> bool:
#     return example_audio["instrument_source_str"] in sources

# # Explicit filters
# def is_brass(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(
#         ["brass"], example_audio
#     )

# def is_brass_like(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(
#         ["brass", "reed"], example_audio
#     )

# def is_string(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(
#         ["string"], example_audio
#     )

# def is_string_like(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(
#         ["string"], example_audio
#     )

# def is_drum(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(["mallet"], example_audio) or has_quality(["percussive"], example_audio)

# def is_drum_like(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(["mallet"], example_audio) or has_quality(["percussive"], example_audio)
        
# def is_blown(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(
#         ["brass", "flute", "reed", "vocal"], example_audio
#         )

# def is_blown_like(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(
#         ["brass", "flute", "reed", "vocal", "organ"], example_audio
#         )

# def is_electronic(example_audio: Dict[str, Any]) -> bool:
#     return has_source(["electronic"], example_audio)

# def is_electronic_like(example_audio: Dict[str, Any]) -> bool:
#     return has_source(["electronic", "synthetic"], example_audio)

# def is_orchestral(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(["string", "brass", "reed", "mallet"], example_audio) and has_source(["acoustic"], example_audio)

# def is_orchestral_like(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(["string", "brass", "reed", "mallet", "vocal"], example_audio) and has_source(["acoustic"], example_audio)

# def is_vocal(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(["vocal"], example_audio) 

# def is_vocal_like(example_audio: Dict[str, Any]) -> bool:
#     return is_in_instrument_family(["vocal"], example_audio)

# def is_distorted(example_audio: Dict[str, Any]) -> bool:
#     return has_quality(["distortion"], example_audio) 

# def is_distorted_like(example_audio: Dict[str, Any]) -> bool:
#     return has_quality(["distortion"], example_audio) 

# # Make this all streamlined in the future please
# # Functions for processing generated data
# def has_aspect(df: pd.DataFrame, aspects: List[str], get_true: bool, logic: str,) -> pd.DataFrame:
#     mask = pd.Series(False, index=df.index)

#     if logic == "or":
#         for aspect in aspects:
#             mask |= (df["tag.aspects"].str.contains(aspect, na=False))
#         mask = mask if get_true else ~mask
#         return df[mask]
    
#     elif logic == "and":
#         for aspect in aspects:
#             mask &= (df["tag.aspects"].str.contains(aspect, na=False))
#         mask = mask if get_true else ~mask
#         return df[mask]
    
#     else:
#         raise("Current logic not supported. Try 'and' or 'or' ")

# # Returns all audio that has brass in it
# def gen_audio_is_brass(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["brass",], get_true=True, logic="or")

# # Returns all audio that doesn't have brass, reed or flute aspects
# def gen_audio_not_is_brass_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["brass",], get_true=False, logic="or")

# def gen_audio_is_string(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["string",], get_true=True, logic="or")

# def gen_audio_not_is_string_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["string",], get_true=False, logic="or")

# def gen_audio_is_blown(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["brass", "flute", "reed", "vocal",], get_true=True, logic="or")

# def gen_audio_not_is_blown_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["brass", "flute", "reed", "vocal", "organ",], get_true=False, logic="or")

# def gen_audio_is_drum(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["mallet", "percussive", "drum", "drums",], get_true=True, logic="or")

# def gen_audio_not_is_drum_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["mallet", "percussive", "drum", "drums",], get_true=False, logic="or")

# def gen_audio_is_orchestral(df: pd.DataFrame) -> pd.DataFrame:
#     inst = has_aspect(df=df, aspects=["brass", "flute", "reed", "strings"], get_true=True, logic="or")
#     return inst
        
# def gen_audio_not_is_orchestral_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["brass", "flute", "reed", "strings", "vocal"], get_true=False, logic="or")

# def gen_audio_is_electronic(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["electronic"], get_true=True, logic="or")

# def gen_audio_not_is_electronic_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["electronic", "synthetic"], get_true=False, logic="or")

# def gen_audio_is_vocal(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["vocal",], get_true=True, logic="or")

# def gen_audio_not_is_vocal_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["vocal",], get_true=False, logic="or")

# def gen_audio_is_distortion(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["distortion",], get_true=True, logic="or")

# def gen_audio_not_is_distortion_like(df: pd.DataFrame) -> pd.DataFrame:
#     return has_aspect(df=df, aspects=["distortion",], get_true=False, logic="or")