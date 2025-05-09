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
    "is_atmospheric":   [{"instrument_family_str": ["organ", "synth_lead", "vocal"],
                         "qualities_str": ["long_release", "reverb", "nonlinear_env"]}, True, "and"],
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
    "is_atmospheric_like":   [{"instrument_family_str": ["organ", "synth_lead", "vocal"],
                              "qualities_str": ["long_release", "reverb", "nonlinear_env"]}, True, "and"],
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
    "is_acoustic_band": [{"tag.aspects": ["bass", "brass", "reed", "keyboard", "guitar",]}, True, "or"],
    "is_percussion":    [{"tag.aspects": ["mallet", "percussive", "drum", "drums"]}, True, "or"],
    "is_techno":        [{"tag.aspects": ["synth_lead", "mallet", "drum", "drums", "brass", "synthetic"]}, True, "or",],
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