from typing import List, Callable, Any, Dict, Optional
import pandas as pd
import importlib.util
from pathlib import Path

# Loading in concept filters
def load_concept_filter(func_name: str, concept_filter_path: Optional[str] = None,) -> Callable:
        concept_filter_path = concept_filter_path if concept_filter_path is not None else "interp/concept_filters.py"
        
        spec = importlib.util.spec_from_file_location(func_name, concept_filter_path)
        func_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(func_file)
        func = getattr(func_file, func_name, None)
        if func == None:
            raise AttributeError("Specified concept filter/padder not found in file")
        else:
            return func

# Functions for processing NSynth
def is_in_instrument_family(families: List[str], example_audio: Dict[str, Any]) -> bool:
    return example_audio["instrument_family_str"] in families

def is_brass(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["brass"], example_audio
    )

def is_brass_like(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["brass", "reed"], example_audio
    )

def is_string(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["string"], example_audio
    )

def is_string_like(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["string"], example_audio
    )

def is_blown(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["brass", "flute", "reed", "vocal"], example_audio
        )

def is_blown_like(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["brass", "flute", "reed", "vocal", "organ"], example_audio
        )

# Make this all streamlined in the future please
# Functions for processing generated data
def has_aspect(df: pd.DataFrame, aspects: List[str], get_true: bool, logic: str,) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)

    if logic == "or":
        for aspect in aspects:
            mask |= (df["tag.aspects"].str.contains(aspect, na=False))
        mask = mask if get_true else ~mask
        return df[mask]
    
    elif logic == "and":
        for aspect in aspects:
            mask &= (df["tag.aspects"].str.contains(aspect, na=False))
        mask = mask if get_true else ~mask
        return df[mask]
    
    else:
        raise("Current logic not supported. Try 'and' or 'or' ")

# Returns all audio that has brass in it
def gen_audio_is_brass(df: pd.DataFrame) -> pd.DataFrame:
    return has_aspect(df=df, aspects=["brass",], get_true=True, logic="or")

# Returns all audio that doesn't have brass, reed or flute aspects
def gen_audio_not_is_brass_like(df: pd.DataFrame) -> pd.DataFrame:
    return has_aspect(df=df, aspects=["brass",], get_true=False, logic="or")

def gen_audio_is_string(df: pd.DataFrame) -> pd.DataFrame:
    return has_aspect(df=df, aspects=["string",], get_true=True, logic="or")

def gen_audio_not_is_string_like(df: pd.DataFrame) -> pd.DataFrame:
    return has_aspect(df=df, aspects=["string",], get_true=False, logic="or")

def gen_audio_is_blown(df: pd.DataFrame) -> pd.DataFrame:
    return has_aspect(df=df, aspects=["brass", "flute", "reed", "vocal",], get_true=True, logic="or")

def gen_audio_not_is_blown_like(df: pd.DataFrame) -> pd.DataFrame:
    return has_aspect(df=df, aspects=["brass", "flute", "reed", "vocal", "organ",], get_true=False, logic="or")