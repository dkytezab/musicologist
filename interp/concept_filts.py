from typing import List, Callable, Any, Dict
from models import BinaryClassifier

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

def is_blown(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["brass", "flute", "reed", "vocal"], example_audio
        )

def is_blown_like(example_audio: Dict[str, Any]) -> bool:
    return is_in_instrument_family(
        ["brass", "flute", "reed", "vocal", "organ", "vocal"], example_audio
        )