#!/usr/bin/env python3
import json
import os
from typing import Dict, Tuple

STATE_DIR = "data_state"

def _get_file_path(agent_type: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"state_{agent_type.lower()}.json")

def load_previous_state(agent_type: str) -> Dict[str, Dict[str, float]]:
    file_path = _get_file_path(agent_type)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Errore lettura stato precedente ({agent_type}): {e}")
    return {}

def save_current_state(agent_type: str, current_data: Dict[str, Dict[str, float]]) -> None:
    file_path = _get_file_path(agent_type)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)
    except Exception as e:
        print(f"❌ Errore salvataggio stato ({agent_type}): {e}")

def calculate_deltas(ticker: str, current_var: float, current_score: float, prev_state: Dict) -> Tuple[float, float]:
    prev = prev_state.get(ticker, {})
    prev_var = prev.get("var_pct", current_var)
    prev_score = prev.get("score", current_score)
    
    delta_var = current_var - prev_var
    delta_score = current_score - prev_score
    return round(delta_var, 2), round(delta_score, 3)
