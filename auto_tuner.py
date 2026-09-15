#!/usr/bin/env python3
"""
Modulo Auto-Tuner (Apprendimento Adattivo).

Valuta la bontà delle previsioni verificate e aggiorna i pesi
degli indicatori per categoria (Portafoglio / Watchlist / ETF).

Regole:
- LEARNING_RATE = 0.02 (conservativo)
- MIN_WEIGHT = 0.03, MAX_WEIGHT = 0.40
- Reward: se l'indicatore puntava nella direzione giusta → peso += lr × |rendimento|
- Penalty: se puntava male → peso -= lr × |rendimento|
- Normalizzazione finale: i pesi sommano sempre a 1.0
"""

import json
from typing import Dict, List, Tuple

from db_manager import (
    get_previsioni_verificate_per_tuning,
    get_pesi_correnti,
    salva_pesi_aggiornati,
    DEFAULT_WEIGHTS,
)


LEARNING_RATE = 0.02
MIN_WEIGHT = 0.03
MAX_WEIGHT = 0.40


# ============================================================================
# UTILITY
# ============================================================================
def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalizza i pesi affinché sommino a 1.0. Applica min/max."""
    # Clipping
    clipped = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for k, v in weights.items()}

    # Normalizzazione
    tot = sum(clipped.values())
    if tot <= 0:
        return clipped

    normalized = {k: v / tot for k, v in clipped.items()}

    # Secondo passaggio di clipping dopo normalizzazione
    clipped2 = {k: max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for k, v in normalized.items()}

    # Ri-normalizza ancora (perché clipping può aver alterato la somma)
    tot2 = sum(clipped2.values())
    if tot2 <= 0:
        return clipped2

    return {k: round(v / tot2, 4) for k, v in clipped2.items()}


# ============================================================================
# AUTO-TUNER PRINCIPALE
# ============================================================================
def evaluate_and_tune(categoria: str, verbose: bool = True) -> bool:
    """
    Verifica le previsioni della categoria e aggiorna i pesi se ci sono
    nuove verifiche rispetto all'ultimo tuning.

    Ritorna True se i pesi sono stati aggiornati, False altrimenti.
    """
    previsioni = get_previsioni_verificate_per_tuning(categoria)

    if not previsioni:
        if verbose:
            print(f"   ⏭️  {categoria}: nessuna previsione verificata da valutare")
        return False

    if verbose:
        print(f"   🎯 {categoria}: {len(previsioni)} previsioni verificate da analizzare")

    # Pesi attuali (o default)
    pesi_attuali = get_pesi_correnti(categoria, DEFAULT_WEIGHTS)

    # Accumulatori per il reward/penalty di ogni indicatore
    delta_pesi: Dict[str, float] = {k: 0.0 for k in pesi_attuali.keys()}

    for prev in previsioni:
        rendimento_pct = prev.get('rendimento_pct', 0.0) or 0.0
        direzione_prevista = prev.get('direzione', 'NEUTRAL')

        if rendimento_pct == 0.0:
            # Nessun movimento → non è informativo, skip
            continue

        try:
            sub_scores = json.loads(prev.get('sub_scores_json', '{}'))
        except Exception:
            continue

        if not sub_scores:
            continue

        # Direzione reale: positiva se rendimento > 0, negativa altrimenti
        direzione_reale = 1 if rendimento_pct > 0 else -1

        # Contributo assoluto (magnitudine)
        reward_magnitude = LEARNING_RATE * abs(rendimento_pct) / 100.0  # normalizzato

        for ind, sub_score in sub_scores.items():
            if ind not in delta_pesi:
                continue

            # Il sub_score è tra 0 e 1. Lo convertiamo in "spinta":
            # - sub_score > 0.5 → spinge verso l'alto (bullish)
            # - sub_score < 0.5 → spinge verso il basso (bearish)
            spinta = (sub_score - 0.5) * 2.0  # range [-1, +1]

            # Se la spinta concorda con la direzione reale → reward
            # Altrimenti → penalty
            if spinta * direzione_reale > 0:
                delta_pesi[ind] += reward_magnitude
            else:
                delta_pesi[ind] -= reward_magnitude

    # Applica i delta
    nuovi_pesi: Dict[str, float] = {}
    for ind, peso in pesi_attuali.items():
        nuovi_pesi[ind] = peso + delta_pesi.get(ind, 0.0)

    # Normalizza e salva
    nuovi_pesi = _normalize_weights(nuovi_pesi)

    salva_pesi_aggiornati(categoria, nuovi_pesi)

    if verbose:
        print(f"   ✅ {categoria}: pesi aggiornati")
        for ind, p in sorted(nuovi_pesi.items(), key=lambda x: -x[1]):
            old = pesi_attuali.get(ind, 0)
            delta = p - old
            arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
            print(f"      {arrow} {ind:15s}: {old:.4f} → {p:.4f}")

    return True
