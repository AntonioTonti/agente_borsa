#!/usr/bin/env python3
"""
Modulo Auto-Tuner (Apprendimento Adattivo).
Valuta la bontà dei segnali passati e aggiusta i pesi degli indicatori.
"""

import json
from typing import Dict, List
import pandas as pd
import numpy as np

from db_manager import (
    get_previsioni_da_verificare, 
    registra_verifica, 
    get_pesi_correnti, 
    salva_pesi_aggiornati
)

LEARNING_RATE = 0.05
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.40

def evaluate_and_tune(ticker: str, frame: str, current_price: float, default_weights: Dict[str, float]):
    """Confronta le previsioni pendenti con il prezzo corrente e aggiorna i pesi."""
    previsioni = get_previsioni_da_verificare(frame)
    if not previsioni:
        return

    pesi_attuali = get_pesi_correnti(ticker, frame, default_weights)
    
    for prev in previsioni:
        if prev['ticker'] != ticker:
            continue
            
        p_in = prev['prezzo_all_emissione']
        if p_in <= 0:
            continue
            
        rendimento_reale = ((current_price - p_in) / p_in) * 100.0
        direzione_reale = "BULLISH" if rendimento_reale > 0.15 else ("BEARISH" if rendimento_reale < -0.15 else "NEUTRAL")
        
        # L'esito è corretto se la direzione prevista coincide con quella reale
        esito_corretto = (prev['direzione'] == direzione_reale)
        
        # Salva la verifica a DB
        registra_verifica(prev['id'], current_price, rendimento_reale, esito_corretto)
        
        # De-serializza i segnali dei singoli indicatori emessi in quella previsione
        valori_ind = json.loads(prev['indicatori_valori_json'])
        
        # Aggiustamento dinamico dei pesi
        nuovi_pesi = {}
        for ind_name, weight in pesi_attuali.items():
            ind_signal = valori_ind.get(ind_name, 0.0) # >0 Bullish, <0 Bearish
            
            # Se l'indicatore spingeva nella direzione giusta, lo premiamo
            if (rendimento_reale > 0 and ind_signal > 0) or (rendimento_reale < 0 and ind_signal < 0):
                reward = LEARNING_RATE * abs(rendimento_reale)
                nuovi_pesi[ind_name] = weight + reward
            else:
                penalty = LEARNING_RATE * abs(rendimento_reale)
                nuovi_pesi[ind_name] = max(MIN_WEIGHT, weight - penalty)
        
        # Normalizzazione dei pesi affinché la somma sia 1.0 (100%)
        tot_weight = sum(nuovi_pesi.values())
        for k in nuovi_pesi:
            nuovi_pesi[k] = round(min(MAX_WEIGHT, nuovi_pesi[k] / tot_weight), 3)
            
        # Re-normalizzazione finale dopo i limiti max
        tot_final = sum(nuovi_pesi.values())
        for k in nuovi_pesi:
            nuovi_pesi[k] = round(nuovi_pesi[k] / tot_final, 3)

        salva_pesi_aggiornati(ticker, frame, nuovi_pesi)
        print(f"🎯 Auto-Tuning [{ticker} - {frame}]: Esito={'CORRETTO' if esito_corretto else 'ERRATO'} | Nuovi Pesi: {nuovi_pesi}")
