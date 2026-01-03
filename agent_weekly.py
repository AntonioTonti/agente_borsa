#!/usr/bin/env python3
"""
Agente di Trading - Analisi Settimanale Completa
Invio: Venerdì 18:00 UTC (19:00 IT)
FEATURES:
- Analisi separata per Portafoglio e Watchlist
- Due invii Telegram distinti
- Gestione errori robusta
- Logging dettagliato
"""

import os
import sys
import time
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configurazione
sys.path.append('.')
from config import (
    load_titoli_csv, load_config, get_recommendation,
    WEEKLY_PERIOD, WEEKLY_INTERVAL, WEEKLY_MIN_POINTS
)

# ============================================================================
# ANALIZZATORE MEDIO-TERMINE
# ============================================================================

class MediumTermAnalyzer:
    """Analizzatore completo per medio termine (3-12 mesi)"""
    
    def __init__(self):
        self.thresholds = load_config()
        
        # Pesi indicatori (somma = 1.0)
        self.weights = {
            'ichimoku': 0.25,      # Trend primario
            'moving_averages': 0.20, # Conferma trend
            'momentum': 0.20,       # Forza movimento
            'volume': 0.15,         # Qualità trend
            'fibonacci': 0.10,      # Livelli tecnici
            'fundamental': 0.10     # Dati fondamentali
        }
        
        # Cache per dati già scaricati
        self.data_cache = {}
    
    def download_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Scarica dati settimanali con cache e gestione errori"""
        if ticker in self.data_cache:
            return self.data_cache[ticker]
        
        try:
            print(f"    📥 Download {ticker}...", end="", flush=True)
            df = yf.download(
                ticker, 
                period=WEEKLY_PERIOD, 
                interval=WEEKLY_INTERVAL, 
                progress=False, 
                timeout=60,
                threads=True
            )
            
            if df.empty:
                print(" ❌ VUOTO")
                return None
            
            # Log dettagliato
            print(f" ✅ {len(df)} righe")
            
            # Controllo flessibile sui dati minimi
            if len(df) < max(10, WEEKLY_MIN_POINTS // 2):
                print(f"    ⚠️  {ticker}: Dati limitati ({len(df)} righe)")
                # Continua comunque con i dati disponibili
            
            # Pulizia dati
            if isinstance(df.columns, pd.MultiIndex):
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            
            # Cache dei dati
            self.data_cache[ticker] = df
            return df
            
        except Exception as e:
            print(f" ❌ ERRORE: {str(e)[:80]}")
            return None
    
    def analyze_ichimoku(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi Ichimoku Cloud"""
        try:
            if len(df) < 52:
                return ("ICHIMOKU: Dati insufficienti", 0.5)
            
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # Calcolo componenti Ichimoku
            tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
            kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
            senkou_a = (tenkan + kijun) / 2
            senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
            
            # Posizione attuale
            price = float(close.iloc[-1])
            cloud_top = max(float(senkou_a.iloc[-26]), float(senkou_b.iloc[-26]))
            cloud_bottom = min(float(senkou_a.iloc[-26]), float(senkou_b.iloc[-26]))
            
            # Determinazione segnale
            if price > cloud_top:
                if tenkan.iloc[-1] > kijun.iloc[-1]:
                    return ("SOPRA CLOUD + TENKAN > KIJUN", 0.9)
                else:
                    return ("SOPRA CLOUD", 0.7)
            elif price < cloud_bottom:
                if tenkan.iloc[-1] < kijun.iloc[-1]:
                    return ("SOTTO CLOUD + TENKAN < KIJUN", 0.9)
                else:
                    return ("SOTTO CLOUD", 0.3)
            else:
                return ("DENTRO CLOUD", 0.5)
                
        except Exception as e:
            print(f"      ⚠️  Ichimoku error: {str(e)[:50]}")
            return ("ICHIMOKU: Errore analisi", 0.5)
    
    def analyze_moving_averages(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi medie mobili settimanali"""
        try:
            close = df['Close']
            if len(close) < 50:
                return ("MA: Dati insufficienti", 0.5)
            
            import ta
            
            ema21 = ta.trend.ema_indicator(close, window=21)
            sma50 = ta.trend.sma_indicator(close, window=50)
            
            if len(ema21) < 3 or len(sma50) < 3:
                return ("MA: Calcolo fallito", 0.5)
            
            ema_now = float(ema21.iloc[-1])
            sma_now = float(sma50.iloc[-1])
            ema_prev = float(ema21.iloc[-2])
            sma_prev = float(sma50.iloc[-2])
            
            # Distanza percentuale
            distance = ((ema_now - sma_now) / sma_now * 100)
            distance_score = min(0.8, abs(distance) / 20)
            
            # Segnale
            if ema_now > sma_now and ema_prev <= sma_prev:
                return (f"CROSSOVER BULLISH (+{distance:.1f}%)", 0.5 + distance_score)
            elif sma_now > ema_now and sma_prev <= ema_prev:
                return (f"CROSSOVER BEARISH ({distance:.1f}%)", 0.5 - distance_score)
            elif ema_now > sma_now:
                return (f"EMA21 > SMA50 (+{distance:.1f}%)", 0.5 + distance_score/2)
            else:
                return (f"SMA50 > EMA21 ({distance:.1f}%)", 0.5 - distance_score/2)
                
        except Exception as e:
            print(f"      ⚠️  MA error: {str(e)[:50]}")
            return ("MA: Errore analisi", 0.5)
    
    def analyze_momentum(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi momentum (RSI + MACD + ADX)"""
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            
            if len(close) < 20:
                return ("MOMENTUM: Dati insufficienti", 0.5)
            
            import ta
            
            # RSI settimanale
            rsi = ta.momentum.rsi(close, window=14)
            rsi_val = float(rsi.iloc[-1]) if len(rsi) > 0 else 50
            
            # MACD settimanale
            macd = ta.trend.MACD(close)
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            
            # ADX (forza trend)
            adx = ta.trend.adx(high, low, close, window=14)
            adx_val = float(adx.iloc[-1]) if len(adx) > 0 else 25
            
            # Calcolo score
            score = 0.5
            
            # RSI contributo
            if 40 < rsi_val < 60:
                score += 0.1  # Neutrale
            elif rsi_val > 60:
                score += 0.15 if rsi_val < 70 else 0.05
            elif rsi_val < 40:
                score -= 0.15 if rsi_val > 30 else 0.05
            
            # MACD contributo
            if len(macd_line) > 0 and len(signal_line) > 0:
                if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]):
                    score += 0.1
                else:
                    score -= 0.1
            
            # ADX contributo (trend forte = buono)
            if adx_val > 25:
                score += 0.05
            if adx_val > 40:
                score += 0.05
            
            # Normalizza
            score = max(0.1, min(0.9, score))
            
            if score > 0.6:
                return (f"MOMENTUM BULLISH (ADX:{adx_val:.0f})", score)
            elif score < 0.4:
                return (f"MOMENTUM BEARISH (ADX:{adx_val:.0f})", score)
            else:
                return (f"MOMENTUM NEUTRO (ADX:{adx_val:.0f})", score)
                
        except Exception as e:
            print(f"      ⚠️  Momentum error: {str(e)[:50]}")
            return ("MOMENTUM: Errore analisi", 0.5)
    
    def analyze_volume(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi volume e liquidità"""
        try:
            volume = df['Volume']
            close = df['Close']
            
            if len(volume) < 10:
                return ("VOLUME: Dati insufficienti", 0.5)
            
            # Volume medio ultime 10 settimane
            avg_volume = float(volume.tail(10).mean())
            if avg_volume == 0:
                return ("VOLUME: Media zero", 0.5)
                
            current_volume = float(volume.iloc[-1])
            volume_ratio = current_volume / avg_volume
            
            # Prezzo ultime 2 settimane
            if len(close) >= 2:
                price_change = ((float(close.iloc[-1]) - float(close.iloc[-2])) / 
                               float(close.iloc[-2]) * 100)
            else:
                price_change = 0
            
            # Valutazione
            if volume_ratio > 1.5 and price_change > 2:
                return (f"VOLUME FORTE +{price_change:.1f}%", 0.8)
            elif volume_ratio > 1.2 and price_change > 0:
                return (f"VOLUME BUONO +{price_change:.1f}%", 0.6)
            elif volume_ratio < 0.8 and price_change < -2:
                return (f"VOLUME DEBOLE {price_change:.1f}%", 0.3)
            elif volume_ratio < 0.6:
                return ("VOLUME MOLTO BASSO", 0.2)
            else:
                return ("VOLUME NORMALE", 0.5)
                
        except Exception as e:
            print(f"      ⚠️  Volume error: {str(e)[:50]}")
            return ("VOLUME: Errore analisi", 0.5)
    
    def analyze_fibonacci(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi livelli Fibonacci"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            if len(df) < 52:
                return ("FIBONACCI: Dati insufficienti", 0.5)
            
            # Massimo e minimo ultimo anno
            yearly_high = float(high.tail(52).max())
            yearly_low = float(low.tail(52).min())
            current = float(close.iloc[-1])
            
            # Range
            total_range = yearly_high - yearly_low
            if total_range == 0:
                return ("FIBONACCI: Range zero", 0.5)
            
            # Posizione corrente
            position = (current - yearly_low) / total_range
            
            # Livelli Fibonacci
            fib_levels = {
                0.236: "SUPPORTO FIB 23.6%",
                0.382: "SUPPORTO FIB 38.2%",
                0.5: "MEZZO RANGE",
                0.618: "RESISTENZA FIB 61.8%",
                0.786: "RESISTENZA FIB 78.6%"
            }
            
            # Trova livello più vicino
            closest_level = min(fib_levels.keys(), key=lambda x: abs(x - position))
            distance = abs(position - closest_level)
            
            # Score basato su vicinanza a livello
            score = 0.5
            if distance < 0.05:  # Molto vicino a livello
                if closest_level >= 0.618:
                    score = 0.3  # Vicino a resistenza
                elif closest_level <= 0.382:
                    score = 0.7  # Vicino a supporto
            
            desc = fib_levels.get(closest_level, f"FIB {closest_level*100:.1f}%")
            return (f"{desc} ({position*100:.1f}%)", score)
            
        except Exception as e:
            print(f"      ⚠️  Fibonacci error: {str(e)[:50]}")
            return ("FIBONACCI: Errore analisi", 0.5)
    
    def analyze_fundamental(self, ticker: str) -> Tuple[str, float]:
        """Analisi dati fondamentali (semplificata)"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            score = 0.5
            
            # P/E Ratio
            pe_ratio = info.get('trailingPE', 0)
            if pe_ratio:
                if 10 < pe_ratio < 20:
                    score += 0.1
                elif pe_ratio > 30:
                    score -= 0.1
            
            # Dividend Yield
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield and dividend_yield > 0.02:
                score += 0.1
            
            # Market Cap
            market_cap = info.get('marketCap', 0)
            if market_cap > 1e9:
                score += 0.05
            
            # Profit Margins
            profit_margins = info.get('profitMargins', 0)
            if profit_margins and profit_margins > 0.1:
                score += 0.1
            
            # Debt to Equity
            debt_to_equity = info.get('debtToEquity', 0)
            if debt_to_equity and debt_to_equity < 1.0:
                score += 0.05
            
            score = max(0.1, min(0.9, score))
            
            if score > 0.6:
                return ("FONDAMENTALI SOLIDI", score)
            elif score < 0.4:
                return ("FONDAMENTALI DEBOLI", score)
            else:
                return ("FONDAMENTALI MEDI", score)
                
        except Exception as e:
            print(f"      ⚠️  Fundamental error: {str(e)[:50]}")
            return ("FONDAMENTALI: Dati non disponibili", 0.5)
    
    def analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """Analisi completa di un ticker"""
        try:
            df = self.download_data(ticker)
            if df is None or len(df) < 10:
                print(f"    ⚠️  {ticker}: Dati insufficienti per analisi")
                return None
            
            # Analisi tutti gli indicatori
            indicators = {
                'ichimoku': self.analyze_ichimoku(df),
                'moving_averages': self.analyze_moving_averages(df),
                'momentum': self.analyze_momentum(df),
                'volume': self.analyze_volume(df),
                'fibonacci': self.analyze_fibonacci(df),
                'fundamental': self.analyze_fundamental(ticker)
            }
            
            # Calcola score totale pesato
            total_score = 0
            total_weight = 0
            
            for name, (desc, score) in indicators.items():
                weight = self.weights.get(name, 0)
                total_score += score * weight
                total_weight += weight
            
            final_score = total_score / total_weight if total_weight > 0 else 0.5
            
            # Raccomandazione
            recommendation, rec_type = get_recommendation(final_score, self.thresholds)
            
            return {
                'ticker': ticker,
                'score': round(final_score, 3),
                'recommendation': recommendation,
                'rec_type': rec_type,
                'indicators': indicators,
                'data_points': len(df)
            }
            
        except Exception as e:
            print(f"    ❌ {ticker}: Errore analisi - {str(e)[:100]}")
            return None

# ============================================================================
# FUNZIONI DI FORMATTAZIONE REPORT
# ============================================================================

def create_portfolio_report(results: List[Dict], descriptions: Dict) -> str:
    """Crea report completo per portafoglio"""
    if not results:
        return "📭 *NESSUN TITOLO NEL PORTAFOGLIO ANALIZZATO*"
    
    # Header
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    header = f"📊 *REPORT SETTIMANALE - PORTAFOGLIO*\n"
    header += f"Data: {timestamp}\n"
    header += "=" * 40 + "\n"
    
    # Statistiche
    stats = []
    if results:
        avg_score = sum(r['score'] for r in results) / len(results)
        bearish = sum(1 for r in results if r['score'] < 0.4)
        neutral = sum(1 for r in results if 0.4 <= r['score'] <= 0.6)
        bullish = sum(1 for r in results if r['score'] > 0.6)
        
        stats.append(f"📈 *STATISTICHE PORTAFOGLIO*\n")
        stats.append(f"• Titoli analizzati: {len(results)}\n")
        stats.append(f"• Score medio: {avg_score:.3f}\n")
        stats.append(f"• 🔴 Allerta: {bearish} titoli\n")
        stats.append(f"• ⚪ Neutri: {neutral} titoli\n")
        stats.append(f"• 🟢 Opportunità: {bullish} titoli\n")
    
    # Analisi titoli (dal PEGGIORE al MIGLIORE)
    sorted_results = sorted(results, key=lambda x: x['score'])
    
    analysis_lines = []
    analysis_lines.append(f"\n💰 *ANALISI DETTAGLIATA* (dal peggiore)")
    analysis_lines.append("-" * 40)
    
    for result in sorted_results:
        ticker = result['ticker']
        score = result['score']
        recommendation = result['recommendation']
        desc = descriptions.get(ticker, ticker)
        
        analysis_lines.append(f"\n*{ticker}* - {desc}")
        analysis_lines.append(f"Score: *{score:.3f}* | {recommendation}")
        
        # Indicatori dettagliati
        for ind_name, (ind_desc, ind_score) in result['indicators'].items():
            # Formatta l'emoji in base allo score
            if ind_score >= 0.7:
                emoji = "🟢"
            elif ind_score <= 0.3:
                emoji = "🔴"
            else:
                emoji = "⚪"
            
            analysis_lines.append(f"  {emoji} {ind_desc} ({ind_score:.0%})")
    
    # Footer
    footer = "\n" + "=" * 40 + "\n"
    footer += "*LEGENDA RACCOMANDAZIONI:*\n"
    footer += "🔴🔴 VENDI SUBITO (score < 0.25)\n"
    footer += "🔴 CONSIGLIA VENDITA (score 0.25-0.35)\n"
    footer += "🟡 MONITORA ATTIVAMENTE (score 0.35-0.45)\n"
    footer += "⚪ MANTIENI POSIZIONE (score 0.45-0.55)\n"
    footer += "🟢 CONSIGLIA ACQUISTO (score 0.55-0.65)\n"
    footer += "🟢🟢 FORTE ACQUISTO (score > 0.65)\n"
    footer += "\n_Periodo dati: 1 anno | Intervallo: settimanale_"
    
    # Combina tutto
    message = header + "\n".join(stats) + "\n".join(analysis_lines) + footer
    
    return message

def create_watchlist_report(results: List[Dict], descriptions: Dict) -> str:
    """Crea report completo per watchlist"""
    if not results:
        return "👁️  *NESSUN TITOLO IN WATCHLIST ANALIZZATO*"
    
    # Header
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    header = f"📊 *REPORT SETTIMANALE - WATCHLIST*\n"
    header += f"Data: {timestamp}\n"
    header += "=" * 40 + "\n"
    
    # Statistiche
    stats = []
    if results:
        avg_score = sum(r['score'] for r in results) / len(results)
        bearish = sum(1 for r in results if r['score'] < 0.4)
        neutral = sum(1 for r in results if 0.4 <= r['score'] <= 0.6)
        bullish = sum(1 for r in results if r['score'] > 0.6)
        
        stats.append(f"📈 *STATISTICHE WATCHLIST*\n")
        stats.append(f"• Titoli monitorati: {len(results)}\n")
        stats.append(f"• Score medio: {avg_score:.3f}\n")
        stats.append(f"• 🔴 Attenzione: {bearish} titoli\n")
        stats.append(f"• ⚪ Neutri: {neutral} titoli\n")
        stats.append(f"• 🟢 Opportunità: {bullish} titoli\n")
    
    # Analisi titoli (dal PEGGIORE al MIGLIORE)
    sorted_results = sorted(results, key=lambda x: x['score'])
    
    analysis_lines = []
    analysis_lines.append(f"\n👁️  *ANALISI DETTAGLIATA* (dal peggiore)")
    analysis_lines.append("-" * 40)
    
    for result in sorted_results:
        ticker = result['ticker']
        score = result['score']
        recommendation = result['recommendation']
        desc = descriptions.get(ticker, ticker)
        
        analysis_lines.append(f"\n*{ticker}* - {desc}")
        analysis_lines.append(f"Score: *{score:.3f}* | {recommendation}")
        
        # Solo indicatori chiave per watchlist (3 principali)
        indicators = result['indicators']
        key_indicators = [
            ('ichimoku', 'Trend'),
            ('moving_averages', 'Medie Mobili'),
            ('fundamental', 'Fondamentali')
        ]
        
        for ind_name, ind_label in key_indicators:
            if ind_name in indicators:
                ind_desc, ind_score = indicators[ind_name]
                # Formatta l'emoji in base allo score
                if ind_score >= 0.7:
                    emoji = "🟢"
                elif ind_score <= 0.3:
                    emoji = "🔴"
                else:
                    emoji = "⚪"
                
                analysis_lines.append(f"  {emoji} {ind_desc.split(' ')[0]} ({ind_score:.0%})")
    
    # Footer
    footer = "\n" + "=" * 40 + "\n"
    footer += "*SCORE INTERPRETATION:*\n"
    footer += "🟢 > 0.65: Forte opportunità\n"
    footer += "🟢 0.55-0.65: Opportunità\n"
    footer += "⚪ 0.45-0.55: Neutrale\n"
    footer += "🟡 0.35-0.45: Monitorare\n"
    footer += "🔴 < 0.35: Attenzione\n"
    footer += "\n_Titoli da monitorare per possibili ingressi_"
    
    # Combina tutto
    message = header + "\n".join(stats) + "\n".join(analysis_lines) + footer
    
    return message

# ============================================================================
# FUNZIONI DI INVIO TELEGRAM
# ============================================================================

def send_telegram_message_safe(token: str, chat_id: str, message: str, 
                               message_type: str = "Report") -> bool:
    """
    Invia un messaggio a Telegram con gestione errori robusta
    """
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        # Limite Telegram
        MAX_LENGTH = 4096
        
        print(f"  📤 Preparazione {message_type}...")
        print(f"    Lunghezza: {len(message)} caratteri")
        
        # Se il messaggio è troppo lungo, dividilo
        if len(message) > MAX_LENGTH:
            print(f"    ⚠️  Messaggio troppo lungo, divido in parti...")
            
            # Strategia di divisione intelligente
            parts = []
            sections = message.split('\n💰 *ANALISI DETTAGLIATA*')
            
            if len(sections) > 1:
                # Parte 1: Header + Statistiche
                part1 = sections[0]
                if len(part1) <= MAX_LENGTH:
                    parts.append(part1)
                else:
                    # Dividi ulteriormente
                    lines1 = part1.split('\n')
                    current_part = []
                    current_len = 0
                    
                    for line in lines1:
                        if current_len + len(line) + 1 > MAX_LENGTH:
                            parts.append('\n'.join(current_part))
                            current_part = [line]
                            current_len = len(line)
                        else:
                            current_part.append(line)
                            current_len += len(line) + 1
                    
                    if current_part:
                        parts.append('\n'.join(current_part))
                
                # Parte 2: Analisi dettagliata
                part2 = '💰 *ANALISI DETTAGLIATA*' + sections[1]
                parts.append(part2)
            else:
                # Divisione semplice per righe
                lines = message.split('\n')
                current_part = []
                current_length = 0
                
                for line in lines:
                    if current_length + len(line) + 1 > MAX_LENGTH:
                        parts.append('\n'.join(current_part))
                        current_part = [line]
                        current_length = len(line)
                    else:
                        current_part.append(line)
                        current_length += len(line) + 1
                
                if current_part:
                    parts.append('\n'.join(current_part))
            
            print(f"    Diviso in {len(parts)} parti")
        else:
            parts = [message]
        
        # Invia tutte le parti
        success_count = 0
        for i, part in enumerate(parts):
            try:
                payload = {
                    "chat_id": chat_id,
                    "text": part,
                    "parse_mode": "Markdown" if i == 0 else None,
                    "disable_web_page_preview": True,
                    "disable_notification": (i > 0)  # Solo prima parte fa notifica
                }
                
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"    ✅ Parte {i+1}/{len(parts)} inviata")
                else:
                    print(f"    ❌ Parte {i+1}: Errore {response.status_code}")
                    print(f"      Response: {response.text[:200]}")
                
                # Pausa tra le parti
                if i < len(parts) - 1:
                    time.sleep(1)
                    
            except requests.exceptions.Timeout:
                print(f"    ⏱️  Timeout parte {i+1}")
            except Exception as e:
                print(f"    ❌ Errore parte {i+1}: {str(e)[:100]}")
        
        return success_count == len(parts)
        
    except Exception as e:
        print(f"❌ Errore critico invio {message_type}: {e}")
        return False

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """Funzione principale con gestione errori completa"""
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("📊 AGENTE DI TRADING - ANALISI SETTIMANALE")
        print(f"Avvio: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("=" * 60)
        
        # 1. Caricamento configurazione
        print("\n📁 CARICAMENTO CONFIGURAZIONE")
        print("-" * 40)
        
        portfolio, watchlist, descriptions = load_titoli_csv()
        print(f"✅ Titoli caricati:")
        print(f"   • Portafoglio: {len(portfolio)} titoli")
        print(f"   • Watchlist: {len(watchlist)} titoli")
        
        if not portfolio and not watchlist:
            print("❌ Nessun titolo da analizzare")
            return
        
        # 2. Inizializzazione analizzatore
        print("\n🔧 INIZIALIZZAZIONE ANALIZZATORE")
        print("-" * 40)
        analyzer = MediumTermAnalyzer()
        
        # 3. Analisi Portafoglio
        portfolio_results = []
        if portfolio:
            print(f"\n💰 ANALISI PORTAFOGLIO")
            print("-" * 40)
            
            for i, ticker in enumerate(portfolio, 1):
                print(f"[{i}/{len(portfolio)}] {ticker}...", end="", flush=True)
                result = analyzer.analyze_ticker(ticker)
                
                if result:
                    portfolio_results.append(result)
                    print(f" ✅ Score: {result['score']:.3f}")
                else:
                    print(f" ❌ Fallita")
        
        # 4. Analisi Watchlist
        watchlist_results = []
        if watchlist:
            print(f"\n👁️  ANALISI WATCHLIST")
            print("-" * 40)
            
            for i, ticker in enumerate(watchlist, 1):
                print(f"[{i}/{len(watchlist)}] {ticker}...", end="", flush=True)
                result = analyzer.analyze_ticker(ticker)
                
                if result:
                    watchlist_results.append(result)
                    print(f" ✅ Score: {result['score']:.3f}")
                else:
                    print(f" ❌ Fallita")
        
        # 5. Verifica risultati
        print("\n📊 RIEPILOGO RISULTATI")
        print("-" * 40)
        print(f"Portafoglio analizzati: {len(portfolio_results)}/{len(portfolio)}")
        print(f"Watchlist analizzati: {len(watchlist_results)}/{len(watchlist)}")
        
        if not portfolio_results and not watchlist_results:
            print("❌ Nessun risultato valido da inviare")
            return
        
        # 6. Invio Telegram
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
            print("⚠️  Credenziali Telegram non configurate")
            print("   TELEGRAM_BOT_TOKEN:", "✅" if token else "❌")
            print("   TELEGRAM_CHAT_ID:", "✅" if chat_id else "❌")
            return
        
        print("\n📤 INVIO REPORT TELEGRAM")
        print("-" * 40)
        
        # INVIO 1: PORTAFOGLIO
        if portfolio_results:
            print("\n1️⃣  INVIO PORTAFOGLIO")
            portfolio_message = create_portfolio_report(portfolio_results, descriptions)
            
            success = send_telegram_message_safe(
                token, chat_id, portfolio_message, "Portafoglio"
            )
            
            if success:
                print("✅ Report Portafoglio inviato con successo!")
            else:
                print("❌ Invio Portafoglio parzialmente fallito")
            
            # Pausa tra i due invii
            time.sleep(3)
        
        # INVIO 2: WATCHLIST
        if watchlist_results:
            print("\n2️⃣  INVIO WATCHLIST")
            watchlist_message = create_watchlist_report(watchlist_results, descriptions)
            
            success = send_telegram_message_safe(
                token, chat_id, watchlist_message, "Watchlist"
            )
            
            if success:
                print("✅ Report Watchlist inviato con successo!")
            else:
                print("❌ Invio Watchlist parzialmente fallito")
        
        # 7. Statistiche finali
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🏁 ANALISI COMPLETATA")
        print("-" * 40)
        print(f"Tempo impiegato: {elapsed_time:.1f} secondi")
        print(f"Titoli totali analizzati: {len(portfolio_results) + len(watchlist_results)}")
        print(f"Data cache: {len(analyzer.data_cache)} ticker")
        print(f"Ora completamento: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  INTERROTTO DALL'UTENTE")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERRORE CRITICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# ESECUZIONE
# ============================================================================

if __name__ == "__main__":
    main()
