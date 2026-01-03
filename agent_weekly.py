#!/usr/bin/env python3
"""
Agente di Trading - Analisi Settimanale Completa
Invio: Venerdì 18:00 UTC (19:00 IT)
"""

import os
import sys
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
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
    """Analizzatore completo per medio termine (3-12 meses)"""
    
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
    
    def download_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Scarica dati settimanali"""
        try:
            df = yf.download(ticker, period=WEEKLY_PERIOD, 
                            interval=WEEKLY_INTERVAL, progress=False, timeout=30)
            
            if df.empty:
                print(f"    ⚠️  {ticker}: DataFrame vuoto")
                return None
            
            # Controllo più flessibile
            if len(df) < WEEKLY_MIN_POINTS:
                print(f"    ⚠️  {ticker}: Solo {len(df)} righe (minimo: {WEEKLY_MIN_POINTS})")
                # Potresti decidere di ritornare comunque df con un warning
                # return df  # ← prova questo se il problema persiste
            
            # Pulizia
            if isinstance(df.columns, pd.MultiIndex):
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            
            return df
        except Exception as e:
            print(f"    ❌ {ticker}: Errore download - {str(e)[:100]}")
            return None
    
    def analyze_ichimoku(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi Ichimoku Cloud"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # Calcolo componenti Ichimoku (semplificato)
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
                
        except:
            return ("ICHIMOKU N/A", 0.5)
    
    def analyze_moving_averages(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi medie mobili settimanali"""
        try:
            close = df['Close']
            import ta
            
            # Medie chiave per medio termine
            ema21 = ta.trend.ema_indicator(close, window=21)  # ~5 meses
            sma50 = ta.trend.sma_indicator(close, window=50)  # ~1 anno
            
            if len(ema21) < 3 or len(sma50) < 3:
                return ("MA INSUFFICIENTI", 0.5)
            
            ema_now = float(ema21.iloc[-1])
            sma_now = float(sma50.iloc[-1])
            ema_prev = float(ema21.iloc[-2])
            sma_prev = float(sma50.iloc[-2])
            
            # Distanza percentuale
            distance = abs((ema_now - sma_now) / sma_now * 100)
            distance_score = min(0.8, distance / 20)
            
            # Segnale
            if ema_now > sma_now and ema_prev <= sma_prev:
                return (f"CROSSOVER BULLISH (+{distance:.1f}%)", 0.5 + distance_score)
            elif sma_now > ema_now and sma_prev <= ema_prev:
                return (f"CROSSOVER BEARISH (-{distance:.1f}%)", 0.5 - distance_score)
            elif ema_now > sma_now:
                return (f"EMA21 > SMA50 (+{distance:.1f}%)", 0.5 + distance_score/2)
            else:
                return (f"SMA50 > EMA21 (-{distance:.1f}%)", 0.5 - distance_score/2)
                
        except:
            return ("MA ERROR", 0.5)
    
    def analyze_momentum(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi momentum (RSI + MACD + ADX)"""
        try:
            close = df['Close']
            import ta
            
            # RSI settimanale
            rsi = ta.momentum.rsi(close, window=14)
            rsi_val = float(rsi.iloc[-1]) if len(rsi) > 0 else 50
            
            # MACD settimanale
            macd = ta.trend.MACD(close)
            macd_line = macd.macd()
            signal_line = macd.macd_signal()
            
            # ADX (forza trend)
            adx = ta.trend.adx(df['High'], df['Low'], close, window=14)
            adx_val = float(adx.iloc[-1]) if len(adx) > 0 else 25
            
            # Calcolo score
            score = 0.5
            
            # RSI contributo
            if 40 < rsi_val < 60:
                score += 0.1  # Neutrale
            elif rsi_val > 60:
                score += 0.15 if rsi_val < 70 else 0.05
            else:
                score -= 0.15 if rsi_val > 30 else 0.05
            
            # MACD contributo
            if len(macd_line) > 0 and len(signal_line) > 0:
                if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]):
                    score += 0.1
                else:
                    score -= 0.1
            
            # ADX contributo
            if adx_val > 25:
                score += 0.05
            if adx_val > 40:
                score += 0.05
            
            # Normalizza e descrivi
            score = max(0.1, min(0.9, score))
            
            if score > 0.6:
                return (f"MOMENTUM BULLISH (ADX:{adx_val:.0f})", score)
            elif score < 0.4:
                return (f"MOMENTUM BEARISH (ADX:{adx_val:.0f})", score)
            else:
                return (f"MOMENTUM NEUTRO (ADX:{adx_val:.0f})", score)
                
        except:
            return ("MOMENTUM ERROR", 0.5)
    
    def analyze_volume(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi volume e liquidità"""
        try:
            volume = df['Volume']
            close = df['Close']
            
            if len(volume) < 10:
                return ("VOLUME INSUFF.", 0.5)
            
            # Volume medio ultime 10 settimane
            avg_volume = float(volume.tail(10).mean())
            current_volume = float(volume.iloc[-1])
            volume_ratio = current_volume / avg_volume
            
            # Prezzo ultime 2 settimane
            price_change = ((float(close.iloc[-1]) - float(close.iloc[-2])) / 
                           float(close.iloc[-2]) * 100)
            
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
                
        except:
            return ("VOLUME ERROR", 0.5)
    
    def analyze_fibonacci(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Analisi livelli Fibonacci"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            if len(df) < 52:
                return ("FIB INSUFF.", 0.5)
            
            # Massimo e minimo ultimo anno
            yearly_high = float(high.tail(52).max())
            yearly_low = float(low.tail(52).min())
            current = float(close.iloc[-1])
            
            # Range
            total_range = yearly_high - yearly_low
            if total_range == 0:
                return ("FIB NO RANGE", 0.5)
            
            # Posizione corrente
            position = (current - yearly_low) / total_range
            
            # Livelli Fibonacci chiave
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
            
        except:
            return ("FIB ERROR", 0.5)
    
    def analyze_fundamental(self, ticker: str) -> Tuple[str, float]:
        """Analisi dati fondamentali (semplificata)"""
        try:
            # Usa yfinance per dati fondamentali base
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
            if dividend_yield and dividend_yield > 0.02:  # > 2%
                score += 0.1
            
            # Market Cap
            market_cap = info.get('marketCap', 0)
            if market_cap > 1e9:  # > 1 miliardo
                score += 0.05
            
            # Profit Margins
            profit_margins = info.get('profitMargins', 0)
            if profit_margins and profit_margins > 0.1:  # > 10%
                score += 0.1
            
            score = max(0.1, min(0.9, score))
            
            if score > 0.6:
                return ("FONDAMENTALI SOLIDI", score)
            elif score < 0.4:
                return ("FONDAMENTALI DEBOLI", score)
            else:
                return ("FONDAMENTALI MEDI", score)
                
        except:
            return ("FONDAMENTALI N/D", 0.5)
    
    def analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """Analisi completa di un ticker"""
        df = self.download_data(ticker)
        if df is None:
            print(f"  ⚠️  {ticker}: Dati insufficienti o download fallito")
            return None
        
        print(f"  ✅ {ticker}: {len(df)} righe scaricate")
        
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
            'indicators': indicators
        }

# ============================================================================
# FUNZIONI DI OUTPUT
# ============================================================================

def format_weekly_analysis(results: List[Dict], group_name: str, descriptions: Dict) -> str:
    """Formatta analisi per un gruppo"""
    if not results:
        return f"\n{group_name}\n" + "-" * 40 + "\n📭 Nessun titolo analizzato\n"
    
    # Ordina dal PEGGIORE al MIGLIORE (score crescente)
    sorted_results = sorted(results, key=lambda x: x['score'])
    
    lines = []
    lines.append(f"\n{group_name}")
    lines.append("-" * 40)
    
    for result in sorted_results:
        ticker = result['ticker']
        score = result['score']
        recommendation = result['recommendation']
        desc = descriptions.get(ticker, f"{ticker} (descrizione non disponibile)")
        
        lines.append(f"\n{ticker} - {desc}")
        lines.append(f"Score: {score:.3f} | {recommendation}")
        
        # Indicatori dettagliati (opzionale)
        for ind_name, (ind_desc, ind_score) in result['indicators'].items():
            lines.append(f"  • {ind_desc} ({ind_score:.1%})")
    
    return "\n".join(lines)

def send_telegram_report(token: str, chat_id: str, message: str):
    """Invia report settimanale"""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        
        # Telegram ha limite di 4096 caratteri per messaggio
        MAX_LENGTH = 4096
        
        # Dividi il messaggio in parti più piccole
        def split_message(msg: str, max_len: int = MAX_LENGTH) -> List[str]:
            """Divide intelligentemente il messaggio"""
            if len(msg) <= max_len:
                return [msg]
            
            parts = []
            lines = msg.split('\n')
            current_part = []
            current_length = 0
            
            for line in lines:
                line_length = len(line) + 1  # +1 per il newline
                
                if current_length + line_length > max_len:
                    # Se la parte corrente ha contenuto, salvala
                    if current_part:
                        parts.append('\n'.join(current_part))
                    
                    # Se anche la singola linea è troppo lunga, dividila
                    if line_length > max_len:
                        # Dividi la linea lunga
                        for i in range(0, len(line), max_len - 100):
                            parts.append(line[i:i + max_len - 100])
                        current_part = []
                        current_length = 0
                    else:
                        current_part = [line]
                        current_length = line_length
                else:
                    current_part.append(line)
                    current_length += line_length
            
            # Aggiungi l'ultima parte
            if current_part:
                parts.append('\n'.join(current_part))
            
            return parts
        
        # Dividi il messaggio
        parts = split_message(message)
        
        print(f"📤 Invio {len(parts)} parti a Telegram...")
        
        # Invia tutte le parti
        for i, part in enumerate(parts):
            print(f"  Parte {i+1}/{len(parts)}: {len(part)} caratteri")
            
            payload = {
                "chat_id": chat_id,
                "text": part,
                "parse_mode": "Markdown" if i == 0 else None,  # Solo la prima parte ha markdown
                "disable_web_page_preview": True
            }
            
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 200:
                    print(f"    ✅ Parte {i+1} inviata")
                else:
                    print(f"    ❌ Parte {i+1}: errore {resp.status_code}")
                    print(f"    Response: {resp.text[:200]}")
            except Exception as e:
                print(f"    ❌ Errore invio parte {i+1}: {e}")
            
            # Piccola pausa tra i messaggi per evitare rate limiting
            if i < len(parts) - 1:
                import time
                time.sleep(1)
        
        return True
        
    except Exception as e:
        print(f"❌ Errore invio Telegram: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("📊 ANALISI SETTIMANALE MEDIO-TERMINE")
    print(f"Data: {datetime.now().strftime('%d/%m/%Y')}")
    print("=" * 60)
    
    # Carica titoli da CSV
    portfolio, watchlist, descriptions = load_titoli_csv()
    
    print(f"💰 Portafoglio: {len(portfolio)} titoli")
    print(f"👁️  Watchlist: {len(watchlist)} titoli")
    
    # Analisi
    analyzer = MediumTermAnalyzer()
    
    portfolio_results = []
    watchlist_results = []
    
    # Analizza portafoglio
    print("\n🔍 Analisi portafoglio...")
    for ticker in portfolio:
        print(f"  {ticker}...", end="", flush=True)
        result = analyzer.analyze_ticker(ticker)
        if result:
            portfolio_results.append(result)
            print(f" ✓ (Score: {result['score']:.3f})")
        else:
            print(f" ✗ (Dati insufficienti)")
    
    # Analizza watchlist
    print("\n🔍 Analisi watchlist...")
    for ticker in watchlist:
        print(f"  {ticker}...", end="", flush=True)
        result = analyzer.analyze_ticker(ticker)
        if result:
            watchlist_results.append(result)
            print(f" ✓ (Score: {result['score']:.3f})")
        else:
            print(f" ✗ (Dati insufficienti)")
    
    # Prepara report
    header = f"📈 *REPORT SETTIMANALE - {datetime.now().strftime('%d/%m/%Y')}*\n"
    header += "Analisi Medio-Termine (3-12 mesi)\n"
    header += f"Periodo dati: {WEEKLY_PERIOD} | Intervallo: {WEEKLY_INTERVAL}\n"
    
    # Statistiche
    stats = []
    if portfolio_results:
        avg_score = sum(r['score'] for r in portfolio_results) / len(portfolio_results)
        bearish = sum(1 for r in portfolio_results if r['score'] < 0.4)
        stats.append(f"📊 Portafoglio: Score medio {avg_score:.3f} | Allerta: {bearish}/{len(portfolio_results)}")
    
    if watchlist_results:
        avg_score = sum(r['score'] for r in watchlist_results) / len(watchlist_results)
        bullish = sum(1 for r in watchlist_results if r['score'] > 0.6)
        stats.append(f"📊 Watchlist: Score medio {avg_score:.3f} | Opportunità: {bullish}/{len(watchlist_results)}")
    
    # Costruisci messaggio
    message = header + "\n" + "\n".join(stats) + "\n"
    
    # Aggiungi analisi portafoglio (PEGGIORI prima)
    message += format_weekly_analysis(portfolio_results, "💰 PORTAFOGLIO ATTIVO (dal peggiore)", descriptions)
    
    # Aggiungi analisi watchlist (PEGGIORI prima)
    message += format_weekly_analysis(watchlist_results, "\n👁️  WATCHLIST (dal peggiore)", descriptions)
    
    # Footer
    message += "\n\n" + "=" * 40
    message += "\n📋 LEGENDA RACCOMANDAZIONI:\n"
    message += "🔴🔴 VENDI SUBITO (score < 0.25)\n"
    message += "🔴 CONSIGLIA VENDITA (score 0.25-0.35)\n"
    message += "🟡 MONITORA ATTIVAMENTE (score 0.35-0.45)\n"
    message += "⚪ MANTIENI POSIZIONE (score 0.45-0.55)\n"
    message += "🟢 CONSIGLIA ACQUISTO (score 0.55-0.65)\n"
    message += "🟢🟢 FORTE ACQUISTO (score > 0.65)\n"
    
    # Invia
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if token and chat_id:
        print("\n📤 Invio report a Telegram...")
        success = send_telegram_report(token, chat_id, message)
        if success:
            print("✅ Report inviato con successo!")
        else:
            print("❌ Errore nell'invio")
    else:
        print("\nℹ️  Credenziali Telegram mancanti")
        print(message)
    
    print(f"\n🏁 Analisi completata alle {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
