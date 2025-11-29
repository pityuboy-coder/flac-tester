import sys
import json
import numpy as np
import librosa
import mutagen.flac
import os

output_data = {
    "status": "error",
    "filename": "",
    "is_original": False,
    "confidence": 0,
    "cutoff_frequency": 0,
    "metadata": {},
    "reason": ""
}

def analyze_audio(file_path):
    try:
        # --- 1. Metaadatok ---
        try:
            audio_meta = mutagen.flac.FLAC(file_path)
            output_data['metadata'] = {
                "artist": audio_meta.get("artist", ["Ismeretlen"])[0],
                "title": audio_meta.get("title", ["Ismeretlen"])[0],
                "sample_rate": audio_meta.info.sample_rate,
                "bitrate": audio_meta.info.bitrate
            }
        except Exception:
            output_data['metadata'] = {"error": "Metaadat hiba"}

        # --- 2. Elemzés (Peak módszerrel) ---
        # Betöltünk 30 másodpercet, de kihagyjuk az elejét (offset=5), hogy a csendet kerüljük
        try:
            y, sr = librosa.load(file_path, sr=None, offset=5.0, duration=30.0)
        except:
            # Ha nagyon rövid a fájl, offset nélkül töltjük
            y, sr = librosa.load(file_path, sr=None, duration=30.0)

        if len(y) == 0:
            raise ValueError("Üres vagy hibás audiofájl.")

        # STFT számítás
        stft = np.abs(librosa.stft(y))
        
        # Átlagolás HELYETT: Percentilis (Csúcstartás)
        # Ez a kulcs: nem az átlagot nézzük, hanem a leghangosabb pontokat az adott frekvencián.
        # A 95-ös percentilis kiszűri az egyszeri hibákat, de megtartja a cintányérokat.
        db_values = librosa.amplitude_to_db(stft, ref=np.max)
        peak_power = np.percentile(db_values, 95, axis=1) # mean helyett 95%
        
        freqs = librosa.fft_frequencies(sr=sr)

        # --- 3. Vágási pont keresése (Intelligensebb logika) ---
        
        # Küszöbérték: -80dB (Mivel most a "csúcsokat" nézzük, lemehetünk mélyre a zajszintig)
        noise_floor_db = -80.0
        
        # Megkeressük az összes olyan frekvenciát, ami hangosabb a zajnál
        valid_indices = np.where(peak_power > noise_floor_db)[0]
        
        if len(valid_indices) > 0:
            # A legmagasabb frekvencia, ahol még volt jel
            last_valid_idx = valid_indices[-1]
            cutoff_freq = freqs[last_valid_idx]
        else:
            cutoff_freq = 0

        output_data['cutoff_frequency'] = int(cutoff_freq)

        # --- 4. Kiértékelés (Enyhébb határok) ---
        
        # CD minőség (44.1kHz) esetén a Nyquist frekvencia 22050 Hz.
        # Egy MP3 (LAME 320kbps) általában 20000-20500 Hz-nél vág meredeken.
        # Egy FLAC általában elmegy 21000-22000 Hz-ig, de fokozatosan halkul.

        if cutoff_freq >= 21000:
            output_data['is_original'] = True
            output_data['confidence'] = 98
            output_data['reason'] = f"Teljes spektrum ({int(cutoff_freq)} Hz). Eredeti."
            output_data['status'] = "success"
            
        elif cutoff_freq >= 20000:
            # Ez a 'szürke zóna'. Sok eredeti master itt véget ér, de a legjobb MP3-ak is.
            # Mivel a felhasználó "in dubio pro reo" (kétség esetén eredeti) elvet szeretne:
            output_data['is_original'] = True 
            output_data['confidence'] = 80
            output_data['reason'] = f"Magas frekvencia ({int(cutoff_freq)} Hz). Valószínűleg eredeti, vagy kiváló transcode."
            output_data['status'] = "success"
            
        elif cutoff_freq > 16000:
            # 16k - 19.5k között biztosan veszteséges (MP3 128-192-320kbps vegyesen)
            output_data['is_original'] = False
            output_data['confidence'] = 95
            output_data['reason'] = f"Vágott spektrum ({int(cutoff_freq)} Hz). MP3 forrás."
            output_data['status'] = "success"
        else:
            output_data['is_original'] = False
            output_data['confidence'] = 99
            output_data['reason'] = f"Alacsony minőség ({int(cutoff_freq)} Hz)."
            output_data['status'] = "success"

    except Exception as e:
        output_data['reason'] = str(e)
        output_data['status'] = "error"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_data['filename'] = os.path.basename(file_path)
        if os.path.exists(file_path):
            analyze_audio(file_path)
        else:
            output_data['reason'] = "Fájl nem található"
    else:
        output_data['reason'] = "Nincs fájl megadva"
    
    print(json.dumps(output_data))
