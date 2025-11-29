import sys
import json
import numpy as np
import librosa
import mutagen.flac
import os

# Kimeneti struktúra
output_data = {
    "status": "error",
    "filename": "",
    "is_original": False,
    "confidence": 0,
    "cutoff_frequency": 0,
    "check_frequency_21k_db": 0, # Debug infó: mennyi jel van 21kHz-nél
    "metadata": {},
    "reason": ""
}

def analyze_audio(file_path):
    try:
        # --- Metaadatok (marad a régi) ---
        try:
            audio_meta = mutagen.flac.FLAC(file_path)
            output_data['metadata'] = {
                "artist": audio_meta.get("artist", ["Ismeretlen"])[0],
                "title": audio_meta.get("title", ["Ismeretlen"])[0],
                "sample_rate": audio_meta.info.sample_rate,
                "bitrate": audio_meta.info.bitrate
            }
        except:
            pass

        # --- Elemzés ---
        # Betöltjük a fájlt (offset 10mp, hogy ne a csendes elejét nézzük)
        y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)
        
        # Ha rövid a fájl, betöltjük az egészet
        if len(y) == 0:
             y, sr = librosa.load(file_path, sr=None, duration=30.0)

        # STFT előállítása
        n_fft = 2048
        hop_length = 512
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        # Átkonvertálás dB-re
        db_data = librosa.amplitude_to_db(stft, ref=np.max)
        
        # FONTOS VÁLTOZÁS: 
        # Nem átlagot (mean) számolunk, hanem a 95%-os percentilisét vesszük.
        # Ez azt jelenti: "Mennyi a hangerő a leghangosabb pillanatokban?"
        # Így a rövid cintányér ütések nem vesznek el az átlagolásban.
        peak_power = np.percentile(db_data, 95, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # --- Vágás keresése ---
        
        cutoff_freq = 0
        # Ez a küszöb határozza meg, mit tekintünk "jelnek" és mit "zajnak".
        # Eredeti FLAC-nál a magas frekvenciák is lehetnek halkak (-70, -80dB).
        noise_floor_db = -85.0 

        # Visszafelé pásztázunk a legmagasabb frekvenciától
        for i in range(len(freqs)-1, 0, -1):
            if peak_power[i] > noise_floor_db:
                cutoff_freq = freqs[i]
                break

        output_data['cutoff_frequency'] = int(cutoff_freq)

        # Debug: Megnézzük konkrétan a 21kHz körüli erőt
        idx_21k = np.argmin(np.abs(freqs - 21000))
        power_at_21k = peak_power[idx_21k]
        output_data['check_frequency_21k_db'] = float(round(power_at_21k, 2))

        # --- Kiértékelés (Logika finomhangolva) ---
        
        # 1. eset: Van érdemi jel 20.5 kHz felett -> Eredeti
        if cutoff_freq >= 20500:
            output_data['is_original'] = True
            output_data['confidence'] = 98
            output_data['reason'] = f"Teljes spektrum ({int(cutoff_freq)} Hz). Eredeti CD/Lossless minőség."
            output_data['status'] = "success"

        # 2. eset: Vágás 19kHz és 20.5kHz között -> Gyanús (MP3 320 / AAC)
        # Az MP3 LAME enkóder jellemzően 20kHz vagy 20.5kHz-nél vág meredeken.
        elif 19000 <= cutoff_freq < 20500:
            # Itt egy extra ellenőrzést végzünk: 
            # Ha 21kHz-en a zajszint extrém alacsony (pl -100dB), akkor biztosan vágott.
            # Ha -86dB (épp a küszöb alatt), akkor lehet, hogy csak halk eredeti.
            
            if power_at_21k < -95:
                output_data['is_original'] = False
                output_data['confidence'] = 90
                output_data['reason'] = f"Digitális csend 21kHz-nél ({power_at_21k} dB). Valószínűleg MP3 320kbps forrás."
            else:
                # Határeset, de inkább eredetinek tippeljük, ha nincs 'hard cut'
                output_data['is_original'] = True 
                output_data['confidence'] = 60
                output_data['reason'] = f"A spektrum vége ({int(cutoff_freq)} Hz) határeset, de valószínűleg eredeti."
            
            output_data['status'] = "success"

        # 3. eset: Egyértelmű vágás 19kHz alatt -> Biztosan veszteséges
        else:
            output_data['is_original'] = False
            output_data['confidence'] = 99
            output_data['reason'] = f"Alacsony vágás ({int(cutoff_freq)} Hz). MP3 128-192kbps vagy transzkód."
            output_data['status'] = "success"

    except Exception as e:
        output_data['reason'] = str(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_data['filename'] = os.path.basename(file_path)
        if os.path.exists(file_path):
            analyze_audio(file_path)
        else:
            output_data['reason'] = "Fajl nem talalhato"
    else:
        output_data['reason'] = "Nincs fajl megadva"
    
    print(json.dumps(output_data))
