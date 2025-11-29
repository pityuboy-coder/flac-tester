import sys
import json
import numpy as np
import librosa
import mutagen.flac
import os

# --- Alap kimeneti struktúra ---
output_data = {
    "status": "error",
    "filename": "",
    "is_original": False,
    "confidence": 0,
    "cutoff_frequency": 0,
    "check_frequency_21k_db": 0.0, # Energia 21kHz-nél (dB)
    "db_difference_19k_21k": 0.0, # Kontraszt (19kHz vs 21kHz)
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
            # Nem kritikus hiba, ha a metaadat sérült, de a fájl olvasható
            pass

        # --- 2. Betöltés és STFT elemzés ---
        # Betöltjük a fájlt (offset 10mp, duration 30s – elkerülve a csendes intrót)
        y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)
        
        # Ha rövid a fájl, betöltjük az egészet
        if len(y) == 0:
             y, sr = librosa.load(file_path, sr=None, duration=30.0)

        n_fft = 2048
        stft = np.abs(librosa.stft(y, n_fft=n_fft))
        
        # Átkonvertálás dB-re (a leghangosabb pont legyen a 0 dB)
        db_data = librosa.amplitude_to_db(stft, ref=np.max)
        
        # FONTOS VÁLTOZÁS: 95%-os percentilis a dinamikus csúcsokhoz
        # Ez a valódi zenei csúcsokat (pl. cintányérok) fogja mérni, nem az átlagos halkságot.
        peak_power = np.percentile(db_data, 95, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # --- 3. Vágás keresése (Kritikus pont) ---
        
        cutoff_freq = 0
        # Küszöb a csúcsérték kereséséhez. -85 dB elég érzékeny, de elkerüli a legalacsonyabb zajokat.
        noise_floor_db = -85.0 

        # Visszafelé pásztázunk a legmagasabb frekvenciától
        for i in range(len(freqs)-1, 0, -1):
            if peak_power[i] > noise_floor_db:
                cutoff_freq = freqs[i]
                break
        
        output_data['cutoff_frequency'] = int(cutoff_freq)

        # --- 4. Kontraszt Elemzés (Meredekség ellenőrzése) ---
        
        # 1. Energia 19kHz-nél (Referencia: ahol még erős a zene)
        idx_19k = np.argmin(np.abs(freqs - 19000))
        power_at_19k = peak_power[idx_19k]
        
        # 2. Energia 21kHz-nél (Ellenőrző pont: ahol a fake FLAC-nek "halottnak" kell lennie)
        idx_21k = np.argmin(np.abs(freqs - 21000))
        power_at_21k = peak_power[idx_21k]
        output_data['check_frequency_21k_db'] = float(round(power_at_21k, 2))
        
        # Kiszámoljuk a dB különbséget. Nagy különbség = Meredek vágás (MP3).
        db_difference = power_at_19k - power_at_21k
        output_data['db_difference_19k_21k'] = float(round(db_difference, 2))
        
        # --- 5. Eredmény kiértékelése (Kontraszt alapú logika) ---
        
        # A kontraszt küszöb (20 dB): ha 19k és 21k között 20 dB-nél meredekebb a zuhanás, az mesterséges.
        CONTRAST_THRESHOLD = 20.0
        
        # 1. Eset: Egyértelműen teljes spektrum (>20.5 kHz)
        if cutoff_freq >= 20500:
            
            if db_difference < CONTRAST_THRESHOLD:
                # Magas cutoff ÉS lassú esés = Tiszta eredeti
                output_data['is_original'] = True
                output_data['confidence'] = 99
                output_data['reason'] = f"Teljes spektrum ({int(cutoff_freq)} Hz) és természetes esés (<{CONTRAST_THRESHOLD} dB kontraszt)."
            else:
                # Magas cutoff, de MEREKEN vágás (ritka, de lehet zajosított MP3)
                output_data['is_original'] = False
                output_data['confidence'] = 75
                output_data['reason'] = f"Gyanús: Magas cutoff, de nagyon meredek vágás ({round(db_difference, 1)} dB kontraszt). Lehet transzkódolt."
            
            output_data['status'] = "success"

        # 2. Eset: MP3 vágási zóna (19 kHz - 20.5 kHz között)
        elif 19000 <= cutoff_freq < 20500:
            
            # DÖNTŐ PONT: Ha a kontraszt túl nagy, akkor meredek a vágás, és hamis.
            if db_difference > CONTRAST_THRESHOLD: 
                output_data['is_original'] = False
                output_data['confidence'] = 95
                output_data['reason'] = f"Meredek spektrum esés ({round(db_difference, 1)} dB kontraszt). MP3/Transzkód (320kbps)."
            else:
                # Lassú esés (természetes) esetén elfogadjuk, hogy csak halk a zene
                output_data['is_original'] = True
                output_data['confidence'] = 60
                output_data['reason'] = f"Határeset: Lassú esés (<{CONTRAST_THRESHOLD} dB kontraszt), valószínűleg eredeti, halk magas hangokkal."
            
            output_data['status'] = "success"

        # 3. Eset: Egyértelmű vágás 19 kHz alatt
        else:
            output_data['is_original'] = False
            output_data['confidence'] = 99
            output_data['reason'] = f"Alacsony vágás ({int(cutoff_freq)} Hz). Kisebb bitrátájú MP3/Transzkód."
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
    
    # A végeredmény kiírása JSON formátumban
    print(json.dumps(output_data, indent=4))
