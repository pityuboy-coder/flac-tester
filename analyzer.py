import sys
import json
import numpy as np
import librosa
import mutagen.flac
import os

# Alap kimenet
output_data = {
    "status": "error",
    "filename": "",
    "is_original": False,
    "confidence": 0,
    "cutoff_frequency": 0,
    "rolloff_frequency": 0, # Új adatmező
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
                "album": audio_meta.get("album", ["Ismeretlen"])[0],
                "sample_rate": audio_meta.info.sample_rate,
                "bits_per_sample": audio_meta.info.bits_per_sample,
                "bitrate": audio_meta.info.bitrate
            }
        except Exception as meta_error:
            # Ha a metaadat sérült, ne álljon meg a program, csak logolja
            output_data['metadata'] = {"error": str(meta_error)}

        # --- 2. Betöltés és Elemzés ---
        # 30 helyett 10-40 mp közötti részt nézünk, hogy elkerüljük a csendes intrót
        y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)
        
        # Ha a fájl rövidebb mint 10mp, töltsük be az elejétől
        if len(y) == 0:
             y, sr = librosa.load(file_path, sr=None, duration=30.0)

        stft = np.abs(librosa.stft(y))
        
        # Átlagos dB számítás
        # A ref=np.max biztosítja, hogy a leghangosabb pont legyen a 0 dB
        avg_power = np.mean(librosa.amplitude_to_db(stft, ref=np.max), axis=1)
        freqs = librosa.fft_frequencies(sr=sr)

        # --- 3. Kétlépcsős Vágás Keresés ---

        # "A" Módszer: Hagyományos pásztázás (Szigorúbb küszöbbel!)
        cutoff_freq_scan = 0
        # -60 dB-re emeltem a küszöböt. 
        # A transzkódolt fájlokban gyakran van -70/-80dB zaj a magasban, ezt most figyelmen kívül hagyjuk.
        threshold_db = -60.0 
        
        for i in range(len(freqs)-1, 0, -1):
            if avg_power[i] > threshold_db:
                cutoff_freq_scan = freqs[i]
                break

        # "B" Módszer: Spectral Rolloff (Statisztikai megközelítés)
        # Megkeresi azt a frekvenciát, amely alatt az energia 99%-a van.
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
        rolloff_freq_mean = np.mean(rolloff)

        # A végleges cutoff a két módszer átlaga vagy a szigorúbb (kisebb) érték
        # A biztonság kedvéért a pásztázott értéket vesszük alapnak, de logoljuk a másikat is
        final_cutoff = int(cutoff_freq_scan)
        
        output_data['cutoff_frequency'] = final_cutoff
        output_data['rolloff_frequency'] = int(rolloff_freq_mean)

        # --- 4. Eredmény kiértékelése (Kalibrálva) ---
        
        nyquist = sr / 2
        
        # Ellenőrizzük a mintavételezést is.
        # Ha 44.1kHz-es a fájl, a max frekvencia 22050Hz.
        
        if final_cutoff > 20000:
            # CD minőség (44.1kHz) esetén a 20kHz feletti jel már jónak számít
            output_data['is_original'] = True
            output_data['confidence'] = 98
            output_data['reason'] = f"Teljes spektrum ({final_cutoff} Hz). Valós veszteségmentes."
            output_data['status'] = "success"
            
        elif final_cutoff > 18500:
            # MP3 320kbps vagy AAC általában itt vág (19.5kHz - 20.5kHz körül)
            # De néha régi felvételeknél az eredeti is lehet ilyen.
            output_data['is_original'] = False # Inkább legyünk szigorúak
            output_data['confidence'] = 85
            output_data['reason'] = f"Gyanús vágás 20kHz alatt ({final_cutoff} Hz). Valószínűleg MP3 320kbps vagy régi master."
            output_data['status'] = "success"
            
        elif final_cutoff > 15000:
            output_data['is_original'] = False
            output_data['confidence'] = 99
            output_data['reason'] = f"Erős vágás ({final_cutoff} Hz). MP3 128-192kbps."
            output_data['status'] = "success"
            
        else:
            output_data['is_original'] = False
            output_data['confidence'] = 100
            output_data['reason'] = f"Nagyon alacsony sávszélesség ({final_cutoff} Hz)."
            output_data['status'] = "success"

    except Exception as e:
        output_data['reason'] = str(e)
        # Hiba esetén írjuk ki a konzolra is, ha debugolni kell
        # print(f"Hiba történt: {e}", file=sys.stderr)

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
