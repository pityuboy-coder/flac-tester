import sys
import json
import numpy as np
import librosa
import mutagen.flac
import os

# Alap JSON struktúra
output_data = {
    "status": "error",
    "filename": "",
    "is_original": False,
    "confidence": 0,
    "cutoff_frequency": 0,
    "rolloff_frequency": 0,
    "hf_slope": 0,
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
            output_data['metadata'] = {"error": str(meta_error)}

        # --- 2. Audio betöltés ---
        y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)

        if len(y) == 0:
            y, sr = librosa.load(file_path, sr=None, duration=30.0)

        # STFT
        stft_full = np.abs(librosa.stft(y, n_fft=4096))
        power_spectrum = np.mean(stft_full, axis=1)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        # --- 3. Frekvenciasávok vizsgálata ---
        bands = {
            "14k": (14000, 16000),
            "16k": (16000, 18000),
            "18k": (18000, 20000),
            "20k": (20000, 22050)
        }

        band_energy = {}

        for name, (f_low, f_high) in bands.items():
            idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            if len(idx) > 0:
                band_energy[name] = float(np.mean(power_spectrum[idx]))
            else:
                band_energy[name] = 0.0

        # --- 4. Cutoff frekvencia becslése ---
        if band_energy["18k"] < band_energy["16k"] * 0.1:
            final_cutoff = 18000
        elif band_energy["20k"] < band_energy["18k"] * 0.2:
            final_cutoff = 20000
        else:
            final_cutoff = 22050

        output_data["cutoff_frequency"] = final_cutoff

        # --- 5. Brickwall meredekség ---
        slope = band_energy["18k"] - band_energy["20k"]
        output_data["hf_slope"] = float(slope)

        # --- 6. Spectral rolloff (statisztikai) ---
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
        roll_mean = float(np.mean(roll))
        output_data["rolloff_frequency"] = int(roll_mean)

        # --- 7. Eredetiség kiértékelése ---
        # Brickwall MP3/AAC detektálás
        if slope < -0.7:
            output_data['is_original'] = False
            output_data['confidence'] = 99
            output_data['reason'] = f"Erős brickwall (slope={slope:.2f}), cutoff: {final_cutoff} Hz → MP3 transzkód."
            output_data['status'] = "success"
            return

        # 16–19 kHz körüli vágás → nagy valószínűségű MP3
        if final_cutoff < 19000:
            output_data['is_original'] = False
            output_data['confidence'] = 95
            output_data['reason'] = f"Magas frekvenciás energiahiány, cutoff {final_cutoff} Hz → MP3/veszteséges forrás."
            output_data['status'] = "success"
            return

        # 19–20 kHz → gyanús / AAC
        if final_cutoff < 21000:
            output_data['is_original'] = False
            output_data['confidence'] = 90
            output_data['reason'] = f"20 kHz alatti vágás ({final_cutoff} Hz). Valószínűleg AAC/MP3 transzkód."
            output_data['status'] = "success"
            return

        # Ha a 20–22 kHz tartományban is van jel → eredeti FLAC
        if final_cutoff > 21000 and slope > -0.3:
            output_data['is_original'] = True
            output_data['confidence'] = 98
            output_data['reason'] = "Teljes spektrum (20–22 kHz), nincs brickwall → Valós FLAC."
            output_data['status'] = "success"
            return

        # Egyéb ismeretlen eset
        output_data['is_original'] = False
        output_data['confidence'] = 80
        output_data['reason'] = "Gyanús spektrum, nem tipikus FLAC mintázat."
        output_data['status'] = "success"

    except Exception as e:
        output_data["reason"] = str(e)
        output_data["status"] = "error"


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
