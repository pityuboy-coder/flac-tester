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
    "rolloff_frequency": 0,
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

        # --- 2. Betöltés ---
        y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)
        if len(y) == 0:
            y, sr = librosa.load(file_path, sr=None, duration=30.0)

        stft = np.abs(librosa.stft(y))
        avg_power = np.mean(librosa.amplitude_to_db(stft, ref=np.max), axis=1)
        freqs = librosa.fft_frequencies(sr=sr)

        # --- 3. Cutoff keresés ---
        cutoff_freq_scan = 0
        threshold_db = -80.0

        for i in range(len(freqs)-1, 0, -1):
            if avg_power[i] > threshold_db:
                cutoff_freq_scan = freqs[i]
                break

        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
        rolloff_freq_mean = int(np.mean(rolloff))

        output_data['cutoff_frequency'] = int(cutoff_freq_scan)
        output_data['rolloff_frequency'] = rolloff_freq_mean

        final_cutoff = int(cutoff_freq_scan)

        # --- 4. ÚJ – MP3 transzkód felismerés ---
        # 16–20 kHz energia mérése
        hf_start = np.where(freqs >= 16000)[0][0]
        hf_end = np.where(freqs >= 20000)[0][0]

        hf_energy = np.mean(avg_power[hf_start:hf_end])
        total_energy = np.mean(avg_power)

        hf_ratio = hf_energy - total_energy  # dB különbség

        # Spektrális lejtés 16k felett
        slope = avg_power[hf_start] - avg_power[hf_end]

        # --- 5. DÖNTÉS ---
        # HAMIS FLAC FELISMERÉSE (MP3-ból)
        # Ha cutoff magas, de nincs energiája → hamis
        if final_cutoff > 20000 and (hf_ratio < -20 or slope > 12):
            output_data['is_original'] = False
            output_data['confidence'] = 99
            output_data['reason'] = f"Hamis FLAC: 16kHz felett összeesik az energiája (cutoff {final_cutoff} Hz)."
            output_data['status'] = "success"
            return

        # --- eredeti logika + finomítás ---
        if final_cutoff > 20000:
            output_data['is_original'] = True
            output_data['confidence'] = 98
            output_data['reason'] = f"Teljes spektrum ({final_cutoff} Hz). Valós veszteségmentes."
            output_data['status'] = "success"

        elif final_cutoff > 18500:
            output_data['is_original'] = False
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
