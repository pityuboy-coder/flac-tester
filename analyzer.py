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
    "metadata": {},
    "reason": ""
}

def analyze_audio(file_path):
    try:
        # Metaadatok
        audio_meta = mutagen.flac.FLAC(file_path)
        output_data['metadata'] = {
            "artist": audio_meta.get("artist", ["Ismeretlen"])[0],
            "title": audio_meta.get("title", ["Ismeretlen"])[0],
            "album": audio_meta.get("album", ["Ismeretlen"])[0],
            "sample_rate": audio_meta.info.sample_rate,
            "bits_per_sample": audio_meta.info.bits_per_sample,
            "bitrate": audio_meta.info.bitrate
        }

        # Elemzés (első 30 mp)
        y, sr = librosa.load(file_path, sr=None, duration=30.0)
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        avg_power = np.mean(librosa.amplitude_to_db(stft, ref=np.max), axis=1)

        # Vágás keresése
        cutoff_freq = 0
        noise_floor_db = -65
        for i in range(len(freqs)-1, 0, -1):
            if avg_power[i] > noise_floor_db:
                cutoff_freq = freqs[i]
                break

        output_data['cutoff_frequency'] = int(cutoff_freq)

        # Eredmény kiértékelése
        if cutoff_freq > 21000:
            output_data['is_original'] = True
            output_data['confidence'] = 95
            output_data['reason'] = "Teljes spektrum (>21kHz). Eredeti."
            output_data['status'] = "success"
        elif cutoff_freq > 18000:
             output_data['is_original'] = False
             output_data['confidence'] = 80
             output_data['reason'] = "Gyanus vagas 18-20kHz kozott (MP3 320kbps)."
             output_data['status'] = "success"
        else:
            output_data['is_original'] = False
            output_data['confidence'] = 99
            output_data['reason'] = f"Alacsony vagas ({int(cutoff_freq)} Hz). MP3/Transcode."
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
