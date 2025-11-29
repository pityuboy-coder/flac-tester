import sys
import json
import numpy as np
import librosa
import mutagen.flac
import os

# --- Kimeneti struktúra (EREDETI FORMÁBAN MEGTARTVA) ---
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

# Segédfüggvény: átlagos dB egy frekvenciasávban
def mean_db_in_band(power_spectrum_db, freqs, low_hz, high_hz):
    idx = np.where((freqs >= low_hz) & (freqs < high_hz))[0]
    if len(idx) == 0:
        return None
    return float(np.mean(power_spectrum_db[idx]))


def analyze_audio(file_path):
    try:
        # --- 1) METAADATOK ---
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
            sr_meta = audio_meta.info.sample_rate
        except Exception as e:
            output_data['metadata'] = {"error": str(e)}
            sr_meta = None

        # --- 2) AUDIO BETÖLTÉSE ---
        try:
            y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)
            if len(y) == 0:
                y, sr = librosa.load(file_path, sr=None, duration=30.0)
        except Exception:
            y, sr = librosa.load(file_path, sr=None, duration=30.0)

        # --- 3) SPEKTRUM ---
        n_fft = 4096
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft//4))
        power_spectrum = np.mean(S, axis=1)
        power_spectrum_db = librosa.amplitude_to_db(power_spectrum, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        nyquist = sr / 2.0

        # --- 4) FREKISÁVOK VIZSGÁLATA ---
        bands_to_check = [
            ("14_16k", 14000, 16000),
            ("16_18k", 16000, 18000),
            ("18_20k", 18000, 20000),
            ("20_22k", 20000, 22050)
        ]

        band_vals = {}
        for name, lo, hi in bands_to_check:
            if lo >= nyquist:
                band_vals[name] = None
            else:
                hi_eff = min(hi, nyquist - 1)
                band_vals[name] = mean_db_in_band(power_spectrum_db, freqs, lo, hi_eff)

        # --- Zajpadló ---
        noise_floor = np.percentile(power_spectrum_db, 5)

        # --- Rolloff ---
        try:
            roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
            roll_mean = float(np.mean(roll))
            output_data["rolloff_frequency"] = int(roll_mean)
        except:
            roll_mean = 0.0
            output_data["rolloff_frequency"] = 0

        # --- KÖZÉPFREKI REFERENCIA ---
        mid_idx = np.where((freqs >= 4000) & (freqs <= min(8000, nyquist-1)))[0]
        if len(mid_idx) > 0:
            mid_db = float(np.mean(power_spectrum_db[mid_idx]))
        else:
            mid_db = float(np.mean(power_spectrum_db))

        def rel(x): return None if x is None else (x - mid_db)

        rel_16_18 = rel(band_vals["16_18k"])
        rel_18_20 = rel(band_vals["18_20k"])
        rel_20_22 = rel(band_vals["20_22k"])

        # --- FEJLETT HAMISÍTÁS FELISMERÉS ---
        score_fake = 0
        score_real = 0
        reasons = []

        # 16–18 kHz erős esés → MP3 transzkód
        if rel_16_18 is not None and rel_16_18 <= -12:
            score_fake += 30
            reasons.append("Erős esés 16–18 kHz között.")

        # 18–20 kHz esés
        if rel_18_20 is not None and rel_18_20 <= -12:
            score_fake += 20
            reasons.append("18–20 kHz gyenge.")

        # 20–22 kHz zajkiterjesztés
        if band_vals["20_22k"] is not None:
            if band_vals["20_22k"] > (noise_floor + 2) and rel_16_18 <= -10:
                score_fake += 25
                reasons.append("Zajkiterjesztés 20–22 kHz között.")

        # Magas tartomány jó → eredeti
        high_ok = 0
        for r in (rel_16_18, rel_18_20, rel_20_22):
            if r is not None and r > -6:
                high_ok += 1
        if high_ok >= 2:
            score_real += 60
            reasons.append("Magas frekvenciák megfelelő energiával.")

        # --- VALÓDI DÖNTÉS ---
        total = score_fake + score_real
        if total == 0:
            fake_prob = 0.5
        else:
            fake_prob = score_fake / total

        is_original = fake_prob < 0.5
        confidence = int(50 + abs(fake_prob - 0.5) * 100)

        # EXTRA SZIGOR
        if band_vals["20_22k"] is not None and rel_16_18 is not None:
            if (band_vals["20_22k"] > (noise_floor + 3)) and (rel_16_18 <= -12):
                is_original = False
                confidence = max(confidence, 95)
                reasons.append("20–22 kHz zaj + 16–18 kHz esés → hamis.")

        # --- CUTOFF MEGHATÁROZÁSA ---
        thresh = noise_floor + 6
        idxs = np.where(power_spectrum_db >= thresh)[0]
        if len(idxs) > 0:
            cutoff_freq = int(freqs[idxs[-1]])
        else:
            cutoff_freq = 0

        # --- KIMENET (EREDETI STRUKTÚRA!) ---
        output_data["cutoff_frequency"] = cutoff_freq
        output_data["is_original"] = is_original
        output_data["confidence"] = int(confidence)
        output_data["reason"] = "; ".join(reasons) if reasons else ("Eredeti" if is_original else "Hamis")
        output_data["status"] = "success"

    except Exception as e:
        output_data["reason"] = str(e)
        output_data["status"] = "error"


# --- FŐPROGRAM ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_data["filename"] = os.path.basename(file_path)
        if os.path.exists(file_path):
            analyze_audio(file_path)
        else:
            output_data["reason"] = "Fájl nem található"
    else:
        output_data["reason"] = "Nincs fájl megadva"

    print(json.dumps(output_data))
