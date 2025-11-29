import sys
import json
import numpy as np
import librosa
import mutagen.flac
import os

# Alap kimenet (EREDETI mezők változatlanul)
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

        # --- 2. Betöltés és alap STFT (ahogy eredetileg volt) ---
        # 30 mp körüli rész (offset-el), de ha rövidebb a fájl, akkor az eleje
        try:
            y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)
            if y is None or len(y) == 0:
                y, sr = librosa.load(file_path, sr=None, duration=30.0)
        except Exception:
            # Ha valamiért az offset nem működik, töltsük be egyszerűen
            y, sr = librosa.load(file_path, sr=None, duration=30.0)

        # Eredeti stft számítás (megtartva az eredeti logikát is)
        stft = np.abs(librosa.stft(y))
        # Átlagos dB számítás (ref=np.max, hogy a leghangosabb pont 0 dB legyen)
        avg_power = np.mean(librosa.amplitude_to_db(stft, ref=np.max), axis=1)
        freqs = librosa.fft_frequencies(sr=sr)

        # --- EREDETI: egyszerű cutoff pásztázás (megtartva) ---
        cutoff_freq_scan = 0
        threshold_db = -80.0
        for i in range(len(freqs)-1, 0, -1):
            if avg_power[i] > threshold_db:
                cutoff_freq_scan = freqs[i]
                break

        # EREDETI: spectral rolloff
        try:
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
            rolloff_freq_mean = int(np.mean(rolloff))
        except Exception:
            rolloff_freq_mean = 0

        # --- BEILLESZTETT FEJLETTEBB SZŰRŐ (a második kódból) ---
        # Pontosabb spektrumanalízis nagyobb n_fft értékkel
        n_fft = 4096
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft//4))
        power_spectrum = np.mean(S, axis=1)
        power_spectrum_db = librosa.amplitude_to_db(power_spectrum, ref=np.max)
        freqs_precise = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        nyquist = sr / 2.0

        # Vizsgálandó sávok (14-16, 16-18, 18-20, 20-22 kHz)
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
                band_vals[name] = mean_db_in_band(power_spectrum_db, freqs_precise, lo, hi_eff)

        # Zajpadló (alsó 5%)
        noise_floor = np.percentile(power_spectrum_db, 5)

        # Rolloff (biztosítva, hogy az eredeti mező is meglegyen)
        output_data['rolloff_frequency'] = rolloff_freq_mean

        # Közép-referencia (4-8 kHz, ha van)
        mid_idx = np.where((freqs_precise >= 4000) & (freqs_precise <= min(8000, nyquist-1)))[0]
        if len(mid_idx) > 0:
            mid_db = float(np.mean(power_spectrum_db[mid_idx]))
        else:
            mid_db = float(np.mean(power_spectrum_db))

        def rel(x): 
            return None if x is None else (x - mid_db)

        rel_14_16 = rel(band_vals.get("14_16k"))
        rel_16_18 = rel(band_vals.get("16_18k"))
        rel_18_20 = rel(band_vals.get("18_20k"))
        rel_20_22 = rel(band_vals.get("20_22k"))

        # Pontozás a hamis/eredeti irányába (második kód logikája)
        score_fake = 0
        score_real = 0
        reasons = []

        # 1) Erős csökkenés 16-18k körül (>=12 dB) -> hamis
        if rel_16_18 is not None and rel_16_18 <= -12.0:
            score_fake += 30
            reasons.append(f"16-18k ~{rel_16_18:.1f} dB-rel alacsonyabb a középhez képest.")

        # 2) 18-20k gyengeség
        if rel_18_20 is not None and rel_18_20 <= -12.0:
            score_fake += 20
            reasons.append(f"18-20k ~{rel_18_20:.1f} dB-rel gyengébb a középhez képest.")

        # 3) 20-22k zaj + 16-18k gyenge -> zaj-kiterjesztés (hamis)
        if band_vals.get("20_22k") is not None and rel_16_18 is not None:
            if band_vals["20_22k"] > (noise_floor + 2.0) and rel_16_18 <= -10.0:
                score_fake += 25
                reasons.append("20-22k sávban kis energia van, miközben 16-18k gyenge -> zaj-kiterjesztés.")

        # 4) Ha magas sávok közel vannak a középhez (> -6 dB) -> eredeti
        high_good_count = 0
        for v in (rel_16_18, rel_18_20, rel_20_22):
            if v is not None and v > -6.0:
                high_good_count += 1
        if high_good_count >= 2:
            score_real += 60
            reasons.append("Magas sávokban megfelelő energia (nem tipikus transzkód).")

        # 5) Nagyon meredek fal (>=15 dB esés) -> nagyon erős hamis
        slope_14_16__16_18 = None
        if band_vals.get("14_16k") is not None and band_vals.get("16_18k") is not None:
            slope_14_16__16_18 = band_vals["16_18k"] - band_vals["14_16k"]
            if slope_14_16__16_18 <= -15.0:
                score_fake += 30
                reasons.append("Nagyon meredek fal a 16k körül (>=15 dB esés).")

        # 6) Rolloff alacsony -> hamis irányba tol
        if output_data["rolloff_frequency"] and output_data["rolloff_frequency"] < 18500:
            score_fake += 15
            reasons.append(f"Alacsony spectral rolloff: {output_data['rolloff_frequency']} Hz")

        # 7) Ha a magas sávok jól teljesítenek -> real pont
        if (rel_16_18 is not None and rel_16_18 > -8.0) and (rel_18_20 is not None and rel_18_20 > -8.0):
            score_real += 30

        # Összesítés
        total = score_fake + score_real
        if total <= 0:
            fake_prob = 0.5
        else:
            fake_prob = score_fake / total

        # Confidence skála 50..99, ahogy korábban használtuk
        confidence = int(min(99, max(50, round(50 + (fake_prob - 0.5) * 100))))
        is_original = False if fake_prob > 0.5 else True

        # EXTRA szigorú szabály (ha 20-22k zaj és 16-18k <= -12)
        if band_vals.get("20_22k") is not None and rel_16_18 is not None:
            if (band_vals["20_22k"] > (noise_floor + 3.0)) and (rel_16_18 <= -12.0):
                is_original = False
                confidence = max(confidence, 95)
                reasons.append("20-22 kHz zaj + 16-18 kHz esés → biztos hamis.")

        # Cutoff meghatározása (eredeti: legmagasabb frekvencia ahol dB >= noise_floor + 6 dB)
        thresh = noise_floor + 6.0
        idxs = np.where(power_spectrum_db >= thresh)[0]
        if len(idxs) > 0:
            cutoff_freq_precise = int(np.ceil(freqs_precise[idxs[-1]]))
        else:
            cutoff_freq_precise = int(cutoff_freq_scan)  # fallback az eredeti pásztázásra

        # --- OUTPUT kitöltése (pontosan az EREDETI mezőkkel) ---
        output_data['cutoff_frequency'] = cutoff_freq_precise
        output_data['is_original'] = bool(is_original)
        output_data['confidence'] = int(confidence)
        output_data['reason'] = ("; ".join(reasons) + " / " + ("Eredeti" if is_original else "Hamis")) if reasons else ("Eredeti" if is_original else "Hamis")
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

    # Visszaadjuk az eredeti JSON struktúrát — pontosan ugyanazokkal a kulcsokkal
    print(json.dumps(output_data))
