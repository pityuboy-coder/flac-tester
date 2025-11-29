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
    "band_db": {},
    "metadata": {},
    "reason": ""
}

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
            sr_meta = audio_meta.info.sample_rate
        except Exception as meta_error:
            output_data['metadata'] = {"error": str(meta_error)}
            sr_meta = None

        # --- 2. Betöltés (10s offset, 30s duration) ---
        try:
            y, sr = librosa.load(file_path, sr=None, offset=10.0, duration=30.0)
            if y is None or len(y) == 0:
                y, sr = librosa.load(file_path, sr=None, duration=30.0)
        except Exception:
            y, sr = librosa.load(file_path, sr=None, duration=30.0)

        # --- 3. STFT és dB-s Spektrum ---
        n_fft = 4096
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft//4))
        power_spectrum = np.mean(S, axis=1)  # linear
        # konvertáljuk dB-be relatif a maxra, hogy zaj + jel jól látszódjon
        power_spectrum_db = librosa.amplitude_to_db(power_spectrum, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        nyquist = sr / 2.0

        # --- 4. Definiált sávok (figyelem: ha sr kicsi, a felső sávokat kihagyjuk) ---
        bands_to_check = [
            ("low_mid", 4000, 8000),
            ("14_16k", 14000, 16000),
            ("16_18k", 16000, 18000),
            ("18_20k", 18000, 20000),
            ("20_22k", 20000, 22050)
        ]

        band_db = {}
        for name, lo, hi in bands_to_check:
            if lo >= nyquist:
                band_db[name] = None
            else:
                hi_eff = min(hi, nyquist - 1.0)
                val = mean_db_in_band(power_spectrum_db, freqs, lo, hi_eff)
                band_db[name] = val

        output_data["band_db"] = band_db

        # --- 5. Rolloff ---
        try:
            roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
            roll_mean = float(np.mean(roll))
            output_data["rolloff_frequency"] = int(roll_mean)
        except Exception:
            output_data["rolloff_frequency"] = 0

        # --- 6. Elemzés: energia arányok és slope-ok ---
        # Referencia: közép-sáv (4-8k)
        mid_db = band_db.get("low_mid")
        db_14_16 = band_db.get("14_16k")
        db_16_18 = band_db.get("16_18k")
        db_18_20 = band_db.get("18_20k")
        db_20_22 = band_db.get("20_22k")

        # Ha nincs közép-sáv adat (pl kis sr) fallback a teljes spektrum középére
        if mid_db is None:
            mid_idx = np.where((freqs >= 2000) & (freqs <= min(8000, nyquist-1)))[0]
            if len(mid_idx) > 0:
                mid_db = float(np.mean(power_spectrum_db[mid_idx]))
            else:
                mid_db = float(np.mean(power_spectrum_db))

        # kis helper: zajszint küszöb (pl -80 dB alatt zajnak tekintjük)
        noise_floor_db = np.percentile(power_spectrum_db, 5)

        # slope a 14-16 vs 16-18 között
        slope_14_16__16_18 = None
        if db_14_16 is not None and db_16_18 is not None:
            slope_14_16__16_18 = db_16_18 - db_14_16

        # slope 18-20 vs 20-22 (ha léteznek)
        slope_18_20__20_22 = None
        if db_18_20 is not None and db_20_22 is not None:
            slope_18_20__20_22 = db_20_22 - db_18_20

        # magas sáv átlag relatív a középhez
        def rel(db_val):
            return None if db_val is None else (db_val - mid_db)

        rel_14_16 = rel(db_14_16)
        rel_16_18 = rel(db_16_18)
        rel_18_20 = rel(db_18_20)
        rel_20_22 = rel(db_20_22)

        # --- 7. Döntési szabályok (szigorúbb detektálás a 16k körüli esésre) ---
        reasons = []
        score_fake = 0  # nagyobb = hamis valószínűbb
        score_real = 0

        # 1) Ha 16-18k nagyon sokkal (>= 12 dB) gyengébb, mint közép -> erős jel a vágásra
        if rel_16_18 is not None and rel_16_18 <= -12.0:
            score_fake += 30
            reasons.append(f"16-18k ~{rel_16_18:.1f} dB-rel alacsonyabb a középhez képest (>-12 dB küszöb).")

        # 2) Ha a 14-16k -> 16-18k között éles esés van (slope <= -8..-12 dB)
        if slope_14_16__16_18 is not None and slope_14_16__16_18 <= -8.0:
            score_fake += 25
            reasons.append(f"Sérülékeny meredekség 14-16k -> 16-18k: {slope_14_16__16_18:.1f} dB (brickwall jelleg).")

        # 3) Ha a 20-22k sávban van minimális, de jelenlévő energia (például -70..-40 dB), és közben 16-18k gyenge => zaj-kiterjesztés (hamis)
        if db_20_22 is not None and rel_16_18 is not None:
            # ha 20-22k > noise_floor + 3dB (van "valami" zaj), de rel_16_18 <= -10 => valószínű műzaj
            if db_20_22 > (noise_floor_db + 3.0) and rel_16_18 <= -10.0:
                score_fake += 20
                reasons.append(f"20-22k sávban van kis energia ({db_20_22:.1f} dB) míg 16-18k gyenge → zaj-kiterjesztés.")

        # 4) Ha 18-20k is erősen visszaesett (<= -12 dB) => erős transzkód
        if rel_18_20 is not None and rel_18_20 <= -12.0:
            score_fake += 20
            reasons.append(f"18-20k ~{rel_18_20:.1f} dB-rel gyengébb a középhez képest (erős vágás).")

        # 5) Ha a magas sávok közel vannak a közép szinthez (>-6 dB), pont a real jelzés felé
        high_good_count = 0
        for v in (rel_16_18, rel_18_20, rel_20_22):
            if v is not None and v > -6.0:
                high_good_count += 1
        if high_good_count >= 2:
            score_real += 60
            reasons.append("Magas sávokban megfelelő energia (nem látszik transzkód).")

        # 6) Brickwall erősség: ha slope_14_16__16_18 <= -15 dB, ez nagyon erős MP3 jel
        if slope_14_16__16_18 is not None and slope_14_16__16_18 <= -15.0:
            score_fake += 30
            reasons.append("Nagyon meredek fal a 16k körül (>=15 dB esés) — tipikus MP3/AAC transzkód.")

        # 7) Ha a rolloff nagyon alacsony (< 18500) → hamis
        if output_data["rolloff_frequency"] and output_data["rolloff_frequency"] < 18500:
            score_fake += 15
            reasons.append(f"Spectral rolloff alacsony: {output_data['rolloff_frequency']} Hz.")

        # 8) Ha összességében magas sáv jól teljesít -> real
        if (rel_16_18 is not None and rel_16_18 > -8.0) and (rel_18_20 is not None and rel_18_20 > -8.0):
            score_real += 30

        # Normáljuk a pontszámot 0-100 között
        total = score_real + score_fake
        if total <= 0:
            # kevés adat, óvatos döntés: gyanús, de nem biztos
            confidence = 60
            is_original = False
            reason = "Elégtelen bizonyíték; gyanús spektrum."
        else:
            # compute probability of fake as ratio
            fake_prob = score_fake / total
            confidence = int(min(99, max(50, round(50 + (fake_prob - 0.5) * 100))))
            # ha fake_prob > 0.5 -> hamis
            is_original = False if fake_prob > 0.5 else True
            reason = ("Hamis valószínűsége magas." if not is_original else "Valószínűleg eredeti (FLAC).")

        # Extra finomhangolás: ha van zaj a 20-22k-ben, de 16-18k nagyon gyenge -> biztos hamis
        if db_20_22 is not None and rel_16_18 is not None:
            if (db_20_22 > (noise_floor_db + 2.0)) and (rel_16_18 <= -12.0):
                is_original = False
                confidence = max(confidence, 95)
                reason = "Zaj-kiterjesztés a 20-22k sávban, miközben 16-18k gyenge — hamis."

        # cutoff_frequency: a legmagasabb frekuncia ahol dB még >= noise_floor + 6dB (értelmes jel)
        cutoff_freq = 0
        thresh = noise_floor_db + 6.0
        idxs = np.where(power_spectrum_db >= thresh)[0]
        if len(idxs) > 0:
            cutoff_freq = int(np.ceil(freqs[idxs[-1]]))
        else:
            cutoff_freq = 0

        # Fill output
        output_data["cutoff_frequency"] = cutoff_freq
        output_data["hf_slope"] = float(slope_14_16__16_18) if slope_14_16__16_18 is not None else 0.0
        output_data["confidence"] = int(confidence)
        output_data["is_original"] = bool(is_original)
        output_data["reason"] = "; ".join(reasons) + " / " + reason if reasons else reason
        output_data["status"] = "success"

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

    print(json.dumps(output_data, ensure_ascii=False, indent=2))
