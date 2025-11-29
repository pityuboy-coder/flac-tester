import sys
import json
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import welch
import mutagen.flac
import os


def downsample_for_plot(x_data, y_data, target_points=250):
    if len(x_data) <= target_points:
        return x_data.tolist(), y_data.tolist()
    
    bin_size = int(len(x_data) / target_points)
    x_binned = []
    y_binned = []
    
    for i in range(0, len(x_data), bin_size):
        x_slice = x_data[i:i+bin_size]
        y_slice = y_data[i:i+bin_size]
        if len(x_slice) > 0:
            x_binned.append(round(np.mean(x_slice))) 
            y_binned.append(round(np.mean(y_slice), 2))
            
    return x_binned, y_binned

def analyze_flac(file_path):
    try:
        # --- JAVÍTÁS KEZDETE ---
        # 1. Először lekérjük az infót, hogy tudjuk a samplerate-et
        info = sf.info(file_path)
        samplerate = info.samplerate
        
        # Most már biztonságosan beolvashatjuk az első 45 másodpercet
        # A read függvény visszaadja az adatot és a rátát is, de a rátát már tudjuk (_ jelzi hogy nem kell)
        data, _ = sf.read(file_path, start=0, stop=int(45 * samplerate))
        # --- JAVÍTÁS VÉGE ---

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 2. Spektrális sűrűség (Welch)
        freqs, psd = welch(data, samplerate, nperseg=8192)

        # 3. dB számítás
        psd_db = 10 * np.log10(psd + 1e-12) 
        
        peak_db = np.max(psd_db)
        threshold_db = peak_db - 65 

        detected_cutoff = samplerate / 2 

        for i in range(len(freqs) - 5, 5, -1):
            if (psd_db[i] > threshold_db and 
                psd_db[i-1] > threshold_db and 
                psd_db[i-2] > threshold_db):
                detected_cutoff = freqs[i]
                break

        # 4. Kiértékelés (20250 Hz határ)
        check_limit = 20250
        confidence = "Magas"
        
        if detected_cutoff > check_limit:
            status = "✅ VALÓDI FLAC (Lossless)"
            details = f"A jel érdemi energiát tartalmaz {check_limit} Hz felett is (Érzékelve eddig: {round(detected_cutoff/1000, 2)} kHz)."
        elif detected_cutoff > 19000:
             status = "⚠️ GYANÚS (Lehet 320kbps MP3)"
             details = f"A jel meredeken esik {round(detected_cutoff/1000, 2)} kHz körül."
             confidence = "Közepes"
        else:
            status = "❌ HAMIS FLAC (Alacsony minőségű)"
            details = f"A magas frekvenciák hiányoznak {round(detected_cutoff/1000, 2)} kHz felett."

        plot_freqs, plot_dbs = downsample_for_plot(freqs, psd_db)

        result = {
            "success": True,
            "samplerate": samplerate,
            "detected_cutoff": round(detected_cutoff, 2),
            "check_limit_used": check_limit,
            "status": status,
            "confidence": confidence,
            "details": details,
            "plot_data": {
                "freqs": plot_freqs,
                "dbs": plot_dbs
            }
        }

    except Exception as e:
        result = {
            "success": False,
            "error": str(e) # Itt küldjük vissza a hibaüzenetet
        }

    print(json.dumps(result))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) > 1:
        analyze_flac(sys.argv[1])
    else:
        print(json.dumps({"success": False, "error": "Nincs fájl megadva"}))
