import librosa
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Load audio ----------
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # sr=None keeps original sample rate
    duration = len(y) / sr
    print(f"Duration: {duration:.2f} seconds, Sample Rate: {sr} Hz")
    return y, sr

# ---------- 2. Get frequency spectrum ----------
def get_frequency_spectrum(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    freq = np.fft.fftfreq(len(fft), 1/sr)
    
    # Only keep positive frequencies
    mask = freq >= 0
    freq = freq[mask]
    magnitude = magnitude[mask]
    
    return freq, magnitude

# ---------- 3. Get dominant frequency ----------
def get_dominant_frequency(y, sr):
    freq, magnitude = get_frequency_spectrum(y, sr)
    index = np.argmax(magnitude)
    dom_freq = freq[index]
    print(f"Dominant frequency: {dom_freq:.2f} Hz")
    return dom_freq

# ---------- 4. Pitch detection ----------
def get_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    index = magnitudes.argmax()
    pitch = pitches.flatten()[index]
    print(f"Detected pitch: {pitch:.2f} Hz")
    return pitch

# ---------- 5. Amplitude at a specific frequency ----------
def amplitude_at_frequency(y, sr, target_freq):
    freq, magnitude = get_frequency_spectrum(y, sr)
    # Find closest frequency index
    idx = (np.abs(freq - target_freq)).argmin()
    amp = magnitude[idx]
    print(f"Amplitude at {target_freq} Hz: {amp:.2f}")
    return amp

# ---------- 6. Plot frequency spectrum ----------
def plot_spectrum(y, sr):
    freq, magnitude = get_frequency_spectrum(y, sr)
    plt.figure(figsize=(10,4))
    plt.plot(freq, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum")
    plt.xlim(0, 5000)  # optional: show only 0-5k Hz
    plt.show()

# ---------- 7. Example usage ----------
file_path = "your-audio.mp3"
y, sr = load_audio(file_path)
get_dominant_frequency(y, sr)
get_pitch(y, sr)
amplitude_at_frequency(y, sr, 440)  # amplitude at A4
plot_spectrum(y, sr)
