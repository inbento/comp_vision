import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

class FFT_Noise:
    def __init__(self, sample_rate=44100, fft_size=2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = fft_size // 4
        self.window = np.hanning(fft_size)
    
    def load_audio(self, filepath):
        audio_data, sr = sf.read(filepath)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data.flatten()
    
    def my_fft(self, signal):
        n = len(signal)
        
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 <<= 1
        
        if next_pow2 != n:
            signal = np.pad(signal, (0, next_pow2 - n), 'constant')
            n = next_pow2
        
        if n == 1:
            return signal
        
        even = self.my_fft(signal[::2])
        odd = self.my_fft(signal[1::2])
        
        k = np.arange(n // 2)
        factor = np.exp(-2j * np.pi * k / n) * odd
        
        result = np.zeros(n, dtype=complex)
        result[:n//2] = even + factor
        result[n//2:] = even - factor
        
        return result
    
    def my_ifft(self, spectrum):
        n = len(spectrum)
        
        if n == 1:
            return spectrum
        
        even = self.my_ifft(spectrum[::2])
        odd = self.my_ifft(spectrum[1::2])
        
        k = np.arange(n // 2)
        factor = np.exp(2j * np.pi * k / n) * odd
        
        result = np.zeros(n, dtype=complex)
        result[:n//2] = even + factor
        result[n//2:] = even - factor
        
        return result / n
    
    def estimate_noise_profile(self, audio, noise_duration=0.5):
        noise_samples = int(noise_duration * self.sample_rate)
        if len(audio) < noise_samples:
            noise_samples = len(audio) // 4
        
        noise_segment = audio[:noise_samples]
        
        noise_frames = []
        for i in range(0, len(noise_segment) - self.fft_size, self.hop_size):
            frame = noise_segment[i:i + self.fft_size]
            if len(frame) == self.fft_size:
                windowed = frame * self.window
                spectrum = self.my_fft(windowed)
                magnitude = np.abs(spectrum[:self.fft_size // 2 + 1])
                noise_frames.append(magnitude)
        
        if not noise_frames:
            padded = np.pad(noise_segment, (0, self.fft_size - len(noise_segment)), 'constant')
            spectrum = self.my_fft(padded * self.window[:len(padded)])
            noise_frames = [np.abs(spectrum[:self.fft_size // 2 + 1])]
        
        noise_frames = np.array(noise_frames)
        
        noise_mean = np.mean(noise_frames, axis=0)
        noise_std = np.std(noise_frames, axis=0)
        
        return noise_mean, noise_std
    
    def compute_perceptual_weights(self, frequencies):
        weights = np.ones_like(frequencies)
        
        speech_mask = (frequencies >= 300) & (frequencies <= 3400)
        weights[speech_mask] = 1.2
        
        bass_mask = (frequencies >= 80) & (frequencies < 300)
        weights[bass_mask] = 1.1
        
        return weights
    
    def balanced_spectral_subtraction(self, audio, noise_mean, noise_std):
        result = np.zeros_like(audio)
        
        num_frames = (len(audio) - self.fft_size) // self.hop_size + 1
        
        freq_bins = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)[:self.fft_size // 2 + 1]
        perceptual_weights = self.compute_perceptual_weights(np.abs(freq_bins))
        
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.fft_size
            
            if end > len(audio):
                break
                
            frame = audio[start:end]
            
            windowed = frame * self.window
            spectrum = self.my_fft(windowed)
            
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            half_len = self.fft_size // 2 + 1
            magnitude_half = magnitude[:half_len]
            
            if len(noise_mean) < half_len:
                noise_mean_ext = np.pad(noise_mean, (0, half_len - len(noise_mean)), 'edge')
                noise_std_ext = np.pad(noise_std, (0, half_len - len(noise_std)), 'edge')
            else:
                noise_mean_ext = noise_mean[:half_len]
                noise_std_ext = noise_std[:half_len]
            
            threshold = noise_mean_ext + 2.5 * noise_std_ext
            cleaned_magnitude_half = np.zeros_like(magnitude_half)
            
            for j in range(len(magnitude_half)):
                mag = magnitude_half[j]
                thr = threshold[j]
                
                if mag > thr:
                    excess_ratio = (mag - thr) / (thr + 1e-10)
                    gain = 1.0 + min(excess_ratio, 1.2)
                    
                    perceptual_gain = perceptual_weights[j] if j < len(perceptual_weights) else 1.0
                    cleaned_magnitude_half[j] = mag * gain * perceptual_gain
                else:
                    cleaned_magnitude_half[j] = mag * 0.001
            
            cleaned_magnitude = np.concatenate([
                cleaned_magnitude_half,
                cleaned_magnitude_half[-2:0:-1]
            ])
            
            cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
            
            cleaned_frame = np.real(self.my_ifft(cleaned_spectrum))
            
            result[start:end] += cleaned_frame * self.window
        
        envelope = np.zeros_like(audio)
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.fft_size
            if end <= len(envelope):
                envelope[start:end] += self.window ** 2
        
        envelope[envelope < 1e-6] = 1
        result = result / envelope
        
        original_peak = np.max(np.abs(audio))
        cleaned_peak = np.max(np.abs(result))
        
        if cleaned_peak > 1e-10:
            target_peak = original_peak * 0.9
            
            gain_needed = target_peak / cleaned_peak
            
            gain_needed = min(gain_needed, 5.0)
            
            result = result * gain_needed
        
        return result
    
    def balanced_noise_reduction(self, audio, noise_duration=1):
        noise_mean, noise_std = self.estimate_noise_profile(audio, noise_duration)
        
        cleaned = self.balanced_spectral_subtraction(audio, noise_mean, noise_std)
        
        if np.max(np.abs(audio)) > 0:
            original_norm = audio / np.max(np.abs(audio))
        else:
            original_norm = audio
            
        if np.max(np.abs(cleaned)) > 0:
            cleaned_norm = cleaned / np.max(np.abs(cleaned))
        else:
            cleaned_norm = cleaned
        
        fft_original = self.my_fft(original_norm[:len(original_norm)//2])
        magnitude_original = np.abs(fft_original)
        
        return cleaned, original_norm, cleaned_norm, magnitude_original, noise_mean
    
    def save_audio(self, audio_data, filename):
        """
        ПРОСТОЕ сохранение аудио с нормализацией для предотвращения клиппинга
        """
        max_val = np.max(np.abs(audio_data))
        
        if max_val > 0:
            audio_to_save = audio_data * 0.95 / max_val
        else:
            audio_to_save = audio_data
        
        sf.write(filename, audio_to_save, self.sample_rate)
        print(f"Сохранено: {filename}")
    
    def plot_results(self, original_norm, cleaned_norm, magnitude_original, noise_mean):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        n = len(magnitude_original)
        freq = np.fft.fftfreq(n * 2, 1/self.sample_rate)[:n]
        
        fft_cleaned = self.my_fft(cleaned_norm[:len(cleaned_norm)//2])
        magnitude_cleaned = np.abs(fft_cleaned)
        
        ax.plot(freq, magnitude_original, alpha=0.7, label='Оригинал', color='blue')
        ax.plot(freq[:len(magnitude_cleaned)], magnitude_cleaned, 
            alpha=0.7, label='Очищенный', color='red')
        ax.set_title("Сравнение спектров до и после шумоподавления", fontsize=14)
        ax.set_xlabel("Частота (Hz)", fontsize=12)
        ax.set_ylabel("Амплитуда", fontsize=12)
        ax.legend(fontsize=12)
        ax.set_xlim(0, 5000)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    
    denoiser = FFT_Noise(sample_rate=44100, fft_size=4096)    
    filepath = input("Введите путь к аудиофайлу: ")
    audio = denoiser.load_audio(filepath)
    duration = len(audio) / denoiser.sample_rate
    print(f"Загружено {duration:.1f} секунд аудио")

    cleaned_raw, original_norm, cleaned_norm, magnitude, noise_profile = denoiser.balanced_noise_reduction(audio)
    
    clean_name = input("Имя для очищенного файла: ").strip()
    if not clean_name.endswith('.wav'):
        clean_name += '.wav'
    
    denoiser.save_audio(cleaned_raw, clean_name)
    denoiser.plot_results(original_norm, cleaned_norm, magnitude, noise_profile)

if __name__ == "__main__":
    main()