"""
Scratch script to play with input parameters,
shapes, and visualize data
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

N_FFT = 4096
WIN_LENGTH = N_FFT//1
HOP_LENGTH = WIN_LENGTH // 16
window = 'hann'

N_MELS = 256

files = librosa.util.find_files('data/tricycle/')
file_index = 500
print(files[file_index])


y, sr = librosa.load(files[file_index], sr=None, offset=0.0)
print(y.shape)
print(sr)
spect = librosa.core.stft(y=y,
                          n_fft=N_FFT,
                          hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH,
                          window=window)


mel = librosa.feature.melspectrogram(y=y,
                                     sr=sr,
                                     n_fft=N_FFT,
                                     hop_length=HOP_LENGTH,
                                     win_length=WIN_LENGTH,
                                     window=window,
                                     n_mels=N_MELS,
                                     center=True,
                                     power=1,
                                     htk=True)


print(sr)
print(mel.shape)
plt.figure(figsize=(10, 10))
#
plt.subplot(2, 2, 1)
librosa.display.specshow(np.abs(spect), y_axis='log', x_axis='time', sr=sr, hop_length=HOP_LENGTH)
plt.colorbar(format='%+2.0f')
plt.title('Spectrogram')
plt.tight_layout()
##
plt.subplot(2, 2, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(spect), ref=np.max), y_axis='log', x_axis='time', sr=sr, hop_length=HOP_LENGTH)
plt.colorbar(format='%+2.0f dB')
plt.title('Power spectrogram')
plt.tight_layout()

mel_db = librosa.amplitude_to_db(mel, ref=np.max)
# pcen_mel = librosa.pcen(mel, eps=1e-10, gain=0.7, bias=0, power=0.125, time_constant=0.25)
pcen_mel = librosa.pcen(mel * (2**31), eps=1e-10, gain=0.7, bias=0, power=0.125, time_constant=0.25)
# pcen_mel = librosa.pcen(mel * (2**31))
plt.subplot(2, 2, 3)
librosa.display.specshow(pcen_mel, y_axis='mel', x_axis='time', sr=sr, hop_length=HOP_LENGTH, fmax=sr / 2.0)
plt.colorbar(format='%+2.0f dB')
plt.title('PCEN spectrogram')
plt.tight_layout()

plt.subplot(2, 2, 4)
librosa.display.specshow(librosa.amplitude_to_db(mel, ref=np.max), y_axis='mel', x_axis='time', sr=sr, hop_length=HOP_LENGTH, fmax=sr / 2.0)
plt.colorbar(format='%+2.0f dB')
plt.title('log Mel spectrogram')
plt.tight_layout()

plt.show()