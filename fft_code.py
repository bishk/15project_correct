import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from IPython.display import Audio

def fft_wrapper(file_name, show_plot = True, n_secs = 90, f_max_plot = 2000, y_lim = .001, pass_title = None):
    sound, samplerate = librosa.load(file_name, sr = None)
    n_frames = n_secs*samplerate
    
    yf = np.fft.fft(sound[:n_frames])
    fs = np.fft.fftfreq(n_frames, 1/samplerate)
    real_scaling = 2.0/n_frames * np.abs(yf[:n_frames//2]) #taking real components, appropriately scaling
    fs_cut = fs[:n_frames//2]
    #T = 1.0 / samplerate
    #yf = scipy.fftpack.fft(sound, n = 1000000)
    #xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    print("fft was complete")
    
    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(fs_cut, real_scaling, '.', markersize = 1, color = 'gray')
        plt.xlim((0, f_max_plot))
        plt.ylim((0, y_lim))
        plt.ylabel("amplitude (AU)")
        plt.xlabel("frequency (inverse seconds)")
        if pass_title is not None:
            plt.title(pass_title)
        #plt.show()
    
    return real_scaling, fs_cut, samplerate