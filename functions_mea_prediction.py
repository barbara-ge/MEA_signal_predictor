from scipy.signal import butter, lfilter

def butter_lowpass(lowcut, fs, order=5):
    return butter(order, lowcut, fs=fs, btype='low')

def butter_lowpass_filter(data, lowcut, fs, order):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y