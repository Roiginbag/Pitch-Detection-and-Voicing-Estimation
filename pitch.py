from __future__ import print_function, division

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, kaiser, hamming, decimate
from keras.models import load_model

# Author: Carlos Roig
# Kickstart code from Professor Jose A. R. Fonollosa

model = load_model("VoicingNet_model.h5")
model.load_weights("VoicingNet_weights.h5")

# Parabolic interpolation method from https://gist.github.com/endolith/255291
def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def autocorr_method(frame, rate, rnn):
    """Estimate pitch using autocorrelation
    """
    # Resize the frame to the value of the training samples
    # if the rnn flag is activated
    if(rnn != 0 and len(frame)<1536):
        frame_d=np.zeros(1536)
        frame_d[0:len(frame)]= frame
    else:
        frame_d = frame
    # If the rnn flag is activated prepare the frame
    if(rnn != 0):
        frame_d = np.array(frame_d).reshape((1,len(frame_d)))
    if(rnn != 0 and model.predict_classes(frame_d) == 0):
        return 0
    else:
        defvalue = (0.0, 1.0)

        frame = frame.astype(np.float)
        frame -= frame.mean()
        #frame = frame/np.max(frame)
        N = len(frame)
        noise = np.max(frame)*(np.random.random_sample(N)-.5)*.05
        frame = frame+noise

        # Calculate autocorrelation using scipy correlate
        amax = np.abs(frame).max()
        if amax > 0:
            frame /= amax
        else:
            return defvalue

        corr = correlate(frame, frame)
        # keep the positive part
        corr = corr[int(len(corr)/2):]

        # Find the first minimum
        dcorr = np.diff(corr)
        rmin = np.where(dcorr > 0)[0]
        if len(rmin) > 0:
            rmin1 = rmin[0]
        else:
            return defvalue

        # Find the next peak
        peak = np.argmax(corr[rmin1:]) + rmin1
        rmax = corr[peak]/corr[0]
        f0 = rate / peak

        if rmax > 0.6 and f0 > 50 and f0 < 550:
            return f0
        else:
            return 0;

def hps_method(frame, rate):
    N = len(frame)
    
    # De-mean the frame
    frame -= np.mean(frame,dtype=np.int16)
    
    # Apply a window to the frame
    windowed_frame = frame*kaiser(N,100)
    
    # Compute the real fft and get the spectrum
    spectrum = np.log(np.abs(np.fft.rfft(windowed_frame)))
    
    hps = np.copy(spectrum)
    # Downsample the spectrum and add the log of spectrum instead
    # of multiplying
    for i in range(2,5):
        y = decimate(spectrum, i)
        hps[:len(y)] += y
    
    # Find the highest peak and interpolate to get a more accurate result
    peak = np.argmax(hps[:len(y)])
    interp = parabolic(hps, peak)[0]
    
    # Convert to the corresponding frequency
    f0 = rate*interp/N

    # Process the interpolated result
    if np.max(hps) > 45 and f0 > 50 and f0 < 550:
        return f0
    else:
        return 0

def cepstrum_method(frame, rate, rnn):
    # Resize the frame to the value of the training samples
    # if the rnn flag is activated
    if(rnn != 0 and len(frame)<1536):
        frame_d=np.zeros(1536)
        frame_d[0:len(frame)]= frame
    else:
        frame_d = frame
    # If the rnn flag is activated prepare the frame
    if(rnn != 0):
        frame_d = np.array(frame_d).reshape((1,len(frame_d)))
    if(rnn != 0 and model.predict_classes(frame_d) == 0):
        return 0
    else:
        # Normalize the frame to -1 to 1
        frame = frame/np.max(frame)
        N = len(frame)
        noise = (np.random.random_sample(N)-.5)*.05
        frame = frame+noise
        
        # Apply a window to the frame
        windowed_frame = frame*hamming(N)
        # Compute ceptrsum
        cepstrum = np.abs(np.fft.irfft(np.log(np.abs(np.fft.rfft(windowed_frame)))))
        start = int(N/12)
        end = int(N/2)
        
        # Find the highest peak and interpolate to get a more accurate result
        peak = np.argmax(cepstrum[start:end])
        
        # Convert to the corresponding frequency
        f0 = rate/(start+peak)
        
        if f0 > 60 and f0 < 450:
            return(f0)
        else:
            return(0)

def pitch_estimator(frame, rate, method, rnn):
    if(method == "hps"):
        return(hps_method(frame, rate))
    elif(method == "cepstrum"):
        return(cepstrum_method(frame, rate, rnn))
    elif(method == "autocorr"):
        return(autocorr_method(frame, rate, rnn))
    else:
        return(autocorr_method(frame, rate))

def wav2f0(options, gui):
    with open(gui) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            filename = os.path.join(options.datadir, line + ".wav")
            f0_filename = os.path.join(options.datadir, line + ".f0")
            print("Processing:", filename, '->', f0_filename)
            rate, data = wavfile.read(filename)
            with open(f0_filename, 'wt') as f0file:
                nsamples = len(data)

                # From miliseconds to samples
                ns_windowlength = int(round((options.windowlength * rate) / 1000))
                ns_framelength = int(round((options.framelength * rate) / 1000))
                for ini in range(0, nsamples - ns_windowlength + 1, ns_framelength):
                    frame = data[ini:ini+ns_windowlength]
                    f0 = pitch_estimator(frame, rate, options.method, options.rnn)
                    print(f0, file=f0file)               


def main(options, args):
    wav2f0(options, args[0])

if __name__ == "__main__":
    import optparse
    optparser = optparse.OptionParser()
    optparser.add_option(
        '-w', '--windowlength', type='float', default=32,
        help='windows length (ms)')
    optparser.add_option(
        '-f', '--framelength', type='float', default=15,
        help='frame shift (ms)')
    optparser.add_option(
        '-d', '--datadir', type='string', default='data',
        help='data folder')
    optparser.add_option(
        '-r', '--rnn', type='int', default=False,
        help='use rnn for voicing (0 or 1)')
    optparser.add_option(
        '-m', '--method', type='string', default='autocorr',
        help='pitch estimator method: hps, cepstrum, autocorr(default)')

    options, args = optparser.parse_args()

    main(options, args)
