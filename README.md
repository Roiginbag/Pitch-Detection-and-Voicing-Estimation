# Pitch Detection and Voicing Estimation
Pitch Detection and Voicing Estimation implementation of Master's Course Speech Technologies

Requirements
-----------
* Python 3.5
* Numpy 1.12.0
* Scipy 0.18.1
* Keras 1.2.1
* Theano 0.8.2

Usage
-----------
```
Usage: pitch.py [options]

Options:
  -h, --help            show this help message and exit
  -w WINDOWLENGTH, --windowlength=WINDOWLENGTH
                        windows length (ms)
  -f FRAMELENGTH, --framelength=FRAMELENGTH
                        frame shift (ms)
  -d DATADIR, --datadir=DATADIR
                        data folder
  -r RNN, --rnn=RNN     use rnn for voicing (0 or 1)
  -m METHOD, --method=METHOD
                        pitch estimator method: hps, cepstrum,
                        autocorr(default)
```
