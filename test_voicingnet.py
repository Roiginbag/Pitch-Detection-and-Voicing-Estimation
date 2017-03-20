import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, kaiser, decimate
from keras.models import load_model
from keras.preprocessing import sequence

model = load_model("VoicingNet_model.h5")
model.load_weights("VoicingNet_weights.h5")

X_db = []
y_db = []
# Load the database in memory
# We used a subset of the PTDB_TUG with 500 audio files
for line in open("pitch/pitch/ST201701_pitch/pda_ue.gui"):
    line = line.strip()
    if len(line) == 0:
        continue
    filename = os.path.join("path-to-data", line + ".wav")
    f0_filename = os.path.join("path-to-data", line + ".f0ref")
    rate, data = wavfile.read(filename)
    with open(f0_filename) as f0file:
        f0file = f0file.readlines()
        nsamples = len(data)
        # From miliseconds to samples
        ns_windowlength = int(round((32 * rate) / 1000))
        ns_framelength = int(round((10 * rate) / 1000))
        i = 0
        X_db = np.zeros([32860,1536])
        y_db = np.zeros(32860)
        for ini in range(0, nsamples - ns_windowlength + 1, ns_framelength):
            X_db[i][0:640] = data[ini:ini+ns_windowlength]
            if(i < len(f0file)):
                if(f0file[i] == '0\n'):
                    y_db[i] = 0
                else:
                    y_db[i] = 1
            else:
                y_db[i] = 1
            i+=1

print("Testing samples shape:", X_db.shape)
print("Testing labels shape:", y_db.shape)

scores = model.evaluate(X_db, y_db, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
