import os
import numpy as np
from scipy.io import wavfile
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM

X_db = []
y_db = []
# Load the database in memory
# We used a subset of the PTDB_TUG with 500 audio files
for line in open("ptdb_tug_sorted.gui"):
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
        for ini in range(0, nsamples - ns_windowlength + 1, ns_framelength):
            X_db.append(data[ini:ini+ns_windowlength])
            if(i < len(f0file)):
                if(f0file[i] == '0.0\n'):
                    y_db.append(0)
                else:
                    y_db.append(1)
            else:
                y_db.append(1)
            i+=1

# The resulting number of frames was more than 350000 frames
# Which were split into training and validation
X_db = np.array(X_db)
y_db = np.array(y_db)
X_train = X_db[:300000]
y_train = y_db[:300000]
X_test = X_db[300000:]
y_test = y_db[300000:]

print(X_db.shape)
print(y_db.shape)

s_w_voiced = (len(y_db)-np.sum(y_db))/len(y_db)
s_w_unvoiced = (np.sum(y_db))/len(y_db)

class_weight = {0 : s_w_unvoiced, 1: s_w_voiced}
print(class_weight)

max_features = 41834
embedding_vecor_length = 32

# Define the model with a 1D Convolutional layer and LSTM
model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=1536))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(LSTM(output_dim=10, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the network
model.fit(X_train, y_train, batch_size=128, class_weight=class_weight, nb_epoch=3)

# Evaluate the accuracy of the network with the validation set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Finally save the weights of the model
model.save_weights('VoicingNet_weights.h5')
model.save('VoicingNet_model.h5')