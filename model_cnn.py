from load_data import Load_dataset as Ld
import os
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, Dense
from keras import Sequential



os.chdir(r'C:\Users\praop\Desktop\Research\ECG_FEATURE_EXTRACTION\DATASET')

file_path = 'mit-bih-arrhythmia-database-1.0.0/'

patients = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111',
            '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
            '200', '201', '202', '203', '205', '207', '208', '209', '210','212', '213','214','215',
            '217', '219', '220', '221', '222', '223','228', '230', '231', '232','233','234']



abnormal = ['L', 'R', 'V', '/', 'A', 'f' , 'F', 'j', 'a', 'E', 'J', 'e', 'S']

window = 3
frequency = 360

X_total, Y_total, sym_total = Ld.create_dataset(patients, window, frequency, abnormal)

X_train, X_valid, y_train, y_valid = train_test_split(X_total, Y_total, test_size=0.33, random_state=42)

X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_valid_cnn = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))


model = Sequential()
model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape = (2160,1)))
model.add(Dropout(rate = 0.25))
model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape = (2160,1)))
model.add(Dropout(rate = 0.25))
model.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape = (2160,1)))
model.add(Dropout(rate = 0.25))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

# compile the model - use categorical crossentropy, and the adam optimizer
model.compile(
                loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])


model.fit(X_train_cnn, y_train, batch_size = 4080, epochs= 5 , verbose = 1)