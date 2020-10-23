# An Optimal Deep Learning Framework for Detecting Abnormal Heartbeats Using ECG Signals
The following project detects abnormal heart beats using Electrocardiogram (ECG) signals. A Convolutional Neural Network is used to predict if the given heartbeat has arrhythmia using a 6 second window.

This project was inspired from Andrew Ng’s team’s work on heart arrhythmia detector. 
For detailed information please refer to [https://stanfordmlgroup.github.io/projects/ecg/](https://stanfordmlgroup.github.io/projects/ecg/)


## FILES
- [`load_data`](load_data) : This folder contains a python file that extracts ECG signals, labels, and annotations from the dataset and processes it in order to feed it into the CNN model.

- [`model_cnn.py`](model_cnn.py) : Code that trains a CNN model that is used to predict if a given heartbeat has arrhythmia.


## Dependensies

- [Tensorflow](http://tensorflow.org)

- [Keras](http://keras.io)


## DATASET 
We will use the MIH-BIH Arrythmia dataset from https://physionet.org/content/mitdb/1.0.0/ which is made available under the ODC Attribution License.
The dataset consists of 48 half-hour two-channel ECG recordings which is measured at a frequency of 360Hz.


## MODEL 
We have used a 1-D Convolutional Neural Network (CNN).
A CNN is a deep learning model that uses kernals (or filters) and convolutional operators to reduce the number of parameters. Our model uses Dropout to reduce overfitting of the dataset.


## RESULTS
Our model achieved a training accuracy of 98.2% and a testing accuracy of 87%. 


