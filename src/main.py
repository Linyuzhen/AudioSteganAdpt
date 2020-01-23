import numpy as np
import tensorflow as tf
import read_data
from sklearn.model_selection import train_test_split
from tensorflow import keras
from dann import DANN

# Load data&label
cover_dir = r'.\cut_datasets\iPhone 6s_1s'
stego_dir = r'.\Steg_datasets\LSBM_1\Stego_iPhone 6s'
cover_data = read_data.read_datasets(cover_dir)
stego_data = read_data.read_datasets(stego_dir)
X = np.concatenate((cover_data,stego_data),axis=0)
print(X.shape)
y = read_data.get_label(X.shape[0],positive=0.5)

# Unlabeled Data
un_cover_dir = r'.\cut_datasets\Samsung Galaxy S5_1s'
un_stego_dir = r'.\Steg_datasets\LSBM_1\Stego_Samsung Galaxy S5'
un_cover_data = read_data.read_datasets(un_cover_dir)
un_stego_data = read_data.read_datasets(un_stego_dir)
DX = np.concatenate((un_cover_data,un_stego_data),axis=0)
Dy = read_data.get_label(DX.shape[0],positive=0.5)

print(DX.shape)

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)
D_trainX, D_testX, D_train_y, D_test_y = train_test_split(DX,Dy,test_size=0.6,random_state=0)

X_train = X_train.reshape(-1, X.shape[1], 1)
D_trainX = D_trainX.reshape(-1, X.shape[1], 1)
D_testX = D_testX.reshape(-1, X.shape[1], 1)
X_test = X_test.reshape(-1, X.shape[1], 1)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)


# Initiate the model
dann = DANN(summary=True,sample_length=X.shape[1],channels=1, classes=2, features=128, batch_size=64, model_plot=False)
# Train the model
dann.train(X_train, D_trainX, y_train, epochs=50, batch_size=64)
# Test the model
dann.evaluate(testX=X_test,testY=y_test,save_pred='./label_pred.npy',verbose=1)
# Evaluate
dann.evaluate(testX=D_testX,save_pred='./unlabel_pred.npy',verbose=1)



