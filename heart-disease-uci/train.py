import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
from keras import models
from keras import layers

df = pd.read_csv("heart.csv")

# Data standardization
df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['target'] = df['target']
df = df_scaled

X = df.loc[:, df.columns != 'target']
y = df.loc[:, 'target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(13,)) )
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath= dir_path + "/weights/" + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=50000,
                    batch_size=16,
                    callbacks=callbacks_list,
                    validation_data=(X_val, y_val))


preds = model.evaluate(X_test, y_test)
print(preds)

