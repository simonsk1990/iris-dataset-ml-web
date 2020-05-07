#trainning our model

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#TF
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from  tensorflow.keras.callbacks import  EarlyStopping



iris = pd.read_csv('/home/sshkilevich/Desktop/CV/DATA/TensorflowData/TF_2_Notebooks_and_Data/DATA/iris.csv') #flowers dataset
#seperating fetures vs classes
X = iris.drop('species',axis=1) # we are droppin column - left with fetures
y = iris['species']#classes
print(y.unique()) #we have 3 classes
encoder = LabelBinarizer()#instanse of this binarizer
y = encoder.fit_transform(y) #hot encoded to 3 classes

#Pre-proccesing Data
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=101) #splittin data test and train
#scaling data
scaler = MinMaxScaler()
scaler.fit(X_train) # we are fitting only on the trainning set - dont want to assume prior knowladge on the test set
scaled_X_train = scaler.transform(X_train) #scaling
scaled_X_test = scaler.transform(X_test)


#create model
model = Sequential()
model.add(Dense(units=4,activation='relu',input_shape=[4,]))
model.add(Dense(units=3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stop = EarlyStopping(patience=10,restore_best_weights=True) #dont need to worry for the num of epochs - 10 epochs no change will stop


#saving model checkpoints:
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5) #saves 5 epochs checkpoint trainning

model.summary()
model.fit(x=scaled_X_train,y=y_train,epochs=1000,validation_data=(scaled_X_test,y_test),callbacks=[early_stop,cp_callback])
#saving model


metrics = pd.DataFrame(model.history.history)
print('end trainning')
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
print(model.evaluate(scaled_X_test,y_test,verbose=0))


model.save("trainning_model.h5")
plt.show()

print('end')
