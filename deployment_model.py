# Trainning on all the data for deployment


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 4)                 20
# _________________________________________________________________
# dense_1 (Dense)              (None, 3)                 15
# =================================================================
# Total params: 35
# Trainable params: 35
# Non-trainable params: 0



import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

#TF
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

#scaling
scaler = MinMaxScaler()
scaler.fit(X) # we are fitting only on the trainning set - dont want to assume prior knowladge on the test set
scaled_X= scaler.transform(X) #scaling

#building model
model = Sequential()
model.add(Dense(units=4,activation='relu',input_shape=[4,]))
model.add(Dense(units=3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#trainning on all data for deployment
early_stop = EarlyStopping(patience=10,restore_best_weights=True) #dont need to worry for the num of epochs - 10 epochs no change will stop
model.fit(x=scaled_X,y=y,epochs=1000,callbacks=[early_stop])
print('end trainning')


#saves
model.save("deployment_model.h5") #model
dump(scaler,'iris_scaler.pkl') #scaler for new input

print('end')
