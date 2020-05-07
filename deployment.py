from joblib import load
from tensorflow.keras.models import load_model
import numpy as np

flower_model=load_model('deployment_model.h5')
flower_scaler = load('iris_scaler.pkl')



def return_prediction(json,model=flower_model,scalar=flower_scaler):

    #break components from json
    s_len = json['sepal_length']
    s_wid = json['sepal_width']
    p_len = json['petal_length']
    p_wid = json['petal_width']


    flower = [[s_len,s_wid,p_len,p_wid]] #attributes input
    classes = np.array(['setosa', 'versicolor', 'virginica'])

    flower = scalar.transform(flower) #transforming the input

    class_index = model.predict_classes(flower) #predicting the class index based on the attributes


    return classes[class_index][0]


