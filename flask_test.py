####testing
import requests
flower_example  = {
'sepal_length':5.1,
'sepal_width':3.5,
'petal_length':1.4,
'petal_width':0.2
}

result = requests.post(url='http://localhost:5000',json=flower_example)

print(result.status_code)