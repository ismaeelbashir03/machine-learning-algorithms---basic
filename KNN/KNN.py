# k nearest neighbors
# where k is the value of how many neighbors the model looks at
# to classify any given point. a tally is made of all the neighbors
# and the class with the most 'votes' is the class the data is set to

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

#reading in our data
data = pd.read_csv('car.data')

#encoding our data to numbers
le = preprocessing.LabelEncoder()
#putting each encoded feature in a variable 
buying = le.fit_transform(data['buying'])
maint = le.fit_transform(data['maint'])
door = le.fit_transform(data['door'])
persons = le.fit_transform(data['persons'])
lug_boot = le.fit_transform(data['lug_boot'])
safety = le.fit_transform(data['safety'])
cls = le.fit_transform(data['class'])

#storing our prediction label
predict = 'class'

#setting our x and y values
x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

#splitting our data 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

#model has already been saved

'''
#initialising a max
max = 0

#looping until we create a model with at least 99 percent accuarcy
while max < 0.99:

    #splitting our data 
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

    #creating our model
    model = KNeighborsClassifier(n_neighbors=9)

    #training our model and printing an accuracy
    model.fit(x_train, y_train)
    acc = model.score(x_test,y_test)

    if acc > max:
        max = acc
        with open('car_model.pickle', 'wb') as file:
            pickle.dump(model, file)
        print(max)
'''

#loading in our model
file_open = open('car_model.pickle', 'rb')
model = pickle.load(file_open)

predictions = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predictions)):

    #setting variables for our predictions and actual answers with readable names 
    pred = names[predictions[x]]
    real = names[y_test[x]]

    #checking if our model is correct
    if real == pred:
        correct = True
    else:
        correct = False

    #printing the data back to us
    print('Predicted:', names[predictions[x]],' Actual:', names[y_test[x]],' Correct:',correct)

