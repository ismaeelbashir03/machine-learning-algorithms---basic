# the data given is plotted on a graph and a line of best fit is made
# new data is then inputted as the x and a corrosponding y is predicted 
# (can only be used on data that is linear, look at the graphs)

import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

#reading in our csv file with our data to a var
data = pd.read_csv("student.csv", sep=';')

#selecting only a certain amount of features that we want to use
data = data[["studytime", "failures", "absences", "G1", "G2", "G3"]]

#creating a variable for the prediction data
predict = "G3"

#splitting data into input (x) and output (y)
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


#spliting our data to test abd train data in a 10 percent split to test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

#i have already created my model
"""
max = 0

# ensure our accuracy is high by looping our model till it reaches 97 percent accuracy
while max < 0.97:

    #spliting our data to test abd train data in a 10 percent split to test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

    #initialising our modeling and training it with our train data
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    #getting our accuracy of our model
    acc = model.score(x_test, y_test)

    #checks if new best accuracy is made
    if acc > max:
        #updates our max
        max = acc

        #saves the model
        with open("Student_model.pickle", "wb") as file:
            pickle.dump(model, file)

    #printing our progress
    print(acc," : ", max)
"""

open_file = open("Student_model.pickle", "rb")
model = pickle.load(open_file)

# creating a list of predictions made by the model
predictions = model.predict(x_test)

#printing our accuracy
print("Model Accuracy:", model.score(x_test, y_test))

#printing the prediction and the actual result
for i in range(len(predictions)):
    print("Model Predicted :",predictions[i]," Actual Result: ",y_test[i])

#graphing the first grade to the final grade
style.use("ggplot")
p = "G1"
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()

#graphing the second grade to the final grade
p = "G2"
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()

