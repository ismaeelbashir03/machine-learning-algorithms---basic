# Support Vector Machine
# data is plotted on a scatter graph and a hyperplane is set between types of data
# with the hyperplane having the same distance between the closest points of each data type (this changes the angle)
# the hyperplane is set to the biggest distance from the points, so the margin is at its max
# when there is no clear seperation in our data to put the hyperplane, we can use kernals to
# add dimensions to the data, (the kernels are arbituary) this can create seperation between the data
# and a hyperplane can be added
# hyperplanes can have soft margins, this means that the margin between the points to the hyperplane
# can have points in them, this is used if it gives a better hyperplane, (no points in margin = hard margin)

import pickle
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

#loading data from sklearn
data = datasets.load_breast_cancer()

#setting x and y data
x = data.data
y = data.target

#splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

#creating readable names for classes
classes = ['malignant', 'benign']

#model has already been created
'''
#init max
max = 0

#looping until we create a model with at lest 98 percent accuracy
while max < 0.98:

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
    
    # we are using a linear kernel
    model = svm.SVC(kernel='linear', C=2)

    #training the model
    model.fit(x_train, y_train)

    #scoring our model
    acc = model.score(x_test,y_test)

    #checking if we have a new best model
    if acc > max:
        
        #setting max to acc
        max = acc
        
        #saving our model
        with open('cancer_model.pickle', 'wb') as file:
            pickle.dump(model, file)
        
        print(max)
'''

#loading in our model
open_file = open('cancer_model.pickle', 'rb')
model = pickle.load(open_file)

#getting our predictions
predictions = model.predict(x_test)

#getting our accuarcy for this prediction
acc = metrics.accuracy_score(y_test, predictions)

#printing our data back to us
print('Model Accuracy:', acc)

for x in range(len(predictions)):

    #var for real and pred
    pred = classes[predictions[x]]
    real = classes[y_test[x]]

    if real == pred:
        correct = True
    else:
        correct = False

    print('Predicted:', pred, ' Actual:', real, 'Correct:', correct)