##################################################################################
# SciKit-Learn based breast cancer class categorization algorithm                #
#                                                                                #
# This example k-nearest neighbor algorithm categorize the class of              #
# breast cancer test results with the trained model from hundreds of data sets   #
#                                                                                #
# What do we use here,                                                           #
#                                                                                #
# numpy                  - to build data in array format which is feasible to    #
#                          apply machine learning model                          #
# pandas                 - to read data set from a CSV file                      #
# sklearn (scikit-learn) - to use existing KNeighborsClassifier algorithm        #
#                          and to pre-process data set                           #
#                                                                                #
# The data set we use here is from University of California for learning purpose #
# The last field says the class: 2 for Benign and 4 for Malignant                #
#                                                                                #
##################################################################################

import numpy as np
import pandas as pandas
from sklearn import model_selection, neighbors

filename = 'breast-cancer-wisconsin.data'  # data set file

# Start of data set processing
df = pandas.read_csv(filename)
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# End of data set pre-processing

clf = neighbors.KNeighborsClassifier()  # Choosing machine learning model
clf.fit(X_train, y_train)  # training happens here

accuracy = clf.score(X_test, y_test)  # testing happens here

print(accuracy * 100)  # This shows the accuracy of our model in percent

new_data = [[4, 2, 1, 1, 1, 2, 3, 2, 1], [6, 7, 4, 3, 2, 7, 6, 8, 4]]  # Class categorizing unknown data set

example_measures = np.array(new_data)  # pre-processing new data set predict using our trained model

example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)  # Prediction happens here

print(prediction)  # Prints the class of the breast cancer test results for the new data
