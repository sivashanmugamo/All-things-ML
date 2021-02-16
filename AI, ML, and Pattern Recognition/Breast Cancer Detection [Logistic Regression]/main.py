#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression (with scikit-learn)

# In[ ]:


'''
Importing required libraries

Pandas library is used to read the data in a CSV format and store 
it as a DataFrame
'''
import pandas as pd

'''
Scikit learn library contains functions to learn a model, predict
'''
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# In[ ]:


'''
Reading the CSV file into a DataFrame
'''
wdbc_data_path = "./wdbc_data.csv"

pd.set_option("display.max_colwidth", 50)
wdbc_data_df = pd.read_csv(wdbc_data_path, delimiter=",", encoding="utf-8")


# In[ ]:


'''
Rearranging columns to make the datareadable by train_test_split
'''

target = wdbc_data_df["diagnosis"]
Y = [1 if i == 'M' else 0 for i in target] # Replacing 'M' by 1 & 'B' by 0

del wdbc_data_df['id']
del wdbc_data_df['diagnosis']


# In[ ]:


'''
Performing scaling to bring the values of the features to the same 
level of magnitude
'''

scaler = StandardScaler()
X = scaler.fit_transform(wdbc_data_df)

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

full_df = pd.concat([X, Y], axis = 1)


# In[ ]:


'''
Splitting 20% of data for testing and the rest for training the model
'''

X_train, X_test, Y_train, Y_test = train_test_splitt(X, Y, test_size=0.2)


# In[ ]:


'''
Training a Logistic Regression model
'''

model = LogisticRegression(solver = "liblinear")
model.fit(X_train, Y_train)

Y_predicted = model.predict(X_test)

print(classification_report(Y_test, Y_predicted))
print(confusion_matrix(Y_test, Y_predicted))


# # Logistic Regression using Gradient Descent

# In[ ]:


'''
Importing required libraries
'''

import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# In[ ]:


'''
Learning rate determines the rate at which the slope and intercept are
learnt
'''

L = 0.01


# In[ ]:


def get_data(path):
    '''
    This function is to load the wdbc.data into a Dataframe
    '''
    
    pd.set_option("display.max_colwidth", 50)
    data = pd.read_csv(path, delimiter = ",", encoding = "utf-8", header = None)
    
    Y = np.asarray([1 if i=='M' else 0 for i in data[1]])
    
    del data[0]
    del data[1]
    
    X = feature_scaling(data)
    
    return X, Y


# In[ ]:


def model(X, Y):
    '''
    This function is to perform logistic regression using gradient descent method
    '''
    
    m = np.zeros(30)[np.newaxis]
    c = 0
    
    Entropy = []
    
    for i in range(500):
        Y_cap = np.add(np.matmul(X, m.T), np.array([c]))
        
        Entropy.append(((-1)/len(X))*(np.sum((Y * np.log(Y_cap)))+((1-Y)*(np.log(1-Y_cap)))))
        
        D_m = (-1)*(1/len(X))*(np.sum(X * np.subtract(Y, Y_cap)))
        D_c = (-1)*(1/len(X))*(np.sum(np.subtract(Y, Y_cap)))
        m = np.subtract(m, (np.multiply(L, D_m)))
        c = np.subtract(c, (np.multiply(L, D_c)))
        
    return m, c


# In[ ]:


def feature_scaling(data):
    '''
    This function is to scale the values of the features to the same
    order of magnitude
    '''
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data


# In[ ]:


def predict(X, m, c):
    '''
    This function is to predict the value for Y from the given test input
    '''
    
    Y_i = 1/(1 + np.exp(np.matmul(X, m.T) + c))
    Y_i = np.asarray([1 if i > 0.5 else 0 for i in Y_i])[:, np.newaxis]
    
    return Y_i


# In[ ]:


def metrics_report(predicted_data, test_data):
    '''
    This function is to calculate the accuracy, precision, recall and
    confusion matrix
    '''
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for j in range(0, len(test_data)):
        if(predicted_data[j] == 1 and test_data[j] == 1):
            true_positive = true_positive + 1
        elif(predicted_data[j] == 0 and test_data[j] == 1):
            false_negative = false_negative + 1
        elif(predicted_data[j] == 1 and test_data[j] == 0):
            false_positive = false_positive + 1
        elif(predicted_data[j] == 0 and test_data[j] == 0):
            true_negative = true_negative + 1
            
    confusion_matrix = np.asarray([[true_positive, false_positive],[false_negative, true_negative]])

    accuracy = ((true_positive + true_negative)/(true_positive + false_positive + false_negative + true_negative))*100
    precision = (true_positive)/(true_positive + false_positive)
    recall = (true_positive)/(true_positive + false_negative)

    report = "Accuracy : "+str(round(accuracy, 2))+"%"+"\n"+"Precision : "+str(round(precision, 3))+"\n"+"Recall : "+str(round(recall, 3))+"\n"

    return report, confusion_matrix


# In[ ]:


data_path = "./Input/wdbc.data"

X, Y = get_data(data_path)

X_train = X[:(math.ceil(len(X)*0.8))]
Y_train = Y[:(math.ceil(len(Y)*0.8))][:, np.newaxis]

X_test = X[(math.ceil(len(X)*0.8)):]
Y_test = Y[(math.ceil(len(Y)*0.8)):][:, np.newaxis]

slope, intercept = model(X_train, Y_train)

Y_predict = predict(X_test, slope.flatten(), np.asscalar(intercept))

report, confusion_matrix = metrics_report(Y_predict, Y_test)

print(report)
print(confusion_matrix)

