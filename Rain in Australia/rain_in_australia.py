# -*- coding: utf-8 -*-
"""
Created on Sat May 29 21:25:14 2021

@author: bymeh
"""
#%% IMPORT LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#%% Read the dataset

data = pd.read_csv("weatherAUS.csv")

#%% Exploratory Data Analysis (EDA)

# what are the data columns ?
data.columns 

# information about the dataset
data.info()
data.describe()


#%%  Finding Missing Values


data_len = len(data)
print(data_len)

# To find missing values
data.columns[data.isnull().any()]
data.isnull().sum() # Here , how many missing values are in the dataset ?



#%% Filling the missing values

# Here , I have been taken a decision to fill missing values with average values for the numerical features.

numerical_variables = ["MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","WindSpeed3pm","WindSpeed3pm","Humidity9am","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm"]

a = len(numerical_variables)

def fill_missing_values(a,numerical_variables):
    for a in numerical_variables:
        data[numerical_variables] = data[numerical_variables].fillna(data[numerical_variables].mean())

fill_missing_values(a,numerical_variables)



#%% Here , I have been droped any Nan values in the dataset.

# we droped categorical "Nan" values

data = data.dropna(how = "any",axis = 0)
data.isnull().sum() # when we check the dataset is it have a nan values ? , outcome is no !!


#%% Categorical Variables

# Here I have been converted categorical variables to numerical variables

# WindGustDir , WindDir9am ,WindDir3pm ,WindSpeed9am , Humidity3pm ,RainToday , RainTomorrow , Location 



from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()



data["WindGustDir"] = label_encoder.fit_transform(data["WindGustDir"])
data["WindDir9am"] = label_encoder.fit_transform(data["WindDir9am"])
data["WindDir3pm"] = label_encoder.fit_transform(data["WindDir3pm"])
data["WindSpeed9am"] = label_encoder.fit_transform(data["WindSpeed9am"])
data["Humidity3pm"] = label_encoder.fit_transform(data["Humidity3pm"])
data["RainToday"] = label_encoder.fit_transform(data["RainToday"])
data["RainTomorrow"] = label_encoder.fit_transform(data["RainTomorrow"])
data["Location"] = label_encoder.fit_transform(data["Location"])


#%%
# In this dataset , RainTomorrow is a target that is why i will change the column name of "RainTomorrow" as a "target"

data = data.rename({'RainTomorrow': 'Target'},axis = 1)

#%% Outlier detection

from collections import Counter

def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers

data.loc[detect_outliers(data,["MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm"])]



#%% Drop outliers
# axis = 0 , which means delete rows.
# axis = 1 , which means delete columns
data = data.drop(detect_outliers(data,["MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm"]),axis = 0).reset_index(drop = True)


#%% Logistic Regression Implemtation

# Beginning of the logistic regression implentation , I'm going to drop "Date" and "Location" columns because of fact that I will not use this column

# Remember ! axis = 1 -> Column , axis = 0 -> Row
data.drop(["Date"],axis = 1,inplace = True) # inplace , which means "drop and save inside of the dataset"
data.drop(["Location"],axis = 1,inplace = True)

y = data.Target.values # Target is RainTomorrow , ı changed the name of this column as a "target"
x_data = data.drop(["Target"],axis = 1)


# %% normalization
# o ile 1 arasında scale etmek (taşımak)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


#%% train test split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size =0.2,random_state=42 )

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


#%% parameter initialize and sigmoid function

# dimension = 30 (nummber of features)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

#w,b = initialize_weights_and_bias(30)

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# %%
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients


#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.title("Number of Iterarion & Cost ")
    plt.show()
    return parameters, gradients, cost_list


#%%  # prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# %% logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 3000)    


#%% sklearn with LR
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("Test Accuracy {}".format(lr.score(x_test.T,y_test.T)))


