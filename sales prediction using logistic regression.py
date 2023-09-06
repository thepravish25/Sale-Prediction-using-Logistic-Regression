#import pandas library useful for loading the datatset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#load the datasets
dataset=pd.read_csv("C:\\Users\\ASUS\\Downloads\\ML_Day_3_Materials\\DigitalAd_dataset.csv")
print(dataset)

#summarize dataset
print(dataset.shape) #no of rows and coloumns
print(dataset.head(10)) #top 10 values of the data

#seggregate the data X(dependent value) and Y(independent value)
X=dataset.iloc[:,:-1].values
X

Y=dataset.iloc[:,-1].values
Y

#splitting the dataset into training and testing and validation
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


#"Feature Scaling"
#we scale our data to make all the features contribute equally to the result
#Fit_Transform - fit method is calculating the mean and variance of each of the features present in our data
#Transform - Transform method is transforming all the features using the respective mean and variance, 
#We want our test data to be a completely new and a surprise set for our model

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

#training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, Y_train)

#Predicting, wheather new customer with Age & Salary will Buy or Not
age = int(input("Enter New Customer Age: "))
sal = int(input("Enter New Customer Salary: "))
newCust = [[age,sal]]
result = model.predict(sc.transform(newCust))
print(result)
if result == 1:
  print("Customer will Buy")
else:
  print("Customer won't Buy")


#Prediction for all Test Data
Y_pred = model.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))


#EVALUATING MODEL: CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix: ")
print(cm)

print("Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_pred)*100))


