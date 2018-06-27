import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

data = pd.read_csv('Churn_Modelling.csv')
# print data.head()
data.drop(['RowNumber'],1,inplace = True)
data.drop(['CustomerId'],1,inplace = True)
data.drop(['Surname'],1,inplace = True)
# print data.head()
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values

labelencode1 = LabelEncoder()
X[:,1] = labelencode1.fit_transform(X[:,1])
labelencode2 = LabelEncoder()
X[:,2] = labelencode2.fit_transform(X[:,2])
onehotencode = OneHotEncoder(categorical_features = [1])
X = onehotencode.fit_transform(X).toarray()
X = X[:,1:]

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)

feature_scaling = StandardScaler()
X_train = feature_scaling.fit_transform(X_train)
X_test = feature_scaling.fit_transform(X_test)


neural_network = Sequential()

neural_network.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))
neural_network.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
neural_network.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

neural_network.compile(optimizer = 'adam',loss = 'binary_crossentropy' ,metrics =['accuracy'])

neural_network.fit(X_train,Y_train ,batch_size = 10,nb_epoch = 10)

predict = neural_network.predict(X_test)
predict = (predict > 0.5)

confusion_matrix_result = confusion_matrix(Y_test,predict)

print confusion_matrix_result
