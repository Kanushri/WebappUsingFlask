import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['dayofweek'] = df['effective_date'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Feature = df[['Principal','terms','age','Gender','weekend']]
print(df['Principal'].dtype)
print(df['Principal'].value_counts())
print()
print(df['terms'].dtype)
print(df['terms'].value_counts())
print()
print(df['age'].dtype)
print(df['age'].value_counts())
print()
print(df['Gender'].dtype)
print(df['Gender'].value_counts())
print()
print(df['weekend'].dtype)
print(df['weekend'].value_counts())
print()
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.rename(columns={'High School or Below':'HighSchoolorBelow'},inplace=True)
print(Feature.columns) 
X = Feature
y = df['loan_status'].values
X= preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
k = 7
neigh7 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh7.predict(X_test)
ynee=neigh7.predict([[1000,15,32,0,1,1,0,0]])
print(ynee)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh7.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
with open('bike_model_xgboost.pkl', 'wb') as file:pickle.dump(neigh7, file)

