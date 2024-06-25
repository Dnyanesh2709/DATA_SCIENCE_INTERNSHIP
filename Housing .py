import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE   #Synthetic Minority Oversampling (SMOTE) oversampling
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("C:/Users/Lucky/Documents/Housing.csv")

print(df.isnull().sum())


Y = df['MEDV']
X = df.drop(columns=['MEDV'])
le = LabelEncoder()
df['CHAS'] = le.fit_transform(df['CHAS'])
df['NOX'] = le.fit_transform(df['NOX'])
df['LSTAT'] = le.fit_transform(df['LSTAT'])

print(X['CHAS'])
print(X['NOX'])
print(X['LSTAT'])

df['MEDV']=pd.cut(df['MEDV'],5,labels=['0','1','2','3','4'])
df['CHAS'] = df['CHAS'].astype(float)
df['NOX'] = df['NOX'].astype(float)
df['LSTAT'] = df['LSTAT'].astype(float)
df['MEDV'] = df['MEDV'].astype(float)


X = df.drop(columns=['MEDV'])
Y = df['MEDV']
print(X)

from imblearn.over_sampling import SMOTE                 #Synthetic Minority Oversampling (SMOTE) oversampling
sms=SMOTE(random_state=0)
X, Y=sms.fit_resample(X,Y)

print(Counter(Y))
from imblearn.over_sampling import SMOTE                 #Synthetic Minority Oversampling (SMOTE) oversampling
sms=SMOTE(random_state=0)
X, Y=sms.fit_resample(X,Y)

scaler = MinMaxScaler()
df[['CRIM', 'ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','BLACK','LSTAT' ]] = scaler.fit_transform(df[['CRIM', 'ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','BLACK','LSTAT']])

print(X)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']

print(featuresScores)

from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dt=tree.DecisionTreeClassifier()
pca=PCA(n_components=13)
pca.fit(X)
X=pca.transform(X)

#print(X)

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

train=dt.fit(X_train,y_train)

y_pred=dt.predict(X_test)

print(accuracy_score(y_test,y_pred))

# 5. Random Forest

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
pca=PCA(n_components=13)
pca.fit(X)
X=pca.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

train=rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

print("Random Forest",accuracy_score(y_test,y_pred))

# 6. Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gbm=GradientBoostingClassifier()
pca=PCA(n_components=13)
pca.fit(X)
X=pca.transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

gbm.fit(X_train,Y_train)

y_pred=gbm.predict(X_test)

print("GBM: ",accuracy_score(Y_test,y_pred))

# 3. KNN

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn=KNeighborsClassifier(n_neighbors=5)
pca=PCA(n_components=13)
pca.fit(X)
X=pca.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

train=knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print("KNN:",accuracy_score(y_test,y_pred))

#NBC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

nb=GaussianNB()
pca=PCA(n_components=13)
pca.fit(X)
X=pca.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1,test_size=0.2)

nb.fit(X_train,y_train)

y_pred1=nb.predict(X_test)

print("Naive Bayes: ",accuracy_score(y_test,y_pred1))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
pca=PCA(n_components=13)

pca.fit(X)
X=pca.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=1,test_size=0.3)

logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(accuracy_score(y_test,y_pred))