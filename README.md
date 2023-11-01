# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
## DATA PREPROCESSING BEFORE FEATURE SELECTION:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/a73701d1-e33a-4409-accf-123151342664)

## checking data
```
df.isnull().sum()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/1f67200e-39db-48f3-a525-8ea99daf3eaa)

## removing unnecessary data variables
```
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/795decc9-49c6-4033-91a1-fca31636a587)

## cleaning data
```
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/0e45e20f-cf13-406c-ad55-8501993ac22a)

## removing outliers 
```
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/de490215-8a08-49c7-a83c-0c63128d201b)

```
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/161919e2-804d-458b-a5d3-b3e0da813b49)

```
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/3c16a0ec-276e-4787-85d3-e66d9f1976e1)

```
from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/de72d74f-df55-40be-a0ec-fafb7371568b)

```
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/01727210-ad2b-4220-ac3f-cbf93b3e8d56)

```
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)
```
```
df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/30d6cf75-888f-4546-83fb-69f937616be8)

```
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
```
```
X = df1.drop("Survived",1) 
y = df1["Survived"] 
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/1da87d80-925c-44ad-a007-b34582632441)

##  FEATURE SELECTION:
##  FILTER METHOD:
```
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/ddc3827d-d737-41ae-a5fb-6a357b337c86)

## HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
```
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/0a19b57a-c118-4362-83d5-31c23c19a433)

## BACKWARD ELIMINATION:
```
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/3a44909d-5aa5-4ff9-9ace-70d1f5062406)

## RFE (RECURSIVE FEATURE ELIMINATION):
```
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/75e19eb1-f8ea-4976-b945-370279290c93)

## OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
```
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/be59d3f7-c6dd-466a-b0c3-88b5bfb2b482)

## FINAL SET OF FEATURE:
```
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/2450afef-95cf-4bd3-a76e-21274f70ae2b)

## EMBEDDED METHOD:
```
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![image](https://github.com/Sathya-006/ODD2023-Datascience-Ex-07/assets/121661327/c10492e7-5881-4cba-9407-b2d9d09618f0)

# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
