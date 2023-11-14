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
```
Developed by:VASUNDRA SRI R
Reg no:212222230168
```
## Importing library
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
```
## Data loading:
```
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()
```
## Now, we are checking start with a pairplot, and check for missing values
```
sns.heatmap(data.isnull(),cbar=False)
```
## Data Cleaning and Data Drop Process
```
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
```
## Change to categoric column to numeric
```
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
```
## Instead of nan values
```
data['Embarked']=data['Embarked'].fillna('S')
```
## Change to categoric column to numeric
```
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
```
## Drop unnecessary columns
```
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
data.head(11)
```
## Heatmap for train dataset
```
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
```
## Now, data is clean and read to a analyze
```
sns.heatmap(data.isnull(),cbar=False)
```
## How many people survived or not... %60 percent died %40 percent survived
```
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
```
## Age with survived
```
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()
```
## Count the pessenger class
```
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
```
## Split the data into training and test sets
```
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
```
## Create a Random Forest classifier
```
my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5, criterion='entropy')
```
## Fit the model to the training data
```
my_forest.fit(X_train, y_train)
```
## Make predictions on the test data
```
target_predict = my_forest.predict(X_test)
```
## Evaluate the model's performance
```
accuracy = accuracy_score(y_test, target_predict)
mse = mean_squared_error(y_test, target_predict)
r2 = r2_score(y_test, target_predict)

print("Random forest accuracy: ", accuracy)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2) Score: ", r2)
```
# OUPUT
### Initial data
![Screenshot 2023-11-14 111622](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/60b4e4ba-341b-4c3f-a97d-ac7d19843d2d)

### Null values
![Screenshot 2023-11-14 111734](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/c7c6e220-3b3a-41d2-ba29-5fa23752a7b4)

### Describing the data
![Screenshot 2023-11-14 111823](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/eeac0377-3142-44d8-9b31-c606c03adacf)

### Missing values
![Screenshot 2023-11-14 111917](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/814153fd-5937-4e7e-99be-eb7680eaa676)

### Data after cleaning
![Screenshot 2023-11-14 111917](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/467d206e-5a8a-4fca-8eeb-dc932de071d6)

### Data on Heatmap
![Screenshot 2023-11-14 112112](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/53e6ddda-3f26-49f2-a869-9117e105092a)

### Report of(people survied & died)
![Screenshot 2023-11-14 112240](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/c1222677-0eb7-4c68-ab89-e500086c24a6)

### Cleaned null values
![Screenshot 2023-11-14 112343](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/e8f26318-26cc-4219-a234-7d492f964a92)

### Report of survied people's age
![Screenshot 2023-11-14 112446](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/1d333ecf-56b3-46d9-a337-de9194772ac8)

### Report of pessengers
![Screenshot 2023-11-14 112604](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/a1c0f26e-cad9-4a32-aa44-49d2b16a2909)

### Report
![Screenshot 2023-11-14 112658](https://github.com/vasundrasriravi/ODD2023-Datascience-Ex-07/assets/119393983/bdab3bce-a2f8-4c00-adec-70fb2e4d145b)

# RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.
