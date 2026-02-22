#2025 Conrad Challenge

import plotly
import imblearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import pickle

warnings.filterwarnings("ignore")
data = pd.read_csv("C:/Users/USER/Desktop/RpROGRAM/heart.failure.csv")

plt.figure(figsize=(10, 5))
sns.countplot(x='HeartDisease', data=data)
plt.title("Distribution of target class")

f, axs = plt.subplots(1, 2, figsize=(20, 5), width_ratios=[3, 1])

sns.heatmap(data=data.corr(method="pearson", numeric_only=True), vmin=-1, vmax=1, annot=True, ax=axs[0])
axs[0].set_title("Correlation matrix")

sns.heatmap(data=data.corr(method="pearson", numeric_only=True)[["HeartDisease"]].sort_values('HeartDisease', ascending=False), vmin=-1, vmax=1, annot=True, ax=axs[1])
axs[1].set_title("Correlation of numerical attributes with HeartDisease")

plt.figure(figsize=(20, 10))
fig = px.histogram(data_frame=data, x="Sex", title="Distribution of Sex")
fig.update_traces(marker={"color": "red", "opacity": 0.6, "line": {"width": 4, "color": "black"}})
fig.show()

plt.figure(figsize=(20, 10))
fig = px.histogram(data_frame=data, x="ChestPainType", title="Distribution of ChestPain Type")
fig.update_traces(marker={"color": "red", "opacity": 0.6, "line": {"width": 4, "color": "black"}})
fig.show()

px.pie(values=data['ChestPainType'].value_counts(),names =data['ChestPainType'].value_counts().index).update_layout(title='Chest Pain Type')

labels=['1','0']
count= data['HeartDisease'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(count,labels=labels,autopct='%.0f',explode=(0,.1), colors=['r','g'])
plt.legend( ['heart disease','Normal'],loc =1)

plt.title('Heart Disease')
plt.show()

px.histogram(data,x='Age',color='HeartDisease').update_layout(title='Age Distribution')

X=data.drop('HeartDisease', axis=1)
y=data['HeartDisease']

data.select_dtypes(include="object").columns

le_model = LabelEncoder()

X['Sex'] = le_model.fit_transform(X['Sex'])
X['ChestPainType'] = le_model.fit_transform(X['ChestPainType'])
X['RestingECG'] = le_model.fit_transform(X['RestingECG'])
X['ExerciseAngina'] = le_model.fit_transform(X['ExerciseAngina'])
X['ST_Slope'] = le_model.fit_transform(X['ST_Slope'])

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train.value_counts()

smote = SMOTE(random_state=0)

X_train, y_train = smote.fit_resample(X_train,y_train)
X_test, y_test = smote.fit_resample(X_test,y_test)

y_train.value_counts()

def evaluate(model):
  model.fit(X_train,y_train)
  pre = model.predict(X_test)
    
  accuracy = accuracy_score(pre,y_test)
  recall = recall_score(pre,y_test)
  f1 = f1_score(pre,y_test)

  sns.heatmap(confusion_matrix(pre,y_test),annot=True)
  print(model)
  print('Accuracy : ',accuracy,'Recall : ',recall,"F1 : ",f1)


model_RFC = RandomForestClassifier()
## Using grid search cv to find the best parameters.
param = {'n_estimators': [10, 20, 30, 40, 50], 'max_depth': [2, 3, 4, 7, 9]}
clf_rfc_cv = GridSearchCV(model_RFC, param, cv=5,scoring='roc_auc', n_jobs=-1)
clf_rfc_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",clf_rfc_cv.best_params_)
print("accuracy :",clf_rfc_cv.best_score_)

rf = RandomForestClassifier(max_depth=7, n_estimators=30)

evaluate(rf)

pickle.dump(clf_rfc_cv.fit, open('heart.pkl', 'wb'))