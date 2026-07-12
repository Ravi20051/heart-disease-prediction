import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("heart.csv")
#heatmap
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
plt.show()
#Distribution
plt.figure(figsize=(5,5))
df["target"].value_counts().plot(kind="bar",color=["steelblue","coral"])
plt.title("Distribution of target variable")
plt.xlabel("target")
plt.ylabel("count")
plt.xticks(rotation=0)
plt.show()
#Age Disstribution 
plt.figure(figsize=(5,5))
plt.hist(df[df["target"]==1]["age"],bins=20,alpha=0.7,label="diesease")
plt.hist(df[df["target"]==0]["age"],bins=20,alpha=0.7,label="no disease")
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution of Heart Disease")
plt.legend()
plt.show()
df.groupby("cp")["target"].mean().plot(kind="bar",color=["steelblue","coral"])
plt.title("Distribution of target variable")
plt.xlabel("target")
plt.ylabel("count")
plt.xticks(rotation=0)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
X=df.drop("target",axis=1)
y=df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
rr=RandomForestClassifier(n_estimators=100,random_state=42)
rr.fit(X_train,y_train)
print(f"Accuracy of RandomForestClassifier: {rr.score(X_test,y_test):.4f}")
model=[LogisticRegression(max_iter=103,random_state=42
),DecisionTreeClassifier(max_depth=3,random_state=42
),RandomForestClassifier(n_estimators=100,random_state=42
),SVC(kernel="rbf",random_state=42
),GaussianNB(),KNeighborsClassifier(n_neighbors=5)]
names=["Logistic Regression","Decision Tree","Random Forest","SVM","Gaussian Naive Bayes","KNN"]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
for model,name in zip(model,names):
    scores=cross_val_score(model,X_scaled,y,cv=5)
    print(f"Accuracy of {name}: {scores.mean():.4f}")

#important features
importance =pd.DataFrame({
    "feature":X.columns,
    "importance":rr.feature_importances_})
importance=importance.sort_values("importance",ascending=False)
print(importance)
plt.figure(figsize=(10,5))
plt.barh(importance["feature"], importance["importance"], color="steelblue")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()
from sklearn.metrics import confusion_matrix,classification_report
y_pred=rr.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test,y_pred))
