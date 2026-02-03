# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset.Convert it into a pandas DataFrame and separate features X and labels y.
2. Divide the data into training and test sets using train_test_split.
3. Initialize the SGDClassifier with a maximum number of iterations.Fit the model on the training data.
4. Predict labels for the test set.Calculate accuracy score and display the confusion matrix.
Program:

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Avantika Krishnadas Kundooly
RegisterNumber: 212224040040
```

## Head and Tail values:
```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
display(df.head())
display(df.tail())
X=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
```
## Output:
<img width="547" height="354" alt="image" src="https://github.com/user-attachments/assets/78a7c538-f60e-45dc-a7f6-88f94584b7aa" />



## Predicted values:
```py
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)

sgd_clf.fit(x_train,y_train)

y_pred=sgd_clf.predict(x_test)

y_pred
```

## Output:
<img width="637" height="46" alt="image" src="https://github.com/user-attachments/assets/96651f27-548c-49db-9615-b859844d8a2f" />



## Accuracy:
```py
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy : {accuracy:.3f}")
```

## Output:
<img width="146" height="24" alt="image" src="https://github.com/user-attachments/assets/f464ed43-514a-48f6-b4e8-4a63aeb93dd4" />


## Confusion matrix:
```py
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:")
print(cm)
```

## Output:
<img width="155" height="77" alt="image" src="https://github.com/user-attachments/assets/fb6289f7-5046-4cac-9ba4-81b1c492c36c" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
