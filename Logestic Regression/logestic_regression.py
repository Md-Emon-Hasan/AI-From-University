# import dependency
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# load tht iris datasets
iris = load_iris()

# convert into the dataframe
iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)

iris_df['species'] = iris['target']

# split the dataset
x = iris_df.drop(columns=['species'])
y = iris_df['species']

# training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# train the Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# evaluate model accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# define the prediction function
def predict_flower():
    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))
        
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(user_input)
        
        if prediction == 1:
            print("The flower is predicted to be Setosa.")
        else:
            print("The flower is predicted to be NOT Setosa.")
    except ValueError:
        print("Invalid input. Please enter numerical values.")
# Get user input and predict the result
predict_flower()