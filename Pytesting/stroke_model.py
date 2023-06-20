import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("stroke_data.csv")

# There's only NaNs in the BMI column so let's replace these with 0
data['bmi']=data['bmi'].fillna(0)

print(f"Class balance: \nNo stroke = 0, stroke = 1 \n{data['stroke'].value_counts()}")


le = LabelEncoder()
d_list = data.select_dtypes(include = ['object']).columns.tolist()
for i in d_list:
    le.fit(data[i])
    data[i] = le.transform(data[i])

x=data.drop('stroke',axis=1)
y=data['stroke']

# Split the data into train and test sets
def train_test_sets(x, y, test_size=0.2):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
  return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_sets(x, y, test_size=0.2)

model=Sequential()

model.add(Dense(7,activation="relu",input_dim=11))
model.add(Dense(7,activation="relu"))
model.add(Dense(14,activation="relu"))
model.add(Dense(28,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy",optimizer="Adam")

model.fit(x_train, y_train, epochs=10, validation_split=0.2)

pred_prob = model.predict(x_test)
pred = np.where(pred_prob > 0.5, 1, 0)

cn=confusion_matrix(y_test,pred)

sns.heatmap(cn,annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

def accuracy(real, pred):
  sklearn_acc = accuracy_score(real, pred)
  return sklearn_acc
  
print(f"Accuracy of model: {accuracy(y_test, pred)}")
