import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()
print(dir(digits))

for i in range(5):
    plt.matshow(digits.images[i])
plt.show()

print(digits.target[0:5])

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

print("Sample training data size = ", len(X_train))
print("Sample testing data size = ",len(X_test))

reg = LogisticRegression(max_iter=1500)
reg.fit(X_train, y_train)

print("Model score = ",reg.score(X_test, y_test))
y_predicted = reg.predict(X_test)
print("Model prediction on test data = ", y_predicted)

plt.matshow(digits.images[108])
plt.show()

print("Image at index 108 is = ", digits.target[108])
print("Image prediction result is = ", reg.predict([digits.data[108]]))

"""Confusion matrix"""
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()