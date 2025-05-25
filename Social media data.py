# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Simulated Class Distribution for Bar Chart (Distribution A)
bar_counts = [40, 25, 10]  # Setosa - 40, Versicolor - 25, Virginica - 10
plt.figure(figsize=(6,4))
plt.bar(class_names, bar_counts, color=['pink', 'blue', 'violet'])
"plt.title(""Simulated Class Distribution A - Bar Chart"")"
"plt.xlabel(""Class"")"
"plt.ylabel(""Count"")"
plt.show()

# Simulated Class Distribution for Pie Chart (Distribution B)
pie_counts = [15, 35, 25]  # Setosa - 15, Versicolor - 35, Virginica - 25
plt.figure(figsize=(6,6))
plt.pie(pie_counts, labels=class_names, autopct='%1.1f%%', colors=['pink', 'blue', 'violet'])
"plt.title(""Simulated Class Distribution B - Pie Chart"")"
plt.show()

# Split original Iris data and train a Random Forest classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
"plt.title(""Confusion Matrix"")"
plt.show()
