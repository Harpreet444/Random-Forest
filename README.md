# Iris Flower Classification using Random Forest Classifier

This project uses the famous Iris flower dataset to predict flower species using the Random Forest classifier. The dataset includes three species: Setosa, Versicolor, and Virginica.

## Requirements

- scikit-learn
- matplotlib
- pandas
- seaborn

## Dataset

The dataset used is the `Iris` dataset from `sklearn.datasets`, which contains measurements of iris flowers.

## Steps

1. **Load Dataset**: Load the iris dataset using `load_iris()` from `sklearn.datasets`.
2. **Data Preparation**: Prepare the dataset by creating a DataFrame and splitting it into training and testing sets.
3. **Model Training**: Train the Random Forest classifier with the default number of estimators (10).
4. **Evaluation**: Measure the prediction score and plot the confusion matrix.
5. **Model Tuning**: Fine-tune the model by changing the number of trees in the classifier and determine the best score.

## Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# Load dataset
data_set = load_iris()
target_array = data_set.target_names

# Create DataFrame
df = pd.DataFrame(data_set.data, columns=data_set.feature_names)
target = pd.DataFrame(data_set.target, columns=['target'])
df = pd.concat([df, target], axis='columns')

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), df['target'], test_size=0.2, random_state=10)

# Train model with default n_estimators
model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Measure prediction score
score = model.score(x_test, y_test)
print(f'Prediction Score with 10 Trees: {score}')

# Plot confusion matrix
sns.heatmap(confusion_matrix(target_array[y_test], target_array[y_pred]), cmap='Greens', annot=True, xticklabels=target_array, yticklabels=target_array)
plt.title("Confusion Matrix - Random Forest Classifier")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# Fine-tune the model
best_score = 0
best_estimators = 0
for n in range(1, 101):
    model = RandomForestClassifier(n_estimators=n)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if score > best_score:
        best_score = score
        best_estimators = n

print(f'Best Score: {best_score} with {best_estimators} Trees')
