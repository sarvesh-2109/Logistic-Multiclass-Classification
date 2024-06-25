# Logistic Multiclass Classification

This repository contains a project that demonstrates the use of logistic regression for multiclass classification using the Iris dataset. The project includes data visualization, model training, and evaluation.

## Output


https://github.com/sarvesh-2109/Logistic-Multiclass-Classification/assets/113255836/d8530a1f-3ee5-4054-bca8-d91b53e58241



## Project Overview

In this project, we use logistic regression to classify iris flowers into three species: Setosa, Versicolour, and Virginica. The Iris dataset, a classic dataset in machine learning, contains 150 samples of iris flowers with four features: sepal length, sepal width, petal length, and petal width.

## Dataset

The Iris dataset is a built-in dataset in the `sklearn.datasets` module. It includes:

- 150 samples
- 4 features per sample: sepal length, sepal width, petal length, and petal width
- 3 classes (species): Setosa, Versicolour, and Virginica

## Files in the Repository

- `Logistic_Multiclass.ipynb`: The Jupyter notebook containing the complete code for the project.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/sarvesh-2109/Logistic-Multiclass-Classification.git
   cd Logistic-Multiclass-Classification
   ```

2. Install the required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook Logistic_Multiclass.ipynb
   ```

## Steps in the Project

1. **Data Loading**: Load the Iris dataset using `sklearn.datasets.load_iris`.
2. **Data Visualization**: Plot the features of the first five samples.
3. **Model Creation and Training**: Create and train a logistic regression model using the training data.
4. **Model Evaluation**: Measure the accuracy of the model on the test data and create a confusion matrix to evaluate the performance.

## Results

- **Accuracy**: The model achieves a high accuracy on the test set.
- **Confusion Matrix**: A heatmap of the confusion matrix is generated to visualize the performance of the model.
- **Sample Predictions**: Print out the features, predicted labels, and actual labels for each sample in the test set.

## Visualizations

- **Feature Plots**: Visualize the features of the first five samples.
- **Confusion Matrix**: A heatmap to display the confusion matrix.

## Example Code

Below is a snippet from the project to demonstrate the training and evaluation process:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()

model = LogisticRegression(max_iter=1000)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

for i in range(len(X_test)):
    print(f'Sample {i+1}:')
    print(f'Features: {X_test[i]}')
    print(f'Predicted Label: {iris.target_names[y_pred[i]]}')
    print(f'Actual Label: {iris.target_names[y_test[i]]}')
    print()
```

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The Iris dataset is available from the UCI Machine Learning Repository.
- Thanks to the developers of `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` for their excellent libraries.
