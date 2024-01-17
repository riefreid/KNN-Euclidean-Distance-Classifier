import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Euclidean distance matrix, 19 by 19
euclidean_matrix = np.array([
    [0.0, 1.5, 1.4, 1.6, 1.7, 1.3, 1.6, 1.5, 1.4, 2.3, 2.9, 3.2, 3.3, 3.4, 4.2, 4.1, 5.9, 6.1, 6.0],  # x1
    [1.5, 0.0, 1.6, 1.4, 1.4, 1.4, 1.3, 1.4, 1.3, 2.4, 2.8, 3.3, 3.4, 3.2, 4.1, 4.1, 6.2, 6.3, 6.1],  # x2
    [1.4, 1.6, 0.0, 1.3, 1.5, 1.4, 1.4, 1.6, 1.4, 2.5, 2.9, 3.2, 3.2, 3.5, 4.1, 4.1, 6.2, 6.2, 6.2],  # x3
    [1.6, 1.4, 1.3, 0.0, 1.5, 1.5, 1.4, 1.3, 1.5, 2.3, 3.0, 3.1, 3.2, 3.4, 4.1, 4.1, 5.8, 5.8, 5.8],  # x4
    [1.7, 1.4, 1.5, 1.5, 0.0, 1.4, 1.5, 1.7, 1.2, 2.6, 2.9, 3.3, 3.3, 3.7, 4.1, 4.1, 6.1, 6.1, 6.1],  # x5
    [1.3, 1.4, 1.4, 1.5, 1.4, 0.0, 1.8, 1.6, 1.4, 2.7, 3.1, 3.4, 3.4, 3.5, 4.1, 4.1, 6.0, 6.0, 6.0],  # x6
    [1.6, 1.3, 1.4, 1.4, 1.5, 1.8, 0.0, 1.4, 1.3, 2.8, 2.9, 3.3, 3.2, 3.6, 4.1, 4.1, 6.1, 6.1, 6.1],  # x7
    [1.5, 1.4, 1.6, 1.3, 1.7, 1.6, 1.4, 0.0, 1.5, 2.7, 3.1, 3.4, 3.3, 3.3, 4.1, 4.1, 5.9, 5.9, 5.9],  # x8
    [1.4, 1.3, 1.4, 1.5, 1.2, 1.4, 1.3, 1.5, 0.0, 3.1, 3.0, 3.5, 3.5, 3.5, 4.1, 4.1, 5.8, 5.8, 5.8],  # x9
    [2.3, 2.4, 2.5, 2.3, 2.6, 2.7, 2.8, 2.7, 3.1, 0.0, 1.5, 3.3, 3.6, 3.6, 4.1, 4.1, 6.0, 6.0, 6.0],  # x10
    [2.9, 2.8, 2.9, 3.0, 2.9, 3.1, 2.9, 3.1, 3.0, 1.5, 0.0, 1.6, 1.4, 1.5, 1.7, 1.6, 2.3, 3.1, 3.0],  # x11
    [3.2, 3.3, 3.2, 3.1, 3.3, 3.4, 3.3, 3.4, 3.5, 3.3, 1.6, 0.0, 1.7, 1.8, 1.6, 1.5, 2.3, 2.7, 2.9],  # x12
    [3.3, 3.4, 3.2, 3.2, 3.3, 3.4, 3.2, 3.3, 3.5, 3.6, 1.4, 1.7, 0.0, 0.5, 0.3, 0.4, 2.5, 2.6, 2.7],  # x13
    [3.4, 3.2, 3.5, 3.4, 3.7, 3.5, 3.6, 3.3, 3.5, 3.6, 1.5, 1.8, 0.5, 0.0, 0.5, 0.5, 2.3, 2.3, 2.4],  # x14
    [4.2, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 1.7, 1.6, 0.3, 0.5, 0.0, 0.4, 2.4, 2.5, 2.5],  # x15
    [4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 1.6, 1.5, 0.4, 0.5, 0.4, 0.0, 2.5, 2.6, 2.8],  # x16
    [5.9, 6.2, 6.2, 5.8, 6.1, 6.0, 6.1, 5.9, 5.8, 6.0, 2.3, 2.3, 2.5, 2.3, 2.4, 2.5, 0.0, 3.0, 3.1],  # x17
    [6.1, 6.3, 6.2, 5.8, 6.1, 6.0, 6.1, 5.9, 5.8, 6.0, 3.1, 2.7, 2.6, 2.3, 2.5, 2.6, 3.0, 0.0, 0.4],  # x18
    [6.0, 6.1, 6.2, 5.8, 6.1, 6.0, 6.1, 5.9, 5.8, 6.0, 3.0, 2.9, 2.7, 2.4, 2.5, 2.8, 3.1, 0.4, 0.0]   # x19
])#    x1   x2   x3   x4   x5   x6   x7   x8   x9  x10   x11  x12  x13  x14  x15  x16  x17  x18  x19

# class labels for samples
# Class 1 is 0, Class 2 is 1
labels = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1])

# training and test sets
train_indices = [0, 4, 8, 12, 16, 2, 6, 10, 14, 18]
X_train = euclidean_matrix[train_indices]
y_train = labels[train_indices]

test_indices = [1, 5, 9, 13, 17, 3, 7, 11, 15]
X_test = euclidean_matrix[test_indices]
y_test = labels[test_indices]

# values of K to use
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# lists to store training and test errors for different K values
train_errors = []
test_errors = []

# for loop that iterates through each value of k
for k in k_values:
    # Create and fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on the training and test sets
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    # Calculate training and test errors (1 - accuracy)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    # Append errors to the lists
    train_errors.append(train_error)
    test_errors.append(test_error)

# Plot the training and test errors
plt.figure(figsize=(8, 6))
plt.plot(k_values, train_errors, label="Training Error", marker='o')
plt.plot(k_values, test_errors, label="Test Error", marker='o')
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Error Rate")
plt.title("KNN Model: Training and Test Errors vs. K")
plt.legend()
plt.grid()
plt.show()
