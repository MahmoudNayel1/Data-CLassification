from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import Data_Exploration
import Data_Preparation
from sklearn.svm import SVC
import numpy as np

print("Without PCA")
# For linear kernel
C_values_linear = [3 , 15 , 50]
accuracy_linear = []
X_train = Data_Preparation.X_train
y_train = Data_Preparation.y_train

X_validate = Data_Preparation.X_val
y_validate = Data_Preparation.y_val

# X_validate = Data_Exploration.Testing_Features
# y_validate = Data_Exploration.Testing_Class

for c in C_values_linear:
    # Train SVM model with linear kernel
    svm_linear = SVC(kernel='linear', C=c, random_state=42)
    svm_linear.fit(X_train, y_train)

    # Evaluate performance
    y_pred_linear = svm_linear.predict(X_validate)
    accuracy_linear.append(accuracy_score(y_validate, y_pred_linear))
print("")
print("linear SVM accuracies =  " + str(accuracy_linear))

print("")

# Visualize the performance
plt.figure(figsize=(12, 6))

# Linear kernel
plt.subplot(1, 2, 1)
plt.plot(C_values_linear, accuracy_linear, marker='o')
plt.title('Accuracy vs C (Linear Kernel)')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.grid(True)



# For RBF kernel
C_values_rbf = [2 , 4 , 51]
gamma_values_rbf = [0.02 , 1 , 11]
accuracy_rbf = []



plt.subplot(1, 2, 2)
for C in C_values_rbf:
    for gamma in gamma_values_rbf:
        # Train SVM model with RBF kernel
        svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm_rbf.fit(X_train, y_train)

        # Evaluate performance
        y_pred_rbf = svm_rbf.predict(X_validate)
        accuracy_rbf.append((C, gamma, accuracy_score(y_validate, y_pred_rbf)))

print("RPF SVM accuracies =  " +str(accuracy_rbf))
plt.subplot(1, 2, 2)
for C in C_values_rbf:
    accuracies = [acc for c, g, acc in accuracy_rbf if c == C]
    plt.plot(gamma_values_rbf, accuracies, marker='o', label=f'C={C}')
plt.title('Accuracy vs C and Gamma (RBF Kernel)')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
