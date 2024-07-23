from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import Data_Exploration
import Data_Preparation
import Classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


print(" ")
print("Training dataset(I) with Normaliztation Only")
Training_Features = Data_Exploration.Normalized_Training_Features
Testing_Features = Data_Exploration.Normalized_Testing_Features

Training_Class = Data_Exploration.Training_Class
Testing_Class = Data_Preparation.Testing_Class

svm_rbf = SVC(kernel='rbf', C = 4, gamma=0.02, random_state=42)
svm_rbf.fit(Training_Features, Training_Class)

y_pred_rbf = svm_rbf.predict(Testing_Features)
Accuracy = accuracy_score(Testing_Class, y_pred_rbf)
cm = confusion_matrix(Testing_Class, y_pred_rbf)

# Plot confusion matrix as heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


print("Testing Accuracy =  " +str(Accuracy))

print(" ")

print("Training dataset(I) with Normaliztation And PCA")

Training_Features2 = Data_Exploration.Train_pca
Testing_Features2 = Data_Exploration.Test_pca

#Data Normalization
scaler_train = StandardScaler()
Training_Features2 = scaler_train.fit_transform(Training_Features2)
Testing_Features2 = scaler_train.transform(Testing_Features2)

svm_rbf2 = SVC(kernel='rbf', C = 4, gamma=0.2, random_state=42)
svm_rbf2.fit(Training_Features2, Training_Class)

y_pred_rbf = svm_rbf2.predict(Testing_Features2)
Accuracy2 = accuracy_score(Testing_Class, y_pred_rbf)
cm = confusion_matrix(Testing_Class, y_pred_rbf)

# Plot confusion matrix as heatmap
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Testing Accuracy =  " +str(Accuracy2))