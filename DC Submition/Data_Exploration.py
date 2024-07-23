import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#Reading Dataset
Training_Dataset = pd.read_csv("wdbc_training.csv")
Testing_Dataset = pd.read_csv("wdbc_test.csv")


Training_Class = Training_Dataset["Diagnosis"]
Testing_Class = Testing_Dataset["Diagnosis"]



Training_Features = Training_Dataset.drop(columns=["ID number", "Diagnosis"])
Testing_Features = Testing_Dataset.drop(columns=["ID number", "Diagnosis"])


# Select two features and the label class column
feature1 =  "Column14"
feature2 = "Column20"
labels = "Diagnosis"


# Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(Training_Dataset[Training_Dataset[labels] == 2][feature1], Training_Dataset[Training_Dataset[labels] == 2][feature2], color='blue', label='benign')
plt.scatter(Testing_Dataset[Testing_Dataset[labels] == 4][feature1], Testing_Dataset[Testing_Dataset[labels] == 4][feature2], color='red', label='malignant')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('Scatter Plot of {} vs {}'.format(feature1, feature2))
plt.legend()
plt.grid(True)
plt.show()



#Data Normalization
scaler_test = StandardScaler()
scaler_train = StandardScaler()
Normalized_Training_Features = scaler_train.fit_transform(Training_Features)
Normalized_Testing_Features = scaler_train.transform(Testing_Features)

Mean = Normalized_Testing_Features[:,0].mean()
STD = Normalized_Testing_Features[:,0].std()
print("The Mean of the first Feature after Normalization = " + str(Mean))
print("The Standard deviation of the first Feature after Normalization = " + str(STD))



# Perform PCA analysis
pca = PCA()
Train_pca = pca.fit_transform(Normalized_Training_Features)
Test_pca = pca.transform(Normalized_Testing_Features)

# Plot the scree plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')


# Plot the projection of training set onto PC1 and PC2
plt.subplot(1, 2, 2)
plt.scatter(Train_pca[:, 0], Train_pca[:, 1], c=Training_Class, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection of Training Set (PC1 vs PC2)')
plt.colorbar(label='Class')

plt.tight_layout()
plt.show()

# Transforming to the best n_components
pca = PCA(n_components=5)
Train_pca = pca.fit_transform(Training_Features)
Test_pca = pca.transform(Testing_Features)
