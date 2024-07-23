from sklearn.model_selection import train_test_split
from Data_Exploration import Training_Features
from Data_Exploration import Testing_Features
from Data_Exploration import Training_Class
from Data_Exploration import Testing_Class
from sklearn.preprocessing import StandardScaler

print("")
# Set the test_size parameter to the desired ratio (0.2 for 20% validation data)
X_train, X_val, y_train, y_val = train_test_split(Training_Features, Training_Class , test_size=0.2, random_state=42)

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_val = Scaler.fit_transform(X_val)

print("")
print("After Normalization of Training Set (II) and Validation set")

print("Training Feature 1 Mean after normalization =  " + str(X_train[:,0].mean()))
print("Training Standard diviation after normalization =  " + str(X_train[:,0].std()))

print("Validation Feature 1 mean after normalization =  " + str (X_val[:,0].mean()))
print("Validation standard diviation after normalization =  "+ str(X_val[:,0].std()))

