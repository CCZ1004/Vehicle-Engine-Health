import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import tensorflow as tf

# Check if GPU is available and use it
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print("Warning: GPU device not found. Training will run on CPU.")
else:
    print(f"Using GPU: {device_name}")

# Load the dataset
dataset = pd.read_csv("C:\\Users\\USER\\Downloads\\engine_data\\engine_data.csv")

# Display dataset structure and summary
print(dataset.info())
print(dataset.describe())

# Scale the feature data
scaler = MinMaxScaler()
X = dataset.iloc[:, :-1].values  # Exclude the target column
y = dataset.iloc[:, -1].values  # The target variable
X_scaled = scaler.fit_transform(X).astype(np.float32)  # Scale and convert to float32

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=222)

# Define the improved model function for cross-validation with added enhancements
def create_improved_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    
    # First hidden layer
    model.add(Dense(512, kernel_regularizer=l2(0.001)))  # L2 regularization to avoid overfitting
    model.add(LeakyReLU(negative_slope=0.01))  # Using LeakyReLU with negative_slope
    model.add(Dropout(0.5))  # Increased dropout rate to combat overfitting
    model.add(BatchNormalization())  # Batch Normalization to speed up training
    
    # Second hidden layer
    model.add(Dense(256, kernel_regularizer=l2(0.001)))  # L2 regularization
    model.add(LeakyReLU(negative_slope=0.01))  # Using LeakyReLU with negative_slope
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    model.add(BatchNormalization())
    
    # Third hidden layer
    model.add(Dense(128, kernel_regularizer=l2(0.001)))  # L2 regularization
    model.add(LeakyReLU(negative_slope=0.01))  # Using LeakyReLU with negative_slope
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    
    # Compile the model with an optimizer and loss function
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# Implement 5-fold cross-validation with loss tracking
kfold = KFold(n_splits=5, shuffle=True, random_state=222)
cv_results = []

for train_index, val_index in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Create and train the model
    model = create_improved_model()
    
    # Fit the model and store the history
    history = model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, verbose=0, validation_data=(X_val_fold, y_val_fold))

    # Predict on validation data
    y_val_pred = model.predict(X_val_fold)
    y_val_pred_binary = (y_val_pred >= 0.47).astype(int)
    
    # Calculate accuracy and append to cv_results
    accuracy = accuracy_score(y_val_fold, y_val_pred_binary)
    cv_results.append(accuracy)

# Print cross-validation results
print(f"Cross-Validation Results: {cv_results}")
print(f"Mean Accuracy: {np.mean(cv_results)} (+/- {np.std(cv_results)})")

# Final model on the entire training dataset
final_model = create_improved_model()
history = final_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Predict using the trained model
y_pred = final_model.predict(X_test)
y_pred_binary = (y_pred >= 0.47).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Plot and save the confusion matrix as an image
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Health', 'Good Health'], yticklabels=['Low Health', 'Good Health'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_test, y_pred_binary, target_names=['Low Health', 'Good Health'])
print("Classification Report:")
print(report)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plot the Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve', color='green')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

x=0
while(x == 0):
    # Test with new inputs
    print("Testing with New Inputs:")
    input_data = []
    for column in dataset.columns[:-1]:
        value = float(input(f"Enter {column}: "))
        input_data.append(value)

    input_df = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_df)

    # Make predictions
    new_output = final_model.predict(input_scaled)

    binary_output = (new_output >= 0.47).astype(int)

    print("Predicted output:")
    print(binary_output)

    if binary_output == 0:
        print("Engine Health Condition is low")
    else:
        print("Engine Health Condition is good")

    