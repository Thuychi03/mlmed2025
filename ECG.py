import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# File paths
train_path = "D:/Downloads/D/USTH/B3/Machine Learning in Medicine/ECG/archive/mitbih_train.csv"
test_path = "D:/Downloads/D/USTH/B3/Machine Learning in Medicine/ECG/archive/mitbih_test.csv"

# Load datasets
df_train = pd.read_csv(train_path, header=None)
df_test = pd.read_csv(test_path, header=None)

# Dataset info
df_train.shape
df_test.shape

# Check for missing values
df_train.isnull().sum().sum()
df_test.isnull().sum().sum()

# Display first few rows
df_train.head()
df_test.head()

# Count label distribution (last column is the label)
train_label_counts = df_train.iloc[:, -1].value_counts()
test_label_counts = df_test.iloc[:, -1].value_counts()


# Plot label distribution
plt.figure(figsize=(12, 5))
sns.barplot(x=train_label_counts.index, y=train_label_counts.values, palette="viridis")
plt.xlabel("Heartbeat Class")
plt.ylabel("Count")
plt.title("Distribution of Heartbeat Classes in Training Set")
plt.show()

(train_shape, test_shape, train_missing, test_missing, df_train_head, df_test_head, train_label_counts, test_label_counts)

# Reshape data for CNN input (Assuming 1D ECG signals)
X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].values
X_test, y_test = df_test.iloc[:, :-1].values, df_test.iloc[:, -1].values

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape to fit CNN input (samples, timesteps, channels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define CNN model
model = keras.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(np.unique(y_train)), activation='softmax')  # Multi-class classification
])


# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Making predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Evaluate using classification report and confusion matrix
from sklearn.metrics import classification_report

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_test, predicted_classes))