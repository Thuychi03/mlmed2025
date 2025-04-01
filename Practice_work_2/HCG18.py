import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,  mean_squared_error, r2_score

train_csv = pd.read_csv("/content/drive/MyDrive/ML Medicine/HCG18/train_set_pixel_size_and_HC.csv")
train_dir = "/content/drive/MyDrive/ML Medicine/HCG18/train_set"

test_csv = pd.read_csv("/content/drive/MyDrive/ML Medicine/HCG18/test_set_pixel_size.csv")
test_dir = "/content/drive/MyDrive/ML Medicine/HCG18/test_set"

train_csv.head()

train_csv.info()

test_csv.head()

len(train_csv)

len(test_csv)


image_ids = train_csv["filename"]
hc_values = train_csv["HC (mm)"].values

def load_image(img_id):
    img_path = os.path.join(train_dir, img_id)
    if not os.path.exists(img_path):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img

X_data = np.array([load_image(img_id) for img_id in image_ids])
y_data = np.array(hc_values)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f" Train size: {X_train.shape}, Validation size: {X_val.shape}")

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="linear")(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanAbsoluteError(), metrics=["mae"])

model.summary()

history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 10, batch_size = 32, verbose = 1)

y_pred = model.predict(X_val).flatten()

mae = mean_absolute_error(y_val, y_pred)

mse = mean_squared_error(y_val, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_val, y_pred)

print(" Regression Model Evaluation:  ")
print(f" MAE  = {mae:.2f} mm")
print(f" MSE  = {mse:.2f}")
print(f" RMSE = {rmse:.2f}")
print(f" R² Score = {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.5, color='blue')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], '--', color='red')
plt.xlabel("Actual HC (mm)")
plt.ylabel("Predicted HC (mm)")
plt.title("Actual vs Predicted HC")
plt.grid(True)
plt.show()

print("\nHead Circumference (HC) Statistics (mm):")
print(train_csv["HC (mm)"].describe())

plt.figure(figsize=(8, 5))
sns.histplot(train_csv["HC (mm)"], bins=30, kde=True)
plt.xlabel("Head Circumference (mm)")
plt.ylabel("Count")
plt.title("Distribution of HC in the Training Set")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Learning Curve")
plt.legend()
plt.show()

residuals = y_val - y_pred.flatten()
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color="blue")
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals (Actual - Predicted HC)")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

test_image_ids = test_csv["filename"]

X_test = np.array([load_image(img_id) for img_id in test_image_ids])

y_test_pred = model.predict(X_test)

test_csv["Predicted_HC (mm)"] = y_test_pred
test_csv.to_csv("predicted_test_HC.csv", index=False)
