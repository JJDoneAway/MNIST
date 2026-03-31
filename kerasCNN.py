import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. Daten laden
# ==========================================
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255, -1)

# ==========================================
# 2. Modell-Architektur
# ==========================================
inputs = keras.Input(shape=(28, 28, 1))

# Data Augmentation
x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(inputs)
x = layers.RandomZoom(height_factor=0.1, width_factor=0.1)(x)
x = layers.RandomRotation(factor=0.05)(x)

# Block 1
x = layers.Conv2D(32, kernel_size=3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, kernel_size=3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.25)(x)

# Block 2
x = layers.Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.25)(x)

# Klassifikations-Kopf
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ==========================================
# 3. Kompilieren & Trainieren
# ==========================================
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

lr_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.00001
)

print("Starte Training...")
# Für einen schnellen Test kannst du die Epochen hier erstmal auf z.B. 5 runtersetzen
model.fit(x_train, y_train, batch_size=86, epochs=30, validation_data=(x_test, y_test), callbacks=[lr_reduction])

# ==========================================
# 4. Modell Evaluieren & Speichern (NEU)
# ==========================================
score = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinale Test Genauigkeit (Accuracy): {score[1]*100:.3f} %")

# Speichert das komplette Modell inkl. Architektur, Gewichten und Optimizer-Zustand
speicherpfad = "mnist_top_model.keras"
model.save(speicherpfad)
print(f"Modell erfolgreich gespeichert unter: {speicherpfad}")

# ==========================================
# 5. Confusion Matrix Auswertung (NEU)
# ==========================================
print("\nErstelle Confusion Matrix...")

# Das Modell trifft Vorhersagen für alle 10.000 Testbilder
y_pred_probs = model.predict(x_test)

# predict() liefert Wahrscheinlichkeiten (z.B. [0.01, 0.98, ...]).
# argmax() sucht den Index mit dem höchsten Wert (also die vorhergesagte Ziffer).
y_pred = np.argmax(y_pred_probs, axis=1)

# Berechne die 10x10 Matrix
cm = confusion_matrix(y_test, y_pred)

# Zeichne die Matrix
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))

# Wir nutzen einen blauen Farbverlauf. 'd' sorgt dafür, dass ganze Zahlen (keine Kommazahlen) stehen.
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

plt.title("Confusion Matrix - MNIST Vorhersagen")
plt.xlabel("Vorhergesagte Ziffer (Modell)")
plt.ylabel("Wahre Ziffer (Label)")
plt.show()
