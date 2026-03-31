import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. Modell und Daten laden
# ==========================================
print("Lade Modell und Daten...")
pfad_zum_modell = "mnist_top_model.keras"

try:
    model = keras.models.load_model(pfad_zum_modell)
except Exception as e:
    print(f"Fehler beim Laden: {e}")
    exit()

(_, _), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test_raw.astype("float32") / 255, -1)

# ==========================================
# 2. Vorhersagen & Confusion Matrix berechnen
# ==========================================
print("Berechne Vorhersagen...")
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

# Fehlerklassen extrahieren (> 1 Fehler, jenseits der Diagonale)
fehler_klassen = []
for wahr in range(10):
    for vorhergesagt in range(10):
        if wahr != vorhergesagt and cm[wahr, vorhergesagt] > 1:
            fehler_klassen.append((wahr, vorhergesagt, cm[wahr, vorhergesagt]))

fehler_klassen.sort(key=lambda x: x[2], reverse=True)

# Wir limitieren auf die Top 5 Fehlerklassen, damit es auf einen Bildschirm passt
max_klassen_anzeigen = min(5, len(fehler_klassen))
max_bilder_pro_klasse = 5

# ==========================================
# 3. Das "All-in-One" Dashboard Layout
# ==========================================
print("Erstelle Dashboard-Grafik...")

# Erstelle ein großes Fenster (z.B. 20 Zoll breit, 10 Zoll hoch)
fig = plt.figure(figsize=(20, 10))
fig.canvas.manager.set_window_title('MNIST Fehleranalyse - Dashboard')

if max_klassen_anzeigen == 0:
    print("Keine Fehler gefunden! Zeige nur die Matrix.")
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix (Perfektes Modell!)")

else:
    # GridSpec: Teilt das Fenster in Zeilen und Spalten auf
    # Wir brauchen so viele Zeilen wie Fehlerklassen, und Spalten für die Matrix (1) + Bilder (max 5)
    # width_ratios sorgt dafür, dass die Matrix links breiter ist als die einzelnen kleinen Bilder rechts
    gs = gridspec.GridSpec(nrows=max_klassen_anzeigen,
                           ncols=max_bilder_pro_klasse + 1,
                           width_ratios=[2.5] + [1] * max_bilder_pro_klasse)

    # --- Linke Seite: Confusion Matrix ---
    # Die Matrix soll sich über ALLE Zeilen (:) in der allerersten Spalte (0) erstrecken
    ax_cm = fig.add_subplot(gs[:, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))

    # colorbar=False spart hier etwas Platz, da die Zahlen ohnehin drinstehen
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d', colorbar=False)
    ax_cm.set_title("Confusion Matrix", fontsize=16)

    # --- Rechte Seite: Die Fehlerbilder ---
    for row_idx in range(max_klassen_anzeigen):
        wahr, vorhergesagt, anzahl = fehler_klassen[row_idx]
        indizes = np.where((y_test == wahr) & (y_pred == vorhergesagt))[0]
        bilder_anzeigen = min(anzahl, max_bilder_pro_klasse)

        for col_idx in range(bilder_anzeigen):
            # Füge jedes Bild in die entsprechende Zeile und Spalte (ab Spalte 1) ein
            ax_img = fig.add_subplot(gs[row_idx, col_idx + 1])
            img_idx = indizes[col_idx]

            ax_img.imshow(x_test_raw[img_idx], cmap="gray")

            # Kurzer Titel über jedem Bild
            ax_img.set_title(f"Wahr: {wahr} | Modell: {vorhergesagt}", fontsize=11, color="red")
            ax_img.axis('off')

# Layout aufräumen, damit nichts überlappt, und anzeigen!
plt.tight_layout()
plt.show()
