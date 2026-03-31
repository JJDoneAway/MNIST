import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Modell laden und eine "8" suchen
# ==========================================
print("Lade Modell...")
model = keras.models.load_model("mnist_top_model.keras")

(_, _), (x_test_raw, y_test) = keras.datasets.mnist.load_data()

# Finde den Index der ersten "8" im Test-Datensatz
idx_8 = np.where(y_test == 8)[0][0]
bild_8 = x_test_raw[idx_8]

# Bild für das Netzwerk vorbereiten
bild_8_input = np.expand_dims(bild_8.astype("float32") / 255, axis=(0, -1))

# ==========================================
# 2. Die relevanten Schichten finden
# ==========================================
# Wir suchen automatisch den letzten Conv2D-Layer und den Dense-Layer mit 256 Neuronen
letzter_conv_layer = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)][-1]
dense_256_layer = [l for l in model.layers if isinstance(l, keras.layers.Dense) and l.units == 256][0]

print(f"Zapfe Formen an: {letzter_conv_layer.name}")
print(f"Zapfe Konzepte an: {dense_256_layer.name}")

# Wir bauen ein Multi-Output-Modell, das uns BEIDE Zwischenergebnisse liefert
analyse_modell = keras.Model(
    inputs=model.inputs, 
    outputs=[letzter_conv_layer.output, dense_256_layer.output]
)

# Wir schicken die "8" durch und fangen die Signale auf
conv_formen, dense_fingerabdruck = analyse_modell.predict(bild_8_input, verbose=0)

# ==========================================
# 3. Visualisierung
# ==========================================
fig = plt.figure(figsize=(15, 10))
fig.canvas.manager.set_window_title('Analyse der Acht')

# --- Plot 1: Das Originalbild ---
ax1 = plt.subplot(2, 3, 1)
ax1.imshow(bild_8, cmap="gray")
ax1.set_title("Originale 8", fontsize=14)
ax1.axis("off")

# --- Plot 2: Der "Fingerabdruck" (Dense 256) ---
# Das ist ein 1D-Array mit 256 Werten. Wir zeigen es als Balkendiagramm (Equalizer)
ax2 = plt.subplot(2, 3, (2, 3)) # Nimmt den Platz von Spalte 2 und 3 ein
ax2.bar(range(256), dense_fingerabdruck[0])
ax2.set_title("Der konzeptionelle Fingerabdruck (Dense 256)", fontsize=14)
ax2.set_xlabel("Neuron ID (0 - 255)")
ax2.set_ylabel("Aktivierungsstärke")

# --- Plot 3: Die erkannten High-Level Formen (Letzter Conv Layer) ---
# Der letzte Conv Layer hat 64 Filter. Wir zeigen die 10 stärksten an.
# Wir berechnen, welche Filter im Durchschnitt am stärksten geleuchtet haben
filter_staerken = np.mean(conv_formen[0], axis=(0, 1))
top_10_filter_indizes = np.argsort(filter_staerken)[-10:][::-1]

plt.figtext(0.5, 0.45, "Was die 10 stärksten Faltungs-Filter (letzter Conv-Layer) in dieser 8 sehen:", 
            ha="center", fontsize=14, fontweight="bold")

# Plotten der 10 stärksten Feature Maps (meist winzige 7x7 Pixel Bilder)
for i, filter_idx in enumerate(top_10_filter_indizes):
    ax = plt.subplot(2, 10, 11 + i) # Startet in der zweiten Hälfte des Fensters
    gefiltertes_bild = conv_formen[0, :, :, filter_idx]
    
    ax.imshow(gefiltertes_bild, cmap="viridis")
    ax.set_title(f"Filter {filter_idx}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()