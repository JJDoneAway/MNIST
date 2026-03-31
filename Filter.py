import os
os.environ["KERAS_BACKEND"] = "torch" 

import keras
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Modell und Test-Bild laden
# ==========================================
print("Lade Modell...")
model = keras.models.load_model("mnist_top_model.keras")

# Lade die Testdaten und schnapp dir das allererste Bild (es ist eine '7')
(_, _), (x_test_raw, _) = keras.datasets.mnist.load_data()
test_bild = x_test_raw[0]

# Das Modell erwartet ein Batch-Format: (Anzahl_Bilder, Höhe, Breite, Farbkanäle)
# Wir wandeln das einzelne Bild (28, 28) in (1, 28, 28, 1) um und normalisieren es
test_bild_input = np.expand_dims(test_bild.astype("float32") / 255, axis=(0, -1))

# ==========================================
# 2. Ein "Abhör-Modell" bauen
# ==========================================
# Wir wollen nicht die finale Ziffer (0-9) wissen, sondern die Signale 
# direkt nach der allerersten Faltungsschicht (Conv2D) abfangen.

# Wir suchen automatisch den Namen der ersten Conv2D Schicht in deinem Modell
erste_conv_schicht = None
for layer in model.layers:
    if isinstance(layer, keras.layers.Conv2D):
        erste_conv_schicht = layer
        break

print(f"Zapfe Schicht an: {erste_conv_schicht.name}")

# Wir bauen ein neues Keras-Modell, das denselben Eingang hat wie dein Hauptmodell,
# aber seine Ausgabe direkt nach der ersten Conv-Schicht ausspuckt.
abhoer_modell = keras.Model(inputs=model.inputs, outputs=erste_conv_schicht.output)

# ==========================================
# 3. Das Bild filtern lassen
# ==========================================
# Wir schicken die '7' durch die Schicht. 
# Da deine Schicht 32 Filter hat, bekommen wir 32 verschiedene Bilder zurück!
feature_maps = abhoer_modell.predict(test_bild_input, verbose=0)

# ==========================================
# 4. Visualisierung der Gehirnströme
# ==========================================
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('Blick ins Gehirn der KI')

# Links oben zeigen wir das Originalbild
ax_orig = plt.subplot(5, 8, 1)
ax_orig.imshow(test_bild, cmap="gray")
ax_orig.set_title("Original", color="red")
ax_orig.axis("off")

# Jetzt plotten wir 31 der 32 gefilterten Bilder daneben/darunter
# Wir fangen bei Index 2 an, da Platz 1 für das Originalbild reserviert ist
for i in range(31):
    ax = plt.subplot(5, 8, i + 2)
    # feature_maps hat das Format (1, 28, 28, 32). Wir greifen uns den Filter i.
    gefiltertes_bild = feature_maps[0, :, :, i]
    
    # 'viridis' ist ein Farbschema, das Aktivierungen gut zeigt (Gelb = starke Aktivierung, Dunkellila = keine)
    ax.imshow(gefiltertes_bild, cmap="viridis")
    ax.set_title(f"Filter {i+1}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()