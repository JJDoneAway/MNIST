# 🧠 Keras 3 & MNIST: Vom "Hello World" zum State-of-the-Art CNN

Dieses Projekt demonstriert den Aufbau, das Training und die tiefgehende visuelle Analyse eines Convolutional Neural Networks (CNN) zur Erkennung handschriftlicher Ziffern (MNIST). Es nutzt das moderne **Keras 3 Multi-Backend-Framework** mit **PyTorch** als Motor, optimiert für Apple Silicon (M1 Pro / MPS).

## 🚀 1. Setup & Installation (Virtuelle Umgebung)
Um Konflikte mit globalen Python-Paketen zu vermeiden, läuft das Projekt in einer isolierten virtuellen Umgebung (`venv`).

### Mac/Linux Terminal-Befehle:
```bash
# 1. In den Projektordner navigieren
cd ~/Downloads/MeinMNISTProjekt

# 2. Virtuelle Umgebung erstellen
python3 -m venv ki_env

# 3. Umgebung aktivieren (Muss vor jedem Arbeiten gemacht werden!)
source ki_env/bin/activate

# 4. Benötigte Pakete installieren
pip install keras torch torchvision numpy matplotlib scikit-learn
```
*(Zum Verlassen der Umgebung später einfach `deactivate` ins Terminal tippen).*

---

## 📂 2. Die Skripte (Unser Workflow)

Wir haben den Prozess in drei logische Python-Skripte unterteilt, die nacheinander ausgeführt werden sollten:

### 📄 Skript 1: `train_sota_cnn.py`
**Zweck:** Trainiert ein tiefes CNN auf ~99,6 % Genauigkeit und speichert es ab.
* **Architektur:** 2 Faltungs-Blöcke (Conv2D -> BatchNorm -> Conv2D -> BatchNorm -> MaxPooling -> Dropout) gefolgt von einem Klassifikationskopf (Flatten -> Dense -> BatchNorm -> Dropout -> Dense).
* **Highlights:** * Nutzt *Data Augmentation* (Translation, Zoom, leichte Rotation), um das Modell robuster zu machen.
  * Nutzt `ReduceLROnPlateau`, um die Lernrate dynamisch zu senken, wenn das Modell stagniert.
* **Output:** Eine Datei namens `mnist_top_model.keras` (Das "Gehirn" auf der Festplatte).

### 📄 Skript 2: `error_dashboard.py`
**Zweck:** Deckt die Schwächen des trainierten Modells auf.
* Lädt das gespeicherte `.keras`-Modell und lässt es auf die 10.000 Testbilder los.
* Berechnet eine **Confusion Matrix**, um zu sehen, welche Ziffern am häufigsten verwechselt werden.
* Filtert automatisch alle Fehler jenseits der Diagonale heraus (z.B. "Wahre 4, aber als 9 erkannt") und plottet diese Fehltritte zusammen mit der Matrix in ein übersichtliches **All-in-One Dashboard** (`GridSpec`).
* Enthält eine Export-Funktion, um das Dashboard als hochauflösendes PNG zu speichern.

### 📄 Skript 3: `explainable_ai.py`
**Zweck:** Schaut "in den Kopf" der KI, um zu verstehen, *wie* sie lernt.
* **Feature Maps:** Zapft den letzten `Conv2D`-Layer an und visualisiert als kleine Bilder, wo das Netzwerk geometrische Formen (wie Linien, Ecken, Schleifen) in einer Ziffer (z.B. einer "8") erkannt hat.
* **Konzeptioneller Fingerabdruck:** Zapft den `Dense(256)`-Layer an und zeigt in einem Balkendiagramm (Equalizer), welche spezifischen Neuronen "aufleuchten", um das abstrakte Konzept einer Ziffer zu repräsentieren, bevor die finale Entscheidung gefällt wird.

---

## 🧠 3. Cheat Sheet: Wichtige Konzepte zum Memorieren

### Das Backend-Setup
In Keras 3 wird das Backend ganz zu Beginn über Umgebungsvariablen definiert. Für Apple M-Chips ist PyTorch ideal:
```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
```

### Die wichtigsten Schichten (Layers) erklärt
* **`Conv2D`:** Die Lupe. Sucht nach lokalen Mustern (Kanten, Kurven) im Bild.
* **`MaxPooling2D`:** Der Kompressor & Rauschfilter. Verkleinert das Bild (meist um die Hälfte), behält aber die stärksten Merkmale. Macht das Modell tolerant gegenüber verschobenen Ziffern.
* **`Flatten`:** Der Übersetzer. Macht aus 2D-Bilddaten eine flache 1D-Liste, damit die Dense-Schicht sie lesen kann. Hierbei geht die räumliche Information verloren.
* **`Dense`:** Das "klassische" neuronale Netz. Verbindet alle Eingangswerte miteinander, um logische Konzepte abzuwiegen und die finale Entscheidung zu treffen.
* **`BatchNormalization`:** Der Tempomacher. Standardisiert die Daten zwischen den Schichten. Verhindert, dass die Werte "ausreißen" und macht das Training extrem schnell und stabil.
* **`Dropout`:** Der Schutz vor Auswendiglernen. Schaltet beim Training zufällig Neuronen ab (z.B. 50%), damit sich das Netzwerk nicht auf einzelne Pixel verlässt, sondern das "große Ganze" lernen muss.

### Metriken & Loss
* **Loss (Verlust):** Der Fehlerwert (z.B. `0.041`). Das ist der Kompass für die KI. Der *Optimizer* (`adam`) versucht, diesen Wert so nah wie möglich an die Null zu drücken.
* **Accuracy (Genauigkeit):** Ein Prozentwert (z.B. `99.6%`). Ist nur für uns Menschen da, um die Leistung einfach abzulesen. Das Modell lernt *nicht* aus der Accuracy.

### Der One-Hot Trick
Statt die Ziel-Labels (0-9) mühsam in Vektoren umzuwandeln (One-Hot-Encoding, z.B. `[0,0,0,1,0,...]`), nutzen wir **`sparse_categorical_crossentropy`** als Loss-Funktion. Dadurch erlaubt uns Keras, einfach die nackten Zahlen (z.B. `3`) als Label zu übergeben. Das spart extrem viel Arbeitsspeicher.

---
*Erstellt im März 2026 - Lokales Training auf Apple M1 Pro.*
