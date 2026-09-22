<div align="center">

<br/>

```
 ██████╗██╗      █████╗ ███████╗██╗███████╗██╗ ██████╗ █████╗ ██████╗  ██████╗ ██████╗ 
██╔════╝██║     ██╔══██╗██╔════╝██║██╔════╝██║██╔════╝██╔══██╗██╔══██╗██╔═══██╗██╔══██╗
██║     ██║     ███████║███████╗██║█████╗  ██║██║     ███████║██║  ██║██║   ██║██████╔╝
██║     ██║     ██╔══██║╚════██║██║██╔══╝  ██║██║     ██╔══██║██║  ██║██║   ██║██╔══██╗
╚██████╗███████╗██║  ██║███████║██║██║     ██║╚██████╗██║  ██║██████╔╝╚██████╔╝██║  ██║
 ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝
                                                                                          
██████╗ ███████╗    ██████╗ ███████╗███████╗██╗██████╗ ██╗   ██╗ ██████╗ ███████╗       
██╔══██╗██╔════╝    ██╔══██╗██╔════╝██╔════╝██║██╔══██╗██║   ██║██╔═══██╗██╔════╝       
██║  ██║█████╗      ██████╔╝█████╗  ███████╗██║██║  ██║██║   ██║██║   ██║███████╗       
██║  ██║██╔══╝      ██╔══██╗██╔══╝  ╚════██║██║██║  ██║██║   ██║██║   ██║╚════██║       
██████╔╝███████╗    ██║  ██║███████╗███████║██║██████╔╝╚██████╔╝╚██████╔╝███████║       
╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═════╝  ╚═════╝  ╚═════╝╚══════╝       
```

### _Clasificación automática de residuos mediante visión por computador y transfer learning_

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![DenseNet121](https://img.shields.io/badge/Model-DenseNet121-4B0082?style=for-the-badge)](https://keras.io/api/applications/densenet/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Real--time-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Conda](https://img.shields.io/badge/Entorno-Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)](https://docs.conda.io/)

<br/>

> **Proyecto de Visión por Computador** · Ingeniería Robótica · 2024

</div>

---

## ♻️ ¿De qué trata el proyecto?

Sistema de **clasificación automática de residuos en tiempo real** usando la cámara del ordenador. El modelo identifica si un objeto es **cartón, metal, plástico o vidrio**, e indica en pantalla en qué contenedor debe depositarse.

El proyecto pasó por **4 versiones** tanto del código como del dataset, evolucionando desde arquitecturas básicas hasta transfer learning con DenseNet121, alcanzando una precisión final del **96%**.

---

## 🗑️ Clases y Contenedores

| Residuo | Contenedor | Color |
|---------|------------|-------|
| 🟫 Cartón | Contenedor azul | `(255, 0, 0)` |
| 🔩 Metal | Contenedor amarillo | `(0, 255, 255)` |
| 🧴 Plástico | Contenedor amarillo | `(0, 255, 255)` |
| 🫙 Vidrio | Contenedor verde | `(0, 255, 0)` |

---

## 🧠 Arquitectura del Modelo

```
Input (224×224×3)
       │
       ▼
┌─────────────────────────────────┐
│         DenseNet121             │  ← Preentrenado en ImageNet
│    (capas congeladas)           │     Extracción de características
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│   Conv2D(32, 3×3) + ReLU        │
│   MaxPooling2D(2×2)             │
│   Conv2D(64, 3×3) + ReLU        │
│   MaxPooling2D(2×2)             │
│   Flatten                       │
│   Dense(32, ReLU)               │
│   Dense(4, Softmax)             │  ← 4 clases de residuos
└─────────────────────────────────┘
       │
       ▼
  [cartón, metal, plástico, vidrio]
```

**Compilación:** Adam · `categorical_crossentropy` · 7 épocas · batch=64 · val_split=0.2

---

## 📦 Evolución del Dataset

El dataset creció progresivamente con cada versión del proyecto:

| Versión | Fuente principal | Tamaño aprox. | Mejora |
|---------|-----------------|---------------|--------|
| v1 | Dataset público de Kaggle | ~2.500 imgs | Baseline |
| v2 | + Imágenes propias | ~4.000 imgs | Más variedad |
| v3 | + Frames de vídeos | ~6.000 imgs | Mayor diversidad |
| v4 | + Ampliación y limpieza | ~8.000 imgs | Dataset final |

> Las imágenes de vídeos se extrajeron automáticamente con `video_to_images.py`, capturando un frame cada 0.05 segundos.

---

## 🔄 Evolución del Código

```
v1  →  CNN básica desde cero (Conv2D + Dense)
v2  →  Mejora de arquitectura + augmentation
v3  →  InceptionResNetV2 (primer intento con transfer learning)
v4  →  DenseNet121 + capas convolucionales adicionales  ✅ Final
```

---

## 📁 Estructura del Proyecto

```
ClasificadorResiduos/
│
├── version4.py          # Entrenamiento del modelo (script principal)
├── camara.py            # Detector en tiempo real con webcam
├── video_to_images.py   # Extracción de frames de vídeos para el dataset
├── best_model.h5        # Mejor modelo guardado (checkpoint)
├── saved_model.pb       # Modelo exportado en formato SavedModel
│
├── videos/              # Vídeos utilizados para ampliar el dataset
│
└── DataSheet/
    └── GarbageClassification/
        ├── train/
        │   ├── Carton/
        │   ├── Metal/
        │   ├── Plastico/
        │   └── Vidrio/
        └── test/
            ├── Carton/
            ├── Metal/
            ├── Plastico/
            └── Vidrio/
```

---

## 🚀 Cómo ejecutarlo

### Entrenamiento (`version4.py`)

```bash
# 1. Activar entorno conda
conda activate <nombre_entorno>

# 2. Cambiar las rutas en version4.py
train_path = 'ruta/a/GarbageClassification/train'
test_path  = 'ruta/a/GarbageClassification/test'

# 3. Ejecutar
python version4.py
```

### Detector en tiempo real (`camara.py`)

```bash
python camara.py
```

| Tecla | Acción |
|-------|--------|
| `S` | Activar / desactivar escaneo |
| `Q` | Salir |

---

## 🎬 Demo

<div align="center">

[![Demo Clasificador de Residuos](https://img.youtube.com/vi/C4SCLI8aqXw/maxresdefault.jpg)](https://www.youtube.com/watch?v=C4SCLI8aqXw)

</div>

---

## 📊 Resultados

- **Precisión final:** ~96% en test
- **Matriz de confusión:** generada automáticamente con Seaborn al ejecutar `version4.py`
- **Checkpoint automático:** `ModelCheckpoint` guarda el mejor modelo en `best_model.h5` según `val_loss`

---

## 🛠️ Tecnologías

- **TensorFlow / Keras** — entrenamiento y carga del modelo
- **DenseNet121** — transfer learning desde ImageNet
- **OpenCV** — captura de vídeo y visualización en tiempo real
- **scikit-learn** — `LabelEncoder`, `confusion_matrix`
- **Seaborn / Matplotlib** — visualización de resultados
- **Pillow (PIL)** — carga y redimensionado de imágenes

---

---

<div align="center">

_Proyecto desarrollado para la asignatura de Visión por Computador · Grado en Ingeniería Robótica_

</div>
