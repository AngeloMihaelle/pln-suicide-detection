# 🧠 Proyecto PLN - Clasificación de Pensamientos Suicidas en Tweets

Este proyecto utiliza técnicas de **Procesamiento de Lenguaje Natural (PLN)** para detectar pensamientos suicidas en publicaciones de Twitter. Se han implementado dos enfoques de clasificación: un modelo basado en **BERT** (Transformers) y otro con **Regresión Logística** usando características TF-IDF. Además, se incluye una interfaz gráfica desarrollada en **Tkinter** para facilitar la interacción del usuario.

---

## 🌟 Objetivo

Clasificar textos (tweets) en dos categorías:

* `0` → **Pensamiento NO suicida**
* `1` → **Pensamiento suicida**

Este proyecto puede ser usado como base para herramientas de apoyo en salud mental, monitoreo de redes sociales y análisis de lenguaje con fines preventivos.

---

## 📁 Estructura del Proyecto

```
Proyecto_PLN-main/
│
├── .gitattributes              # Atributos para Git
├── .gitignore                  # Archivos ignorados por Git
├── README.md                   # Documentación del proyecto
├── bitacora.txt                # Registro de avances y decisiones
│
├── create_split.py            # Preprocesamiento y división del dataset
├── data_raw.csv               # Tweets crudos con etiquetas
├── train_split.csv            # Conjunto de entrenamiento
├── validation_split.csv       # Conjunto de validación
├── test_split.csv             # Conjunto de prueba
│
├── train_BERT.py              # Entrenamiento de modelo BERT
├── train_LR.py                # Entrenamiento con Regresión Logística
├── load_and_predict.py        # Clasificación de textos con modelos entrenados
├── ui.py                      # Interfaz gráfica con Tkinter
│
└── PLN_Proyecto/
    └── pato_Angello.jpg       # Imagen usada en la interfaz gráfica
```

---

## ⚙️ Requisitos

Asegúrate de tener **Python 3.8+** instalado. Instala las dependencias principales con:

```bash
pip install transformers pandas scikit-learn torch pillow
```

---

## 🚀 ¿Cómo usar el proyecto?

### 1. 📄 Preprocesar datos

```bash
python create_split.py
```

Divide el dataset original (`data_raw.csv`) en:

* `train_split.csv`
* `validation_split.csv`
* `test_split.csv`

---

### 2. 🧠 Entrenamiento con BERT (`train_BERT.py`)

Este script entrena un modelo **BERT** multilingüe especializado en español.

#### Características destacadas:

* Modelo base: `dccuchile/bert-base-spanish-wwm-uncased`
* Clasificación binaria (`suicida`, `no_suicida`)
* Checkpoints automáticos
* Soporte para GPU y Early stopping
* Guarda el modelo en `./trained_model`

#### Ejecución:

```bash
python train_BERT.py
```

#### Salida esperada:

* Métricas (accuracy, precision, recall, F1-score) para validación y prueba
* Guardado del modelo y del objeto `Trainer` como `trained_model.pkl`

---

### 3. 🔢 Entrenamiento con Regresión Logística (`train_LR.py`)

Modelo clásico basado en TF-IDF + Regresión Logística.

#### Características:

* Usa `TfidfVectorizer` con unigramas y bigramas
* Submuestreo a 80,000 muestras para rapidez
* Muestra `classification_report` sobre el conjunto de validación

#### Ejecución:

```bash
python train_LR.py
```

---

### 4. 🔍 Clasificación de Texto (`load_and_predict.py`)

Permite clasificar un texto usando el modelo BERT entrenado.

#### Ejecución:

```bash
python load_and_predict.py
```

#### Características:

* Usa tokenizer y modelo desde `trained_model`
* Devuelve `suicida` o `no_suicida`
* Dispositivo: GPU si está disponible

---

### 5. 💻 Interfaz Gráfica (Tkinter) (`ui.py`)

Lanza una interfaz con campos de entrada y resultados visuales.

#### Ejecución:

```bash
python ui.py
```

#### Características:

* Ventana con menú lateral y contenido principal
* Ingreso de texto, botón de clasificar y resultado
* Usa la imagen `pato_Angello.jpg` en la interfaz

---

## 📊 Dataset

El archivo principal `data_raw.csv` contiene:

* `text`: texto del tweet
* `label`: `suicida` o `no_suicida`

Los scripts generan:

* `train_split.csv`
* `validation_split.csv`
* `test_split.csv`

---

## 📃 Bitácora

Consulta `bitacora.txt` para conocer:

* Cambios realizados
* Problemas encontrados
* Decisiones de diseño

---

## ⚠️ Notas Éticas

Este proyecto trata información sensible relacionada con la salud mental. Su uso debe hacerse con responsabilidad, empatía y siempre con fines académicos, investigativos o de ayuda profesional. **No sustituye el diagnóstico ni tratamiento de un profesional de la salud.**

---

## 👤 Autor

Desarrollado por **Angelo Ojeda** y **Laura Estrada** como parte de un proyecto de PLN.

---

## 🔹 Licencia

Este proyecto se distribuye bajo la Licencia MIT.
