# Sistem Pakar Penilaian Kualitas Biji Kopi API

Sistem Pakar untuk menilai kualitas biji kopi menggunakan **Backpropagation Neural Network**.

## 🧠 Algoritma Backpropagation

Model menggunakan Multi-Layer Perceptron dengan algoritma Backpropagation:

### Arsitektur Jaringan

```
Input Layer (10 neurons)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Hidden Layer 3 (16 neurons, ReLU)
    ↓
Output Layer (3 neurons, Softmax)
```

### Proses Training

1. **Forward Propagation**: Input → Hidden Layers → Output
2. **Compute Loss**: Cross-entropy loss
3. **Backward Propagation**: Hitung gradien dengan chain rule
4. **Update Weights**: Adam optimizer dengan adaptive learning rate

## 📁 Struktur Project

```
sistem-pakar-biji-kopi-api/
├── src/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py       # Health check endpoints
│   │       ├── prediction.py   # Prediction endpoints
│   │       ├── model.py        # Model management
│   │       └── reference.py    # Reference data
│   ├── core/
│   │   ├── config.py          # Configuration
│   │   └── exceptions.py      # Custom exceptions
│   ├── models/
│   │   └── backpropagation.py # Neural network model
│   ├── schemas/
│   │   ├── request.py         # Request schemas
│   │   └── response.py        # Response schemas
│   └── main.py                # FastAPI application
├── data/
│   └── df_arabica_clean.csv   # Dataset
├── trained_models/            # Saved models (generated)
├── requirements.txt
├── run.py
└── README.md
```

## 🚀 Instalasi & Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python run.py
```

API akan berjalan di `http://localhost:8000`

## 📚 API Endpoints

### Health

| Method | Endpoint               | Deskripsi       |
| ------ | ---------------------- | --------------- |
| GET    | `/api/v1/health`       | Health check    |
| GET    | `/api/v1/health/ready` | Readiness check |
| GET    | `/api/v1/health/live`  | Liveness check  |

### Prediction

| Method | Endpoint                  | Deskripsi              |
| ------ | ------------------------- | ---------------------- |
| POST   | `/api/v1/predict`         | Prediksi kualitas kopi |
| POST   | `/api/v1/predict/batch`   | Batch prediction       |
| POST   | `/api/v1/predict/analyze` | Analisis detail        |

### Model

| Method | Endpoint                     | Deskripsi                 |
| ------ | ---------------------------- | ------------------------- |
| GET    | `/api/v1/model/info`         | Info model                |
| GET    | `/api/v1/model/architecture` | Arsitektur neural network |
| GET    | `/api/v1/model/metrics`      | Training metrics          |
| POST   | `/api/v1/model/train`        | Train/retrain model       |

### Reference

| Method | Endpoint                          | Deskripsi           |
| ------ | --------------------------------- | ------------------- |
| GET    | `/api/v1/reference/grades`        | Info grade kualitas |
| GET    | `/api/v1/reference/features`      | Info fitur input    |
| GET    | `/api/v1/reference/cupping-guide` | Panduan cupping     |

## 📝 Contoh Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "aroma": 8.0,
    "flavor": 8.0,
    "aftertaste": 7.5,
    "acidity": 8.0,
    "body": 7.5,
    "balance": 8.0,
    "uniformity": 10.0,
    "clean_cup": 10.0,
    "sweetness": 10.0,
    "moisture_percentage": 11.0
  }'
```

## 📊 Grade Classification

| Grade | Score      | Label           |
| ----- | ---------- | --------------- |
| A     | ≥ 85       | Specialty Grade |
| B     | 80 - 84.99 | Premium Grade   |
| C     | < 80       | Below Premium   |

## 📖 Dokumentasi

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
