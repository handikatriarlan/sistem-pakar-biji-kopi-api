# ☕ Sistem Pakar Penilaian Kualitas Biji Kopi API

Sistem Pakar untuk menilai kualitas biji kopi menggunakan **Jaringan Saraf Tiruan Backpropagation** berdasarkan standar protokol cupping SCA (Specialty Coffee Association).

## ✨ Fitur

- **Prediksi Kualitas**: Klasifikasi kopi ke Grade A, B, atau C menggunakan jaringan saraf tiruan
- **Pemrosesan Batch**: Prediksi beberapa sampel dalam satu permintaan (maksimal 100)
- **Analisis Mendetail**: Dapatkan breakdown komprehensif setiap atribut sensorik
- **Analisis Fitur**: Identifikasi kekuatan dan area yang perlu ditingkatkan
- **Rekomendasi AI**: Terima saran peningkatan kualitas
- **Manajemen Model**: Training, retraining, dan monitoring performa model
- **Data Referensi**: Akses panduan cupping SCA dan informasi fitur

## 🧠 Algoritma Backpropagation

Model menggunakan Multi-Layer Perceptron dengan algoritma Backpropagation:

### Arsitektur Jaringan

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER (10 neuron)                  │
│  [Aroma, Flavor, Aftertaste, Acidity, Body, Balance,       │
│   Uniformity, Clean Cup, Sweetness, Moisture]              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              HIDDEN LAYER 1 (64 neuron, ReLU)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              HIDDEN LAYER 2 (32 neuron, ReLU)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              HIDDEN LAYER 3 (16 neuron, ReLU)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            OUTPUT LAYER (3 neuron, Softmax)                 │
│                  [Grade A, Grade B, Grade C]                │
└─────────────────────────────────────────────────────────────┘
```

### Proses Training

| Langkah | Proses | Deskripsi |
|---------|--------|-----------|
| 1 | **Forward Propagation** | Input → Hidden Layers → Output |
| 2 | **Hitung Loss** | Perhitungan cross-entropy loss |
| 3 | **Backward Propagation** | Hitung gradien menggunakan chain rule |
| 4 | **Update Bobot** | Adam optimizer dengan adaptive learning rate |

## 📁 Struktur Proyek

```
sistem-pakar-biji-kopi-api/
├── src/
│   ├── api/routes/           # Endpoint API
│   ├── core/                 # Konfigurasi & exceptions
│   ├── models/               # Implementasi jaringan saraf tiruan
│   ├── schemas/              # Skema Request/Response
│   └── main.py               # Aplikasi FastAPI
├── data/                     # Dataset training
├── trained_models/           # Model tersimpan (auto-generated)
├── bruno/                    # Koleksi test API
├── requirements.txt
├── run.py
├── README.md
└── API_DOCUMENTATION.md      # Dokumentasi API lengkap
```

## 🚀 Instalasi & Menjalankan

### Prasyarat

- Python 3.10+
- pip

### Instalasi

```bash
# Clone repository
git clone https://github.com/handikatriarlan/sistem-pakar-biji-kopi-api.git
cd sistem-pakar-biji-kopi-api

# Buat virtual environment (disarankan)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Menjalankan API

```bash
python run.py
```

API akan tersedia di:
- **API**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📚 Endpoint API

| Kategori | Method | Endpoint | Deskripsi |
|----------|--------|----------|-----------|
| Health | GET | `/api/v1/health` | Cek kesehatan API |
| Health | GET | `/api/v1/health/ready` | Cek kesiapan layanan |
| Health | GET | `/api/v1/health/live` | Cek ketersediaan layanan |
| Prediksi | POST | `/api/v1/predict` | Prediksi tunggal |
| Prediksi | POST | `/api/v1/predict/batch` | Prediksi batch |
| Prediksi | POST | `/api/v1/predict/analyze` | Analisis detail |
| Model | GET | `/api/v1/model/info` | Informasi model |
| Model | GET | `/api/v1/model/architecture` | Arsitektur jaringan |
| Model | GET | `/api/v1/model/metrics` | Metrik training |
| Model | GET | `/api/v1/model/status` | Status model |
| Model | POST | `/api/v1/model/train` | Training model |
| Model | POST | `/api/v1/model/retrain` | Retraining model |
| Referensi | GET | `/api/v1/reference/grades` | Informasi grade |
| Referensi | GET | `/api/v1/reference/features` | Informasi fitur |
| Referensi | GET | `/api/v1/reference/cupping-guide` | Panduan cupping SCA |

> 📖 **Untuk dokumentasi API lengkap, lihat [API_DOCUMENTATION.md](API_DOCUMENTATION.md)**

## 📝 Contoh Quick Start

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

## 📊 Klasifikasi Grade

Berdasarkan standar SCA (Specialty Coffee Association):

| Grade | Rentang Skor | Label | Deskripsi |
|-------|--------------|-------|-----------|
| **A** | ≥ 85 | Specialty Grade | Kualitas premium, layak untuk specialty market |
| **B** | 80 - 84.99 | Premium Grade | Kualitas baik, cocok untuk pasar umum premium |
| **C** | < 80 | Below Premium | Di bawah standar premium, perlu peningkatan kualitas |

## 📝 Fitur Input

| Fitur | Rentang | Kategori | Deskripsi |
|-------|---------|----------|-----------|
| aroma | 0-10 | Sensorik | Intensitas dan kualitas aroma |
| flavor | 0-10 | Sensorik | Rasa keseluruhan |
| aftertaste | 0-10 | Sensorik | Rasa yang tertinggal |
| acidity | 0-10 | Sensorik | Tingkat kecerahan |
| body | 0-10 | Sensorik | Mouthfeel/tekstur |
| balance | 0-10 | Sensorik | Keharmonisan atribut |
| uniformity | 0-10 | Kualitas | Konsistensi antar cup |
| clean_cup | 0-10 | Kualitas | Ketiadaan defect |
| sweetness | 0-10 | Kualitas | Kemanisan alami |
| moisture_percentage | 0-20 | Fisik | Kelembaban biji % |

## 🧪 Testing dengan Bruno

Proyek ini menyertakan koleksi test API Bruno di folder `/bruno`. Import koleksi di Bruno untuk menjalankan test API secara komprehensif.