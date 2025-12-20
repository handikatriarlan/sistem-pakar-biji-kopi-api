# 📖 Dokumentasi API

Dokumentasi API lengkap untuk Sistem Pakar Penilaian Kualitas Biji Kopi.

## 📑 Daftar Isi

- [Ringkasan](#ringkasan)
- [Base URL](#base-url)
- [Autentikasi](#autentikasi)
- [Format Response](#format-response)
- [Endpoint Health](#endpoint-health)
- [Endpoint Prediksi](#endpoint-prediksi)
- [Endpoint Manajemen Model](#endpoint-manajemen-model)
- [Endpoint Data Referensi](#endpoint-data-referensi)
- [Penanganan Error](#penanganan-error)

---

## Ringkasan

API ini menyediakan endpoint untuk memprediksi kualitas biji kopi menggunakan Jaringan Saraf Tiruan Backpropagation. Sistem mengklasifikasikan kopi ke dalam tiga grade (A, B, C) berdasarkan standar protokol cupping SCA (Specialty Coffee Association).

### Versi API

Versi saat ini: `v1`

### Content Type

Semua request dan response menggunakan format JSON:

```
Content-Type: application/json
```

---

## Base URL

| Environment | Base URL |
|-------------|----------|
| Development | `http://localhost:8000/api/v1` |
| Production | `https://sistem-pakar-biji-kopi-api.fly.dev/api/v1` |

---

## Autentikasi

Saat ini, API tidak memerlukan autentikasi. Semua endpoint dapat diakses secara publik.

---

## Format Response

### Response Sukses

Semua response sukses mengikuti struktur ini:

```json
{
  "success": true,
  "message": "Pesan sukses deskriptif",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    // Data response
  }
}
```

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| success | boolean | Selalu `true` untuk request yang berhasil |
| message | string | Pesan sukses yang dapat dibaca manusia |
| timestamp | datetime | Timestamp format ISO 8601 |
| data | object | Payload response (bervariasi per endpoint) |

### Response Error

```json
{
  "detail": "Pesan error yang menjelaskan apa yang salah"
}
```

Atau untuk error validasi:

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "nama_field"],
      "msg": "Deskripsi error",
      "input": "nilai_invalid"
    }
  ]
}
```

---

## Endpoint Health

Endpoint health check untuk monitoring status layanan.

### 1. Health Check

Cek kesehatan API dan status model secara keseluruhan.

**Endpoint:**

```http
GET /health
```

**Headers:**

| Header | Nilai | Wajib |
|--------|-------|-------|
| Accept | application/json | Tidak |

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response:**

```json
{
  "status": "healthy",
  "model_status": "ready",
  "version": "1.0.0",
  "timestamp": "2025-12-20T10:30:00.000000"
}
```

**Field Response:**

| Field | Tipe | Deskripsi | Nilai yang Mungkin |
|-------|------|-----------|---------------------|
| status | string | Kesehatan layanan keseluruhan | `healthy`, `unhealthy` |
| model_status | string | Status model jaringan saraf | `ready`, `not_ready` |
| version | string | Versi API (format semver) | contoh: `1.0.0` |
| timestamp | datetime | Waktu pembuatan response | Format ISO 8601 |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Layanan sehat |

---

### 2. Readiness Check

Cek apakah layanan siap menerima request prediksi.

**Endpoint:**

```http
GET /health/ready
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/health/ready"
```

**Response (Siap):**

```json
{
  "ready": true,
  "message": "Service is ready"
}
```

**Response (Belum Siap):**

```json
{
  "ready": false,
  "message": "Model not trained yet"
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| ready | boolean | Apakah layanan dapat menerima request |
| message | string | Deskripsi status |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Pengecekan selesai (lihat field `ready` untuk status) |

**Kasus Penggunaan:**

Gunakan endpoint ini di Kubernetes readiness probes atau health check load balancer untuk memastikan traffic hanya diarahkan ke instance yang dapat memproses request.

---

### 3. Liveness Check

Pengecekan sederhana untuk memverifikasi proses layanan masih berjalan.

**Endpoint:**

```http
GET /health/live
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/health/live"
```

**Response:**

```json
{
  "alive": true
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| alive | boolean | Selalu `true` jika layanan berjalan |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Layanan aktif |

**Kasus Penggunaan:**

Gunakan endpoint ini di Kubernetes liveness probes untuk mendeteksi dan me-restart container yang tidak responsif.

---

## Endpoint Prediksi

Endpoint untuk memprediksi grade kualitas kopi.

### 1. Prediksi Tunggal

Prediksi grade kualitas kopi untuk satu sampel menggunakan Jaringan Saraf Tiruan Backpropagation.

**Endpoint:**

```http
POST /predict
```

**Headers:**

| Header | Nilai | Wajib |
|--------|-------|-------|
| Content-Type | application/json | Ya |

**Request Body:**

```json
{
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
}
```

**Parameter Request:**

| Parameter | Tipe | Rentang | Wajib | Deskripsi |
|-----------|------|---------|-------|-----------|
| aroma | float | 0-10 | ✅ Ya | Intensitas dan kualitas aroma kopi. Dievaluasi saat fase kering (fragrance) dan basah (aroma). |
| flavor | float | 0-10 | ✅ Ya | Rasa keseluruhan termasuk aroma retro-nasal. Kombinasi taste buds dan persepsi aromatik. |
| aftertaste | float | 0-10 | ✅ Ya | Kualitas rasa yang tertinggal setelah ditelan atau dibuang. Durasi dan kesenangan dari rasa yang tersisa. |
| acidity | float | 0-10 | ✅ Ya | Kecerahan dan keaktifan kopi. Acidity yang baik digambarkan sebagai "bright" bukan "sour". |
| body | float | 0-10 | ✅ Ya | Mouthfeel dan tekstur fisik. Sensasi taktil kopi di lidah dan langit-langit. |
| balance | float | 0-10 | ✅ Ya | Keharmonisan antara flavor, aftertaste, acidity, dan body. Tidak ada atribut yang mendominasi secara negatif. |
| uniformity | float | 0-10 | ✅ Ya | Konsistensi rasa antar cup. 2 poin per cup yang konsisten (5 cups = maksimal 10 poin). |
| clean_cup | float | 0-10 | ✅ Ya | Ketiadaan defect dan off-flavors. 2 poin per cup yang bersih (5 cups = maksimal 10 poin). |
| sweetness | float | 0-10 | ✅ Ya | Persepsi kemanisan alami. 2 poin per cup yang manis (5 cups = maksimal 10 poin). |
| moisture_percentage | float | 0-20 | ✅ Ya | Kandungan kelembaban biji kopi hijau. Rentang ideal: 9-12%. |

**Contoh Request:**

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

**Response:**

```json
{
  "success": true,
  "message": "Prediksi berhasil dilakukan",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "prediction": {
      "grade": "A",
      "confidence": 0.92,
      "probabilities": {
        "A": 0.92,
        "B": 0.07,
        "C": 0.01
      },
      "grade_info": {
        "label": "Specialty Grade",
        "description": "Kopi dengan kualitas premium, layak untuk specialty market"
      }
    },
    "feature_analysis": [
      {
        "name": "aroma",
        "value": 8.0,
        "status": "excellent",
        "benchmark": 7.5,
        "recommendation": null
      },
      {
        "name": "flavor",
        "value": 8.0,
        "status": "excellent",
        "benchmark": 7.5,
        "recommendation": null
      },
      {
        "name": "aftertaste",
        "value": 7.5,
        "status": "good",
        "benchmark": 7.5,
        "recommendation": "Tingkatkan aftertaste untuk hasil yang lebih optimal"
      },
      {
        "name": "body",
        "value": 7.5,
        "status": "good",
        "benchmark": 7.5,
        "recommendation": "Tingkatkan body untuk hasil yang lebih optimal"
      }
    ],
    "recommendations": [
      "Pertahankan kualitas aroma yang sudah excellent",
      "Pertahankan kualitas flavor yang sudah excellent",
      "Tingkatkan aftertaste untuk mencapai grade yang lebih tinggi",
      "Tingkatkan body untuk hasil yang lebih optimal"
    ],
    "backpropagation_info": {
      "model_accuracy": 0.857,
      "prediction_method": "Forward propagation through trained neural network"
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.prediction.grade | string | Grade yang diprediksi: `A`, `B`, atau `C` |
| data.prediction.confidence | float | Skor kepercayaan (0-1) |
| data.prediction.probabilities | object | Distribusi probabilitas untuk setiap grade |
| data.prediction.grade_info | object | Label dan deskripsi grade yang dapat dibaca manusia |
| data.feature_analysis | array | Analisis setiap fitur input |
| data.feature_analysis[].name | string | Nama fitur |
| data.feature_analysis[].value | float | Nilai input |
| data.feature_analysis[].status | string | `excellent`, `good`, atau `needs_improvement` |
| data.feature_analysis[].benchmark | float | Nilai benchmark referensi |
| data.feature_analysis[].recommendation | string | Saran perbaikan (null jika excellent) |
| data.recommendations | array | Daftar rekomendasi perbaikan |
| data.backpropagation_info | object | Informasi jaringan saraf |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Prediksi berhasil |
| 422 | Error validasi - nilai input tidak valid |
| 503 | Model belum siap - perlu training |

**Aturan Validasi:**

- Semua atribut sensorik harus antara 0 dan 10
- Persentase kelembaban harus antara 0 dan 20
- Semua field wajib diisi

---

### 2. Prediksi Batch

Prediksi kualitas untuk beberapa sampel kopi dalam satu request.

**Endpoint:**

```http
POST /predict/batch
```

**Headers:**

| Header | Nilai | Wajib |
|--------|-------|-------|
| Content-Type | application/json | Ya |

**Request Body:**

```json
{
  "samples": [
    {
      "aroma": 8.5,
      "flavor": 8.5,
      "aftertaste": 8.25,
      "acidity": 8.5,
      "body": 8.25,
      "balance": 8.5,
      "uniformity": 10.0,
      "clean_cup": 10.0,
      "sweetness": 10.0,
      "moisture_percentage": 10.5
    },
    {
      "aroma": 7.5,
      "flavor": 7.5,
      "aftertaste": 7.25,
      "acidity": 7.5,
      "body": 7.5,
      "balance": 7.5,
      "uniformity": 10.0,
      "clean_cup": 10.0,
      "sweetness": 10.0,
      "moisture_percentage": 11.0
    },
    {
      "aroma": 6.5,
      "flavor": 6.5,
      "aftertaste": 6.25,
      "acidity": 6.5,
      "body": 6.5,
      "balance": 6.5,
      "uniformity": 8.0,
      "clean_cup": 8.0,
      "sweetness": 8.0,
      "moisture_percentage": 12.5
    }
  ]
}
```

**Parameter Request:**

| Parameter | Tipe | Wajib | Deskripsi |
|-----------|------|-------|-----------|
| samples | array | ✅ Ya | Array sampel kopi (1-100 item) |
| samples[] | object | ✅ Ya | Struktur sama dengan request prediksi tunggal |

**Batasan:**

- Minimum sampel: 1
- Maksimum sampel: 100

**Contoh Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "aroma": 8.5, "flavor": 8.5, "aftertaste": 8.25,
        "acidity": 8.5, "body": 8.25, "balance": 8.5,
        "uniformity": 10.0, "clean_cup": 10.0, "sweetness": 10.0,
        "moisture_percentage": 10.5
      },
      {
        "aroma": 7.5, "flavor": 7.5, "aftertaste": 7.25,
        "acidity": 7.5, "body": 7.5, "balance": 7.5,
        "uniformity": 10.0, "clean_cup": 10.0, "sweetness": 10.0,
        "moisture_percentage": 11.0
      }
    ]
  }'
```

**Response:**

```json
{
  "success": true,
  "message": "Batch prediction berhasil untuk 3 sampel",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "predictions": [
      {
        "sample_index": 0,
        "prediction": {
          "grade": "A",
          "confidence": 0.95,
          "probabilities": {"A": 0.95, "B": 0.04, "C": 0.01},
          "grade_info": {
            "label": "Specialty Grade",
            "description": "Kopi dengan kualitas premium"
          }
        },
        "feature_analysis": [...],
        "recommendations": [...]
      },
      {
        "sample_index": 1,
        "prediction": {
          "grade": "B",
          "confidence": 0.88,
          "probabilities": {"A": 0.10, "B": 0.88, "C": 0.02},
          "grade_info": {
            "label": "Premium Grade",
            "description": "Kopi dengan kualitas baik"
          }
        },
        "feature_analysis": [...],
        "recommendations": [...]
      },
      {
        "sample_index": 2,
        "prediction": {
          "grade": "C",
          "confidence": 0.76,
          "probabilities": {"A": 0.05, "B": 0.19, "C": 0.76},
          "grade_info": {
            "label": "Below Premium Grade",
            "description": "Kopi di bawah standar premium"
          }
        },
        "feature_analysis": [...],
        "recommendations": [...]
      }
    ],
    "summary": {
      "total_samples": 3,
      "grade_distribution": {
        "A": 1,
        "B": 1,
        "C": 1
      },
      "average_confidence": 0.863
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.predictions | array | Array hasil prediksi |
| data.predictions[].sample_index | int | Index sampel dalam array input |
| data.predictions[].prediction | object | Hasil prediksi (sama dengan prediksi tunggal) |
| data.summary.total_samples | int | Jumlah total sampel yang diproses |
| data.summary.grade_distribution | object | Jumlah setiap grade |
| data.summary.average_confidence | float | Rata-rata confidence semua prediksi |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Prediksi batch berhasil |
| 422 | Error validasi - input tidak valid |
| 503 | Model belum siap |

**Kasus Penggunaan:**

- Kontrol kualitas untuk beberapa batch
- Analisis perbandingan sampel kopi
- Pemrosesan massal untuk penilaian inventaris
- A/B testing metode pemrosesan berbeda

---

### 3. Analisis Detail

Dapatkan analisis komprehensif dengan breakdown skor, ringkasan kualitas, dan rekomendasi detail.

**Endpoint:**

```http
POST /predict/analyze
```

**Headers:**

| Header | Nilai | Wajib |
|--------|-------|-------|
| Content-Type | application/json | Ya |

**Request Body:**

Sama dengan prediksi tunggal.

**Contoh Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/predict/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "aroma": 8.0,
    "flavor": 7.75,
    "aftertaste": 7.5,
    "acidity": 8.0,
    "body": 7.5,
    "balance": 7.75,
    "uniformity": 10.0,
    "clean_cup": 10.0,
    "sweetness": 10.0,
    "moisture_percentage": 10.8
  }'
```

**Response:**

```json
{
  "success": true,
  "message": "Analisis berhasil",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "prediction": {
      "grade": "B",
      "confidence": 0.85,
      "probabilities": {"A": 0.12, "B": 0.85, "C": 0.03},
      "grade_info": {
        "label": "Premium Grade",
        "description": "Kopi dengan kualitas baik, cocok untuk pasar umum premium"
      }
    },
    "score_breakdown": {
      "sensory_total": 76.5,
      "individual_scores": {
        "aroma": 8.0,
        "flavor": 7.75,
        "aftertaste": 7.5,
        "acidity": 8.0,
        "body": 7.5,
        "balance": 7.75,
        "uniformity": 10.0,
        "clean_cup": 10.0,
        "sweetness": 10.0
      },
      "moisture": 10.8
    },
    "feature_analysis": [
      {
        "name": "aroma",
        "value": 8.0,
        "status": "excellent",
        "benchmark": 7.5,
        "recommendation": null
      },
      {
        "name": "aftertaste",
        "value": 7.5,
        "status": "good",
        "benchmark": 7.5,
        "recommendation": "Tingkatkan aftertaste untuk hasil yang lebih optimal"
      },
      {
        "name": "body",
        "value": 7.5,
        "status": "good",
        "benchmark": 7.5,
        "recommendation": "Tingkatkan body untuk hasil yang lebih optimal"
      }
    ],
    "recommendations": [
      "Pertahankan kualitas aroma yang sudah excellent",
      "Pertahankan kualitas acidity yang sudah excellent",
      "Tingkatkan aftertaste untuk mencapai grade A",
      "Tingkatkan body untuk meningkatkan mouthfeel"
    ],
    "quality_summary": {
      "strengths": [
        "aroma",
        "acidity",
        "uniformity",
        "clean_cup",
        "sweetness"
      ],
      "areas_for_improvement": [
        "aftertaste",
        "body"
      ]
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.prediction | object | Prediksi grade dengan confidence |
| data.score_breakdown | object | Breakdown skor detail |
| data.score_breakdown.sensory_total | float | Jumlah semua skor sensorik |
| data.score_breakdown.individual_scores | object | Skor untuk setiap atribut sensorik |
| data.score_breakdown.moisture | float | Persentase kelembaban |
| data.feature_analysis | array | Analisis detail per fitur |
| data.recommendations | array | Saran perbaikan yang actionable |
| data.quality_summary | object | Kategorisasi kekuatan dan kelemahan |
| data.quality_summary.strengths | array | Fitur dengan status excellent |
| data.quality_summary.areas_for_improvement | array | Fitur yang perlu ditingkatkan |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Analisis berhasil |
| 422 | Error validasi |
| 503 | Model belum siap |

**Kasus Penggunaan:**

- Laporan penilaian kualitas detail
- Mengidentifikasi area spesifik untuk perbaikan
- Perencanaan peningkatan kualitas
- Sertifikat kualitas untuk pelanggan

---

## Endpoint Manajemen Model

Endpoint untuk mengelola model jaringan saraf.

### 1. Informasi Model

Dapatkan informasi komprehensif tentang model Jaringan Saraf Tiruan Backpropagation yang sudah ditraining.

**Endpoint:**

```http
GET /model/info
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/model/info"
```

**Response:**

```json
{
  "success": true,
  "message": "Model information retrieved successfully",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "architecture": {
      "input_neurons": 10,
      "hidden_layers": [
        {"layer": 1, "neurons": 64, "activation": "relu"},
        {"layer": 2, "neurons": 32, "activation": "relu"},
        {"layer": 3, "neurons": 16, "activation": "relu"}
      ],
      "output_neurons": 3,
      "output_activation": "softmax",
      "total_parameters": 3411
    },
    "metrics": {
      "accuracy": 0.857,
      "precision_macro": 0.556,
      "recall_macro": 0.604,
      "f1_macro": 0.566,
      "cv_accuracy_mean": 0.845,
      "cv_accuracy_std": 0.032
    },
    "training_info": {
      "training_date": "2025-12-20T10:00:00",
      "training_samples": 207,
      "grade_distribution": {"A": 48, "B": 156, "C": 3}
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.architecture | object | Detail arsitektur jaringan saraf |
| data.architecture.input_neurons | int | Jumlah fitur input (10) |
| data.architecture.hidden_layers | array | Konfigurasi hidden layer |
| data.architecture.output_neurons | int | Jumlah kelas output (3) |
| data.architecture.total_parameters | int | Total parameter yang dapat ditraining |
| data.metrics | object | Metrik performa model |
| data.metrics.accuracy | float | Akurasi keseluruhan (0-1) |
| data.metrics.precision_macro | float | Precision rata-rata makro |
| data.metrics.recall_macro | float | Recall rata-rata makro |
| data.metrics.f1_macro | float | F1 score rata-rata makro |
| data.metrics.cv_accuracy_mean | float | Rata-rata akurasi cross-validation |
| data.metrics.cv_accuracy_std | float | Standar deviasi akurasi cross-validation |
| data.training_info | object | Metadata training |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Informasi berhasil diambil |
| 503 | Model belum ditraining |

---

### 2. Arsitektur Jaringan

Dapatkan arsitektur jaringan saraf dan konfigurasi training secara detail.

**Endpoint:**

```http
GET /model/architecture
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/model/architecture"
```

**Response:**

```json
{
  "success": true,
  "message": "Network architecture retrieved",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "algorithm": "Backpropagation Neural Network",
    "architecture": {
      "input_neurons": 10,
      "hidden_layers": [
        {"layer": 1, "neurons": 64, "activation": "relu"},
        {"layer": 2, "neurons": 32, "activation": "relu"},
        {"layer": 3, "neurons": 16, "activation": "relu"}
      ],
      "output_neurons": 3,
      "activation_function": "relu",
      "output_activation": "softmax",
      "learning_rate": 0.001,
      "optimizer": "adam",
      "total_parameters": 3411
    },
    "training_config": {
      "learning_rate": 0.001,
      "max_iterations": 500,
      "optimizer": "adam",
      "activation": "relu",
      "regularization_alpha": 0.0001,
      "early_stopping": true,
      "validation_fraction": 0.1,
      "n_iter_no_change": 10,
      "batch_size": "auto"
    },
    "description": {
      "forward_propagation": "Input → Hidden Layers (ReLU) → Output (Softmax)",
      "backward_propagation": "Compute gradients via chain rule, update weights",
      "optimization": "Adam optimizer with adaptive learning rate"
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.algorithm | string | Nama algoritma |
| data.architecture | object | Struktur jaringan |
| data.training_config | object | Hyperparameter training |
| data.training_config.learning_rate | float | Learning rate untuk update bobot |
| data.training_config.max_iterations | int | Maksimum epoch training |
| data.training_config.optimizer | string | Algoritma optimisasi |
| data.training_config.regularization_alpha | float | Kekuatan regularisasi L2 |
| data.training_config.early_stopping | bool | Apakah early stopping diaktifkan |
| data.description | object | Deskripsi algoritma yang dapat dibaca manusia |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Arsitektur berhasil diambil |

---

### 3. Metrik Training

Dapatkan metrik training dan evaluasi detail termasuk confusion matrix dan metrik per kelas.

**Endpoint:**

```http
GET /model/metrics
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/model/metrics"
```

**Response:**

```json
{
  "success": true,
  "message": "Metrics retrieved successfully",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "overall_metrics": {
      "accuracy": 0.857,
      "precision_macro": 0.556,
      "recall_macro": 0.604,
      "f1_macro": 0.566
    },
    "cross_validation": {
      "cv_accuracy_mean": 0.845,
      "cv_accuracy_std": 0.032,
      "cv_n_splits": 3
    },
    "confusion_matrix": [
      [10, 0, 0],
      [5, 26, 1],
      [0, 0, 0]
    ],
    "per_class_metrics": {
      "A": {
        "precision": 0.667,
        "recall": 1.0,
        "f1-score": 0.8,
        "support": 10
      },
      "B": {
        "precision": 1.0,
        "recall": 0.812,
        "f1-score": 0.897,
        "support": 32
      },
      "C": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0
      }
    },
    "training_info": {
      "iterations": 156,
      "final_loss": 0.234,
      "best_loss": 0.198
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.overall_metrics | object | Metrik performa agregat |
| data.cross_validation | object | Skor cross-validation |
| data.cross_validation.cv_n_splits | int | Jumlah fold CV yang digunakan |
| data.confusion_matrix | array[array] | Confusion matrix 3x3 [A, B, C] |
| data.per_class_metrics | object | Metrik untuk setiap kelas grade |
| data.per_class_metrics.{grade}.precision | float | Precision untuk grade ini |
| data.per_class_metrics.{grade}.recall | float | Recall untuk grade ini |
| data.per_class_metrics.{grade}.f1-score | float | F1 score untuk grade ini |
| data.per_class_metrics.{grade}.support | int | Jumlah sampel untuk grade ini |
| data.training_info | object | Informasi proses training |
| data.training_info.iterations | int | Jumlah iterasi training |
| data.training_info.final_loss | float | Nilai loss di akhir training |
| data.training_info.best_loss | float | Loss terbaik yang dicapai selama training |

**Interpretasi Confusion Matrix:**

```
              Diprediksi
              A    B    C
Aktual  A  [ 10,   0,   0 ]
        B  [  5,  26,   1 ]
        C  [  0,   0,   0 ]
```

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Metrik berhasil diambil |
| 503 | Model belum ditraining |

---

### 4. Status Model

Cek status training model saat ini dan keberadaan file.

**Endpoint:**

```http
GET /model/status
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/model/status"
```

**Response:**

```json
{
  "success": true,
  "message": "Model status retrieved",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "is_trained": true,
    "training_date": "2025-12-20T10:00:00",
    "training_samples": 207,
    "model_files_exist": {
      "model": true,
      "scaler": true,
      "metadata": true
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.is_trained | boolean | Apakah model sudah ditraining dan siap |
| data.training_date | datetime | Kapan model terakhir ditraining |
| data.training_samples | int | Jumlah sampel yang digunakan untuk training |
| data.model_files_exist | object | Status file model yang tersimpan |
| data.model_files_exist.model | boolean | File bobot jaringan saraf ada |
| data.model_files_exist.scaler | boolean | File feature scaler ada |
| data.model_files_exist.metadata | boolean | File metadata model ada |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Status berhasil diambil |

---

### 5. Training Model

Training model Jaringan Saraf Tiruan Backpropagation dengan dataset.

**Endpoint:**

```http
POST /model/train
```

**Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/model/train"
```

**Response:**

```json
{
  "success": true,
  "message": "Model berhasil ditraining menggunakan Backpropagation",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "training_completed": true,
    "samples_used": 207,
    "metrics": {
      "accuracy": 0.857,
      "precision_macro": 0.556,
      "recall_macro": 0.604,
      "f1_macro": 0.566,
      "cv_accuracy_mean": 0.845,
      "cv_accuracy_std": 0.032,
      "cv_n_splits": 3,
      "confusion_matrix": [[10, 0, 0], [5, 26, 1], [0, 0, 0]],
      "classification_report": {...}
    },
    "training_history": {
      "n_iterations": 156,
      "final_loss": 0.234,
      "best_loss": 0.198,
      "loss_curve": [0.8, 0.6, 0.4, ...]
    },
    "network_architecture": {
      "input_neurons": 10,
      "hidden_layers": [...],
      "output_neurons": 3,
      "total_parameters": 3411
    },
    "grade_distribution": {
      "A": 48,
      "B": 156,
      "C": 3
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.training_completed | boolean | Apakah training selesai dengan sukses |
| data.samples_used | int | Jumlah sampel training |
| data.metrics | object | Metrik evaluasi pada test set |
| data.training_history | object | Riwayat proses training |
| data.training_history.n_iterations | int | Jumlah iterasi sampai konvergen |
| data.training_history.loss_curve | array | Nilai loss per iterasi |
| data.network_architecture | object | Arsitektur jaringan final |
| data.grade_distribution | object | Jumlah sampel per grade dalam dataset |

**Proses Training:**

1. Load dataset dari `data/df_arabica_clean.csv`
2. Preprocessing dan normalisasi fitur menggunakan StandardScaler
3. Split ke 80% training / 20% test set (stratified)
4. Training MLPClassifier dengan backpropagation
5. Evaluasi pada test set
6. Simpan file model ke `trained_models/`

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Training berhasil |
| 404 | File dataset tidak ditemukan |
| 500 | Training gagal |

---

### 6. Retraining Model

Alias untuk endpoint train - melatih ulang model dengan data baru.

**Endpoint:**

```http
POST /model/retrain
```

**Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/model/retrain"
```

**Response:**

Sama dengan `/model/train`

**Kasus Penggunaan:**

- Retraining setelah update dataset
- Reset bobot model
- Refresh performa model

---

## Endpoint Data Referensi

Endpoint untuk mengakses informasi referensi.

### 1. Informasi Grade

Dapatkan informasi klasifikasi grade kualitas kopi berdasarkan standar SCA.

**Endpoint:**

```http
GET /reference/grades
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/reference/grades"
```

**Response:**

```json
{
  "success": true,
  "message": "Grade information retrieved",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "grades": [
      {
        "grade": "A",
        "min_score": 85,
        "max_score": 100,
        "label": "Specialty Grade",
        "description": "Kopi dengan kualitas premium, layak untuk specialty market",
        "color": "#22c55e"
      },
      {
        "grade": "B",
        "min_score": 80,
        "max_score": 84.99,
        "label": "Premium Grade",
        "description": "Kopi dengan kualitas baik, cocok untuk pasar umum premium",
        "color": "#eab308"
      },
      {
        "grade": "C",
        "min_score": 0,
        "max_score": 79.99,
        "label": "Below Premium Grade",
        "description": "Kopi di bawah standar premium, perlu peningkatan kualitas",
        "color": "#ef4444"
      }
    ],
    "scoring_system": "SCA (Specialty Coffee Association) Cupping Protocol",
    "total_possible_points": 100
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.grades | array | Array definisi grade |
| data.grades[].grade | string | Identifier grade (A, B, C) |
| data.grades[].min_score | float | Threshold skor minimum |
| data.grades[].max_score | float | Threshold skor maksimum |
| data.grades[].label | string | Label yang dapat dibaca manusia |
| data.grades[].description | string | Deskripsi grade |
| data.grades[].color | string | Kode warna hex untuk UI |
| data.scoring_system | string | Referensi standar penilaian |
| data.total_possible_points | int | Skor maksimum yang mungkin |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Informasi berhasil diambil |

---

### 2. Informasi Fitur

Dapatkan informasi detail tentang semua fitur input.

**Endpoint:**

```http
GET /reference/features
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/reference/features"
```

**Response:**

```json
{
  "success": true,
  "message": "Feature information retrieved",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "features": [
      {
        "name": "aroma",
        "display_name": "Aroma",
        "description": "Intensitas dan kualitas aroma kopi, dinilai saat dry dan wet",
        "min": 0,
        "max": 10,
        "step": 0.25,
        "category": "sensory",
        "weight": "high",
        "tips": "Nilai aroma dari fragrance (kering) dan aroma (basah)"
      },
      {
        "name": "flavor",
        "display_name": "Flavor",
        "description": "Rasa keseluruhan yang mencakup taste dan aroma retro-nasal",
        "min": 0,
        "max": 10,
        "step": 0.25,
        "category": "sensory",
        "weight": "high",
        "tips": "Kombinasi taste buds dan aroma yang naik ke hidung"
      },
      {
        "name": "aftertaste",
        "display_name": "Aftertaste",
        "description": "Rasa yang tertinggal setelah kopi ditelan atau dibuang",
        "min": 0,
        "max": 10,
        "step": 0.25,
        "category": "sensory",
        "weight": "medium",
        "tips": "Durasi dan kualitas rasa yang bertahan"
      },
      {
        "name": "acidity",
        "display_name": "Acidity",
        "description": "Tingkat keasaman yang memberikan brightness pada kopi",
        "min": 0,
        "max": 10,
        "step": 0.25,
        "category": "sensory",
        "weight": "high",
        "tips": "Acidity yang baik terasa bright, bukan sour"
      },
      {
        "name": "body",
        "display_name": "Body",
        "description": "Ketebalan dan tekstur kopi di mulut (mouthfeel)",
        "min": 0,
        "max": 10,
        "step": 0.25,
        "category": "sensory",
        "weight": "medium",
        "tips": "Sensasi fisik kopi di lidah dan langit-langit"
      },
      {
        "name": "balance",
        "display_name": "Balance",
        "description": "Keseimbangan antara flavor, aftertaste, acidity, dan body",
        "min": 0,
        "max": 10,
        "step": 0.25,
        "category": "sensory",
        "weight": "high",
        "tips": "Tidak ada atribut yang mendominasi secara negatif"
      },
      {
        "name": "uniformity",
        "display_name": "Uniformity",
        "description": "Konsistensi rasa antar cup dalam satu sampel",
        "min": 0,
        "max": 10,
        "step": 2,
        "category": "quality",
        "weight": "medium",
        "tips": "2 poin per cup yang konsisten (5 cups = 10 poin)"
      },
      {
        "name": "clean_cup",
        "display_name": "Clean Cup",
        "description": "Kebersihan rasa tanpa defect atau off-flavors",
        "min": 0,
        "max": 10,
        "step": 2,
        "category": "quality",
        "weight": "medium",
        "tips": "2 poin per cup yang bersih (5 cups = 10 poin)"
      },
      {
        "name": "sweetness",
        "display_name": "Sweetness",
        "description": "Tingkat kemanisan alami dari kopi",
        "min": 0,
        "max": 10,
        "step": 2,
        "category": "quality",
        "weight": "medium",
        "tips": "2 poin per cup dengan sweetness (5 cups = 10 poin)"
      },
      {
        "name": "moisture_percentage",
        "display_name": "Moisture",
        "description": "Persentase kelembaban biji kopi hijau",
        "min": 0,
        "max": 20,
        "step": 0.1,
        "category": "physical",
        "weight": "low",
        "tips": "Ideal: 9-12%. Terlalu rendah = stale, terlalu tinggi = risiko jamur"
      }
    ],
    "total_features": 10,
    "categories": {
      "sensory": "Atribut sensorik yang dinilai melalui cupping",
      "quality": "Atribut kualitas berdasarkan konsistensi",
      "physical": "Karakteristik fisik biji kopi"
    }
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.features | array | Array definisi fitur |
| data.features[].name | string | Nama field API |
| data.features[].display_name | string | Nama yang dapat dibaca manusia |
| data.features[].description | string | Deskripsi detail |
| data.features[].min | float | Nilai minimum yang diizinkan |
| data.features[].max | float | Nilai maksimum yang diizinkan |
| data.features[].step | float | Increment yang disarankan |
| data.features[].category | string | Kategori fitur |
| data.features[].weight | string | Bobot kepentingan (high/medium/low) |
| data.features[].tips | string | Tips evaluasi |
| data.total_features | int | Jumlah total fitur |
| data.categories | object | Deskripsi kategori |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Informasi berhasil diambil |

---

### 3. Panduan Cupping

Dapatkan panduan protokol cupping SCA sebagai referensi.

**Endpoint:**

```http
GET /reference/cupping-guide
```

**Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/reference/cupping-guide"
```

**Response:**

```json
{
  "success": true,
  "message": "Cupping guide retrieved",
  "timestamp": "2025-12-20T10:30:00.000000",
  "data": {
    "protocol": "SCA Cupping Protocol",
    "steps": [
      {
        "step": 1,
        "name": "Fragrance/Aroma",
        "description": "Evaluasi aroma kopi kering dan setelah ditambah air panas"
      },
      {
        "step": 2,
        "name": "Breaking the Crust",
        "description": "Pecahkan kerak dan evaluasi aroma yang keluar"
      },
      {
        "step": 3,
        "name": "Tasting",
        "description": "Slurp kopi untuk mengevaluasi flavor, aftertaste, acidity, body, balance"
      },
      {
        "step": 4,
        "name": "Scoring",
        "description": "Berikan skor untuk setiap atribut (6-10 untuk specialty)"
      }
    ],
    "scoring_scale": {
      "6.00-6.75": "Good",
      "7.00-7.75": "Very Good",
      "8.00-8.75": "Excellent",
      "9.00-9.75": "Outstanding"
    },
    "specialty_threshold": 80,
    "reference_url": "https://sca.coffee/research/protocols-best-practices"
  }
}
```

**Field Response:**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| data.protocol | string | Nama protokol |
| data.steps | array | Langkah-langkah prosedur cupping |
| data.steps[].step | int | Nomor langkah |
| data.steps[].name | string | Nama langkah |
| data.steps[].description | string | Deskripsi langkah |
| data.scoring_scale | object | Interpretasi rentang skor |
| data.specialty_threshold | int | Skor minimum untuk grade specialty |
| data.reference_url | string | URL referensi resmi SCA |

**Kode Status:**

| Kode | Deskripsi |
|------|-----------|
| 200 | Panduan berhasil diambil |

---

## Penanganan Error

### Kode Status HTTP

| Kode | Nama | Deskripsi |
|------|------|-----------|
| 200 | OK | Request berhasil |
| 422 | Unprocessable Entity | Error validasi - input tidak valid |
| 500 | Internal Server Error | Error sisi server |
| 503 | Service Unavailable | Model belum siap |

### Error Validasi (422)

Dikembalikan ketika request body gagal validasi.

**Contoh - Nilai di luar rentang:**

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "aroma"],
      "msg": "Input should be less than or equal to 10",
      "input": 15.0,
      "ctx": {"le": 10}
    }
  ]
}
```

**Contoh - Field wajib tidak ada:**

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "sweetness"],
      "msg": "Field required",
      "input": {...}
    }
  ]
}
```

**Contoh - Tipe tidak valid:**

```json
{
  "detail": [
    {
      "type": "float_parsing",
      "loc": ["body", "aroma"],
      "msg": "Input should be a valid number",
      "input": "invalid"
    }
  ]
}
```

### Model Belum Siap (503)

Dikembalikan ketika mencoba prediksi tanpa model yang sudah ditraining.

```json
{
  "detail": "Model belum ditraining. Silakan training model terlebih dahulu."
}
```

### Internal Server Error (500)

Dikembalikan untuk error server yang tidak terduga.

```json
{
  "detail": "Internal server error occurred"
}
```

---

## Rate Limiting

Saat ini, tidak ada rate limit yang diimplementasikan. Untuk deployment production, pertimbangkan untuk mengimplementasikan rate limiting sesuai kebutuhan Anda.

---

## Changelog

### Versi 1.0.0

- Rilis awal
- Model Jaringan Saraf Tiruan Backpropagation
- 15 endpoint API
- Grading kualitas berbasis SCA

---

## Dukungan

Untuk masalah dan permintaan fitur, silakan buat issue di repository GitHub.
