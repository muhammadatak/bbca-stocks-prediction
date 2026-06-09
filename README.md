# 📈 EQ Prediction — Stock Data Pipeline (BBCA.JK)

Pipeline sederhana untuk mengunduh, membersihkan, dan memproses data saham BBCA.JK menggunakan Yahoo Finance.

---

## 📁 Struktur Proyek

```
eq-prediction/
├── data/
│   ├── raw/               # Data mentah hasil download (CSV per timestamp)
│   └── processed/         # Data hasil preprocessing
├── src/
│   ├── ingest_data.py     # Script download data dari Yahoo Finance
│   ├── preprocess.py      # Fungsi pembersihan & transformasi data
│   └── train.py           # Training Auto-ARIMA + logging MLflow
├── pyproject.toml
└── README.md
```

---

## ⚙️ Prasyarat

- Python `3.10+`
- [uv](https://github.com/astral-sh/uv) (package manager) — sudah dikonfigurasi via `pyproject.toml`

Install dependensi:

```bash
uv sync
```

Atau jika menggunakan `pip`:

```bash
pip install yfinance pandas scikit-learn pmdarima mlflow
```

---

## 🚀 Cara Menjalankan

### 1. Ingest Data (Download dari Yahoo Finance)

Script ini mengunduh data harga saham BBCA.JK untuk hari ini dan menyimpannya ke `data/raw/` dengan nama file berdasarkan timestamp.

```bash
cd src
python ingest_data.py
```

Output:

```
Saved: data/raw/20240601_083045.csv
```

> File CSV akan tersimpan otomatis di folder `data/raw/`.

---

### 2. Preprocessing Data Training

Sebelum menjalankan `main.py`, pastikan kamu sudah menyiapkan file data mentah training dan menaruhnya di:

```
data/raw/raw_training.csv
```

Kemudian jalankan:

```bash
cd src
python main.py
```

Output file hasil preprocessing akan tersimpan di:

```
data/processed/processed_training.csv
```

---

### 3. Training Model + Logging Eksperimen (MLflow)

Script `src/train.py` melatih model Auto-ARIMA menggunakan dataset preprocess:

```
data/processed/processed_training.csv
```

Contoh 1 run:

```bash
uv run python src/train.py --experiment-name eq-prediction-arima --run-name baseline
```

Contoh minimal 3 run dengan parameter berbeda:

```bash
uv run python src/train.py --experiment-name eq-prediction-arima --run-name run-1 --max-p 2 --max-q 2 --d 1 --D 1 --seasonal true
uv run python src/train.py --experiment-name eq-prediction-arima --run-name run-2 --max-p 3 --max-q 3 --d 1 --D 0 --seasonal false
uv run python src/train.py --experiment-name eq-prediction-arima --run-name run-3 --max-p 4 --max-q 1 --d 2 --D 1 --seasonal true
```

Setiap run akan mencatat:
- **Parameter** via MLflow (`mlflow.log_param`)
- **Metrik** regression (RMSE, MAE) via MLflow (`mlflow.log_metric`)
- **Model artifact** Auto-ARIMA via MLflow (`mlflow.pmdarima.log_model`)

Menjalankan MLflow UI:

```bash
uv run mlflow ui --backend-store-uri mlruns --port 5000
```

Lalu buka:

```
http://127.0.0.1:5000
```

Di UI, bandingkan run pada experiment `eq-prediction-arima`, pilih RMSE terendah, lalu lakukan **Register Model** secara manual ke Model Registry.

---

## 🚢 Serve Model dari MLflow Registry (FastAPI)

`app/main.py` sudah menggunakan URI registry, bukan hardcoded `RUN_ID`.

Urutan operasional:
1. Train model (run tercatat di MLflow).
2. Di MLflow UI, register model ke nama registry (contoh: `bbca-xgboost-predictor`).
3. Di MLflow versi terbaru, tetapkan alias model (contoh: `champion`) ke versi yang ingin dipakai.
4. FastAPI memuat model lewat URI `models:/bbca-xgboost-predictor@champion`.

Konfigurasi environment di API:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=bbca-xgboost-predictor
MODEL_ALIAS=champion
```

Jika `MODEL_ALIAS` tidak ditemukan, `app/main.py` akan fallback ke `MODEL_STAGE` (legacy) lalu ke versi model terbaru.

Contoh URI alias:

```bash
models:/<MODEL_NAME>@<MODEL_ALIAS>
```

Request `/predict` sekarang harus mengirim histori OHLCV (minimal **31 bar**) agar preprocessing fitur (`EMA/SMA/RSI/MACD`) konsisten dengan training. Contoh format (dipersingkat):

```json
{
  "history": [
    {
      "Date": "2026-05-01",
      "Open": 9200.0,
      "High": 9300.0,
      "Low": 9150.0,
      "Close": 9280.0,
      "Volume": 1200000
    }
  ]
}
```

---

## 🔄 Alur Pipeline

```
Yahoo Finance
     │
     ▼
ingest_data.py  ──►  data/raw/{timestamp}.csv
                               │
                               ▼  (rename/copy ke raw_training.csv)
                         main.py
                               │
                               ▼
                  data/processed/processed_training.csv
```

---

## 🧹 Apa yang Dilakukan Preprocessing?

Fungsi `preprocess_data()` di `preprocess.py` melakukan langkah-langkah berikut:

| Langkah         | Keterangan                                                       |
| --------------- | ---------------------------------------------------------------- |
| Load CSV        | Membaca file dengan `skiprows=2` dan menamai kolom secara manual |
| Drop NaN        | Menghapus baris dengan nilai kosong                              |
| Parse tanggal   | Mengkonversi kolom `Date` ke tipe `datetime`                     |
| Cast tipe data  | Memastikan semua kolom harga & volume bertipe `float`            |
| Sort by date    | Mengurutkan data dari tanggal terlama ke terbaru                 |
| Drop duplicates | Menghapus baris dengan tanggal yang sama                         |

---

## 📝 Catatan

- Script `ingest_data.py` menggunakan `interval='1d'` dan `period='1d'`, artinya hanya mengunduh data **hari ini saja**. Ubah parameter `period` jika ingin data historis (contoh: `period='1y'` untuk 1 tahun).
- Folder `data/raw/` dan `data/processed/` akan dibuat otomatis jika belum ada.

---

## 🧬 Integrasi DVC (Data Version Control)

Proyek ini sudah diintegrasikan dengan DVC untuk melacak dataset tanpa menyimpan file data besar langsung di Git.

### 1) Inisialisasi DVC

```bash
uv add dvc
uv run dvc init
git add .dvc .dvcignore pyproject.toml uv.lock
git commit -m "Initialize DVC"
```

### 2) Tracking dataset awal

Contoh dataset utama: `data/raw/raw_training.csv`.

```bash
git rm --cached data/raw/raw_training.csv
uv run dvc add data/raw/raw_training.csv
git add data/raw/raw_training.csv.dvc data/raw/.gitignore
git commit -m "Track raw_training.csv with DVC"
```

### 3) Simulasi continual learning (data baru)

Jalankan ingest untuk mengambil data terbaru, lalu tambahkan baris baru ke `raw_training.csv`.

```bash
uv run python src/ingest_data.py
# append baris terbaru ke raw_training.csv (tanpa duplikasi tanggal)
uv run dvc add data/raw/raw_training.csv
git add data/raw/raw_training.csv.dvc
git commit -m "Update dataset version"
```

### 4) Audit perbedaan versi data

Bandingkan metadata versi lama vs baru:

```bash
uv run dvc diff
git --no-pager diff data/raw/raw_training.csv.dvc
```

Perubahan hash/size di file `raw_training.csv.dvc` menunjukkan dataset berubah, sementara file CSV besar tetap disimpan di cache DVC, bukan history Git.

---

## 🤖 Otomasi MLOps (Code as a Trigger)

Workflow GitHub Actions sudah disiapkan di:

```
.github/workflows/mlops-automation.yaml
```

Trigger otomatis:
- `push` ke branch `main`
- `pull_request` menuju branch `main`

Rangkaian job:
1. **Stage 1 — Automated Testing**: menjalankan `pytest`.
2. **Stage 2 — Automated Training**: `dvc pull` data terbaru, preprocessing, lalu retraining via `src/train.py`.
3. **Stage 3 — Evaluation & Validation**: validasi performa model memakai threshold di `model/metadata/validation_threshold.json`.
4. **Stage 4 — Auto Registry Update**: jika validasi lolos, model otomatis didaftarkan ke MLflow Model Registry pada stage **Staging**.

### Secret yang dibutuhkan di GitHub repository

- `DVC_REMOTE_URL` → URL remote storage DVC yang bisa diakses runner.
- `MLFLOW_TRACKING_URI` → endpoint MLflow Tracking/Registry server.
- `MLFLOW_TRACKING_USERNAME` (opsional, jika pakai basic auth)
- `MLFLOW_TRACKING_PASSWORD` (opsional, jika pakai basic auth)
- `MLFLOW_TRACKING_TOKEN` (opsional, jika pakai token auth)

### Simulasi perubahan (untuk pembuktian end-to-end)
\