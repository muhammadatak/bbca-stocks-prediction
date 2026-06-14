# Continuous Training Trigger Thresholds

## Overview

Sistem Continuous Training (CT) dirancang untuk melakukan pelatihan ulang (*retraining*) model secara otomatis ketika terdeteksi penurunan performa model atau perubahan karakteristik data yang signifikan. Tujuan utama dari mekanisme ini adalah mencegah terjadinya *model decay* akibat perubahan kondisi data pada lingkungan produksi.

Retraining dapat dipicu oleh tiga jenis kondisi, yaitu berdasarkan performa model, data drift, dan jadwal berkala.

---

## 1. Performance-Based Trigger

Sistem memantau metrik performa model menggunakan Prometheus dan Grafana. Retraining akan dijalankan secara otomatis apabila nilai akurasi model turun di bawah ambang batas yang telah ditentukan.

### Threshold

| Metric   | Threshold | Action             |
| -------- | --------- | ------------------ |
| Accuracy | < 0.80    | Trigger Retraining |

### Rationale

Nilai akurasi di bawah 80% menunjukkan bahwa model mulai kehilangan kemampuannya dalam melakukan prediksi secara optimal terhadap data terbaru. Kondisi ini dapat mengindikasikan adanya perubahan pola data yang tidak lagi sesuai dengan data pelatihan sebelumnya.

---

## 2. Data Drift Trigger

Sistem melakukan pemantauan terhadap distribusi data menggunakan metode Population Stability Index (PSI). Metrik ini digunakan untuk mengukur tingkat perubahan distribusi antara data referensi dan data terbaru.

### Threshold

| PSI Value         | Interpretation       | Action             |
| ----------------- | -------------------- | ------------------ |
| PSI < 0.10        | No significant drift | No action          |
| 0.10 ≤ PSI < 0.25 | Moderate drift       | Monitoring         |
| PSI ≥ 0.25        | Significant drift    | Trigger Retraining |

### Rationale

Nilai PSI di atas 0.25 menunjukkan adanya perubahan distribusi data yang signifikan sehingga model berpotensi mengalami penurunan performa apabila tidak diperbarui.

---

## 3. Schedule-Based Trigger

Selain pemicu berbasis performa dan data drift, sistem juga menjalankan retraining secara berkala melalui GitHub Actions.

### Schedule

* Setiap hari Minggu (Weekly Retraining)
* Retraining hanya dijalankan apabila terdapat versi dataset baru pada DVC Repository.

### Rationale

Pendekatan ini memastikan model tetap diperbarui secara rutin tanpa harus menunggu terjadinya penurunan performa yang signifikan.

---

## Retraining Workflow

1. Monitoring system mendeteksi penurunan performa atau data drift.
2. Grafana/Prometheus mengirimkan alert melalui webhook.
3. GitHub Actions menerima trigger.
4. Dataset terbaru diambil dari DVC.
5. Pipeline `train.py` dijalankan.
6. Model baru dievaluasi dan dicatat ke MLflow.
7. Model baru dibandingkan dengan model Production saat ini.
8. Model baru dipromosikan ke Production hanya jika memiliki performa yang lebih baik.

---

## Model Promotion Rule

Model hasil retraining akan dipromosikan ke status **Production** apabila memenuhi kondisi berikut:

```text
New Accuracy > Current Production Accuracy

AND

New F1-Score >= Current Production F1-Score
```

Jika syarat tersebut tidak terpenuhi, maka model lama tetap digunakan sebagai model Production.

---

## Summary

| Trigger Type      | Metric          | Threshold    | Action     |
| ----------------- | --------------- | ------------ | ---------- |
| Performance-Based | Accuracy        | < 0.80       | Retraining |
| Data Drift        | PSI             | ≥ 0.25       | Retraining |
| Schedule-Based    | New DVC Dataset | Weekly Check | Retraining |

Dengan konfigurasi ini, sistem Continuous Training mampu menjaga kualitas model secara otomatis dan meminimalkan risiko penurunan performa akibat perubahan data pada lingkungan produksi.
