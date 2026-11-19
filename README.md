# ☕ Coffee Quality AI

**Pengembangan Aplikasi Prediksi Kualitas Kopi Berbasis Machine Learning Menggunakan Framework Django**
Mini Proyek Kelompok – Modul 14, Mata Kuliah Sistem Cerdas.

Aplikasi web yang mengestimasi skor kualitas kopi (*Total Cup Points*) dari asal, varietas, metode pengolahan, dan kondisi fisik biji, memakai model Machine Learning (scikit-learn) yang disajikan lewat Django.

> ⚠️ **Catatan framework:** Modul 14 menggunakan Streamlit. Proyek ini memakai Django sebagai alternatif. **Konfirmasikan ke dosen** apabila Streamlit diwajibkan.
> Perbedaan: Streamlit = satu skrip Python, UI dibuat otomatis, cepat untuk prototipe. Django = arsitektur MVT (URL → view → template), form tervalidasi, ORM/database, lebih cocok untuk aplikasi multi-halaman. Tujuan deployment sama: model dapat dipakai lewat antarmuka web.

## 1. Latar belakang
Skor cupping kopi dihasilkan panelis terlatih (Q-grader) dan memerlukan biaya serta waktu. Proyek ini mengeksplorasi seberapa jauh skor dapat **diperkirakan dari informasi non-rasa** (asal, pengolahan, kondisi fisik) sebagai alat bantu awal, bukan pengganti penilaian resmi.

## 2. Tujuan
1. Menganalisis dataset kualitas kopi.
2. Membangun dan membandingkan model regresi untuk memprediksi *Total Cup Points*.
3. Men-deploy model lewat aplikasi web Django yang dapat dijalankan lokal.

## 3. Dataset
- File: `dataset/kualitas_kopi.csv` – 207 baris × 41 kolom, tanpa baris duplikat.
- **Sumber dan lisensi: `[ISI — tulis asal unduhan dataset dan lisensinya]`.** File CSV tidak memuat informasi ini, jadi mohon dilengkapi anggota kelompok sebelum dikumpulkan.

### Temuan penting: data leakage
`Total Cup Points` adalah **jumlah dari 10 skor sensorik** (Aroma, Flavor, Aftertaste, Acidity, Body, Balance, Uniformity, Clean Cup, Sweetness, Overall); selisih maksimum hanya 0,02 (pembulatan). Memakainya sebagai fitur menghasilkan R² ≈ 0,97–0,99 yang menyesatkan karena model hanya menghafal rumus penjumlahan. Maka kesepuluh kolom itu **tidak dipakai**.

### Fitur dan target
| Fitur | Jenis | Pembersihan |
|---|---|---|
| Country of Origin | Kategorikal | – |
| Variety | Kategorikal | kosong/unknown → `unknown`; campuran → `Blend` |
| Processing Method | Kategorikal | 10 nilai dirapikan jadi 4 grup + `unknown` |
| Color | Kategorikal | 12 ejaan dirapikan jadi 5 grup |
| Altitude | Numerik | rentang → rata-rata; nilai > 3000 diasumsikan **feet** → meter |
| Moisture Percentage | Numerik | 0 % dianggap missing |
| Category One/Two Defects, Quakers | Numerik | – |

**Target:** `Total Cup Points` (regresi). Kolom ID, teks bebas (nama kebun, perusahaan, kontak), konstan (Status, Clean Cup, Sweetness, Defects), serta tanggal juga dibuang.

## 4. Algoritma
Linear Regression, Decision Tree Regressor (max_depth=4), dan Random Forest Regressor (300 pohon). Semuanya dibungkus `Pipeline` (imputasi → scaling/one-hot → model). Data dibagi 80/20 (`random_state=42`); pemilihan model memakai 5-fold CV **hanya pada data latih**; model terbaik: **Random Forest**. Model akhir di-fit ulang pada seluruh data lalu disimpan dengan Joblib.

## 5. Hasil evaluasi (dari `python ml/train_model.py`)
| Model | CV MAE | CV RMSE | CV R² | Uji MAE | Uji MSE | Uji RMSE | Uji R² |
|---|---|---|---|---|---|---|---|
| Linear Regression | 1.212 | 1.566 | 0.172 | 1.360 | 3.569 | 1.889 | -0.430 |
| Decision Tree | 1.258 | 1.523 | 0.223 | 1.288 | 2.534 | 1.592 | -0.015 |
| Random Forest | 1.172 | 1.456 | 0.275 | 1.124 | 1.838 | 1.356 | 0.264 |
| Baseline (prediksi rata-rata) | – | – | – | 1.218 | 2.588 | 1.609 | -0.037 |

Feature importance (Random Forest): Variety 26.4%, Altitude 24.2%, Country of Origin 13.8%, Moisture Percentage 13.7%, Category Two Defects 8.0%, Quakers 7.2%, Processing Method 3.6%, Color 2.9%, Category One Defects 0.2%.

**Interpretasi jujur:** hanya Random Forest yang mengalahkan baseline pada data uji (RMSE 1.36 vs 1.61), tetapi R² uji hanya 0.26. Artinya informasi asal dan kondisi fisik menjelaskan sebagian kecil variasi skor; rasa sebenarnya ditentukan cupper. Data uji hanya 42 sampel, sehingga angka ini dapat berubah bila pembagian data berbeda. Jangan memakai hasil aplikasi sebagai kualitas objektif atau sertifikasi.

## 6. Teknologi
Python, Django, scikit-learn, pandas, NumPy, Matplotlib, Joblib, HTML/CSS, Bootstrap 5 (via CDN, butuh internet saat demo), SQLite (riwayat).

## 7. Struktur folder
```
coffee_quality_ai/
├── manage.py
├── requirements.txt
├── README.md
├── config/                 # settings.py, urls.py, wsgi.py
├── coffee_app/             # views, urls, forms, models, services, admin, tests, templates/
├── ml/
│   ├── preprocessing.py    # cleaning + pipeline (dipakai training & Django)
│   ├── train_model.py      # training, evaluasi, simpan model & grafik
│   └── coffee_quality_model.joblib
├── dataset/kualitas_kopi.csv
└── static/                 # css/, js/, images/ (grafik hasil training)
```

## 8. Instalasi
```bash
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Konfigurasi Gemini AI
Salin `.env.example` menjadi `.env`, lalu isi `GEMINI_API_KEY` dengan API key dari Google AI Studio. File `.env` tidak dimasukkan ke Git. Model utama dan cadangan dapat diubah lewat `GEMINI_MODEL` (default: `gemini-3.6-flash`) dan `GEMINI_FALLBACK_MODEL` (default: `gemini-3.5-flash-lite`). Jika model utama sementara sibuk (HTTP 429/503), aplikasi mencoba ulang sekali lalu beralih ke model cadangan. Fitur analisis AI memerlukan koneksi internet dan API key yang valid.

## 9. Menjalankan training (opsional, model sudah disertakan)
```bash
python ml/train_model.py
```
Menghasilkan ulang `ml/coffee_quality_model.joblib` dan grafik di `static/images/`.

## 10. Menjalankan aplikasi Django
```bash
python manage.py migrate
python manage.py runserver
```
Buka http://127.0.0.1:8000/. Model dimuat dari file `.joblib` (tidak ada training ulang). Uji alur: `python manage.py test` (8 tes).

## 11. Pembagian tugas
| Anggota | NIM | Tugas |
|---|---|---|
| `[Nama 1]` | `[NIM]` | Analisis dataset, EDA, preprocessing (`ml/preprocessing.py`) |
| `[Nama 2]` | `[NIM]` | Training dan evaluasi model (`ml/train_model.py`) |
| `[Nama 3]` | `[NIM]` | Backend Django (`views.py`, `forms.py`, `services.py`, `models.py`) |
| `[Nama 4]` | `[NIM]` | Frontend/UI (templates, CSS) dan pengujian |
| `[Semua]` | – | Dokumentasi dan persiapan demo |

## 12. Panduan demo (±5 menit)
1. **Dashboard:** tunjukkan statistik dataset dan informasi model.
2. **Jelaskan leakage:** buka *Visualisasi & Info*, coba tombol analisis Gemini pada grafik, lalu jelaskan korelasi sensorik dan alasan skor sensorik dibuang.
3. **Prediksi:** isi form, klik *Prediksi*, tunjukkan skor, rentang ± RMSE, dan peringatan bahwa ini estimasi.
4. **Kasus batas:** masukkan ketinggian 99999 (ditolak), atau cacat kategori 2 = 90 (peringatan di luar rentang data latih).
5. **Riwayat:** tunjukkan data tersimpan di SQLite.
6. **Kode:** jelaskan alur `forms.py` → `services.predict_score()` → `preprocessing.clean_features()` → pipeline → `views.py`.

Pertanyaan yang mungkin muncul: *mengapa R² rendah?* (leakage dihindari, fitur non-rasa terbatas); *mengapa Django?* (lihat catatan di atas); *mengapa Random Forest?* (RMSE CV terendah).
