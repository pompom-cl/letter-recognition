# Klasifikasi Huruf
Proposal Machine Learning

## Bab 1: Pendahuluan

### 1.1 Latar Belakang
Seiring berkembangnya zaman, data yang disimpan secara digital jauh lebih mudah diolah daripada data tulisan tangan secara fisik. Banyak lembaga melakukan digitalisasi data karena hal ini. Untuk mempercepat proses pemindahan data menjadi data digital, diperlukan sistem yang dapat mengenali teks dari gambar. Teknologi ini sangat berguna agar proses digitalisasi tidak perlu dilakukan secara manual.

Ada berbagai macam sistem yang bisa mendeteksi teks dari gambar, salah satunya dengan model Machine Learning. Ada banyak algoritma Machine Learning yang bisa mengklasifikasi huruf. Dalam proposal ini, hanya tiga model yang akan dibandingkan untuk membaca teks dari gambar.

Proposal ini akan menjelaskan proses dari tiga model Machine Learning untuk mendeteksi pola dari sebuah dataset yang dihasilkan oleh berbagai karakter dan menampilkan akurasi dari hasil prediksi dari data yang diujikan.

### 1.2 Rumusan Masalah
- Bagaimana cara kerja sistem yang dapat mengklasifikasikan huruf dari gambar?
- Algoritma machine learning apa yang paling akurat untuk mengklasifikasi huruf tulisan tangan?
- Apa kelebihan dan kekurangan dari sistem dan algoritma yang digunakan?

### 1.3 Tujuan
- Mengetahui langkah-langkah sistem yang dapat mengklasifikasikan huruf dari gambar.
- Mengetahui algoritma machine learning yang paling akurat untuk mengklasifikasi huruf tulisan tangan.
- Mengetahui kelebihan dan kekurangan dari sistem dan algoritma yang digunakan.

## Bab 2: Metode Penelitian

### 2.1 Metode Pengumpulan Data
Data training adalah dataset yang sudah tersedia dan digunakan untuk melatih model. Data testing adalah data yang digunakan untuk menguji model. Dalam proyek ini, sumber data adalah data sekunder dari UCI Machine Learning: Letter Recognition. Dataset berisi 20.000 data unik dengan 16 atribut numerik, hasil distorsi acak terhadap 20 jenis huruf kapital.

### 2.2 Informasi Dataset
URL: https://archive.ics.uci.edu/dataset/59/letter+recognition

Dataset memiliki 17 fitur:
- lettr: huruf kapital (target klasifikasi)
- x-box, y-box, width, high, onpix, x-bar, y-bar, x2bar, y2bar, xybar, x2ybr, xy2br, x-ege, xegvy, y-ege, yegvx: fitur numerik (0-15)

### 2.3 Model Machine Learning Classification
Dataset dilatih dan diuji dengan 3 model klasifikasi: Random Forest, K-Nearest Neighbors (KNN), dan XGBoost. Metode train-test-split digunakan untuk membagi data: 80% training, 20% testing.

```python
from sklearn.model_selection import train_test_split

data["lettr"] = data["lettr"].astype("category").cat.codes
X = data.drop(columns=["lettr"])
y = data["lettr"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2.3.1 Random Forest Classifier
Random Forest adalah algoritma ensemble learning dengan kombinasi banyak decision tree. Menggunakan teknik bootstrap aggregating (bagging) dan voting majority untuk prediksi.

Akurasi: 96.15%
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
```

### 2.3.2 K-Nearest Neighbors Classifier
KNN mengklasifikasikan data berdasarkan tetangga terdekat. Tidak melakukan training eksplisit, tetapi menyimpan data dan menggunakan metrik jarak (Euclidean) saat prediksi.

Akurasi: 95.2%
```python
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
```

### 2.3.3 XGBoost Classifier
XGBoost membangun decision tree secara bertahap untuk meminimalkan error. Menggunakan regularisasi untuk menghindari overfitting dan meningkatkan performa.

Akurasi: 95.1%
```python
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=26,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
```

## Bab 3: Kesimpulan

### 3.1 Hasil Perbandingan 3 Model Machine Learning
| Model | Akurasi |
|-------|---------|
| Random Forest | 96.15% |
| KNN | 95.2% |
| XGBoost | 95.1% |

Random Forest memberikan hasil akurasi tertinggi di antara tiga model. KNN dan XGBoost masih menunjukkan performa tinggi, namun kalah dari Random Forest dalam hal akurasi.

### 3.2 Pekerjaan di Masa Depan
- Menggunakan fitur dari gambar aktual huruf (bukan hanya data numerik)
- Menambah variasi dataset agar lebih representatif terhadap tulisan tangan nyata
- Mencoba model deep learning seperti CNN untuk pengenalan huruf dari gambar

## Referensi
- https://archive.ics.uci.edu/dataset/59/letter+recognition
- https://scikit-learn.org/stable/
- https://xgboost.readthedocs.io
- https://geeksforgeeks.org

