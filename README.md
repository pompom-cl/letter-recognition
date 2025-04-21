# Klasifikasi Huruf
Proposal proyek untuk kelas CDS2122 - Pembelajaran Mesin

## Bab 1: Pendahuluan

### 1.1 Latar Belakang
Seiring berkembangnya zaman, data yang disimpan secara digital jauh lebih mudah diolah daripada data tulisan tangan secara fisik. Banyak lembaga melakukan digitalisasi data karena hal ini. Untuk mempercepat proses pemindahan data menjadi data digital, diperlukan sistem yang dapat mengenali teks dari gambar. Teknologi ini sangat berguna agar proses digitalisasi tidak perlu dilakukan secara manual. 
Ada berbagai macam sistem yang bisa mendeteksi teks dari gambar, salah satunya dengan model Machine Learning. Ada banyak algoritma Machine Learning yang bisa mengklasifikasi huruf. Dalam proposal ini, hanya tiga model yang akan dibandingkan untuk membaca teks dari gambar.
Proposal ini akan menjelaskan proses dari empat model Machine Learning untuk mendeteksi pola dari sebuah dataset yang dihasilkan oleh berbagai karakter dan menampilkan akurasi dari hasil prediksi dari data yang diujikan. 


### 1.2 Rumusan Masalah
1. Bagaimana cara kerja sistem yang dapat mengklasifikasikan huruf dari gambar?
2. Algoritma machine learning apa yang paling akurat untuk mengklasifikasi huruf tulisan tangan?
3. Apa kelebihan dan kekurangan dari sistem dan algoritma yang digunakan?

### 1.3 Tujuan
1. Mengetahui langkah-langkah sistem yang dapat mengklasifikasikan huruf dari gambar.
2. Mengetahui algoritma machine learning yang paling akurat untuk mengklasifikasi huruf tulisan tangan.
3. Mengetahui kelebihan dan kekurangan dari sistem dan algoritma yang digunakan.

## Bab 2: Metode Penelitian

### 2.1 Metode Pengumpulan Data
Data training adalah dataset yang sudah tersedia dan digunakan untuk melatih model. Data testing adalah data yang digunakan untuk menguji model. Dalam proyek ini, sumber data adalah data sekunder (raw data yang masih perlu diproses). Data diambil dari Kaggle: A-Z Handwritten Alphabets dalam format csv. Dalam data file, terdapat lebih dari 370.000 data gambar berukuran 28x28 piksel, yang artinya terdapat 784 fitur.
Sumber data tersebut dihasilkan melalui gambar dari NIST (https://www.nist.gov/srd/nist-special-database-19), MNIST, serta sumber lainnya yang kemudian dipusatkan pada kotak berukuran 20x20 piksel sehingga menghasilkan sebuah file yang terdiri dari 370.000 data yang unik. Setiap data berisi 785 atribut, atribut pertama merupakan label klasifikasi huruf dan 784 atribut berikutnya merupakan indikator grey level dari piksel gambar (0-255)


### 2.2 Informasi Dataset
Sumber Dataset: [link](https://archive.ics.uci.edu/dataset/59/letter+recognition)
Dataset yang digunakan masih perlu diproses agar tahap training, testing, dan prediksi lebih efisien. Dataset yang digunakan masih perlu diproses agar tahap training, testing, dan prediksi lebih efisien. Dari 370.00 raw data, diambil 29.169 data yang diharapkan dapat merepresentasi 26 huruf. Proyek UCI Machine Learning: Letter Recognition ([link](https://archive.ics.uci.edu/dataset/59/letter+recognition)) juga melakukan klasifikasi huruf, proyek ini menggunakan dataset yang terdiri dari 17 fitur yang diekstrak dari gambar huruf. Data memiliki 17 fitur dengan tipe data pada setiap fitur adalah integer, kecuali fitur lettr, yaitu kategorikal.

- lettr: huruf kapital (target klasifikasi)
- x-box, y-box, width, high, onpix, x-bar, y-bar, x2bar, y2bar, xybar, x2ybr, xy2br, x-ege, xegvy, y-ege, yegvx: fitur numerik (0-15)

### 2.3 Model Machine Learning Classification
Dataset akan diproses menjadi dataset baru yang terdiri dari 17 fitur karakteristik huruf, kemudian dilatih dan diuji menggunakan 4 model classification, antara lain Random Forest Classifier, K-Nearest Neighbors Classifier, Multilayer Perceptron, dan XGBoost Classifier. Sebelum itu, metode train-test-split digunakan untuk membagi data training dan data testing. Proyek ini memisahkan 20% data untuk diuji dan 80% data untuk dilatih. Berikut implementasinya dalam bahasa python:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2.3.1 Random Forest Classifier

Random Forest Classifier merupakan salah satu algoritma machine learning yang menghasilkan model dari berbagai pohon keputusan. Kumpulan pohon keputusan akan dilatih secara independen dengan teknik bootstrap aggregating (bagging). Bootstrap adalah teknik statistik yang memungkinkan estimasi distribusi sampel dengan melakukan resampling (pengambilan sampel berulang kali).
Untuk klasifikasi dan prediksi, Random Forest dapat menghitung tingkat kepentingan (feature importance), salah satunya adalah metode Mean Decrease in Accuracy. Metode Mean Decrease in Accuracy akan menghitung akurasi pada data OOB. Terdapat sekitar 36.8% dari data asli yang tidak termasuk dalam bootstrap sample untuk setiap pohon tertentu yang disebut sebagai Out-of-Bag (OOB) samples. Jika akurasi turun secara signifikan, maka fitur tersebut penting. Pengetahuan ini akan mempengaruhi bagaimana setiap pohon menentukan prediksi. Setelah setiap pohon memberikan hasil prediksinya, hasil prediksi yang sama dan terbanyak akan dipilih sebagai hasil prediksi final. Teknik ini disebut sebagai Voting Majority:


$$
\hat{y} = \arg\max_c \sum_{i=1}^{n} I(h_i(X) = c)
$$

Keterangan:
- $\hat{y}$ = hasil prediksi akhir  
- $c$ = label kelas hasil klasifikasi  
- $I(h_i(X) = c)$ = fungsi indikator (bernilai 1 jika pohon ke-$i$ memprediksi $c$, 0 jika tidak)  
- $n$ = jumlah pohon dalam Random Forest


Dalam implementasinya menggunakan bahasa python untuk menunjukkan proses pelatihan model Random Forest Classifier dengan pipeline, hyperparameter tuning (GridSearchCV), dan evaluasi model menggunakan Cross Validation (baik K-Fold biasa maupun StratifiedKFold), berikut akurasi dari model dengan berbagai metode evaluasi:

### 2.3.2 K-Nearest Neighbors Classifier

K-Nearest Neighbors adalah cara mengklasifikasi suatu data dengan melihat data di sekelilingnya. Algoritma tidak secara langsung belajar dari data training, tetapi menyimpan dataset dan baru mempelajarinya pada saat klasifikasi. Nilai k pada k-Nearest Neighbors adalah angka yang menunjukkan jumlah tetangga yang perlu dilihat waktu mengambil keputusan. 
Algoritma K-Nearest Neighbors yang digunakan pada proyek ini adalah dari sklearn.neighbors. Secara default, metode perhitungan jarak untuk mengidentifikasi tetangga terdekat adalah Euclidean Distance yang mengikut rumus berikut.
$$
\text{distance}(x, X_i) = \sqrt{ \sum_{j=1}^{d} (x_j - X_{ij})^2 }
$$

Keterangan:
- $x$ = vektor fitur dari data yang ingin diprediksi  
- $X_i$ = vektor fitur dari data latih ke-$i$  
- $d$ = jumlah fitur (dimensi)  
- $(x_j - X_{ij})^2$ = selisih kuadrat antara fitur ke-$j$ dari $x$ dan $X_i$


Dalam implementasinya menggunakan bahasa python untuk menunjukkan proses pelatihan model K-Nearest Neighbors (KNN) dengan pipeline, hyperparameter tuning (GridSearchCV), dan evaluasi model menggunakan Cross Validation (baik K-Fold biasa maupun StratifiedKFold), berikut akurasi dari model dengan berbagai metode evaluasi:

| **Metode Evaluasi**           | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|------------------------------|--------------|----------------|------------|--------------|
| K-Fold CV                    | 80.75%       | 81.07%         | 80.74%     | 80.67%       |
| Stratified K-Fold CV         | 80.46%       | 80.76%         | 80.46%     | 80.38%       |
| Stratified + Best Model      | 81.46%       | 81.66%         | 81.46%     | 81.36%       |

### 2.3.3 Multilayer Perceptron (MLP)

Multilayer Perceptron (MLP) adalah sebuah jaringan saraf artifisial yang terdiri dari beberapa lapisan neuron. MLP terdiri dari satu input layer, satu atau lebih hidden layer, dan satu output layer. Input layer terdiri dari neuron yang menerima data input awal, dengan tiap neuron mewakili fitur dimensi data input. Sehingga, jumlah neuron dalam lapisan input ditentukan oleh jumlah fitur data input. Setelah input layer, terdapat satu atau lebih lapisan neuron yang disebut dengan hidden layer, jumlah lapisan ini merupakan hyperparameter yang dapat diubah sesuai kebutuhan. Terakhir, output layer, merupakan lapisan yang terdiri dari neuron sebagai output akhir. Jumlah neuronnya bergantung pada jenis prediksi yang dibutuhkan.

Dalam implementasinya menggunakan bahasa python untuk menunjukkan proses pelatihan model Multilayer Perceptron Classifier dengan pipeline, hyperparameter tuning (GridSearchCV), dan evaluasi model menggunakan Cross Validation (baik K-Fold biasa maupun StratifiedKFold), berikut akurasi dari model dengan berbagai metode evaluasi:


### 2.3.4 XGBoost Classifier

XGBoost merupakan kombinasi beberapa algoritma untuk mendapatkan hasil yang optimal dan mengurangi loss function. Algoritma ini bekerja dengan membangun sekumpulan pohon keputusan (decision tree) secara berurutan. Setiap pohon baru mengoreksi kesalahan yang dibuat oleh pohon sebelumnya. Algoritma ini menggunakan teknik optimisasi lanjutan dan metode regularisasi yang mengurangi overfitting dan meningkatkan kinerja model. XGBoost membutuhkan beberapa parameter, antara lain

1. objective	: menentukan jenis tugas yang dilakukan
2. num_class	: jumlah kelas (parameter ini harus ada jika objective=”multi:softmax”)
3. n_estimator	: jumlah pohon dalam boosting
4. max_depth	: kedalaman maksimum setiap pohon
5. learning_rate	: seberapa besar langkah perubahan tiap iterasi
6. subsample	: persentase data untuk membangun setiap pohon
7. random_state	: untuk memastikan hasil yang sama pada setiap eksekusi

Dalam implementasinya menggunakan bahasa python untuk menunjukkan proses pelatihan model Ensemble Learning, yaitu XGBoost Classifier, dengan pipeline, hyperparameter tuning (GridSearchCV), dan evaluasi model menggunakan Cross Validation (baik K-Fold biasa maupun StratifiedKFold), berikut akurasi dari model dengan berbagai metode evaluasi:


## Bab 3: Kesimpulan

### 3.1 Hasil Perbandingan 3 Model Machine Learning

Dari ketiga model Machine Learning yang dilatih dan diuji menggunakan dataset dari UCI, dihasilkan bahwa model Random Forest Classification memiliki akurasi yang tertinggi, yaitu mencapai 96.15% diikuti oleh K-Nearest Neighbors Classifier yang mencapai tingkat akurasi 95.2% dan XGBoost Classifier yang mencapai tingkat akurasi 95.1%. Ketiga model ini memiliki keunggulan dalam aspek akurasi dan performa dan dapat mengatasi serta menghindari overfitting. Akan tetapi, masing-masing model memiliki kekurangan. Random Forest Classification memerlukan komputasi yang tinggi dan potensi bias terhadap kelas mayoritas. K-Nearest Neighbors Classification membutuhkan komputasi yang tinggi pada data dengan skala besar dan memori yang besar. Apabila nilai k yang dipilih tidak optimal, maka akan terjadi overfitting dan underfitting. XGBoost Classification juga memiliki sejumlah parameter yang perlu ditetapkan untuk mencapai kinerja yang optimal. Dengan demikian, terdapat berbagai aspek dari segi waktu, komputasi, memori, dan lain-lain yang perlu dipertimbangkan dalam memilih model yang tepat untuk melakukan prediksi bagi data-data baru yang akan diujikan di masa depan. 



### 3.2 Pekerjaan di Masa Depan

Proyek ini termasuk klasifikasi karena tugas utamanya mengelompokkan gambar huruf ke dalam kategori yang telah ditentukan dengan keterangan berikut.

1. Input	: gambar huruf
2. Output	: label huruf yang sesuai.

Input dengan tipe data gambar akan melalui algoritma preprocessing yang akan mengekstrak data gambar menjadi 16 fitur. Fitur-fitur ini sesuai fitur pada dataset tanpa fitur yang pertama. Data ini akan diuji oleh tiga model machine learning di atas untuk diklasifikasi. Output data berupa label yang berasal dari hasil klasifikasi setiap model.


## Referensi

Iskandar, D., Yulianto, E., & Rizal, A. (2022). Artificial Intelligence for Sentiment Analysis Using Machine Learning Algorithm. Journal of Informatics and Visualization, 8(1), 36-43. DOI: 10.62527/joiv.8.1.1707

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. DOI: 10.1023/A:1010933404324

Efron, B., & Tibshirani, R. (1993). Bootstrap Methods and Applications. IEEE Signal Processing Magazine, 24(4), 50–60. DOI: 10.1109/MSP.2007.4286560

JBMR Plus. (2023). Artificial Intelligence in Medical Imaging: A Review of Machine Learning Applications in Radiology. JBMR Plus, 7(8), e10757. DOI: 10.1002/jbm4.10757

Chourasiya, V. K., & Sahu, P. K. (2022). A review on analysis of K-Nearest Neighbor classification machine learning algorithms based on supervised learning. International Journal of Engineering Trends and Technology, 70(7), 111–117. https://doi.org/10.14445/22315381/IJETT-V70I7P205

Chen, T., & Guestrin, C. (2019). A comparative analysis of XGBoost. arXiv preprint, arXiv:1911.01914. https://doi.org/10.48550/arXiv.1911.01914

DataCamp. (n.d.). Multilayer perceptrons in machine learning. https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning
