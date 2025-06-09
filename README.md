# Laporan Proyek Machine Learning - Yulia Khoirunnisa
## Project Overview
![book](https://github.com/user-attachments/assets/f511329e-fad5-406a-908a-3f18bb388b11)
### Latar Belakang
Sistem rekomendasi merupakan salah satu penerapan *machine learning* yang paling populer dan banyak ditemui dalam kehidupan sehari-hari.  Mulai dari platform *e-commerce*, layanan *streaming* musik dan film, hingga portal berita, sistem ini membantu pengguna menemukan item yang relevan di antara jutaan pilihan yang tersedia. Dalam industri perbukuan, baik toko buku daring maupun perpustakaan digital, sistem rekomendasi memegang peranan krusial dalam meningkatkan pengalaman pengguna. Dengan merekomendasikan buku yang sesuai dengan selera atau kebutuhan, pengguna dapat lebih mudah menemukan bacaan baru yang menarik, yang pada akhirnya dapat meningkatkan keterlibatan (*engagement*) dan loyalitas pengguna. 

Proyek ini bertujuan untuk membangun sebuah sistem rekomendasi buku menggunakan dataset publik "Book-Crossing: User review ratings". Pentingnya proyek ini adalah untuk menerapkan dan membandingkan dua pendekatan utama dalam sistem rekomendasi, yaitu *Content-Based Filtering* dan *Collaborative Filtering*.  Dengan memahami dan mengimplementasikan kedua pendekatan ini, kita dapat melihat bagaimana masing-masing metode bekerja dalam memberikan rekomendasi yang dipersonalisasi serta menganalisis kelebihan dan kekurangannya dalam konteks data buku.

### Referensi 
* Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In *Recommender Systems Handbook* (pp. 1-35). Springer, Boston, MA. [https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1)
* Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer. [https://link.springer.com/book/10.1007/978-3-319-29659-3](https://link.springer.com/book/10.1007/978-3-319-29659-3)

## Business Understanding
### Problem Statements 
Berdasarkan latar belakang yang telah diuraikan, masalah utama yang ingin diselesaikan adalah:
* Bagaimana cara memberikan rekomendasi buku kepada pengguna berdasarkan kemiripan konten (judul, penulis, penerbit) dari buku yang pernah mereka sukai?
* Bagaimana cara menyajikan rekomendasi buku yang dipersonalisasi untuk seorang pengguna berdasarkan pola rating dari pengguna lain yang memiliki selera serupa?

### Goals 
Tujuan dari proyek ini adalah untuk menjawab pernyataan masalah tersebut, yaitu:
* Membangun model sistem rekomendasi menggunakan pendekatan *Content-Based Filtering* yang mampu menemukan buku-buku serupa berdasarkan atributnya.
* Mengembangkan model sistem rekomendasi menggunakan pendekatan *Collaborative Filtering* untuk memberikan daftar Top-N rekomendasi buku kepada pengguna.
* Mengevaluasi kinerja model *Collaborative Filtering* menggunakan metrik yang sesuai.

### Solution Approach 
Untuk mencapai tujuan yang telah ditentukan, diusulkan dua pendekatan solusi sebagai berikut:
1. **Content-Based Filtering**: Pendekatan ini akan merekomendasikan buku dengan cara menganalisis kemiripan fitur atau konten dari buku itu sendiri, seperti judul, penulis, dan penerbit. Fitur-fitur teks ini akan digabungkan dan diubah menjadi representasi numerik menggunakan `TfidfVectorizer`. Kemudian, kemiripan antar buku dihitung menggunakan *Cosine Similarity* yang diimplementasikan melalui `NearestNeighbors` untuk menemukan buku-buku yang paling mirip.
2. **Collaborative Filtering**: Pendekatan ini akan memberikan rekomendasi berdasarkan data interaksi (rating) dari semua pengguna.  Algoritma yang digunakan adalah *Singular Value Decomposition* (SVD) dari library `scikit-surprise`. Model ini akan mempelajari pola tersembunyi (*latent factors*) dari rating pengguna untuk memprediksi rating pada buku yang belum pernah dibaca oleh seorang pengguna, lalu merekomendasikan buku dengan prediksi rating tertinggi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah **Book-Crossing Dataset** dari [Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) yang terdiri dari tiga file: `BX-Books.csv`, `BX-Book-Ratings.csv`, dan `BX-Users.csv`.  Dataset ini berisi informasi mengenai buku, rating yang diberikan oleh pengguna, dan data demografis pengguna.

Setelah melalui proses pembersihan data, jumlah data yang digunakan adalah sebagai berikut:

![dataset books](https://github.com/user-attachments/assets/bddc1e27-8ca7-4874-8891-a628acaf4c27)
* **Books**: 270,494 records
  
![dataset rating](https://github.com/user-attachments/assets/e599f1e2-d5a5-4f95-9902-ca646064ebe7)
* **Ratings**: 433,659 records (hanya rating > 0)
  
![dataset user](https://github.com/user-attachments/assets/2d07a607-de0f-41f1-816d-47fbd01a73ac)
* **Users**: 166,422 records (usia antara 5-100 tahun)

### Variabel pada Data 
Berikut adalah penjelasan mengenai variabel-variabel utama yang digunakan dari setiap file:

**1. BX-Books.csv**
* `ISBN`: International Standard Book Number, kode unik untuk setiap buku.
* `Book-Title`: Judul dari buku.
* `Book-Author`: Nama penulis buku.
* `Year-Of-Publication`: Tahun buku diterbitkan.
* `Publisher`: Nama penerbit buku.
* `Image-URL-S`: URL ke gambar sampul buku ukuran S.
* `Image-URL-M`: URL ke gambar sampul buku dalam ukuran M.
* `Image-URL-L`: URL ke gambar sampul buku dalam ukuran L.

**2. BX-Book-Ratings.csv**
* `User-ID`: ID unik untuk setiap pengguna.
* `ISBN`: ISBN buku yang diberi rating.
* `Book-Rating`: Rating yang diberikan oleh pengguna dengan skala 1-10 (rating implisit 0 telah dibuang).

**3. BX-Users.csv**
* `User-ID`: ID unik untuk setiap pengguna.
* `Location`: Lokasi (kota, provinsi, negara) dari pengguna.
* `Age`: Usia pengguna.

### Exploratory Data Analysis (EDA) 

**Distribusi Usia Pengguna**

![distribusi usia ](https://github.com/user-attachments/assets/150b0c1c-ac81-4b72-aaf4-ecf4df43705c)

Visualisasi data usia pengguna menunjukkan bahwa mayoritas pengguna berada di rentang usia 20 hingga 40 tahun, dengan puncak distribusi di sekitar usia 25 tahun.
Tentu, ini adalah kode Markdown lengkap untuk laporan Anda. Anda bisa menyalin dan menempelkannya ke dalam file .md.

_Insight_: Karena mayoritas pengguna berada dalam rentang usia dewasa muda, rekomendasi yang dihasilkan mungkin lebih condong ke selera demografis ini.

**10 Lokasi Pengguna Terbanyak**

![lokasi pengguna](https://github.com/user-attachments/assets/599fceb5-a604-47e7-850c-2293b7ffe87f)

Grafik menunjukkan bahwa pengguna terbanyak berasal dari kota-kota besar di negara berbahasa Inggris seperti London (Inggris), Sydney (Australia), dan berbagai kota di Amerika Serikat serta Kanada.

_Insight_: Sebaran geografis pengguna ini mengindikasikan bahwa koleksi buku dan preferensi rating kemungkinan besar didominasi oleh buku-buku berbahasa Inggris.

# Data Preparation
Tahapan data preparation dilakukan untuk membersihkan dan menyusun data agar siap digunakan untuk modeling. Langkah-langkah yang dilakukan adalah sebagai berikut:
1. **Pembersihan Data Awal:**
   - Pada data `books`, baris yang semua nilainya kosong (`NaN`) dihapus.
   - Pada data `ratings`, rating bernilai 0 (menandakan rating implisit) dihapus karena untuk model collaborative filtering ini hanya rating eksplisit (1-10) yang relevan.
   - Pada data `users`, data pengguna difilter untuk menjaga usia yang logis, yaitu antara 5 hingga 100 tahun.
2. **Penggabungan Data:** DataFrame `ratings` dan `books` digabungkan (merge) berdasarkan kolom `ISBN`. Ini bertujuan agar setiap rating terhubung dengan informasi detail bukunya.
```
data = ratings.merge(books, on='ISBN')
```
3. **Seleksi Fitur:** Hanya kolom-kolom yang relevan untuk kedua model yang dipilih, yaitu `User-ID`, `Book-Title`, `Book-Author`, `Publisher`, dan `Book-Rating`. Ini dilakukan untuk menyederhanakan dataset dan mengurangi penggunaan memori.
```
data = data[['User-ID', 'Book-Title', 'Book-Author', 'Publisher', 'Book-Rating']]
```
4. **Persiapan untuk Content-Based Filtering:** 
   - Duplikasi data berdasarkan `Book-Title` dihapus untuk membuat matriks kemiripan yang unik per buku.
   - Fitur teks (`Book-Title`, `Book-Author`, `Publisher`) digabungkan menjadi satu kolom `combined`. Ini dilakukan agar `TfidfVectorizer` dapat memproses semua informasi konten secara bersamaan.
   - Nilai `NaN` yang mungkin ada di kolom `combined` diisi dengan string kosong untuk menghindari error pada saat vektorisasi. <!-- end list -->
```
content_data = data.drop_duplicates(subset=['Book-Title']).copy()
content_data['combined'] = content_data['Book-Title'] + ' ' + content_data['Book-Author'] + ' ' + content_data['Publisher']
content_data['combined'] = content_data['combined'].fillna('')
```
5. **Persiapan untuk Collaborative Filtering:**
   - Data rating dibagi menjadi 80% data latih dan 20% data uji. Ini penting untuk melatih model pada satu set data dan mengevaluasinya pada set data lain yang belum pernah dilihatnya.
   - Dataset diubah ke dalam format yang dapat dibaca oleh library `surprise` menggunakan `Reader` dan `Dataset.load_from_df()`.
Tahapan ini sangat penting karena kualitas data yang bersih dan terstruktur akan sangat menentukan performa model yang akan dibangun.

# Modeling
### **1. Content-Based Filtering**
Model ini dibuat untuk merekomendasikan buku berdasarkan kemiripan kontennya. 
**- Proses:** Fitur gabungan (`combined`) dari setiap buku diubah menjadi vektor TF-IDF. Matriks TF-IDF ini merepresentasikan pentingnya setiap kata dalam dokumen (buku). Selanjutnya, model `NearestNeighbors` dilatih pada matriks ini untuk menemukan buku-buku dengan vektor terdekat (kemiripan kosinus tertinggi).
**- Kelebihan:**  Tidak memerlukan data rating dari pengguna lain (mengatasi masalah _cold_ start untuk item baru) dan dapat memberikan rekomendasi untuk item yang spesifik/kurang populer.
**- Kekurangan:**  Terbatas pada fitur yang ada (tidak dapat menemukan item baru yang tak terduga atau _serendipity_) dan bisa menjadi terlalu spesifik (jika pengguna menyukai satu genre, ia hanya akan direkomendasikan genre itu saja).

**Hasil Rekomendasi (Top-N Recommendation)** 
Berikut adalah hasil 10 rekomendasi teratas untuk buku "Harry Potter and the Sorcerer's Stone (Book 1)":
![tpo N](https://github.com/user-attachments/assets/90e9711f-2655-40ac-bb7e-4ea135675643)

### 2. Collaborative Filtering
