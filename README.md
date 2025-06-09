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

Markdown

# Laporan Proyek Machine Learning - Sistem Rekomendasi Buku

## Project Overview

[cite_start]Sistem rekomendasi merupakan salah satu penerapan *machine learning* yang paling populer dan banyak ditemui dalam kehidupan sehari-hari.  Mulai dari platform *e-commerce*, layanan *streaming* musik dan film, hingga portal berita, sistem ini membantu pengguna menemukan item yang relevan di antara jutaan pilihan yang tersedia. [cite_start]Dalam industri perbukuan, baik toko buku daring maupun perpustakaan digital, sistem rekomendasi memegang peranan krusial dalam meningkatkan pengalaman pengguna.  [cite_start]Dengan merekomendasikan buku yang sesuai dengan selera atau kebutuhan, pengguna dapat lebih mudah menemukan bacaan baru yang menarik, yang pada akhirnya dapat meningkatkan keterlibatan (*engagement*) dan loyalitas pengguna. 

[cite_start]Proyek ini bertujuan untuk membangun sebuah sistem rekomendasi buku menggunakan dataset publik "BX-Book-Crossing".  [cite_start]Pentingnya proyek ini adalah untuk menerapkan dan membandingkan dua pendekatan utama dalam sistem rekomendasi, yaitu *Content-Based Filtering* dan *Collaborative Filtering*.  Dengan memahami dan mengimplementasikan kedua pendekatan ini, kita dapat melihat bagaimana masing-masing metode bekerja dalam memberikan rekomendasi yang dipersonalisasi serta menganalisis kelebihan dan kekurangannya dalam konteks data buku.

### [cite_start]Referensi 
* Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. In *Recommender Systems Handbook* (pp. 1-35). Springer, Boston, MA. [https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1)
* Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer. [https://link.springer.com/book/10.1007/978-3-319-29659-3](https://link.springer.com/book/10.1007/978-3-319-29659-3)

## Business Understanding

### [cite_start]Problem Statements 
Berdasarkan latar belakang yang telah diuraikan, masalah utama yang ingin diselesaikan adalah:
* Bagaimana cara memberikan rekomendasi buku kepada pengguna berdasarkan kemiripan konten (judul, penulis, penerbit) dari buku yang pernah mereka sukai?
* Bagaimana cara menyajikan rekomendasi buku yang dipersonalisasi untuk seorang pengguna berdasarkan pola rating dari pengguna lain yang memiliki selera serupa?

### [cite_start]Goals 
Tujuan dari proyek ini adalah untuk menjawab pernyataan masalah tersebut, yaitu:
* Membangun model sistem rekomendasi menggunakan pendekatan *Content-Based Filtering* yang mampu menemukan buku-buku serupa berdasarkan atributnya.
* Mengembangkan model sistem rekomendasi menggunakan pendekatan *Collaborative Filtering* untuk memberikan daftar Top-N rekomendasi buku kepada pengguna.
* Mengevaluasi kinerja model *Collaborative Filtering* menggunakan metrik yang sesuai.

### [cite_start]Solution Approach 
Untuk mencapai tujuan yang telah ditentukan, diusulkan dua pendekatan solusi sebagai berikut:
1.  [cite_start]**Content-Based Filtering**: Pendekatan ini akan merekomendasikan buku dengan cara menganalisis kemiripan fitur atau konten dari buku itu sendiri, seperti judul, penulis, dan penerbit.  Fitur-fitur teks ini akan digabungkan dan diubah menjadi representasi numerik menggunakan `TfidfVectorizer`. Kemudian, kemiripan antar buku dihitung menggunakan *Cosine Similarity* yang diimplementasikan melalui `NearestNeighbors` untuk menemukan buku-buku yang paling mirip.
2.  [cite_start]**Collaborative Filtering**: Pendekatan ini akan memberikan rekomendasi berdasarkan data interaksi (rating) dari semua pengguna.  Algoritma yang digunakan adalah *Singular Value Decomposition* (SVD) dari library `scikit-surprise`. Model ini akan mempelajari pola tersembunyi (*latent factors*) dari rating pengguna untuk memprediksi rating pada buku yang belum pernah dibaca oleh seorang pengguna, lalu merekomendasikan buku dengan prediksi rating tertinggi.

## Data Understanding
[cite_start]Dataset yang digunakan dalam proyek ini adalah **Book-Crossing Dataset** yang terdiri dari tiga file: `BX-Books.csv`, `BX-Book-Ratings.csv`, dan `BX-Users.csv`.  Dataset ini berisi informasi mengenai buku, rating yang diberikan oleh pengguna, dan data demografis pengguna.

Setelah melalui proses pembersihan data, jumlah data yang digunakan adalah sebagai berikut:
* **Books**: 270,494 records
* **Ratings**: 433,659 records (hanya rating > 0)
* **Users**: 166,422 records (usia antara 5-100 tahun)

[cite_start]**Sumber Dataset**: [cite: 30] Dataset ini dapat diunduh dari Kaggle, meskipun sumber aslinya berasal dari [Institut für Informatik, Universität Freiburg](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

### [cite_start]Variabel pada Data 
Berikut adalah penjelasan mengenai variabel-variabel utama yang digunakan dari setiap file:

**1. BX-Books.csv**
* `ISBN`: International Standard Book Number, kode unik untuk setiap buku.
* `Book-Title`: Judul dari buku.
* `Book-Author`: Nama penulis buku.
* `Year-Of-Publication`: Tahun buku diterbitkan.
* `Publisher`: Nama penerbit buku.
* `Image-URL-S/M/L`: URL ke gambar sampul buku dalam berbagai ukuran.

**2. BX-Book-Ratings.csv**
* `User-ID`: ID unik untuk setiap pengguna.
* `ISBN`: ISBN buku yang diberi rating.
* `Book-Rating`: Rating yang diberikan oleh pengguna dengan skala 1-10 (rating implisit 0 telah dibuang).

**3. BX-Users.csv**
* `User-ID`: ID unik untuk setiap pengguna.
* `Location`: Lokasi (kota, provinsi, negara) dari pengguna.
* `Age`: Usia pengguna.

### [cite_start]Exploratory Data Analysis (EDA) 

**Distribusi Usia Pengguna**

![distribusi usia ](https://github.com/user-attachments/assets/5d6d952a-fcc4-443e-b93b-c052fb9f613b)

Visualisasi data usia pengguna menunjukkan bahwa mayoritas pengguna berada di rentang usia 20 hingga 40 tahun, dengan puncak distribusi di sekitar usia 25 tahun.
Insight: Karena mayoritas pengguna berada dalam rentang usia dewasa muda, rekomendasi yang dihasilkan mungkin lebih condong ke selera demografis ini.
