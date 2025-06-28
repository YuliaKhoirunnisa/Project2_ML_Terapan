import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from collections import defaultdict
from IPython.display import display
from sklearn.neighbors import NearestNeighbors

books = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip', engine='python')
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip', engine='python')
users = pd.read_csv('BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip', engine='python')

books.dropna(how='all', inplace=True)
ratings = ratings[ratings['Book-Rating'] > 0]
users = users[(users['Age'].fillna(0) >= 5) & (users['Age'].fillna(0) <= 100)]
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

print("=== Info Dataset Books ===")
print(books.info())
print(books.describe())
display(books[['ISBN', 'Book-Title', 'Book-Author', 'Publisher']].head())
print(f"Jumlah record di Books: {books.shape[0]}")

print("\n=== Info Dataset Ratings ===")
print(ratings.info())
print(ratings.describe())
display(ratings[['User-ID', 'ISBN', 'Book-Rating']].head())
print(f"Jumlah record di Ratings: {ratings.shape[0]}")

print("\n=== Info Dataset Users ===")
print(users.info())
print(users.describe())
display(users[['User-ID', 'Location', 'Age']].head())
print(f"Jumlah record di Users: {users.shape[0]}")

plt.figure(figsize=(10,6))
users['Age'].dropna().astype(int).plot.hist(bins=30, edgecolor='black')
plt.title('Distribusi Usia Pengguna')
plt.xlabel('Usia')
plt.ylabel('Jumlah Pengguna')
plt.show()

top_locations = users['Location'].value_counts().head(10)
plt.figure(figsize=(10,6))
top_locations.plot(kind='bar')
plt.title('10 Lokasi Pengguna Terbanyak')
plt.xlabel('Lokasi')
plt.ylabel('Jumlah Pengguna')
plt.show()

data = ratings.merge(books, on='ISBN')
data = data[['User-ID', 'Book-Title', 'Book-Author', 'Publisher', 'Book-Rating']]

content_data = data.drop_duplicates(subset=['Book-Title']).copy()
content_data['combined'] = content_data['Book-Title'] + ' ' + content_data['Book-Author'] + ' ' + content_data['Publisher']
content_data['combined'] = content_data['combined'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_data['combined'])

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(tfidf_matrix)
indices = pd.Series(content_data.index, index=content_data['Book-Title']).drop_duplicates()

def recommend_books(title, model_knn=model_knn, n_recommendations=10):
    idx = indices[title]
    distances, indices_nn = model_knn.kneighbors(tfidf_matrix[idx], n_neighbors=n_recommendations+1)
    recommended_indices = indices_nn.flatten()[1:]
    return content_data['Book-Title'].iloc[recommended_indices]

print("Rekomendasi untuk 'Harry Potter and the Sorcerer's Stone (Book 1)':")
print(recommend_books("Harry Potter and the Sorcerer's Stone (Book 1)"))

def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    intersection = recommended_set.intersection(relevant_set)
    precision = len(intersection) / k
    return precision

recommended_books = recommend_books("Harry Potter and the Sorcerer's Stone (Book 1)")
user_id = 160681
relevant_books = list(data[data['User-ID'] == user_id]['Book-Title'].unique())

prec_at_10 = precision_at_k(recommended_books, relevant_books, k=10)
print(f"Precision@10 untuk Content-Based Filtering: {prec_at_10:.2f}")

reader = Reader(rating_scale=(1, 10))
data_surprise = Dataset.load_from_df(data[['User-ID', 'Book-Title', 'Book-Rating']], reader)

trainset, testset = train_test_split(
    data[['User-ID', 'Book-Title', 'Book-Rating']], 
    test_size=0.2, 
    random_state=42
)

trainset_surprise = Dataset.load_from_df(trainset, reader).build_full_trainset()

algo = SVD(random_state=42)
algo.fit(trainset_surprise)

cross_validate(algo, data_surprise, measures=['RMSE'], cv=3, verbose=True)

testset_surprise = [tuple(x) for x in testset.values]
predictions = algo.test(testset_surprise)
print("RMSE:", accuracy.rmse(predictions))

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=10)

user_id = list(top_n.keys())[0]
recommendations = top_n[user_id]

rekomendasi_df = pd.DataFrame(recommendations, columns=['Book Title', 'Predicted Rating'])
rekomendasi_df.index = range(1, len(rekomendasi_df) + 1) # Biar mulai dari 1
print(f"Top 10 rekomendasi untuk User {user_id}:")
display(rekomendasi_df)