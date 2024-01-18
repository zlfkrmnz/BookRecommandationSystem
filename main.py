import requests
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

column_names_books = ['Sıra No', 'Demirbaş No', 'Tür Adı', 'Alt Tür Adı', 'Eser Adı', 'Yazar Adı Soyadı',
                      'Konu Başlığı', 'Konu', 'Yayın Dili', 'Yayın Yılı', 'Yayın Evi']

df_books = pd.read_csv("Data/books.csv", sep=';', quotechar='"', names=column_names_books)

books = df_books[['Demirbaş No', 'Tür Adı', 'Alt Tür Adı', 'Eser Adı',
                      'Yazar Adı Soyadı', 'Konu Başlığı', 'Konu', 'Yayın Dili', 'Yayın Yılı',
                      'Yayın Evi']]

def preprocess(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

# creating a new column 'processed' with the pre-processing function applied on the 'Konu Başlığı'
df_books["processed"] = df_books["Konu Başlığı"].apply(preprocess)

# getting stopwords from a GitHub repo
stopwords = ''
try:
    response = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-tr/master/stopwords-tr.txt")
    stopwords = response.text.split()
except requests.exceptions.RequestException as e:
    print(e)

# creating the numerical feature matrix
vectorizer = TfidfVectorizer(stop_words=stopwords)
tfid_matrix = vectorizer.fit_transform(df_books["processed"])

# setting up the ML algorithm (nearest neighbor)
n_neighbor = 6
model = NearestNeighbors(n_neighbors=n_neighbor, metric='cosine')
model.fit(tfid_matrix)

# creating a mapping from book titles to index

title_index = pd.Series(df_books.index, index=df_books["Eser Adı"].apply(lambda x: x.lower()))

def recommendation(title, model=model, df=df_books, title_index=title_index):
    # Girdi olarak verilen kitabın indeksini al
    idx = title_index.get(title.lower())

    # Eğer kitap bulunamazsa, hata mesajı döndür
    if idx is None:
        return "Kitap bulunamadı."

    # Modeli kullanarak komşuları bul
    distances, indices = model.kneighbors(tfid_matrix[idx])

    # Önerilen kitapların başlıklarını al
    recommended_titles = df['Eser Adı'].iloc[indices[0]].tolist()

    # Girdi kitabını öneri listesinden çıkar ve tekrar edenleri filtrele
    recommended_titles = [title for title in recommended_titles if title.lower() != title.lower()]
    unique_recommended_titles = list(dict.fromkeys(recommended_titles))

    # Yeterli sayıda benzersiz öneri sağlamak için kontrol
    additional_indices = 1
    max_index = len(indices[0]) - 1  # Maksimum indeks değeri
    while len(unique_recommended_titles) < n_neighbor - 1 and additional_indices <= max_index:
        new_index = indices[0][additional_indices]
        new_title = df_books['Eser Adı'].iloc[new_index]
        if new_title.lower() != title.lower() and new_title not in unique_recommended_titles:
            unique_recommended_titles.append(new_title)
        additional_indices += 1

    return unique_recommended_titles


def main():
    while True:
        book_title = input("Bir eser adı giriniz (Programdan çıkmak için 'çıkış' yazınız): ").lower()
        if book_title == 'çıkış':
            break
        if book_title not in title_index:
            print("Üzgünüz! Girdiğiniz kitap veritabanında mevcut değil. Başka bir kitap deneyiniz.")
            continue
        rec = recommendation(book_title)
        print("\nİşte '{}' adlı kitap için öneriler: ".format(book_title.capitalize()))
        print(rec)
        print("\n")


if __name__ == "__main__":
    main()
