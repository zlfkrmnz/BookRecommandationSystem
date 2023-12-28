import requests
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

column_names = ['Sıra No', 'Demirbaş No', 'Tür Adı', 'Alt Tür Adı', 'Eser Adı', 'Yazar Adı Soyadı', 'Konu Başlığı', 'Konu', 'Yayın Dili', 'Yayın Yılı', 'Yayın Evi']
df = pd.read_csv("books.csv", sep=';', quotechar='"', names=column_names)

def preprocess(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

# creating a new column 'processed' with the pre-processing function applied on the 'Konu Başlığı'
df["processed"] = df["Konu Başlığı"].apply(preprocess)

# getting stopwords from a GitHub repo
stopwords = ''
try:
    response = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-tr/master/stopwords-tr.txt")
    stopwords = response.text.split()
except requests.exceptions.RequestException as e:
    print(e)

# creating the numerical feature matrix
vectorizer = TfidfVectorizer(stop_words=stopwords)
tfid_matrix = vectorizer.fit_transform(df["processed"])
print(tfid_matrix.shape)

# setting up the ML algorithm (nearest neighbor)
n_neighbor = 6
model = NearestNeighbors(n_neighbors=n_neighbor, metric='cosine')
model.fit(tfid_matrix)

# creating a mapping from book titles to index

title_index = pd.Series(df.index, index=df["Eser Adı"].apply(lambda x: x.lower()))


def recommendation(title, model=model, df=df, title_index=title_index):
    # Girdi olarak verilen kitabın indeksini al
    idx = title_index.get(title)

    # Eğer kitap bulunamazsa, hata mesajı döndür
    if idx is None:
        return "Kitap bulunamadı."

    # Modeli kullanarak komşuları bul
    distances, indices = model.kneighbors(tfid_matrix[idx])

    # İlk sonucu atlayarak önerilen kitapların indekslerini al
    indices = indices[0][1:]

    # Önerilen kitapların başlıklarını döndür
    return df['Eser Adı'].iloc[indices].values.tolist()


def main():
    while True:
        book_title = input("Bir eser adı giriniz (Programdan çıkmak için 'çıkış' yazınız): ").lower()
        if book_title == 'çıkış':
            break
        if book_title not in title_index:
            print("Üzgünüz! Girdiğiniz kitap veritabanında mevcut değil. Başka bir kitap deneyiniz.")
            continue
        rec = recommendation(book_title)
        print("\nİşte sizin için öneriler: '{}'".format(book_title.capitalize()))
        print(rec)
        print("\n")


if __name__ == "__main__":
    main()
