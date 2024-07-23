import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
import ssl

app = Flask(__name__)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class DocumentClustering:
    def __init__(self, file_path='data/newsdata.csv'):
        self.file_path = file_path
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)
        self.kmeans = None
        self.cluster_names = {
            0: 'Business',
            1: 'Health',
            2: 'Sports',
            3: 'Politics',
            4: 'Entertainment'
        }
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File '{self.file_path}' not found.")
            return False
        return True

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and token not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def train_model(self, n_clusters=5):
        if self.df is None:
            self.load_data()
        if self.df is not None:
            self.df['processed_text'] = self.df['description'].apply(self.preprocess_text)
            documents = self.df['processed_text'].dropna().tolist()
            X = self.vectorizer.fit_transform(documents)
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.kmeans.fit(X)

            # Create a dataframe to store the results
            results = pd.DataFrame()
            results['document'] = self.df['processed_text']
            results['cluster'] = self.kmeans.labels_

            # Evaluate clustering
            labels = self.kmeans.labels_
            silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
            print(f"Silhouette Score: {silhouette_score}")

            cluster_counts = pd.Series(labels).value_counts().sort_index()
            print("Cluster Counts:")
            print(cluster_counts)

            # Print top terms per cluster
            self.print_top_terms_per_cluster(X)

    def print_top_terms_per_cluster(self, X, top_n=10):
        feature_names = self.vectorizer.get_feature_names_out()
        order_centroids = self.kmeans.cluster_centers_.argsort()[:, ::-1]
        for i in range(len(order_centroids)):
            print(f"Cluster {i} Top Terms:")
            for ind in order_centroids[i, :top_n]:
                print(f"{feature_names[ind]}")
            print()

    def predict_cluster(self, new_document):
        if self.kmeans is None:
            self.train_model()
        if self.kmeans is not None:
            processed_doc = self.preprocess_text(new_document)
            new_X = self.vectorizer.transform([processed_doc])
            predicted_cluster = self.kmeans.predict(new_X)[0]
            return predicted_cluster
        else:
            return None

    def cluster_document(self, new_document):
        predicted_cluster = self.predict_cluster(new_document)
        if predicted_cluster is not None:
            cluster_name = self.cluster_names.get(predicted_cluster, 'unknown')
            response = {
                'document': new_document,
                'predicted_cluster': cluster_name
            }
            return response
        else:
            return {'error': 'Failed to predict cluster.'}, 500


document_clustering = DocumentClustering()


@app.route('/')
def index():
    return render_template('cluster.html', title='Document Clustering')


@app.route('/cluster_result', methods=['POST'])
def cluster_result():
    document = request.form['document']
    result = document_clustering.cluster_document(document)
    return jsonify(result)


if __name__ == '__main__':
    if document_clustering.load_data():
        document_clustering.train_model(n_clusters=5)
    app.run(debug=True)
