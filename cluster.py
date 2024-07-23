import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from flask import Flask, render_template, request, jsonify
import ssl

app = Flask(__name__)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample documents
documents = [
    "The economy is improving with better fiscal policies.",
    "The new movie has great visual effects and a compelling story.",
    "Political tensions are rising in the country.",
    # Add more documents here
]


# Preprocess documents
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


processed_docs = [preprocess_text(doc) for doc in documents]

# Vectorize documents
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_docs)

# Apply K-means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Print documents in each cluster
labels = kmeans.labels_
for i in range(num_clusters):
    print(f"Cluster {i}:")
    for idx, label in enumerate(labels):
        if label == i:
            print(f" - {documents[idx]}")

# Print cluster centroids
print("\nCluster centroids:")
terms = vectorizer.get_feature_names_out()
for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}:")
    for idx in centroid.argsort()[-10:]:  # print top 10 terms per cluster
        print(f" {terms[idx]}")


# Function to predict the cluster for a new document
def predict_cluster(new_doc):
    processed_doc = preprocess_text(new_doc)
    X_new = vectorizer.transform([processed_doc])
    return kmeans.predict(X_new)[0]


# # Test the model with a new document
# new_document = "The latest economic report shows growth in the GDP."
# predicted_cluster = predict_cluster(new_document)
# print(f'The new document belongs to cluster: {predicted_cluster}')


# Route for the home page
@app.route('/')
def index():
    return render_template('cluster.html', title='Document Clustering')


# Route to handle form submission and clustering
@app.route('/cluster_result', methods=['POST'])
def cluster_result():
    document = request.form['document']
    predicted_cluster = predict_cluster(document)
    return jsonify({'document': document, 'predicted_cluster': f"Cluster {predicted_cluster}"})


if __name__ == '__main__':
    app.run(debug=True)