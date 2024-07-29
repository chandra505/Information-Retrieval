import requests
from bs4 import BeautifulSoup
import json
import re
from collections import defaultdict
import logging
from flask import Flask, render_template, request, redirect, url_for
import time
from math import sqrt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from apscheduler.schedulers.background import BackgroundScheduler
import ssl
import nltk

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Download NLTK stopwords data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Web Crawler
class CMDSPublicationsCrawler:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited_urls = set()
        self.publications = []

    def crawl(self, url):
        if url in self.visited_urls:
            return

        response = requests.get(url)
        self.visited_urls.add(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            self.extract_publications(soup)

            # Find next page link and continue crawling
            next_page_tag = soup.find('a', class_='next')
            if next_page_tag and 'href' in next_page_tag.attrs:
                next_page_url = next_page_tag['href']
                time.sleep(1)  # Delay to avoid hitting the server too fast
                self.crawl(next_page_url)
        else:
            logger.warning(f"Failed to retrieve URL: {url} with status code: {response.status_code}")

    def extract_publications(self, soup):
        for pub in soup.find_all('div', class_='result-container'):
            title_tag = pub.find('h3', class_='title')
            title = title_tag.text.strip() if title_tag else 'No Title'

            authors_tags = pub.find_all('a', class_='link person')
            authors = [a.text for a in authors_tags] if authors_tags else []

            year_tag = pub.find('span', class_='date')
            if year_tag:
                year = year_tag.text.strip()
            else:
                year = "Unknown"

            pub_url_tag = pub.find('a', class_='link')
            if pub_url_tag:
                pub_url = pub_url_tag['href']
            else:
                logger.warning("URL not found in publication")
                continue

            auth_url_tags = pub.find_all('a', class_='link person')
            if auth_url_tags:
                auth_urls = [a['href'] for a in auth_url_tags]
            else:
                logger.warning("URL not found in author profiles")
                continue

            self.publications.append({
                'title': title,
                'authors': authors,
                'year': year,
                'url': pub_url,
                'authorsProfiles': auth_urls
            })

    def save_to_file(self, filename='data/publications.json'):
        with open(filename, 'w') as file:
            json.dump(self.publications, file, indent=4)

# Inverted Index
class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def build_index(self, documents):
        for doc_id, doc in enumerate(documents):
            for term in re.findall(r'\w+', doc['title'].lower()):
                self.index[term].append(doc_id)

    def save_to_file(self, filename='data/inverted_index.json'):
        with open(filename, 'w') as file:
            json.dump(self.index, file, indent=4)

    def load_from_file(self, filename='data/inverted_index.json'):
        with open(filename, 'r') as file:
            self.index = json.load(file)

# Query Processor
class QueryProcessor:
    def __init__(self, index_filename='data/inverted_index.json', documents_filename='data/publications.json'):
        self.index = self.load_json(index_filename)
        self.documents = self.load_json(documents_filename)
        self.stop_words = set(stopwords.words('english'))

    def load_json(self, filename):
        with open(filename, 'r') as file:
            return json.load(file)

    def preprocess_text(self, text):
        # Tokenize text and remove stopwords
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return ' '.join(filtered_tokens)

    def build_tfidf_model(self):
        # Preprocess titles and build TF-IDF model
        titles = [self.preprocess_text(doc['title']) for doc in self.documents]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(titles)
        return vectorizer, tfidf_matrix

    def cosine_similarity(self, vec1, vec2):
        # Compute cosine similarity between two vectors
        dot_product = sum(p*q for p, q in zip(vec1, vec2))
        magnitude = sqrt(sum([val**2 for val in vec1])) * sqrt(sum([val**2 for val in vec2]))
        if not magnitude:
            return 0
        return dot_product/magnitude

    def search(self, query):
        start_time = time.time()
        preprocessed_query = self.preprocess_text(query)
        vectorizer, tfidf_matrix = self.build_tfidf_model()
        query_vector = vectorizer.transform([preprocessed_query])

        scores = defaultdict(float)
        for doc_id, document in enumerate(self.documents):
            title_vector = tfidf_matrix.getrow(doc_id)
            similarity_score = self.cosine_similarity(query_vector.toarray()[0], title_vector.toarray()[0])
            scores[doc_id] = similarity_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = [{'doc': self.documents[doc_id], 'score': score} for doc_id, score in sorted_results if score > 0.0]

        search_time = (time.time() - start_time) * 1000
        return results, search_time

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query')
    if not query:
        return render_template('results.html', query='', results=[], search_time=0, zip=zip)

    qp = QueryProcessor()
    results, search_time = qp.search(query)
    return render_template('results.html', query=query, results=results, search_time=search_time, zip=zip)

def scheduled_crawl():
    base_url = 'https://pureportal.coventry.ac.uk/en/organisations/eec-school-of-computing-mathematics-and-data-sciences-cmds/publications'
    crawler = CMDSPublicationsCrawler(base_url)
    crawler.crawl(base_url)
    crawler.save_to_file()

    with open('data/publications.json', 'r') as file:
        documents = json.load(file)

    inverted_index = InvertedIndex()
    inverted_index.build_index(documents)
    inverted_index.save_to_file()

# Main execution
if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_crawl, 'interval', weeks=1)
    scheduler.start()

    try:
        app.run(debug=True)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
