from flask import Flask, request, jsonify
import pandas as pd
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re

app = Flask(__name__)

# @app.route('/search', methods=['POST'])
# def search():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "Request body must be JSON"}), 400
        
#         query = data.get('query', '')
#         records = data.get('records', [])
#         threshold = data.get('confidence', 0.3)
        
#         if not query or not records:
#             return jsonify({"error": "Both 'query' and 'records' are required"}), 400
        
#         df = pd.DataFrame(records)
#         df["text"] = (
#             df["name"] + " " + df["description"]
#         ).str.replace("[^a-zA-Z0-9]", " ", regex=True)
        
#         Vectorize = TfidfVectorizer()
#         Tfvect = Vectorize.fit_transform(df["text"])
        
#         sub_match = re.sub("[^a-zA-Z0-9]", " ", query.lower())
#         Query_vec = Vectorize.transform([sub_match])
        
#         Similarity = cosine_similarity(Query_vec, Tfvect).flatten()
#         df['similarity'] = Similarity

#         filtered = df[df['similarity'] >= threshold]
#         filtered = filtered.sort_values('similarity', ascending=False)
#         results = filtered.to_dict(orient='records')

#         return jsonify(results)
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Hi Mom"

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400
        
        query = data.get('query', '')
        records = data.get('records', [])
        threshold = data.get('confidence', 0.3)
        
        if not query or not records:
            return jsonify({"error": "Both 'query' and 'records' are required"}), 400
        
        df = pd.DataFrame(records)
        df["text"] = (
            df["name"] + " " + df["description"]
        ).str.replace("[^a-zA-Z0-9]", " ", regex=True).str.lower()
        
        tokenized_corpus = df["text"].apply(lambda x: x.split()).tolist()
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = re.sub("[^a-zA-Z0-9]", " ", query.lower()).split()
        
        scores = bm25.get_scores(tokenized_query)
        df['similarity'] = scores

        df['similarity'] = (df['similarity'] - df['similarity'].min()) / (df['similarity'].max() - df['similarity'].min())
        
        filtered = df[df['similarity'] >= threshold].drop(columns=['text'], axis=1)
        filtered = filtered.sort_values('similarity', ascending=False)
        results = filtered.to_dict(orient='records')

        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
