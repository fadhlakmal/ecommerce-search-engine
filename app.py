from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)

model = SentenceTransformer("clip-ViT-B-32")
data = []
text_emb = None
DATA_FILE = "data.json"

def load_initial_data():
    global data, text_emb
    
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as file:
                product_data = json.load(file)
                data = product_data
                text_data = [f"{item['name']}, {item['description']}" for item in data]
                text_emb = model.encode(text_data)
                print(f"Successfully loaded {len(data)} products on startup")
        else:
            print("No data file found on startup")
            data = []
            text_emb = None
    except Exception as e:
        print(f"Error loading initial data: {str(e)}")
        data = []
        text_emb = None

load_initial_data()

def save_data_to_file(product_data):
    with open(DATA_FILE, "w") as file:
        json.dump(product_data, file, indent=4)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/updateData', methods=['POST'])
def updateData():
    global data, text_emb

    if not request.json or 'data' not in request.json:
        return jsonify({'error': 'No data payload provided'}), 400

    product_data = request.json['data']
    save_data_to_file(product_data)
    text_data = [f"{item['name']}, {item['description']}" for item in product_data]
    data = product_data
    text_emb = model.encode(text_data)

    return jsonify({'message': 'Search engine initialized successfully', 'product_count': len(data)})

@app.route("/search", methods=["POST"])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    if text_emb is None:
        return jsonify({'error': 'Search engine not initialized. Please load data first.'}), 500

    image = request.files['image']
    try:
        img_emb = model.encode(Image.open(image))
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

    similarity_scores = model.similarity(img_emb, text_emb).numpy().flatten()
    sorted_indices = np.argsort(similarity_scores)[::-1]

    results = []
    for idx in sorted_indices:
        item = data[idx]
        results.append({
            'name': item['name'],
            'description': item['description'],
            'price': item['price'],
            'stock': item['stock'],
            'image_url': item['image_url'],
            'rating': item['rating'],
            'city': item['city'],
            'categories': item['categories'],
            'similarity_score': float(similarity_scores[idx])
        })

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)