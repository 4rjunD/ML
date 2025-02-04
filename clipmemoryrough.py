import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import pickle
import time
from collections import OrderedDict
import psutil
import numpy as np
from datetime import datetime
import gc
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageDatabase:
    def __init__(self, cache_size=100, max_batch_size=32, memory_threshold=0.90, cache_memory_percent=0.3):
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.database = {}
        self.last_accessed = {}
        self.database_file = "image_database.pkl"
        self.metadata_file = "database_metadata.pkl"

        # Load existing database if available
        self.load_database()

    def add_images_to_database(self, folder_path):
        image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                inputs = self.processor(images=[image], return_tensors="pt", padding=True)
                with torch.no_grad():
                    embedding = self.model.get_image_features(**inputs)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                filename = os.path.basename(path)
                self.database[filename] = embedding
                self.last_accessed[filename] = datetime.now()

            except Exception as e:
                logging.error(f"Error processing {path}: {str(e)}")

        logging.info(f"Added {len(image_paths)} images to the database!")
        self.save_database()

    def query_image_database(self, query_image_path, top_k=5):
        try:
            query_image = Image.open(query_image_path).convert("RGB")
            inputs = self.processor(images=[query_image], return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = self.model.get_image_features(**inputs)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

            return self._find_similar_images(query_embedding, top_k)

        except Exception as e:
            logging.error(f"Error during image query: {str(e)}")
            return []

    def query_text_database(self, query_text, top_k=5):
        try:
            inputs = self.processor(text=[query_text], return_tensors="pt", padding=True)
            with torch.no_grad():
                text_embedding = self.model.get_text_features(**inputs)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

            return self._find_similar_images(text_embedding, top_k)

        except Exception as e:
            logging.error(f"Error during text query: {str(e)}")
            return []

    def _find_similar_images(self, query_embedding, top_k):
        similarities = {}
        for filename, stored_embedding in self.database.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, stored_embedding)
            similarities[filename] = similarity.item()
            self.last_accessed[filename] = datetime.now()

        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def save_database(self):
        try:
            with open(self.database_file, "wb") as f:
                pickle.dump(self.database, f)

            metadata = {'last_accessed': self.last_accessed, 'timestamp': datetime.now()}
            with open(self.metadata_file, "wb") as f:
                pickle.dump(metadata, f)

            logging.info("Database and metadata saved successfully")

        except Exception as e:
            logging.error(f"Error saving database: {str(e)}")

    def load_database(self):
        try:
            if os.path.exists(self.database_file):
                with open(self.database_file, "rb") as f:
                    self.database = pickle.load(f)

                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, "rb") as f:
                        metadata = pickle.load(f)
                        self.last_accessed = metadata['last_accessed']

                logging.info(f"Loaded database with {len(self.database)} images")
                return True
            return False

        except Exception as e:
            logging.error(f"Error loading database: {str(e)}")
            return False

def display_results(query, results, folder_path):
    try:
        if not results:
            logging.warning("No results found to display")
            return
            
        fig = plt.figure(figsize=(15, 5))
        
        n_images = len(results) + 1
        axes = []
        for i in range(n_images):
            ax = fig.add_subplot(1, n_images, i + 1)
            axes.append(ax)
        
        # Display query image or text
        if isinstance(query, str) and query.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            query_image = Image.open(query).convert("RGB")
            axes[0].imshow(query_image)
            axes[0].set_title("Query Image")
        else:
            axes[0].text(0.5, 0.5, query, fontsize=12, ha='center', va='center')
            axes[0].set_title("Query Text")
        
        axes[0].axis("off")
        
        # Display top matching images
        for i, (filename, similarity) in enumerate(results):
            result_image_path = os.path.join(folder_path, filename)
            result_image = Image.open(result_image_path).convert("RGB")
            axes[i + 1].imshow(result_image)
            axes[i + 1].set_title(f"{filename}\nSim: {similarity:.4f}")
            axes[i + 1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logging.error(f"Error displaying results: {str(e)}")

if __name__ == "__main__":
    try:
        image_folder = "images_to_add"
        query_image_path = "query_image.jpg"
        query_text = "a sunset over the mountains"

        db = ImageDatabase()

        if os.path.exists(image_folder):
            db.add_images_to_database(image_folder)

        if os.path.exists(query_image_path):
            results = db.query_image_database(query_image_path, top_k=5)
            display_results(query_image_path, results, image_folder)

        results_text = db.query_text_database(query_text, top_k=5)
        display_results(query_text, results_text, image_folder)

    except Exception as e:
        logging.error(f"Main program error: {str(e)}")
