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

class MemoryMonitor(threading.Thread):
    def __init__(self, threshold=0.90, check_interval=60):
        super().__init__()
        self.threshold = threshold
        self.check_interval = check_interval
        self.stop_flag = threading.Event()
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def run(self):
        while not self.stop_flag.is_set():
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.threshold:
                logging.warning(f"Memory usage ({memory_percent*100:.1f}%) exceeded threshold ({self.threshold*100:.1f}%)")
                for callback in self.callbacks:
                    callback()
            time.sleep(self.check_interval)

    def stop(self):
        self.stop_flag.set()

class LRUCache:
    def __init__(self, capacity, max_memory_percent=0.3):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.max_memory_percent = max_memory_percent
        self.memory_usage = 0

    def _get_item_size(self, value):
        return value.element_size() * value.nelement()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.memory_usage -= self._get_item_size(self.cache[key])
            self.cache.move_to_end(key)
        
        item_size = self._get_item_size(value)
        
        # Check if adding this item would exceed memory threshold
        while self.cache and (psutil.virtual_memory().percent / 100 > self.max_memory_percent):
            _, removed_value = self.cache.popitem(last=False)
            self.memory_usage -= self._get_item_size(removed_value)
            gc.collect()

        self.cache[key] = value
        self.memory_usage += item_size

        # Remove oldest items if capacity exceeded
        while len(self.cache) > self.capacity:
            _, removed_value = self.cache.popitem(last=False)
            self.memory_usage -= self._get_item_size(removed_value)

class ImageDatabase:
    def __init__(self, cache_size=100, max_batch_size=32, memory_threshold=0.90, cache_memory_percent=0.3):
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.database = {}
        self.cache = LRUCache(cache_size, max_memory_percent=cache_memory_percent)
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.last_accessed = {}
        self.database_file = "image_database.pkl"
        self.metadata_file = "database_metadata.pkl"
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(threshold=memory_threshold)
        self.memory_monitor.add_callback(self._cleanup_old_embeddings)
        self.memory_monitor.start()
        
        # Load existing database if available
        self.load_database()

    def __del__(self):
        self.memory_monitor.stop()
        self.save_database()

    def _cleanup_old_embeddings(self, age_threshold_days=30):
        current_time = datetime.now()
        keys_to_remove = []
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        for filename, last_access in self.last_accessed.items():
            age = (current_time - last_access).days
            if age > age_threshold_days:
                keys_to_remove.append(filename)
        
        for key in keys_to_remove:
            del self.database[key]
            del self.last_accessed[key]
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_freed = memory_before - memory_after
        
        if keys_to_remove:
            logging.info(f"Cleaned up {len(keys_to_remove)} old embeddings, freed {memory_freed:.2f}MB of memory")

    def add_images_to_database(self, folder_path, batch_size=None):
        if batch_size is None:
            batch_size = self.max_batch_size

        image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]

        total_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size else 0)
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            logging.info(f"Processing batch {current_batch}/{total_batches}")
            
            try:
                batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
                
                # Generate embeddings for batch
                inputs = self.processor(images=batch_images, return_tensors="pt", padding="max_length", truncation=True)
                with torch.no_grad():
                    embeddings = self.model.get_image_features(**inputs)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                # Store embeddings and update metadata
                for path, embedding in zip(batch_paths, embeddings):
                    filename = os.path.basename(path)
                    self.database[filename] = embedding
                    self.last_accessed[filename] = datetime.now()
                    self.cache.put(filename, embedding)
                
                # Periodic garbage collection
                if current_batch % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logging.error(f"Error processing batch {current_batch}: {str(e)}")
                continue

        logging.info(f"Added {len(image_paths)} images to the database!")
        self.save_database()

    def query_database(self, query_image_path, top_k=5):
        try:
            # Check cache first
            cached_result = self.cache.get(os.path.basename(query_image_path))
            if cached_result is not None:
                query_embedding = cached_result
                logging.info("Using cached embedding for query")
            else:
                query_image = Image.open(query_image_path).convert("RGB")
                inputs = self.processor(images=[query_image], return_tensors="pt", padding=True)
                with torch.no_grad():
                    query_embedding = self.model.get_image_features(**inputs)
                    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

            similarities = {}
            for filename, stored_embedding in self.database.items():
                similarity = torch.nn.functional.cosine_similarity(query_embedding, stored_embedding)
                similarities[filename] = similarity.item()
                self.last_accessed[filename] = datetime.now()

            sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]
            
        except Exception as e:
            logging.error(f"Error during query: {str(e)}")
            return []

    def save_database(self):
        try:
            # Save embeddings
            with open(self.database_file, "wb") as f:
                pickle.dump(self.database, f)
            
            # Save metadata
            metadata = {
                'last_accessed': self.last_accessed,
                'timestamp': datetime.now()
            }
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

def display_results(query_image_path, results, folder_path):
    try:
        if not results:
            logging.warning("No results found to display")
            return
            
        query_image = Image.open(query_image_path).convert("RGB")
        fig = plt.figure(figsize=(15, 5))
        
        # Create a grid of subplots
        n_images = len(results) + 1
        axes = []
        for i in range(n_images):
            ax = fig.add_subplot(1, n_images, i + 1)
            axes.append(ax)
        
        # Display query image
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
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
        # Define folder paths
        image_folder = "images_to_add"
        query_image_path = "query_image.jpg"
        
        # Initialize database with memory management
        db = ImageDatabase(
            cache_size=100,
            max_batch_size=32,
            memory_threshold=0.90,
            cache_memory_percent=0.3
        )
        
        # Add images to database
        if os.path.exists(image_folder):
            db.add_images_to_database(image_folder)
        
        # Perform query if query image exists
        if os.path.exists(query_image_path):
            results = db.query_database(query_image_path, top_k=5)
            display_results(query_image_path, results, image_folder)
        
    except Exception as e:
        logging.error(f"Main program error: {str(e)}")
