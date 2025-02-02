# ML
# CLIPMemory: Efficient Image Search and Retrieval using CLIP

CLIPMemory is a simple image search and retrieval system built on OpenAI's CLIP (Contrastive Language-Image Pre-training) model. It enables efficient semantic search of images using both text queries and image-based similarity search.

## Features

- Semantic image search using natural language queries
- Image-to-image similarity search
- Efficient memory management with LRU caching
- Batch processing for optimal performance
- Memory usage monitoring and automatic cleanup
- Persistent storage of image embeddings

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from CLIPMemory import ImageDatabase

# Initialize the database
db = ImageDatabase(cache_size=100)

# Add images to the database
db.add_image("path/to/image.jpg")

# Search using text
results = db.query_text("a dog playing in the park", top_k=5)
for score, path in results:
    print(f"{path}: {score}")

# Search using another image
results = db.query_image("path/to/query_image.jpg", top_k=5)
```

## Architecture

CLIPMemory consists of three main components:

1. **ImageDatabase**: The main interface for adding images and performing queries
2. **LRUCache**: Efficient caching system for frequently accessed embeddings
3. **MemoryMonitor**: Background thread monitoring system memory usage

## Memory Management

CLIPMemory implements several strategies for efficient memory usage:

- LRU (Least Recently Used) caching of embeddings
- Automatic cleanup of old embeddings when memory threshold is reached
- Batch processing of images for optimal GPU memory usage
- Persistent storage to disk for long-term storage

## Usage Examples

### Adding Multiple Images

```python
# Add all images from a directory
db = ImageDatabase()
for image_path in os.listdir("images_directory"):
    if image_path.endswith(('.jpg', '.png', '.jpeg')):
        db.add_image(os.path.join("images_directory", image_path))
```

### Text-Based Search

```python
# Search for nature scenes
results = db.query_text("beautiful nature landscape", top_k=10)
```

### Image-Based Search

```python
# Find similar images
similar_images = db.query_image("query_image.jpg", top_k=5)
```

## Performance Considerations

- Use appropriate `cache_size` based on available system memory
- Adjust `max_batch_size` based on GPU memory
- Monitor memory usage with `memory_threshold` parameter
- Use `display_results()` for visualizing search results
