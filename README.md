# TextCLIP: Text-to-Image Search using CLIP

A powerful text-to-image search system built on OpenAI's CLIP (Contrastive Language-Image Pre-training) model. This system enables semantic image search using natural language queries, allowing users to find relevant images using text descriptions.

## Features

- 🔍 Semantic image search using natural language queries
- 💾 Efficient database management with persistent storage
- 🚀 Batch processing for optimal performance
- 📊 Built-in visualization of search results
- 🔄 Automatic database saving and loading
- 📝 Comprehensive logging system

## Requirements

The system requires the following main dependencies:
```bash
torch
transformers
Pillow
matplotlib
```

## Quick Start

```python
from textCLIP import ImageDatabase, display_results

# Initialize the database
db = ImageDatabase()

# Add images to the database
db.add_images_to_database("path/to/image/folder")

# Search using text query
query_text = "a sunset over the mountains"
results = db.query_text_database(query_text, top_k=5)

# Display results
display_results(query_text, results, "path/to/image/folder")
```

## Core Components

### ImageDatabase Class

The main class that handles:
- Loading and initializing the CLIP model
- Managing the image database
- Processing text queries
- Calculating image similarities
- Persistent storage of embeddings

### Display Function

The `display_results` function provides visualization of:
- Query text
- Top matching images
- Similarity scores

## Database Management

The system automatically manages:
- Persistent storage of image embeddings
- Loading of previous database state
- Metadata tracking including last accessed times
- Database saving after modifications

## Usage Examples

### Adding Images to Database

```python
db = ImageDatabase()
db.add_images_to_database("images_folder")
```

### Text-Based Search

```python
# Search for specific concepts
results = db.query_text_database("a dog playing in the park", top_k=5)

# Search for abstract concepts
results = db.query_text_database("happiness", top_k=5)

# Search for specific objects
results = db.query_text_database("red car", top_k=5)
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

## Error Handling

The system includes comprehensive error handling and logging for:
- Image loading and processing
- Database operations
- Query processing
- Result visualization

## License

This project is available under the MIT License.
