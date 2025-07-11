# 🥭 Durian Image Retrieval System

A comprehensive Streamlit application for image similarity search and data management for durian disease and pest classification.

### 🤖 Available Embedding Models

| Model | Description | Embedding Size | Model Size | Purpose |
|-------|-------------|----------------|------------|---------|
| **SigLIP2 Base** | Google's SigLIP2 Base model | 768 | ~1.2GB | General-purpose image embedding |
| **SigLIP2 Large** | Google's SigLIP2 Large model | 1024 | ~2.5GB | High-performance image embedding |
| **CLIP ViT-B/32** | OpenAI's CLIP ViT-B/32 | 512 | ~150MB | Efficient image-text understanding |
| **CLIP ViT-L/14** | OpenAI's CLIP ViT-L/14 | 768 | ~1.7GB | High-quality image understanding |
| **DINOv2 ViT-B/14** | Facebook's DINOv2 Base | 768 | ~1.1GB | Self-supervised image learning |
| **DINOv2 ViT-L/14** | Facebook's DINOv2 Large | 1024 | ~2.4GB | High-capacity self-supervised learning |

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image_retrieval
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Qdrant server** (using Docker):
   ```bash
   docker-compose up -d
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

### Data Ingestion Workflow

1. **Navigate to Data Ingestion Page**: Select "Data Ingestion" from the sidebar navigation
2. **Configure Settings**: 
   - Set dataset directory path
   - Select categories to ingest (diseases, pests, or both)
   - Choose batch size and distance metric
3. **Start Ingestion**: Click "Start Ingestion" to begin processing
4. **Monitor Progress**: Track real-time progress and view results
5. **Access Database**: Use provided links to access Qdrant dashboard and collections

### Sidebar Configuration

1. **Vector Store Settings**:
   - Configure Qdrant URI (default: `http://localhost:6333/`)
   - Set collection name prefix (default: `durian`)
   - Check database status for existing collections

2. **Model Selection**:
   - Choose from 6 pre-configured embedding models
   - View detailed model specifications
   - Automatic model reloading when changed

3. **Hardware Settings**:
   - Select device (CPU, CUDA, MPS)
   - Optimize for your hardware configuration

4. **Search Configuration**:
   - Set number of results (top-k)
   - Toggle metadata display
   - Configure search parameters

### Image Search

1. **Upload Image**: Use the file uploader to select a query image
2. **Crop (Optional)**: Adjust the region of interest using the cropping tool
3. **Search**: Click "Search Similar Images" to find similar images
4. **View Results**: Browse results with similarity scores and metadata

## 🏗️ Architecture

```
image_retrieval/
├── app.py                 # Main Streamlit application
├── src/
│   ├── config.py         # Configuration settings
│   ├── ingest_data.py    # Data ingestion pipeline
│   ├── embeddings/
│   │   └── image_embedding.py  # Image embedding models
│   └── vector_stores/
│       └── qdrant.py     # Qdrant vector store integration
├── dataset/              # Image dataset directory
├── figures/              # Temporary image storage
└── requirements.txt      # Python dependencies
```

## 🔧 Configuration

### Environment Variables
- `QDRANT_URI`: Qdrant server URL (default: `http://localhost:6333/`)
- `COLLECTION_NAME_PREFIX`: Prefix for collection names (default: `durian`)
- `DEVICE`: Hardware device for model inference (default: `cpu`)
- `DATASET_DIR`: Path to processed dataset directory

### Model Configuration
Each embedding model includes:
- **Description**: Brief overview of the model
- **Purpose**: Main use case and capabilities
- **Architecture**: Technical architecture details
- **Embedding Size**: Vector dimension output
- **Model Size**: Approximate disk space required
- **Training Data**: Dataset used for training
- **Performance**: Expected performance characteristics

## 📊 Database Schema

### Collections
- `{prefix}_diseases`: Disease image embeddings and metadata
- `{prefix}_pests`: Pest image embeddings and metadata

### Metadata Fields
- `image_name`: Original image filename
- `disease`: Disease classification label
- `pest`: Pest classification label
- Additional custom metadata fields

## 🚀 Performance Tips

1. **Model Selection**: Choose smaller models (CLIP ViT-B/32) for faster processing
2. **Batch Size**: Adjust batch size based on available memory
3. **Device**: Use GPU (CUDA) or Apple Silicon (MPS) for faster inference
4. **Distance Metric**: Cosine similarity works well for most image similarity tasks

## 🔍 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**:
   - Ensure Qdrant server is running: `docker-compose up -d`
   - Check URI configuration in sidebar settings

2. **Model Loading Issues**:
   - Verify internet connection for model downloads
   - Check available disk space for model files
   - Ensure compatible device selection

3. **Ingestion Failures**:
   - Verify dataset directory path exists
   - Check file permissions
   - Ensure proper image and metadata file structure

### Logs
- Application logs: Check Streamlit console output
- Ingestion logs: `ingestion.log` file in project root
- Qdrant logs: Docker container logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SigLIP2**: Google's state-of-the-art image-text model
- **CLIP**: OpenAI's contrastive language-image pre-training
- **DINOv2**: Facebook's self-supervised vision model
- **Qdrant**: Vector similarity search engine
- **Streamlit**: Web application framework
