# 🥭 Durian Image Retrieval System

A beautiful and intuitive web application for searching similar durian disease and pest images using advanced AI embedding models.

## ✨ Features

### 🎨 Simple UI/UX
- **Clean Design**: Uses Streamlit's default styling for simplicity
- **Responsive Layout**: Optimized for different screen sizes
- **Interactive Elements**: Progress bars, metrics, and visual feedback

### 📊 Multi-Page Navigation
1. **Disease Report Page**: 
   - Count files in each disease category folder
   - Interactive charts and statistics
   - Summary metrics with clean cards
   - Top categories ranking

2. **Pest Report Page**:
   - Count files in each pest category folder
   - Interactive charts and statistics
   - Summary metrics with clean cards
   - Top categories ranking

3. **Image Search Page**:
   - Upload and crop query images
   - Multiple embedding model selection
   - Real-time similarity search
   - Clean result display with metadata

### 🤖 Advanced Model Selection
Choose from multiple state-of-the-art embedding models:
- **SigLIP2 Base/Large**: Google's latest vision-language model
- **CLIP ViT-B/32/L/14**: OpenAI's contrastive learning model
- **DINOv2 ViT-B/14/L/14**: Facebook's self-supervised vision model

### 🔍 Enhanced Search Features
- **Image Cropping**: Optional ROI selection for focused search
- **Flexible Parameters**: Adjustable number of results and metadata display
- **Category Selection**: Search in disease or pest datasets
- **Real-time Results**: Instant similarity scores and visual feedback

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Navigate the Interface**:
   - Use the sidebar to switch between pages
   - Select your preferred embedding model
   - Upload images and explore the dataset

## 📁 Dataset Structure

The application expects the following dataset structure:
```
dataset/
├── images/
│   ├── original_dataset/
│   │   ├── diseases/
│   │   │   ├── 1_Pytophthora_Patch_Canker_Root_Rot/
│   │   │   ├── 2_Pytophthora_Fruit_Rot/
│   │   │   └── ...
│   │   └── pests/
│   └── processed_dataset/
```

## 🛠️ Configuration

Update `src/config.py` to configure:
- Qdrant vector database URI
- Dataset directory paths
- Default embedding model
- Device settings (CPU/GPU)

## 🎯 Usage Guide

### Disease Report
1. Navigate to "Disease Report" page
2. View file counts for each disease category
3. Explore interactive charts and statistics
4. Refresh counts as needed

### Pest Report
1. Navigate to "Pest Report" page
2. View file counts for each pest category
3. Explore interactive charts and statistics
4. Refresh counts as needed

### Image Search
1. Navigate to "Image Search" page
2. Select your preferred embedding model
3. Choose search category (disease/pest)
4. Upload a query image
5. Optionally crop the image for focused search
6. Click "Search Similar Images"
7. View results with similarity scores

## 🎨 UI Components

- **Simple Headers**: Clean page titles using Streamlit's default styling
- **Metric Displays**: Summary statistics using Streamlit's metric components
- **Upload Areas**: Standard file upload components
- **Result Grids**: Simple image result displays
- **Progress Bars**: Visual similarity score indicators
- **Interactive Charts**: Plotly-powered data visualizations with color

## 🔧 Technical Stack

- **Frontend**: Streamlit with custom CSS
- **AI Models**: Hugging Face Transformers
- **Vector Database**: Qdrant
- **Data Visualization**: Plotly
- **Image Processing**: Pillow
- **Deep Learning**: PyTorch

## 📈 Performance

- **Lazy Loading**: Models loaded only when needed
- **Caching**: Efficient result caching
- **Responsive Design**: Fast UI interactions
- **Memory Efficient**: Optimized image processing

## 🤝 Contributing

Feel free to contribute to enhance the UI/UX or add new features!

## 📄 License

This project is open source and available under the MIT License.
