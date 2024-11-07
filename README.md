# airbnb_rag
RAG pipeline built with SQL preprocessing for AIRBNB dataset

## Overview
This project contains a Python-based Airbnb property assistant chatbot leveraging Llama's language generation capabilities and a FAISS-powered vector store for efficient document retrieval.

The chatbot provides responses to user queries related to Airbnb properties, such as property details, reviews, pricing, and more, using context retrieved from a pre-built vector store.

## Quick Start Guide

### Prerequisites
- Python 3.8+
- Virtual environment setup (recommended)
- Required libraries installed via `requirements.txt`

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Chatbot
1. **Run the `main.py` script**:
   ```bash
   python main.py
   ```
   This will start the Gradio interface, accessible at `http://0.0.0.0:7860`.

### Creating the Vector Store
To create the FAISS vector store from the Airbnb datasets:

1. **Place datasets**:
   Ensure `calendar.csv`, `listings.csv`, and `reviews.csv` are in the `Dataset1` folder.

2. **Run the vector store creation script**:
   ```bash
   python create_vector_store.py
   ```
   This will:
   - Load and process the datasets
   - Generate text documents from the data
   - Create embeddings and build the FAISS index
   - Save the index and documents in the `unified_faiss_index` directory

### Key Files
- **`chatbot.py`**: Script to run the Gradio interface for the chatbot.
- **`create_vector_store.py`**: Script for creating the FAISS index from Airbnb datasets.
- **`unified_faiss_index`**:
  - `index.faiss`: The FAISS index file.
  - `documents.json`: The processed documents.
  - `metadata.json`: Metadata for the vector store.

## Project Structure
```plaintext
├── Dataset1
│   ├── calendar.csv
│   ├── listings.csv
│   └── reviews.csv
├── unified_faiss_index
│   ├── index.faiss
│   ├── documents.json
│   └── metadata.json
├── chatbot.py
├── create_vector_store.py
├── requirements.txt
└── README.md
```

## Example Queries
- "What properties are available in downtown?"
- "Tell me about properties with good reviews"
- "Show me properties under $100 per night"

## Notes
- The chatbot supports queries related to property information, reviews, pricing, and amenities.
- Ensure that your Hugging Face authentication token is correctly configured.

## Troubleshooting
- **Error loading FAISS index**:
  Ensure the `index.faiss` and `documents.json` files are present in the `unified_faiss_index` directory.
- **Out-of-memory errors**:
  Consider reducing the batch size in the `create_vector_store.py` script.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

