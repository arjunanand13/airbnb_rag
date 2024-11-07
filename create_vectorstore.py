import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from tqdm import tqdm

def create_vector_store():
    # Create output directory
    os.makedirs("unified_faiss_index", exist_ok=True)
    
    print("Loading data...")
    # Load individual dataframes
    calendars_df = pd.read_csv('Dataset1/calendar.csv')
    listings_df = pd.read_csv('Dataset1/listings.csv')
    reviews_df = pd.read_csv('Dataset1/reviews.csv')
    
    # Merge reviews with listings
    reviews_enriched = reviews_df.merge(
        listings_df[['id', 'name', 'description', 'neighbourhood', 'property_type', 
                    'room_type', 'price', 'amenities']],
        left_on='listing_id',
        right_on='id',
        how='left'
    )
    
    print("Creating text documents...")
    documents = []
    
    # Process listings with progress bar
    print("Processing listings...")
    for _, listing in tqdm(listings_df.iterrows(), total=len(listings_df)):
        listing_text = f"""
        LISTING INFORMATION:
        Property: {listing['name']}
        Type: {listing['property_type']} - {listing['room_type']}
        Location: {listing['neighbourhood']}
        Price: ${listing['price']}
        Description: {listing['description']}
        Amenities: {listing['amenities']}
        Host Information:
        - Name: {listing['host_name']}
        - Since: {listing['host_since']}
        - Location: {listing['host_location']}
        - About: {listing['host_about']}
        """
        documents.append(listing_text)
    
    # Process reviews with progress bar
    print("Processing reviews...")
    for _, review in tqdm(reviews_enriched.iterrows(), total=len(reviews_enriched)):
        review_text = f"""
        REVIEW INFORMATION:
        Property: {review['name']}
        Type: {review['property_type']} - {review['room_type']}
        Location: {review['neighbourhood']}
        Price: ${review['price']}
        Review Date: {review['date']}
        Reviewer: {review['reviewer_name']}
        Comments: {review['comments']}
        """
        documents.append(review_text)
    
    print(f"Created {len(documents)} documents")
    
    # Save raw documents
    print("Saving documents...")
    with open("unified_faiss_index/documents.json", "w", encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    # Create embeddings
    print("Creating embeddings...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Process embeddings in batches to manage memory
    batch_size = 128
    embeddings = []
    
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings)
    
    # Create and save FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]  # Should be 384 for MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print("Saving FAISS index...")
    faiss.write_index(index, "unified_faiss_index/index.faiss")
    
    # Save metadata
    metadata = {
        "total_documents": len(documents),
        "embedding_dimension": dimension,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": batch_size
    }
    
    with open("unified_faiss_index/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Vector store created successfully!")
    print(f"Total documents: {len(documents)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Index size: {os.path.getsize('unified_faiss_index/index.faiss') / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    try:
        create_vector_store()
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")