import os
import json
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def build_index(num_samples=50000, model_name='all-MiniLM-L6-v2', index_path='wikipedia_50k.index', meta_path='wikipedia_meta_50k.json', npy_path='embeddings_50k.npy'):
    print(f"Loading top {num_samples} from MedRAG/wikipedia...")
    # Dùng streaming=True để tránh tải toàn bộ 29 triệu dòng về RAM
    dataset = load_dataset("MedRAG/wikipedia", split="train", streaming=True)
    
    docs = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        docs.append(item)
        if (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1} documents...")

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Encoding documents...")
    texts = [doc['contents'] for doc in docs]
    
    # normalize_embeddings=True để dùng Cosine Similarity (thông qua Inner Product)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    # Dùng IndexFlatIP cho Inner Product
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving numpy embeddings to {npy_path}...")
    np.save(npy_path, embeddings)
    
    print(f"Saving metadata to {meta_path}...")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
        
    print("Indexing completed successfully!")

if __name__ == "__main__":
    build_index()
