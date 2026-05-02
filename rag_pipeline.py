import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from dotenv import load_dotenv

load_dotenv()

class MedRAGPipeline:
    def __init__(self, 
                 index_path='wikipedia.index', 
                 meta_path='wikipedia_meta.json',
                 embed_model_name='all-MiniLM-L6-v2',
                 cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 gemini_model='gemini-3-flash'):
        
        self.index_path = index_path
        self.meta_path = meta_path
        self.gemini_model = gemini_model
        
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        print("Loading Embedding Model...")
        self.embed_model = SentenceTransformer(embed_model_name)
        
        print("Loading Cross-Encoder Model...")
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        
        print("Initializing Gemini Client...")
        # genai.Client() tự động tìm biến môi trường GEMINI_API_KEY
        self.client = genai.Client() 

    def query_rewrite(self, original_query):
        prompt = f"""You are a medical assistant. Your task is to expand and clarify the following user query to improve information retrieval from a medical Wikipedia database.
Provide only the rewritten query, without any explanations.
Original Query: {original_query}
Rewritten Query:"""
        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=prompt
        )
        rewritten_query = response.text.strip()
        print(f"[*] Query Rewrite:\nOriginal: {original_query}\nRewritten: {rewritten_query}\n")
        return rewritten_query

    def ann_search(self, query, top_n=50):
        # Mã hóa query với normalize_embeddings=True tương tự lúc build index
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True)
        
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_n)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append(self.metadata[idx])
                
        print(f"[*] ANN Search: Found {len(results)} initial candidates.")
        return results

    def filter_metadata(self, candidates, min_length=50):
        filtered = []
        for cand in candidates:
            # Lọc bỏ các chunk quá ngắn có thể không mang nhiều ý nghĩa
            if len(cand.get('contents', '')) >= min_length:
                filtered.append(cand)
        print(f"[*] Metadata Filter: Kept {len(filtered)} out of {len(candidates)} candidates.")
        return filtered

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []
            
        pairs = [[query, cand['contents']] for cand in candidates]
        scores = self.cross_encoder.predict(pairs)
        
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = float(scores[i])
            
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        top_results = reranked[:top_k]
        
        print(f"[*] Rerank: Selected Top-{top_k} from {len(candidates)} candidates.")
        return top_results

    def generate_answer(self, original_query, top_k_results):
        context = "\n\n---\n\n".join([f"Source: {res.get('title')}\nContent: {res.get('contents')}" for res in top_k_results])
        
        prompt = f"""You are a helpful medical assistant. Use the following retrieved context from Wikipedia to answer the user's query. 
If the answer is not contained in the context, say "I don't have enough information to answer that based on the provided context."

Context:
{context}

Query: {original_query}

Answer:"""
        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=prompt
        )
        return response.text

    def run(self, query):
        print(f"\n========== RAG PIPELINE START ==========")
        # 1. Query rewrite
        rewritten_query = self.query_rewrite(query)
        
        # 2. ANN search (top 50)
        candidates = self.ann_search(rewritten_query, top_n=50)
        
        # 3. Filter metadata
        filtered_candidates = self.filter_metadata(candidates, min_length=50)
        
        # 4. Rerank
        top_results = self.rerank(rewritten_query, filtered_candidates, top_k=5)
        
        # 5. Top-k -> LLM
        print(f"[*] Generating Answer with LLM...")
        answer = self.generate_answer(query, top_results)
        
        print(f"\n========== FINAL ANSWER ==========\n{answer}\n==================================\n")
        return answer

if __name__ == "__main__":
    # Đã tự động load từ file .env
    try:
        pipeline = MedRAGPipeline()
        test_query = "What are the common symptoms of Type 2 Diabetes?"
        pipeline.run(test_query)
    except Exception as e:
        print(f"Error during RAG pipeline execution: {e}")
