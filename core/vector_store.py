import os
import numpy as np
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import faiss
from sentence_transformers import SentenceTransformer

from core.models import Resume, JobDescription


class VectorStore:
    """Vector store for semantic search and similarity matching"""
    
    def __init__(self, persist_directory: str = "data/vector_store"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize collections
        self.resume_collection = self.chroma_client.get_or_create_collection(
            name="resumes",
            metadata={"description": "Resume embeddings for semantic search"}
        )
        
        self.jd_collection = self.chroma_client.get_or_create_collection(
            name="job_descriptions", 
            metadata={"description": "Job description embeddings"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize FAISS index for fast similarity search
        self.faiss_index = None
        self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            # Load existing FAISS index if available
            faiss_path = os.path.join(self.persist_directory, "faiss_index.bin")
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            else:
                # Create new FAISS index (384 dimensions for all-MiniLM-L6-v2)
                self.faiss_index = faiss.IndexFlatIP(384)
        except Exception as e:
            print(f"FAISS initialization failed: {e}")
            self.faiss_index = None
    
    def add_resume(self, resume: Resume) -> str:
        """Add resume to vector store"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(resume.resume_text).tolist()
            
            # Add to ChromaDB
            self.resume_collection.add(
                ids=[resume.resume_id],
                embeddings=[embedding],
                documents=[resume.resume_text],
                metadatas=[{
                    "candidate_name": resume.candidate_name,
                    "email": resume.email,
                    "skills": ",".join(resume.skills),
                    "experience": resume.experience,
                    "file_path": resume.file_path
                }]
            )
            
            # Add to FAISS index
            if self.faiss_index is not None:
                embedding_array = np.array([embedding]).astype('float32')
                self.faiss_index.add(embedding_array)
                self._save_faiss_index()
            
            return resume.resume_id
        except Exception as e:
            print(f"Error adding resume to vector store: {e}")
            return None
    
    def add_job_description(self, jd: JobDescription) -> str:
        """Add job description to vector store"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(jd.description_text).tolist()
            
            # Add to ChromaDB
            self.jd_collection.add(
                ids=[jd.job_id],
                embeddings=[embedding],
                documents=[jd.description_text],
                metadatas=[{
                    "company": jd.company,
                    "role_title": jd.role_title,
                    "location": jd.location,
                    "required_skills": ",".join(jd.required_skills),
                    "preferred_skills": ",".join(jd.preferred_skills)
                }]
            )
            
            return jd.job_id
        except Exception as e:
            print(f"Error adding job description to vector store: {e}")
            return None
    
    def find_similar_resumes(self, jd: JobDescription, top_k: int = 10) -> List[Dict]:
        """Find most similar resumes to a job description"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(jd.description_text)
            
            # Search in ChromaDB
            results = self.resume_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            similar_resumes = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                similar_resumes.append({
                    'resume_id': results['ids'][0][i],
                    'candidate_name': metadata.get('candidate_name', 'Unknown'),
                    'email': metadata.get('email', ''),
                    'similarity_score': round(similarity_score * 100, 2),
                    'resume_text': doc,
                    'skills': metadata.get('skills', '').split(',') if metadata.get('skills') else []
                })
            
            return similar_resumes
        except Exception as e:
            print(f"Error finding similar resumes: {e}")
            return []
    
    def find_similar_jobs(self, resume: Resume, top_k: int = 5) -> List[Dict]:
        """Find most similar job descriptions to a resume"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(resume.resume_text)
            
            # Search in ChromaDB
            results = self.jd_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            similar_jobs = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                similar_jobs.append({
                    'job_id': results['ids'][0][i],
                    'company': metadata.get('company', 'Unknown'),
                    'role_title': metadata.get('role_title', 'Unknown'),
                    'location': metadata.get('location', 'Unknown'),
                    'similarity_score': round(similarity_score * 100, 2),
                    'description_text': doc,
                    'required_skills': metadata.get('required_skills', '').split(',') if metadata.get('required_skills') else []
                })
            
            return similar_jobs
        except Exception as e:
            print(f"Error finding similar jobs: {e}")
            return []
    
    def semantic_search(self, query: str, collection_type: str = "resumes", top_k: int = 10) -> List[Dict]:
        """Perform semantic search across resumes or job descriptions"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Choose collection
            collection = self.resume_collection if collection_type == "resumes" else self.jd_collection
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity_score = 1 - distance
                search_results.append({
                    'id': results['ids'][0][i],
                    'similarity_score': round(similarity_score * 100, 2),
                    'content': doc,
                    'metadata': metadata
                })
            
            return search_results
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        try:
            if self.faiss_index is not None:
                faiss_path = os.path.join(self.persist_directory, "faiss_index.bin")
                faiss.write_index(self.faiss_index, faiss_path)
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            resume_count = self.resume_collection.count()
            jd_count = self.jd_collection.count()
            
            return {
                "resume_count": resume_count,
                "job_description_count": jd_count,
                "total_embeddings": resume_count + jd_count,
                "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}