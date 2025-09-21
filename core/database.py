import sqlite3
import json
from typing import List, Dict

from core.models import JobDescription, Resume, EvaluationResult


class DatabaseManager:
    """Manage database operations"""

    def __init__(self, db_path: str = "data/resume_system.db"):
        self.db_path = db_path
        self.init_database()
        self._create_indexes()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Job Descriptions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                job_id TEXT PRIMARY KEY,
                company TEXT,
                role_title TEXT,
                required_skills TEXT,
                preferred_skills TEXT,
                experience_required TEXT,
                education TEXT,
                location TEXT,
                description_text TEXT,
                created_at TIMESTAMP
            )
        """)

        # Resumes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                resume_id TEXT PRIMARY KEY,
                candidate_name TEXT,
                email TEXT,
                phone TEXT,
                skills TEXT,
                experience TEXT,
                education TEXT,
                projects TEXT,
                certifications TEXT,
                resume_text TEXT,
                file_path TEXT,
                uploaded_at TIMESTAMP
            )
        """)

        # Evaluations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                resume_id TEXT,
                job_id TEXT,
                relevance_score REAL,
                hard_match_score REAL,
                soft_match_score REAL,
                missing_skills TEXT,
                matching_skills TEXT,
                verdict TEXT,
                suggestions TEXT,
                evaluated_at TIMESTAMP,
                FOREIGN KEY (resume_id) REFERENCES resumes (resume_id),
                FOREIGN KEY (job_id) REFERENCES job_descriptions (job_id)
            )
        """)

        conn.commit()
        conn.close()

    def save_job_description(self, jd: JobDescription):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO job_descriptions 
            (job_id, company, role_title, required_skills, preferred_skills,
             experience_required, education, location, description_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            jd.job_id, jd.company, jd.role_title,
            json.dumps(jd.required_skills), json.dumps(jd.preferred_skills),
            jd.experience_required, json.dumps(jd.education),
            jd.location, jd.description_text, jd.created_at
        ))
        conn.commit()
        conn.close()

    def save_resume(self, resume: Resume):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO resumes
            (resume_id, candidate_name, email, phone, skills, experience,
             education, projects, certifications, resume_text, file_path, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            resume.resume_id, resume.candidate_name, resume.email, resume.phone,
            json.dumps(resume.skills), resume.experience,
            json.dumps(resume.education), json.dumps(resume.projects),
            json.dumps(resume.certifications), resume.resume_text,
            resume.file_path, resume.uploaded_at
        ))
        conn.commit()
        conn.close()

    def save_evaluation(self, evaluation: EvaluationResult):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO evaluations
            (evaluation_id, resume_id, job_id, relevance_score, hard_match_score,
             soft_match_score, missing_skills, matching_skills, verdict, suggestions, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation.evaluation_id, evaluation.resume_id, evaluation.job_id,
            evaluation.relevance_score, evaluation.hard_match_score,
            evaluation.soft_match_score, json.dumps(evaluation.missing_skills),
            json.dumps(evaluation.matching_skills), evaluation.verdict.value,
            json.dumps(evaluation.suggestions), evaluation.evaluated_at
        ))
        conn.commit()
        conn.close()

    def get_evaluations_by_job(self, job_id: str, min_score: float = 0) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT e.*, r.candidate_name, r.email, r.phone
            FROM evaluations e
            JOIN resumes r ON e.resume_id = r.resume_id
            WHERE e.job_id = ? AND e.relevance_score >= ?
            ORDER BY e.relevance_score DESC
        """, (job_id, min_score))

        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            result['missing_skills'] = json.loads(result['missing_skills'])
            result['matching_skills'] = json.loads(result['matching_skills'])
            result['suggestions'] = json.loads(result['suggestions'])
            results.append(result)

        conn.close()
        return results

    def _create_indexes(self):
        """Create database indexes for better performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_job_id ON evaluations(job_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_resume_id ON evaluations(resume_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_score ON evaluations(relevance_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_verdict ON evaluations(verdict)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_company ON job_descriptions(company)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resumes_name ON resumes(candidate_name)")
        
        conn.commit()
        conn.close()

    def get_all_jobs(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM job_descriptions ORDER BY created_at DESC")

        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            result['required_skills'] = json.loads(result['required_skills'])
            result['preferred_skills'] = json.loads(result['preferred_skills'])
            result['education'] = json.loads(result['education'])
            results.append(result)

        conn.close()
        return results

    def _create_indexes(self):
        """Create database indexes for better performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_job_id ON evaluations(job_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_resume_id ON evaluations(resume_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_score ON evaluations(relevance_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_verdict ON evaluations(verdict)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_company ON job_descriptions(company)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resumes_name ON resumes(candidate_name)")
        
        conn.commit()
        conn.close()
