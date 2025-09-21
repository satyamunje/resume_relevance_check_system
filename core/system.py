from typing import List, Dict
from functools import lru_cache

from core.parsers import ResumeParser, JobDescriptionParser
from core.scoring import ScoringEngine
from core.database import DatabaseManager
from core.llm_integration import LLMAnalyzer
# from core.vector_store import VectorStore  # Temporarily disabled due to compatibility issues
# from core.langgraph_workflow import ResumeEvaluationWorkflow  # Temporarily disabled due to compatibility issues
from core.models import Resume, JobDescription, EvaluationResult


class ResumeRelevanceSystem:
    """Main system orchestrator"""

    def __init__(self, use_llm: bool = False, llm_api_key: str = None, use_langgraph: bool = False):
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()
        self.scoring_engine = ScoringEngine()
        self.db_manager = self._get_db_manager()

        self.use_llm = use_llm
        self.use_langgraph = use_langgraph

        if use_llm:
            self.llm_analyzer = LLMAnalyzer(api_key=llm_api_key)

        # if use_langgraph and use_llm:
        #     self.langgraph_workflow = ResumeEvaluationWorkflow(api_key=llm_api_key)

    # ✅ cache DB initialization so it only runs once
    @staticmethod
    @lru_cache(maxsize=1)
    def _get_db_manager():
        return DatabaseManager()

    def process_job_description(self, jd_text: str,
                               company: str = "Unknown",
                               location: str = "Unknown") -> JobDescription:
        """Process and save job description"""
        jd = self.jd_parser.parse(jd_text, company, location)
        self.db_manager.save_job_description(jd)
        # self.vector_store.add_job_description(jd)  # disabled
        return jd

    def process_resume(self, file_path: str) -> Resume:
        """Process and save resume"""
        resume = self.resume_parser.parse(file_path)
        self.db_manager.save_resume(resume)
        # self.vector_store.add_resume(resume)  # disabled
        return resume

    def evaluate_resume(self, resume_path: str, jd_text: str,
                        company: str = "Unknown", location: str = "Unknown") -> EvaluationResult:
        """Complete evaluation pipeline"""

        # Parse documents
        resume = self.process_resume(resume_path)
        jd = self.process_job_description(jd_text, company, location)

        # Use scoring engine
        evaluation = self.scoring_engine.calculate_relevance_score(resume, jd)

        # Optional: LLM deep analysis
        if self.use_llm:
            llm_analysis = self.llm_analyzer.analyze(
                resume, jd, evaluation.relevance_score,
                evaluation.missing_skills, evaluation.matching_skills
            )
            if 'recommendations' in llm_analysis:
                evaluation.suggestions.extend(llm_analysis['recommendations'][:3])

        # Save evaluation
        self.db_manager.save_evaluation(evaluation)
        return evaluation

    def batch_evaluate(self, resume_paths: List[str], jd_text: str,
                       company: str = "Unknown", location: str = "Unknown") -> List[EvaluationResult]:
        """Evaluate multiple resumes against a single JD"""

        jd = self.process_job_description(jd_text, company, location)
        results = []

        for resume_path in resume_paths:
            try:
                resume = self.process_resume(resume_path)
                evaluation = self.scoring_engine.calculate_relevance_score(resume, jd)
                self.db_manager.save_evaluation(evaluation)
                results.append(evaluation)
            except Exception as e:
                print(f"Error processing {resume_path}: {e}")
                continue

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def get_shortlist(self, job_id: str, top_n: int = 10, min_score: float = 50) -> List[Dict]:
        """Get top candidates for a job"""
        evaluations = self.db_manager.get_evaluations_by_job(job_id, min_score)
        return evaluations[:top_n]
