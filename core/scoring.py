import re
from typing import Dict, Tuple, List
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.models import Resume, JobDescription, EvaluationResult, Verdict


class ScoringEngine:
    """Hybrid scoring engine combining hard and soft matching"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

    def calculate_relevance_score(self, resume: Resume, jd: JobDescription) -> EvaluationResult:
        """Calculate overall relevance score"""

        # Hard matching (40% weight)
        hard_score, skill_analysis = self._hard_match_scoring(resume, jd)

        # Soft matching (60% weight)
        soft_score = self._soft_match_scoring(resume, jd)

        # Combined score
        final_score = (hard_score * 0.4) + (soft_score * 0.6)

        # Determine verdict
        verdict = self._determine_verdict(final_score)

        # Generate suggestions
        suggestions = self._generate_suggestions(resume, jd, skill_analysis, final_score)

        return EvaluationResult(
            evaluation_id=f"{resume.resume_id}_{jd.job_id}",
            resume_id=resume.resume_id,
            job_id=jd.job_id,
            relevance_score=round(final_score, 2),
            hard_match_score=round(hard_score, 2),
            soft_match_score=round(soft_score, 2),
            missing_skills=skill_analysis['missing'],
            matching_skills=skill_analysis['matching'],
            verdict=verdict,
            suggestions=suggestions
        )

    def _hard_match_scoring(self, resume: Resume, jd: JobDescription) -> Tuple[float, Dict]:
        """Hard matching based on keywords and exact matches"""

        score = 0
        max_score = 100
        matching_skills, missing_skills = [], []

        # Skills matching (40 points)
        for skill in jd.required_skills:
            if any(fuzz.partial_ratio(skill.lower(), res_skill.lower()) > 80
                   for res_skill in resume.skills):
                matching_skills.append(skill)
                score += 30 / len(jd.required_skills) if jd.required_skills else 0
            else:
                missing_skills.append(skill)

        for skill in jd.preferred_skills:
            if any(fuzz.partial_ratio(skill.lower(), res_skill.lower()) > 80
                   for res_skill in resume.skills):
                matching_skills.append(skill)
                score += 10 / len(jd.preferred_skills) if jd.preferred_skills else 0

        # Experience matching (20 points)
        jd_exp = self._parse_experience_years(jd.experience_required)
        resume_exp = self._parse_experience_years(resume.experience)
        if resume_exp >= jd_exp:
            score += 20
        elif resume_exp >= jd_exp * 0.7:
            score += 10

        # Education matching (20 points)
        if self._match_education(resume.education, jd.education):
            score += 20

        # Projects relevance (10 points)
        if resume.projects:
            project_text = ' '.join(resume.projects)
            skill_text = ' '.join(jd.required_skills + jd.preferred_skills)
            if self._text_similarity(project_text, skill_text) > 0.3:
                score += 10

        # Certifications (10 points)
        if resume.certifications:
            cert_text = ' '.join(resume.certifications)
            for skill in jd.required_skills + jd.preferred_skills:
                if skill.lower() in cert_text.lower():
                    score += 5
                    if score >= max_score:
                        break

        return min(score, max_score), {
            'matching': matching_skills,
            'missing': missing_skills
        }

    def _soft_match_scoring(self, resume: Resume, jd: JobDescription) -> float:
        """Semantic matching using embeddings"""

        resume_embedding = self.embedding_model.encode(resume.resume_text)
        jd_embedding = self.embedding_model.encode(jd.description_text)

        similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
        score = similarity * 100

        bm25_score = self._calculate_bm25_score(resume.resume_text, jd.description_text)

        return min((score * 0.7) + (bm25_score * 0.3), 100)

    def _calculate_bm25_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate BM25 relevance score"""

        resume_tokens = word_tokenize(resume_text.lower())
        jd_tokens = word_tokenize(jd_text.lower())

        stop_words = set(stopwords.words('english'))
        resume_tokens = [w for w in resume_tokens if w not in stop_words and w.isalnum()]
        jd_tokens = [w for w in jd_tokens if w not in stop_words and w.isalnum()]

        bm25 = BM25Okapi([resume_tokens])
        score = bm25.get_scores(jd_tokens)[0]

        max_possible = bm25.get_scores(resume_tokens)[0]
        return (score / max_possible) * 100 if max_possible > 0 else 0

    def _parse_experience_years(self, exp_string: str) -> float:
        numbers = re.findall(r'\d+', exp_string)
        return float(numbers[0]) if numbers else 0

    def _match_education(self, resume_edu: List[str], jd_edu: List[str]) -> bool:
        if not jd_edu:
            return True
        resume_text = ' '.join(resume_edu).lower()
        jd_text = ' '.join(jd_edu).lower()
        return any(k in resume_text and k in jd_text for k in
                   ['bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'b.e.', 'm.e.'])

    def _text_similarity(self, text1: str, text2: str) -> float:
        try:
            vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
            return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            return 0

    def _determine_verdict(self, score: float) -> Verdict:
        if score >= 75:
            return Verdict.HIGH
        elif score >= 50:
            return Verdict.MEDIUM
        return Verdict.LOW

    def _generate_suggestions(self, resume: Resume, jd: JobDescription,
                              skill_analysis: Dict, score: float) -> List[str]:
        suggestions = []
        if skill_analysis['missing']:
            suggestions.append(
                f"Consider acquiring these missing skills: {', '.join(skill_analysis['missing'][:5])}"
            )

        jd_exp = self._parse_experience_years(jd.experience_required)
        resume_exp = self._parse_experience_years(resume.experience)
        if resume_exp < jd_exp:
            suggestions.append(
                f"Gain more experience. Required: {jd_exp} years, You have: {resume_exp} years"
            )

        if not resume.projects or len(resume.projects) < 2:
            suggestions.append("Add more relevant projects to demonstrate practical skills")

        if not resume.certifications:
            suggestions.append("Consider obtaining relevant certifications in your domain")

        if score < 50:
            suggestions += [
                "Your profile needs significant improvement to match this role",
                "Focus on building core skills mentioned in the job description"
            ]
        elif score < 75:
            suggestions.append("Your profile is moderately suitable. Focus on filling skill gaps")
        else:
            suggestions.append("Your profile is highly suitable. Minor improvements can make it perfect")

        return suggestions[:5]
