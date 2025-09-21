import os
import json
from typing import List, Dict, Optional, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from core.models import Resume, JobDescription


class LLMAnalyzer:
    """Advanced analysis using Large Language Models"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.3,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

        self.analysis_prompt = PromptTemplate(
            input_variables=["resume_text", "jd_text", "initial_score", "missing_skills", "matching_skills"],
            template="""
            You are an expert recruiter and career counselor analyzing a resume against a job description.

            Resume:
            {resume_text}

            Job Description:
            {jd_text}

            Initial Relevance Score: {initial_score}/100
            Missing Skills: {missing_skills}
            Matching Skills: {matching_skills}

            Provide a comprehensive analysis including:
            1. Key strengths of the candidate for this role (be specific about technical and soft skills)
            2. Major gaps that need to be addressed (prioritize by importance)
            3. Specific, actionable recommendations for improvement (include learning resources if possible)
            4. Overall fit assessment with detailed reasoning
            5. Career development suggestions for long-term growth

            Be encouraging but honest. Focus on actionable advice that can help the candidate improve.

            Format your response as a JSON with keys: strengths, gaps, recommendations, fit_assessment, career_advice
            """
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)

    def analyze(self, resume: Resume, jd: JobDescription, initial_score: float, 
                missing_skills: List[str] = None, matching_skills: List[str] = None) -> Dict:
        """Perform deep analysis using LLM"""
        try:
            response = self.chain.run(
                resume_text=resume.resume_text[:2000],  # Limit context
                jd_text=jd.description_text[:2000],
                initial_score=initial_score,
                missing_skills=', '.join(missing_skills or []),
                matching_skills=', '.join(matching_skills or [])
            )
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"LLM JSON parsing failed: {e}")
            return self._fallback_analysis(resume, jd, initial_score, missing_skills, matching_skills)
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._fallback_analysis(resume, jd, initial_score, missing_skills, matching_skills)

    def _fallback_analysis(self, resume: Resume, jd: JobDescription, initial_score: float,
                          missing_skills: List[str] = None, matching_skills: List[str] = None) -> Dict:
        """Fallback analysis when LLM fails"""
        return {
            "strengths": [f"Strong technical skills: {', '.join(matching_skills or [])[:100]}"],
            "gaps": [f"Missing skills: {', '.join(missing_skills or [])[:100]}"],
            "recommendations": [
                "Focus on acquiring missing technical skills",
                "Add relevant projects to demonstrate expertise",
                "Consider obtaining industry certifications"
            ],
            "fit_assessment": f"Score: {initial_score}/100 - {'High' if initial_score >= 75 else 'Medium' if initial_score >= 50 else 'Low'} suitability",
            "career_advice": "Continue building technical skills and gaining relevant experience"
        }
