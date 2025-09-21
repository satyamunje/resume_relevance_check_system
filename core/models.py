from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List

class Verdict(Enum):
    HIGH = "High Suitability"
    MEDIUM = "Medium Suitability"
    LOW = "Low Suitability"

@dataclass
class JobDescription:
    """Job Description data model"""
    job_id: str
    company: str
    role_title: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_required: str
    education: List[str]
    location: str
    description_text: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Resume:
    """Resume data model"""
    resume_id: str
    candidate_name: str
    email: str
    phone: str
    skills: List[str]
    experience: str
    education: List[str]
    projects: List[str]
    certifications: List[str]
    resume_text: str
    file_path: str
    uploaded_at: datetime = field(default_factory=datetime.now)

@dataclass
class EvaluationResult:
    """Evaluation result data model"""
    evaluation_id: str
    resume_id: str
    job_id: str
    relevance_score: float
    hard_match_score: float
    soft_match_score: float
    missing_skills: List[str]
    matching_skills: List[str]
    verdict: Verdict
    suggestions: List[str]
    evaluated_at: datetime = field(default_factory=datetime.now)
