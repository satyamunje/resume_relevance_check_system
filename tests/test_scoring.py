import pytest
from core.scoring import ScoringEngine
from core.models import Resume, JobDescription

def test_scoring_engine():
    resume = Resume(
        resume_id="r1",
        candidate_name="John Doe",
        email="john@example.com",
        phone="1234567890",
        skills=["Python", "SQL"],
        experience="3 years",
        education=["Bachelors in CS"],
        projects=["Built ML models"],
        certifications=["Python Certification"],
        resume_text="Python and SQL developer with 3 years of experience",
        file_path="resume.pdf"
    )

    jd = JobDescription(
        job_id="j1",
        company="TestCo",
        role_title="Software Engineer",
        required_skills=["Python", "SQL"],
        preferred_skills=["Machine Learning"],
        experience_required="2 years",
        education=["Bachelors"],
        location="Remote",
        description_text="Looking for Python and SQL developer with ML skills"
    )

    engine = ScoringEngine()
    result = engine.calculate_relevance_score(resume, jd)

    assert result.relevance_score > 50
    assert "Python" in result.matching_skills
