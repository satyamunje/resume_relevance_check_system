import os
import pytest
from core.database import DatabaseManager
from core.models import JobDescription, Resume, EvaluationResult, Verdict

def test_database_operations(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(str(db_path))

    jd = JobDescription(
        job_id="j1",
        company="TestCo",
        role_title="Engineer",
        required_skills=["Python"],
        preferred_skills=[],
        experience_required="2 years",
        education=["Bachelors"],
        location="Remote",
        description_text="Job description text"
    )
    db.save_job_description(jd)

    resume = Resume(
        resume_id="r1",
        candidate_name="Alice",
        email="alice@example.com",
        phone="1111111111",
        skills=["Python"],
        experience="2 years",
        education=["Bachelors"],
        projects=[],
        certifications=[],
        resume_text="Python developer",
        file_path="resume.pdf"
    )
    db.save_resume(resume)

    evaluation = EvaluationResult(
        evaluation_id="e1",
        resume_id="r1",
        job_id="j1",
        relevance_score=80,
        hard_match_score=40,
        soft_match_score=40,
        missing_skills=[],
        matching_skills=["Python"],
        verdict=Verdict.HIGH,
        suggestions=["Looks good"]
    )
    db.save_evaluation(evaluation)

    jobs = db.get_all_jobs()
    assert len(jobs) == 1

    evals = db.get_evaluations_by_job("j1")
    assert len(evals) == 1
    assert evals[0]["relevance_score"] == 80
