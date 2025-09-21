import pytest
from core.system import ResumeRelevanceSystem

def test_system_pipeline(tmp_path):
    system = ResumeRelevanceSystem()

    jd_text = "Looking for Data Scientist with Python and SQL"
    jd = system.process_job_description(jd_text, company="TestCo", location="Remote")

    # Create dummy resume
    resume_path = tmp_path / "resume.txt"
    resume_path.write_text("I am a Data Scientist skilled in Python and SQL.")

    evaluation = system.evaluate_resume(str(resume_path), jd_text, company="TestCo")
    assert evaluation.relevance_score > 0
    assert isinstance(evaluation.suggestions, list)
