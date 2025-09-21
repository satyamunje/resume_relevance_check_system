import os
import pytest
from core.parsers import ResumeParser, JobDescriptionParser

def test_resume_parser_pdf(tmp_path):
    # Create a dummy PDF
    pdf_path = tmp_path / "resume.pdf"
    pdf_path.write_text("Dummy resume text with Python and SQL skills.")

    parser = ResumeParser()
    resume = parser.parse(str(pdf_path))

    assert resume.resume_id is not None
    assert "Python" in resume.resume_text or "SQL" in resume.resume_text

def test_job_description_parser():
    jd_text = "We need a Software Engineer with Python, SQL and Machine Learning experience."
    parser = JobDescriptionParser()
    jd = parser.parse(jd_text, company="TestCo", location="Remote")

    assert jd.job_id is not None
    assert "Python" in jd.required_skills
    assert jd.company == "TestCo"
