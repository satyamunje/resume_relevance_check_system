import os
import re
import hashlib
from typing import List
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument
import docx2txt
import spacy

from core.models import Resume, JobDescription

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class DocumentParser:
    """Base class for document parsing"""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""

        # Method 1: PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except:
            pass

        # Method 2: pdfplumber
        if not text.strip():
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
            except:
                pass

        return text.strip()

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            doc = DocxDocument(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except:
            try:
                text = docx2txt.process(file_path)
            except:
                text = ""

        return text.strip()


class ResumeParser(DocumentParser):
    """Parse resume and extract structured information"""

    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')

    def parse(self, file_path: str) -> Resume:
        """Parse resume file and extract structured data"""
        if file_path.endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = self.extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file format")

        resume_data = {
            'resume_id': self._generate_id(text),
            'candidate_name': self._extract_name(text),
            'email': self._extract_email(text),
            'phone': self._extract_phone(text),
            'skills': self._extract_skills(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'projects': self._extract_projects(text),
            'certifications': self._extract_certifications(text),
            'resume_text': text,
            'file_path': file_path
        }

        return Resume(**resume_data)

    def _generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:10]

    def _extract_name(self, text: str) -> str:
        doc = nlp(text[:500])  # Usually name is at the beginning
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return "Unknown"

    def _extract_email(self, text: str) -> str:
        match = self.email_pattern.search(text)
        return match.group(0) if match else ""

    def _extract_phone(self, text: str) -> str:
        match = self.phone_pattern.search(text)
        return match.group(0) if match else ""

    def _extract_skills(self, text: str) -> List[str]:
        tech_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'mongodb', 'react',
            'nodejs', 'django', 'flask', 'tensorflow', 'pytorch', 'sklearn',
            'pandas', 'numpy', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
            'git', 'linux', 'agile', 'scrum', 'machine learning', 'deep learning',
            'nlp', 'computer vision', 'data science', 'data analysis'
        ]
        text_lower = text.lower()
        return [skill for skill in tech_skills if skill in text_lower]

    def _extract_experience(self, text: str) -> str:
        exp_pattern = re.compile(r'(\d+[\+]?\s*years?\s*(?:of\s*)?experience)', re.IGNORECASE)
        match = exp_pattern.search(text)
        return match.group(0) if match else "0 years"

    def _extract_education(self, text: str) -> List[str]:
        education_keywords = [
            'bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'bsc', 'msc', 'mba',
            'degree', 'diploma'
        ]
        education = []
        for line in text.split('\n'):
            if any(keyword in line.lower() for keyword in education_keywords):
                education.append(line.strip())
        return education[:3]

    def _extract_projects(self, text: str) -> List[str]:
        project_section = re.search(
            r'projects?[:\s]+(.*?)(?:experience|education|skills|certification)',
            text, re.IGNORECASE | re.DOTALL
        )
        if project_section:
            projects_text = project_section.group(1)
            projects = re.split(r'[\n•·▪▫◦‣⁃]|(?:\d+\.)', projects_text)
            return [p.strip() for p in projects if len(p.strip()) > 20][:5]
        return []

    def _extract_certifications(self, text: str) -> List[str]:
        cert_keywords = ['certification', 'certified', 'certificate']
        certifications = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in cert_keywords):
                certifications.append(line.strip())
                if i + 1 < len(lines):
                    certifications.append(lines[i + 1].strip())
        return certifications[:5]


class JobDescriptionParser(DocumentParser):
    """Parse job description and extract requirements"""

    def parse(self, jd_text: str, company: str = "Unknown", location: str = "Unknown") -> JobDescription:
        jd_data = {
            'job_id': hashlib.md5(jd_text.encode()).hexdigest()[:10],
            'company': company,
            'role_title': self._extract_role_title(jd_text),
            'required_skills': self._extract_required_skills(jd_text),
            'preferred_skills': self._extract_preferred_skills(jd_text),
            'experience_required': self._extract_experience_requirement(jd_text),
            'education': self._extract_education_requirement(jd_text),
            'location': location,
            'description_text': jd_text
        }
        return JobDescription(**jd_data)

    def _extract_role_title(self, text: str) -> str:
        for line in text.split('\n')[:5]:
            if 5 < len(line) < 100:
                return line.strip()
        return "Software Engineer"

    def _extract_required_skills(self, text: str) -> List[str]:
        required_section = re.search(
            r'(?:required|must.?have|mandatory)[:\s]+skills?[:\s]+(.*?)(?:preferred|good.?to.?have|responsibilities)',
            text, re.IGNORECASE | re.DOTALL
        )
        if required_section:
            return self._parse_skills_list(required_section.group(1))
        return self._extract_general_skills(text)[:5]

    def _extract_preferred_skills(self, text: str) -> List[str]:
        preferred_section = re.search(
            r'(?:preferred|good.?to.?have|nice.?to.?have|desired)[:\s]+skills?[:\s]+(.*?)(?:responsibilities|qualifications|$)',
            text, re.IGNORECASE | re.DOTALL
        )
        if preferred_section:
            return self._parse_skills_list(preferred_section.group(1))
        return []

    def _parse_skills_list(self, skills_text: str) -> List[str]:
        skills = re.split(r'[,\n•·▪▫◦‣⁃;]|(?:\d+\.)', skills_text)
        return [skill.strip().lower() for skill in skills if 5 < len(skill.strip()) < 50][:10]

    def _extract_general_skills(self, text: str) -> List[str]:
        tech_terms = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'sql',
            'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'spring',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'tensorflow',
            'pytorch', 'scikit-learn', 'pandas', 'numpy', 'mongodb', 'postgresql'
        ]
        return [term for term in tech_terms if term in text.lower()]

    def _extract_experience_requirement(self, text: str) -> str:
        exp_pattern = re.compile(r'(\d+[\+\-]?\d*\s*years?\s*(?:of\s*)?experience)', re.IGNORECASE)
        match = exp_pattern.search(text)
        return match.group(0) if match else "0-2 years"

    def _extract_education_requirement(self, text: str) -> List[str]:
        education_keywords = [
            'bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'b.e.', 'm.e.',
            'bsc', 'msc', 'mba', 'degree'
        ]
        requirements = []
        for keyword in education_keywords:
            if keyword in text.lower():
                for sentence in text.split('.'):
                    if keyword in sentence.lower():
                        requirements.append(sentence.strip())
                        break
        return requirements[:3]
