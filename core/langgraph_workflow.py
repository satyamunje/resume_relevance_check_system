from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from core.models import Resume, JobDescription, EvaluationResult
from core.vector_store import VectorStore


class WorkflowState(TypedDict):
    """State for the LangGraph workflow"""
    resume: Resume
    job_description: JobDescription
    extracted_skills: List[str]
    missing_skills: List[str]
    matching_skills: List[str]
    hard_score: float
    soft_score: float
    final_score: float
    verdict: str
    suggestions: List[str]
    llm_analysis: Dict[str, Any]
    evaluation_result: EvaluationResult


class ResumeEvaluationWorkflow:
    """LangGraph-based structured workflow for resume evaluation"""
    
    def __init__(self, llm_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            api_key=llm_api_key
        )
        self.vector_store = VectorStore()
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("extract_skills", self._extract_skills_node)
        workflow.add_node("hard_matching", self._hard_matching_node)
        workflow.add_node("semantic_matching", self._semantic_matching_node)
        workflow.add_node("calculate_score", self._calculate_score_node)
        workflow.add_node("generate_verdict", self._generate_verdict_node)
        workflow.add_node("llm_analysis", self._llm_analysis_node)
        workflow.add_node("generate_suggestions", self._generate_suggestions_node)
        workflow.add_node("create_evaluation", self._create_evaluation_node)
        
        # Add edges
        workflow.set_entry_point("extract_skills")
        workflow.add_edge("extract_skills", "hard_matching")
        workflow.add_edge("hard_matching", "semantic_matching")
        workflow.add_edge("semantic_matching", "calculate_score")
        workflow.add_edge("calculate_score", "generate_verdict")
        workflow.add_edge("generate_verdict", "llm_analysis")
        workflow.add_edge("llm_analysis", "generate_suggestions")
        workflow.add_edge("generate_suggestions", "create_evaluation")
        workflow.add_edge("create_evaluation", END)
        
        self.workflow = workflow.compile()
    
    def _extract_skills_node(self, state: WorkflowState) -> WorkflowState:
        """Extract and analyze skills from resume and job description"""
        resume = state["resume"]
        jd = state["job_description"]
        
        # Extract skills using LLM
        skill_extraction_prompt = PromptTemplate(
            input_variables=["resume_text", "jd_text"],
            template="""
            Extract and categorize skills from the resume and job description.
            
            Resume: {resume_text}
            Job Description: {jd_text}
            
            Return a JSON with:
            {{
                "resume_skills": ["skill1", "skill2", ...],
                "required_skills": ["skill1", "skill2", ...],
                "preferred_skills": ["skill1", "skill2", ...]
            }}
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=skill_extraction_prompt)
        
        try:
            response = chain.run(resume_text=resume.resume_text[:1000], jd_text=jd.description_text[:1000])
            import json
            skills_data = json.loads(response)
            
            state["extracted_skills"] = skills_data.get("resume_skills", [])
            state["missing_skills"] = [
                skill for skill in skills_data.get("required_skills", [])
                if skill not in skills_data.get("resume_skills", [])
            ]
            state["matching_skills"] = [
                skill for skill in skills_data.get("resume_skills", [])
                if skill in skills_data.get("required_skills", []) or skill in skills_data.get("preferred_skills", [])
            ]
        except Exception as e:
            print(f"Error in skill extraction: {e}")
            state["extracted_skills"] = resume.skills
            state["missing_skills"] = [skill for skill in jd.required_skills if skill not in resume.skills]
            state["matching_skills"] = [skill for skill in resume.skills if skill in jd.required_skills + jd.preferred_skills]
        
        return state
    
    def _hard_matching_node(self, state: WorkflowState) -> WorkflowState:
        """Perform hard matching based on keywords and skills"""
        resume = state["resume"]
        jd = state["job_description"]
        matching_skills = state["matching_skills"]
        missing_skills = state["missing_skills"]
        
        # Calculate hard score based on skills, experience, education
        score = 0
        max_score = 100
        
        # Skills matching (40 points)
        if jd.required_skills:
            skill_score = (len(matching_skills) / len(jd.required_skills)) * 40
            score += min(skill_score, 40)
        
        # Experience matching (20 points)
        import re
        jd_exp = float(re.findall(r'\d+', jd.experience_required)[0]) if re.findall(r'\d+', jd.experience_required) else 0
        resume_exp = float(re.findall(r'\d+', resume.experience)[0]) if re.findall(r'\d+', resume.experience) else 0
        
        if resume_exp >= jd_exp:
            score += 20
        elif resume_exp >= jd_exp * 0.7:
            score += 10
        
        # Education matching (20 points)
        if any(edu in ' '.join(resume.education).lower() for edu in ['bachelor', 'master', 'phd', 'b.tech', 'm.tech']):
            score += 20
        
        # Projects relevance (10 points)
        if resume.projects and len(resume.projects) >= 2:
            score += 10
        
        # Certifications (10 points)
        if resume.certifications:
            score += 10
        
        state["hard_score"] = min(score, max_score)
        return state
    
    def _semantic_matching_node(self, state: WorkflowState) -> WorkflowState:
        """Perform semantic matching using embeddings"""
        resume = state["resume"]
        jd = state["job_description"]
        
        # Add to vector store if not already present
        self.vector_store.add_resume(resume)
        self.vector_store.add_job_description(jd)
        
        # Find similar resumes to get semantic score
        similar_resumes = self.vector_store.find_similar_resumes(jd, top_k=1)
        
        if similar_resumes:
            semantic_score = similar_resumes[0]["similarity_score"]
        else:
            # Fallback to direct embedding similarity
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            resume_embedding = model.encode(resume.resume_text)
            jd_embedding = model.encode(jd.description_text)
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
            semantic_score = similarity * 100
        
        state["soft_score"] = semantic_score
        return state
    
    def _calculate_score_node(self, state: WorkflowState) -> WorkflowState:
        """Calculate final weighted score"""
        hard_score = state["hard_score"]
        soft_score = state["soft_score"]
        
        # Weighted combination: 40% hard + 60% soft
        final_score = (hard_score * 0.4) + (soft_score * 0.6)
        state["final_score"] = round(final_score, 2)
        
        return state
    
    def _generate_verdict_node(self, state: WorkflowState) -> WorkflowState:
        """Generate suitability verdict"""
        final_score = state["final_score"]
        
        if final_score >= 75:
            verdict = "High Suitability"
        elif final_score >= 50:
            verdict = "Medium Suitability"
        else:
            verdict = "Low Suitability"
        
        state["verdict"] = verdict
        return state
    
    def _llm_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Perform deep LLM analysis"""
        resume = state["resume"]
        jd = state["job_description"]
        final_score = state["final_score"]
        missing_skills = state["missing_skills"]
        matching_skills = state["matching_skills"]
        
        analysis_prompt = PromptTemplate(
            input_variables=["resume_text", "jd_text", "score", "missing_skills", "matching_skills"],
            template="""
            Analyze this resume against the job description and provide detailed insights.
            
            Resume: {resume_text}
            Job Description: {jd_text}
            Score: {score}/100
            Missing Skills: {missing_skills}
            Matching Skills: {matching_skills}
            
            Provide analysis in JSON format:
            {{
                "strengths": ["strength1", "strength2"],
                "weaknesses": ["weakness1", "weakness2"],
                "recommendations": ["rec1", "rec2"],
                "fit_assessment": "detailed assessment"
            }}
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
        
        try:
            response = chain.run(
                resume_text=resume.resume_text[:1500],
                jd_text=jd.description_text[:1500],
                score=final_score,
                missing_skills=', '.join(missing_skills),
                matching_skills=', '.join(matching_skills)
            )
            import json
            state["llm_analysis"] = json.loads(response)
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            state["llm_analysis"] = {
                "strengths": ["Technical skills present"],
                "weaknesses": ["Some skills missing"],
                "recommendations": ["Focus on missing skills"],
                "fit_assessment": f"Score: {final_score}/100"
            }
        
        return state
    
    def _generate_suggestions_node(self, state: WorkflowState) -> WorkflowState:
        """Generate personalized suggestions"""
        missing_skills = state["missing_skills"]
        llm_analysis = state["llm_analysis"]
        final_score = state["final_score"]
        
        suggestions = []
        
        # Add LLM recommendations
        if "recommendations" in llm_analysis:
            suggestions.extend(llm_analysis["recommendations"][:3])
        
        # Add missing skills suggestions
        if missing_skills:
            suggestions.append(f"Focus on acquiring these skills: {', '.join(missing_skills[:3])}")
        
        # Add score-based suggestions
        if final_score < 50:
            suggestions.append("Consider significant skill development and experience building")
        elif final_score < 75:
            suggestions.append("Minor improvements in key areas will enhance your profile")
        else:
            suggestions.append("Your profile is strong - focus on interview preparation")
        
        state["suggestions"] = suggestions[:5]  # Limit to 5 suggestions
        return state
    
    def _create_evaluation_node(self, state: WorkflowState) -> WorkflowState:
        """Create final evaluation result"""
        from core.models import EvaluationResult, Verdict
        
        # Convert verdict string to enum
        verdict_map = {
            "High Suitability": Verdict.HIGH,
            "Medium Suitability": Verdict.MEDIUM,
            "Low Suitability": Verdict.LOW
        }
        
        evaluation = EvaluationResult(
            evaluation_id=f"{state['resume'].resume_id}_{state['job_description'].job_id}",
            resume_id=state['resume'].resume_id,
            job_id=state['job_description'].job_id,
            relevance_score=state['final_score'],
            hard_match_score=state['hard_score'],
            soft_match_score=state['soft_score'],
            missing_skills=state['missing_skills'],
            matching_skills=state['matching_skills'],
            verdict=verdict_map.get(state['verdict'], Verdict.MEDIUM),
            suggestions=state['suggestions']
        )
        
        state["evaluation_result"] = evaluation
        return state
    
    def evaluate_resume(self, resume: Resume, job_description: JobDescription) -> EvaluationResult:
        """Run the complete evaluation workflow"""
        initial_state = WorkflowState(
            resume=resume,
            job_description=job_description,
            extracted_skills=[],
            missing_skills=[],
            matching_skills=[],
            hard_score=0.0,
            soft_score=0.0,
            final_score=0.0,
            verdict="",
            suggestions=[],
            llm_analysis={},
            evaluation_result=None
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state["evaluation_result"]