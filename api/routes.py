import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from core.system import ResumeRelevanceSystem
from core.models import JobDescription


app = Flask(__name__)
CORS(app)

# Initialize system
system = ResumeRelevanceSystem(use_llm=False)  # Set True if using OpenAI


@app.route('/api/upload_jd', methods=['POST'])
def upload_jd():
    """Upload and process job description"""
    try:
        data = request.json
        jd_text = data.get('jd_text')
        company = data.get('company', 'Unknown')
        location = data.get('location', 'Unknown')

        jd = system.process_job_description(jd_text, company, location)

        return jsonify({
            'success': True,
            'job_id': jd.job_id,
            'role_title': jd.role_title,
            'required_skills': jd.required_skills,
            'preferred_skills': jd.preferred_skills
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/upload_resume', methods=['POST'])
def upload_resume():
    """Upload and evaluate resume"""
    try:
        if 'resume' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['resume']
        job_id = request.form.get('job_id')

        if not job_id:
            return jsonify({'success': False, 'error': 'Job ID required'}), 400

        # Save file temporarily
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)

        # Process resume
        resume = system.process_resume(temp_path)

        # Get JD from database
        jobs = system.db_manager.get_all_jobs()
        jd_data = next((j for j in jobs if j['job_id'] == job_id), None)

        if not jd_data:
            return jsonify({'success': False, 'error': 'Job not found'}), 404

        jd = JobDescription(
            job_id=jd_data['job_id'],
            company=jd_data['company'],
            role_title=jd_data['role_title'],
            required_skills=jd_data['required_skills'],
            preferred_skills=jd_data['preferred_skills'],
            experience_required=jd_data['experience_required'],
            education=jd_data['education'],
            location=jd_data['location'],
            description_text=jd_data['description_text']
        )

        # Evaluate
        evaluation = system.scoring_engine.calculate_relevance_score(resume, jd)
        system.db_manager.save_evaluation(evaluation)

        os.remove(temp_path)

        return jsonify({
            'success': True,
            'evaluation_id': evaluation.evaluation_id,
            'relevance_score': evaluation.relevance_score,
            'verdict': evaluation.verdict.value,
            'missing_skills': evaluation.missing_skills,
            'matching_skills': evaluation.matching_skills,
            'suggestions': evaluation.suggestions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/get_shortlist/<job_id>', methods=['GET'])
def get_shortlist(job_id):
    """Get shortlisted candidates for a job"""
    try:
        top_n = request.args.get('top_n', 10, type=int)
        min_score = request.args.get('min_score', 50, type=float)

        shortlist = system.get_shortlist(job_id, top_n, min_score)

        return jsonify({
            'success': True,
            'job_id': job_id,
            'candidates': shortlist
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """Evaluate multiple resumes"""
    try:
        if 'resumes' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400

        files = request.files.getlist('resumes')
        jd_text = request.form.get('jd_text')
        company = request.form.get('company', 'Unknown')
        location = request.form.get('location', 'Unknown')

        if not jd_text:
            return jsonify({'success': False, 'error': 'JD text required'}), 400

        temp_paths = []
        for file in files:
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            temp_paths.append(temp_path)

        results = system.batch_evaluate(temp_paths, jd_text, company, location)

        for path in temp_paths:
            os.remove(path)

        formatted_results = []
        for r in results:
            formatted_results.append({
                'evaluation_id': r.evaluation_id,
                'resume_id': r.resume_id,
                'relevance_score': r.relevance_score,
                'verdict': r.verdict.value,
                'missing_skills': r.missing_skills,
                'suggestions': r.suggestions
            })

        return jsonify({
            'success': True,
            'total_evaluated': len(results),
            'results': formatted_results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get all job descriptions"""
    try:
        jobs = system.db_manager.get_all_jobs()
        return jsonify({'success': True, 'jobs': jobs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
