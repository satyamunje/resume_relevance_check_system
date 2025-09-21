import pytest
from api.routes import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_jobs_endpoint(client):
    rv = client.get('/api/jobs')
    assert rv.status_code == 200
    assert 'success' in rv.get_json()
