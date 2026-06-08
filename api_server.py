from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from main import CodeReviewSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Code Review System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

review_system = CodeReviewSystem()

class CodeAnalysisRequest(BaseModel):
    code: str
    language: str = "python"
    file_name: str = "code.py"

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>AI Code Review System</title>
    <style>
        body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; margin: 0; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; color: white; margin-bottom: 30px; }
        h1 { font-size: 2.5em; margin: 0; }
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .card { background: white; border-radius: 12px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        label { display: block; margin-bottom: 8px; font-weight: bold; }
        textarea, input, select { width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 8px; font-family: monospace; margin-bottom: 15px; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 1em; font-weight: bold; cursor: pointer; width: 100%; }
        .results-section { background: white; border-radius: 12px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-top: 20px; }
        .scores-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }
        .score-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .score-value { font-size: 2.5em; font-weight: bold; }
        .progress-bar { width: 100%; height: 8px; background: rgba(255,255,255,0.3); border-radius: 4px; margin-top: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: rgba(255,255,255,0.8); }
        .hidden { display: none !important; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 AI Code Review System</h1>
            <p>XGBoost + CodeBERT + GNN + LSTM</p>
        </header>
        <div class="main-grid">
            <div class="card">
                <h2>Enter Code</h2>
                <label>Code:</label>
                <textarea id="code" style="min-height: 200px;">def add(a, b):
    return a + b
print(add(5, 3))</textarea>
                <button onclick="analyzeCode()">Analyze Code</button>
            </div>
            <div class="card">
                <h2>Upload File</h2>
                <input type="file" id="fileUpload" accept=".py">
                <button onclick="uploadFile()">Upload & Analyze</button>
            </div>
        </div>
        <div class="results-section hidden" id="resultsSection">
            <h2>Results</h2>
            <div id="loading">
                <div class="spinner"></div>
                <p>Analyzing code...</p>
            </div>
            <div id="resultsContent" class="hidden">
                <div class="scores-grid">
                    <div class="score-card">
                        <div>Quality</div>
                        <div class="score-value" id="qualityScore">-</div>
                        <div class="progress-bar"><div class="progress-fill" id="qualityBar"></div></div>
                    </div>
                    <div class="score-card">
                        <div>Complexity</div>
                        <div class="score-value" id="complexityScore">-</div>
                        <div class="progress-bar"><div class="progress-fill" id="complexityBar"></div></div>
                    </div>
                    <div class="score-card">
                        <div>Maintainability</div>
                        <div class="score-value" id="maintainabilityScore">-</div>
                        <div class="progress-bar"><div class="progress-fill" id="maintainabilityBar"></div></div>
                    </div>
                </div>
                <div id="bugsSection" class="hidden">
                    <h3>Bugs</h3>
                    <div id="bugsList"></div>
                </div>
                <div id="suggestionsSection" class="hidden">
                    <h3>Suggestions</h3>
                    <div id="suggestionsList"></div>
                </div>
            </div>
            <div id="errorMessage" class="hidden" style="color: red; padding: 10px; border: 1px solid red;"></div>
        </div>
    </div>
    <script>
    function showLoading() {
        document.getElementById('resultsSection').classList.remove('hidden');
        document.getElementById('loading').style.display = 'block';

        document.getElementById('resultsContent').classList.add('hidden');
        document.getElementById('errorMessage').classList.add('hidden');

        document.getElementById('bugsSection').classList.add('hidden');
        document.getElementById('suggestionsSection').classList.add('hidden');
    }

    function analyzeCode() {
        const code = document.getElementById('code').value;

        if (!code.trim()) {
            alert('Please enter code');
            return;
        }

        showLoading();

        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                code: code,
                language: 'python',
                file_name: 'code.py'
            })
        })
        .then(response => response.json())
        .then(data => displayResults(data))
        .catch(error => showError(error.message));
    }

    function uploadFile() {
        const file = document.getElementById('fileUpload').files[0];

        if (!file) {
            alert('Please select a file');
            return;
        }

        showLoading();

        const reader = new FileReader();

        reader.onload = function(event) {

            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    code: event.target.result,
                    language: 'python',
                    file_name: file.name
                })
            })
            .then(response => response.json())
            .then(data => displayResults(data))
            .catch(error => showError(error.message));
        };

        reader.readAsText(file);
    }

    function displayResults(data) {

        document.getElementById('resultsSection').classList.remove('hidden');
        document.getElementById('loading').style.display = 'none';

        if (data.status !== 'success') {
            showError(data.message || 'Analysis failed');
            return;
        }

        document.getElementById('qualityScore').textContent =
            Math.round(data.quality_score * 100) + '%';

        document.getElementById('complexityScore').textContent =
            Math.round(data.complexity_score * 100) + '%';

        document.getElementById('maintainabilityScore').textContent =
            Math.round(data.maintainability_score * 100) + '%';

        document.getElementById('qualityBar').style.width =
            (data.quality_score * 100) + '%';

        document.getElementById('complexityBar').style.width =
            (data.complexity_score * 100) + '%';

        document.getElementById('maintainabilityBar').style.width =
            (data.maintainability_score * 100) + '%';

        if (data.bugs && data.bugs.length > 0) {

            let bugHTML = '';

            data.bugs.forEach(function(bug) {
                bugHTML += `
                    <div style="padding:10px;margin:8px 0;background:#ffebee;border-left:4px solid #f44336;border-radius:4px;">
                        <strong>${bug.type || 'Bug'}</strong>
                        <p>${bug.description || JSON.stringify(bug)}</p>
                    </div>
                `;
            });

            document.getElementById('bugsList').innerHTML = bugHTML;
            document.getElementById('bugsSection').classList.remove('hidden');
        }

        if (data.suggestions && data.suggestions.length > 0) {

            let suggestionsHTML = '';

            data.suggestions.forEach(function(suggestion) {
                suggestionsHTML += `
                    <div style="padding:10px;margin:8px 0;background:#e8f5e9;border-left:4px solid #4caf50;border-radius:4px;">
                        ${suggestion}
                    </div>
                `;
            });

            document.getElementById('suggestionsList').innerHTML =
                suggestionsHTML;

            document.getElementById('suggestionsSection')
                .classList.remove('hidden');
        }

        document.getElementById('resultsContent')
            .classList.remove('hidden');
    }

    function showError(message) {

        document.getElementById('resultsSection')
            .classList.remove('hidden');

        document.getElementById('loading').style.display = 'none';

        const errorDiv = document.getElementById('errorMessage');

        errorDiv.textContent = 'Error: ' + message;
        errorDiv.classList.remove('hidden');

        console.error(message);
    }
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return DASHBOARD_HTML

@app.post("/api/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    try:
        result = review_system.analyze_code(request.code, language=request.language, file_name=request.file_name)
        return {
            "file_name": request.file_name,
            "quality_score": result.code_quality_score,
            "complexity_score": result.complexity_score,
            "maintainability_score": result.maintainability_score,
            "bugs": result.bugs,
            "vulnerabilities": result.vulnerabilities,
            "suggestions": result.suggestions,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("Starting AI Code Review System")
    print("="*70)
    print("\nOpen: http://localhost:8000")
    print("\n" + "="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)