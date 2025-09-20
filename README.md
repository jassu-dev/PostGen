# PostX - Ultimate AI Content Suite (2025 Edition)

PostX is an advanced AI-powered social media content generator built with Flask. It leverages Google Gemini for text generation, Vertex AI Imagen for image generation, Google Cloud Text-to-Speech for voiceovers, and Pillow for image manipulation (e.g., memes). Premium features include multi-variation image generation, voiceovers with customizable voices, meme creation, NFT image generation (demo), and post analysis with suggestions.

## Key Features
- **Content Generation**: Platform-specific posts with customizable tone and length using Gemini 1.5 Flash.
- **AI Image Generation**: Up to 4 variations using Vertex AI Imagen 3.
- **AI Voiceovers**: MP3 audio using Google Cloud TTS with Neural2/Wavenet voices.
- **Meme Creator**: Custom memes with text overlays.
- **NFT Demo**: Generate NFT-style images.
- **Post Analysis**: Demo analysis of social posts with improvements (uses Gemini).
- **Responsive UI**: Dark theme toggle, mobile-friendly.

## Tech Stack
- **Backend**: Flask 3.0.3 (Python 3.11+)
- **AI Services**: Google Gemini, Vertex AI Imagen 3, Google Cloud TTS
- **Frontend**: HTML5, CSS3, Vanilla JS, Chart.js, Font Awesome
- **Image Processing**: Pillow 10.4.0

## Prerequisites
- Python 3.11+ (https://python.org)
- Google Cloud account with billing enabled.
  - Enable APIs: Generative Language API, Vertex AI API, Cloud Text-to-Speech API.
  - Get Gemini API key from https://aistudio.google.com/app/apikey.
  - Create service account with roles: `roles/aiplatform.user`, `roles/cloudtts.user`.
  - Download JSON key and note project ID.
- For Imagen: Project may need whitelisting (contact Google Cloud support if errors occur).

## Setup Instructions

1. **Clone/Download the Project**
   ```
   git clone <your-repo-url>  # Or download ZIP
   cd postx-app
   ```

2. **Create Virtual Environment**
   ```
   python -m venv venv
   ```
   Activate:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Install Dependencies**
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   - If errors: `pip install --force-reinstall -r requirements.txt`
   - Verify: `python -c "import flask; print('Flask OK')"`

4. **Configure Environment**
   - Copy `.env.example` to `.env`:
     ```
     cp .env.example .env
     ```
   - Edit `.env`:
     - `GOOGLE_AI_API_KEY=your_key_from_ai_studio`
     - `GOOGLE_CLOUD_PROJECT_ID=your_project_id` (e.g., grand-fx-471616-h8)
     - `GOOGLE_APPLICATION_CREDENTIALS=./path/to/your-service-account.json`
       - Use the provided `grand-fx-471616-h8-69a37d9d94e8.json` or download your own.
       - Place the JSON in the project root.
     - `SECRET_KEY=generate_a_strong_key` (e.g., use `python -c "import secrets; print(secrets.token_hex(32))"`)
     - `DEBUG=1` for development logs.
   - **Security Note**: Never commit `.env` or JSON keys to Git. Add to `.gitignore`.

5. **Verify Google Services**
   - Run: `python -c "import google.generativeai as genai; genai.configure(api_key='test'); print('Gemini OK')"` (replace with your key).
   - For TTS/Imagen:
     ```
     export GOOGLE_APPLICATION_CREDENTIALS=./your-json.json
     python -c "from google.cloud import texttospeech; client = texttospeech.TextToSpeechClient(); print('TTS OK')"
     ```
     ```
     python -c "from vertexai.preview.vision_models import ImageGenerationModel; model = ImageGenerationModel.from_pretrained('imagegeneration@006'); print('Imagen OK')"
     ```
   - If Imagen fails: Check quotas, roles, or use Imagen 2 (`imagegeneration@002`).
   - Test health: After running app, visit `http://localhost:5000/health`.

6. **Handle Common Errors**
   - **"An error occurred. Please try again."**: Check browser console (F12) and server logs.
     - No API key: Set `GOOGLE_AI_API_KEY`.
     - TTS/Image fail: Verify service account path/permissions. Logs show details (e.g., "TTS not configured").
     - Quota exceeded: Check Google Cloud Console > Quotas.
     - Whitelisting: For Imagen, request access via support if "model not found".
   - **Import Errors**: Reinstall deps: `pip uninstall -r requirements.txt -y && pip install -r requirements.txt`.
   - **Font Issues (Memes)**: Install system fonts (e.g., `sudo apt install fonts-liberation` on Linux).
   - **CORS/Frontend Errors**: Ensure Flask-CORS installed; refresh page.
   - **npm Errors?**: This is a Python/Flask app—no npm needed. Ignore if seen (possible mix-up).
   - Logs: Run with `DEBUG=1` for verbose output. Check for "Service initialization failed".

## Running the Application

1. **Start the Server**
   ```
   python app.py
   ```
   - Output: `* Running on http://127.0.0.1:5000` (or http://localhost:5000).
   - Debug mode: Enabled if `DEBUG=1`. Watch console for init messages (e.g., "Gemini OK").
   - If services fail: App runs but features show errors (e.g., "TTS not available").

2. **Access the App**
   - Open browser: http://localhost:5000
   - Test:
     - **Content**: Enter topic/description, select options, click "Generate Content". Copies to clipboard.
     - **Images**: After content, use premium tab > NFT or integrate button (demo: generates via /api/generate_images).
     - **Voice**: Premium > AI Tools > Select voice/text > Generate. Plays audio.
     - **Meme**: Enter texts > Create Meme > Download.
     - **Analysis**: Premium > Analytics > Enter post URL > Analyze (demo output).
   - Errors: Check server console (e.g., "Failed to generate: permission denied") and fix config.

3. **Stop the Server**
   - Ctrl+C in terminal.

## Troubleshooting Specific Errors

- **Generic "An error occurred"**:
  - Browser: Check Network tab (F12) for 500 errors. Response body shows details (e.g., "API key missing").
  - Server: Look for tracebacks. Common: Missing env vars—double-check `.env`.
  - Test endpoint: `curl http://localhost:5000/api/generate_content -H "Content-Type: application/json" -d '{"topic":"test","description":"test"}'`

- **Image Gen Fails**:
  - Error: "Model not found" → Whitelist project for Imagen 3 or fallback to @002.
  - "Permission denied" → Add `roles/aiplatform.user` to service account.
  - Quota: Check https://console.cloud.google.com/quotas.

- **TTS Fails**:
  - "Client not authenticated" → Verify JSON path and `gcloud auth activate-service-account --key-file=your.json`.
  - Enable API: https://console.cloud.google.com/apis/library/texttospeech.googleapis.com.

- **Content Gen Fails**:
  - "API key invalid" → Regenerate at AI Studio.
  - Rate limit: Wait or upgrade plan.

- **Deployment**:
  - **Local**: As above.
  - **Docker** (Optional):
    Create `Dockerfile`:
    ```
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    # Mount JSON securely in prod
    EXPOSE 5000
    CMD ["python", "app.py"]
    ```
    Build/Run:
    ```
    docker build -t postx .
    docker run -p 5000:5000 -e GOOGLE_AI_API_KEY=your_key -v $(pwd)/your.json:/app/your.json -e GOOGLE_APPLICATION_CREDENTIALS=/app/your.json postx
    ```
  - **Google App Engine/Heroku**: Set env vars in dashboard. Upload JSON as secret.

## Customization
- **Add Voices**: Update `ALLOWED_VOICES` in app.py and HTML select.
- **Platforms/Tones**: Edit prompts in `/api/generate_content`.
- **Real NFT**: Integrate Web3.js + blockchain (e.g., Ethereum) in JS.
- **Real Analysis**: Add scraping (e.g., BeautifulSoup) or platform APIs (Twitter API v2).

## Security Notes
- Use HTTPS in production.
- Rate limit APIs (add Flask-Limiter).
- Validate/sanitize all inputs (already implemented).
- Store JSON securely; use Google Secret Manager in prod.

## License
© 2025 PostX. For educational use. Contribute via GitHub issues with logs/error details.

Happy generating! If issues persist, share exact error + logs. 🚀