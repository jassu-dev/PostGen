from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import google.generativeai as genai
from google.cloud import texttospeech
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from dotenv import load_dotenv
import logging
from werkzeug.utils import secure_filename
import json
import re
import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-prod')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO if os.getenv('DEBUG') == '1' else logging.WARNING)
logger = logging.getLogger(__name__)

# Google AI Configuration
GOOGLE_AI_API_KEY = os.getenv('GOOGLE_AI_API_KEY')
model = None
if GOOGLE_AI_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_AI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        model = None
else:
    logger.warning("GOOGLE_AI_API_KEY not set. Content generation disabled.")
    model = None

# Google Cloud Project ID
GOOGLE_CLOUD_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# TTS Client
tts_client = None
if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
        tts_client = texttospeech.TextToSpeechClient()
        logger.info("TTS client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize TTS client: {e}")
        tts_client = None
else:
    logger.warning("TTS not configured. Voice generation disabled.")

# Stable Diffusion Pipeline
sd_pipeline = None
sd_lock = threading.Lock()

def initialize_stable_diffusion():
    """Initialize Stable Diffusion pipeline in a separate thread"""
    global sd_pipeline
    try:
        # Use a small, fast model - using the base v1-5 which is relatively small
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Use CPU if no GPU available, or smaller datatype to save memory
        if torch.cuda.is_available():
            logger.info("Using CUDA for Stable Diffusion")
            torch_dtype = torch.float16
            device = "cuda"
        else:
            logger.info("Using CPU for Stable Diffusion")
            torch_dtype = torch.float32
            device = "cpu"
        
        # Load the pipeline with optimized settings
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disable safety checker for faster generation
            requires_safety_checker=False
        )
        
        # Use a faster scheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        pipeline = pipeline.to(device)
        
        # Enable attention slicing to reduce memory usage
        if device == "cuda":
            pipeline.enable_attention_slicing()
        
        sd_pipeline = pipeline
        logger.info("Stable Diffusion pipeline initialized successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize Stable Diffusion: {e}")
        sd_pipeline = None

# Initialize Stable Diffusion in background thread
sd_thread = threading.Thread(target=initialize_stable_diffusion, daemon=True)
sd_thread.start()

ALLOWED_VOICES = [
    'en-US-Standard-A', 'en-US-Standard-B', 'en-US-Standard-C', 'en-US-Standard-D',
    'en-US-Neural2-A', 'en-US-Neural2-D', 'en-US-Neural2-F', 'en-US-Neural2-J',
    'en-US-Wavenet-D', 'en-US-Wavenet-F'
]

def validate_input(text, max_length=1000):
    """Validate and sanitize input text."""
    if not text or len(text.strip()) == 0:
        raise ValueError("Input text cannot be empty.")
    if len(text) > max_length:
        raise ValueError(f"Input too long. Max {max_length} characters.")
    # Basic sanitization (remove control characters)
    sanitized = ''.join(c for c in text if ord(c) > 31 or c in '\n\t')
    return sanitized.strip()

def extract_json_from_response(text):
    """Extract JSON from potentially malformed response."""
    # Look for JSON object in the text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None

def generate_image_with_sd(prompt, num_images=1):
    """Generate images using Stable Diffusion"""
    if not sd_pipeline:
        raise ValueError("Stable Diffusion not initialized")
    
    # Generate images
    with sd_lock:  # Thread safety
        results = sd_pipeline(
            prompt=prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=20,  # Reduced steps for faster generation
            guidance_scale=7.5,
            width=512,  # Smaller size for faster generation
            height=512
        )
    
    return results.images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    sd_status = sd_pipeline is not None
    return jsonify({
        "status": "healthy", 
        "gemini": model is not None, 
        "tts": tts_client is not None, 
        "stable_diffusion": sd_status
    })

@app.route('/api/generate_content', methods=['POST'])
def generate_content():
    try:
        data = request.get_json()
        if not model:
            return jsonify({"error": "Content generation not available. Check API key."}), 503

        topic = validate_input(data.get('topic', ''), 200)
        platform = data.get('platform', 'twitter')
        tone = data.get('tone', 'professional')
        char_limit = min(max(data.get('char_limit', 280), 50), 1000)
        description = validate_input(data.get('description', ''), 500)

        platforms = {
            'twitter': 'Twitter/X (280 chars max, concise, hashtags)',
            'instagram': 'Instagram (captions up to 2200 chars, emojis, visual focus)',
            'linkedin': 'LinkedIn (professional, longer form, networking)',
            'facebook': 'Facebook (conversational, shares, questions)'
        }
        platform_desc = platforms.get(platform, platforms['twitter'])

        prompt = f"""
        Generate a social media post for {platform} about '{topic}'.
        Description: {description}
        Tone: {tone}
        Keep under {char_limit} characters.
        Make it engaging for {platform_desc}.
        Output only the post text, no extras.
        """

        response = model.generate_content(prompt)
        content = response.text.strip()

        # Truncate if needed
        if len(content) > char_limit:
            content = content[:char_limit].rsplit(' ', 1)[0] + '...'

        return jsonify({"content": content, "char_count": len(content)})

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return jsonify({"error": "An error occurred generating content."}), 500

@app.route('/api/generate_voice', methods=['POST'])
def generate_voice():
    if not tts_client:
        return jsonify({"error": "TTS not available. Check service account."}), 503

    try:
        data = request.get_json()
        text = validate_input(data.get('text', ''), 5000)  # TTS limit ~5000 chars
        voice_name = data.get('voice', 'en-US-Neural2-F')

        if voice_name not in ALLOWED_VOICES:
            return jsonify({"error": "Invalid voice selected."}), 400

        # Audio config
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )

        # Voice config
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name
        )

        # Synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Generate audio
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Return audio as binary response
        return Response(
            response.audio_content,
            mimetype='audio/mpeg',
            headers={'Content-Disposition': 'attachment; filename=voice.mp3'}
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating voice: {e}")
        return jsonify({"error": "Failed to generate voice."}), 500

@app.route('/api/generate_images', methods=['POST'])
def generate_images():
    if not sd_pipeline:
        return jsonify({"error": "Image generation not available. Stable Diffusion is still loading or failed to initialize."}), 503

    try:
        data = request.get_json()
        prompt = validate_input(data.get('prompt', ''), 1000)
        num_images = min(max(data.get('num_images', 1), 1), 4)  # Limit to 4

        # Generate images using Stable Diffusion
        generated_images = generate_image_with_sd(prompt, num_images)

        images = []
        for img in generated_images:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            images.append(f"data:image/png;base64,{img_base64}")

        return jsonify({"images": images})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        return jsonify({"error": "Failed to generate images."}), 500

@app.route('/api/generate_nft', methods=['POST'])
def generate_nft():
    if not sd_pipeline:
        return jsonify({"error": "NFT generation not available. Stable Diffusion is still loading or failed to initialize."}), 503

    try:
        data = request.get_json()
        prompt = validate_input(data.get('prompt', ''), 1000)

        # Generate single NFT-style image with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Add NFT-specific enhancements to the prompt
                nft_prompt = f"digital art, NFT style, {prompt}, high quality, detailed, vibrant colors"
                
                # Generate image using Stable Diffusion
                generated_images = generate_image_with_sd(nft_prompt, 1)
                
                if not generated_images:
                    raise ValueError("No images were generated")

                # Get the first generated image
                generated_image = generated_images[0]
                
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                generated_image.save(img_byte_arr, format='PNG')
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

                return jsonify({
                    "image_url": f"data:image/png;base64,{img_base64}",
                    "message": "NFT image generated successfully"
                })

            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(1)  # Wait before retrying

    except ValueError as e:
        logger.warning(f"Validation error in NFT generation: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating NFT: {e}")
        return jsonify({"error": f"Failed to generate NFT image. Please try again with a different prompt."}), 500

@app.route('/api/analyze_post', methods=['POST'])
def analyze_post():
    try:
        if not model:
            return jsonify({"error": "Analysis not available. Check API key."}), 503

        data = request.get_json()
        post_text = validate_input(data.get('post_text', ''), 2000)  # Allow longer for posts
        platform = data.get('platform', 'twitter')

        platforms = {
            'twitter': 'Twitter/X - focus on concise, engaging, hashtag-driven content under 280 characters',
            'instagram': 'Instagram - visual, emoji-rich captions, storytelling up to 2200 characters',
            'linkedin': 'LinkedIn - professional, value-driven, networking-oriented posts',
            'facebook': 'Facebook - conversational, question-asking, shareable content'
        }
        platform_guidelines = platforms.get(platform, platforms['twitter'])

        # Strict prompt for JSON output only
        prompt = f"""
        You are a social media analyst. Analyze the following post for the {platform} platform.

        Post content: {post_text}

        Platform-specific guidelines: {platform_guidelines}

        Focus your analysis on:
        1. Strengths and weaknesses in structure, tone, engagement potential, relevance to platform best practices.
        2. Specific improvements to boost visibility, interaction, and alignment with platform algorithms.
        3. Competitor examples: Create 3-5 realistic, high-performing example posts for {platform} on similar topics (e.g., if the post is about marketing, use marketing-related examples). Explain why each works well for engagement.

        Respond with ONLY a valid JSON object in this exact structure. Do not include any other text, explanations, or markdown. Ensure the JSON is well-formed and parsable.

        {{
            "score": <integer 0-100>,
            "score_explanation": "<1-2 sentences explaining the score based on platform fit and engagement potential>",
            "analysis": "<Detailed 2-4 paragraphs on strengths, weaknesses, and overall quality. Focus on content structure, calls to action, hashtags/emojis where relevant, and platform-specific elements.>",
            "improvements": [
                "<Specific suggestion 1>",
                "<Specific suggestion 2>",
                "<Specific suggestion 3>",
                "<Specific suggestion 4>",
                "<Specific suggestion 5>"
            ],
            "better_post": "<Improved version of the original post, optimized for {platform}, keeping similar length and topic. Include platform-appropriate elements like hashtags or emojis.>",
            "competitor_examples": [
                "<Example post 1>",
                "<Example post 2>",
                "<Example post 3>",
                "<Example post 4>",
                "<Example post 5>"
            ],
            "competitor_why": [
                "<Why example 1 works: 1-2 sentences>",
                "<Why example 2 works: 1-2 sentences>",
                "<Why example 3 works: 1-2 sentences>",
                "<Why example 4 works: 1-2 sentences>",
                "<Why example 5 works: 1-2 sentences>"
            ]
        }}
        """

        # Use generation config to force JSON output
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "top_p": 0.8,
            "max_output_tokens": 2048
        }

        response = model.generate_content(prompt, generation_config=generation_config)
        analysis_text = response.text.strip()

        logger.info(f"Raw Gemini response: {analysis_text[:200]}...")  # Log for debugging

        # Try direct JSON load
        try:
            analysis_data = json.loads(analysis_text)
        except json.JSONDecodeError:
            # Try extraction
            analysis_data = extract_json_from_response(analysis_text)
            if not analysis_data:
                # Fallback with structured data using platform
                logger.warning("Gemini did not return valid JSON, using fallback.")
                platform_name = platform.capitalize()
                analysis_data = {
                    "score": 70,
                    "score_explanation": "Moderate quality post with room for improvement in engagement and platform alignment.",
                    "analysis": "The post communicates its message but could better incorporate platform-specific elements like concise phrasing for Twitter or visual calls for Instagram. Strengths include clear topic coverage; weaknesses are lack of strong hooks or interactive elements.",
                    "improvements": [
                        f"Add 2-3 relevant #{platform} hashtags to increase discoverability.",
                        "Include a question or call-to-action to encourage comments and shares.",
                        f"Optimize length and formatting for {platform}'s algorithm preferences.",
                        f"Incorporate emojis or visuals if suitable for {platform}.",
                        f"Ensure tone matches {platform}'s audience expectations."
                    ],
                    "better_post": f"Enhanced {platform} post: {post_text[:100]}... (optimized with CTA and hashtags).",
                    "competitor_examples": [
                        f"Competitor example 1 for {platform_name}: Engaging post with question.",
                        f"Competitor example 2 for {platform_name}: Hashtag-driven content.",
                        f"Competitor example 3 for {platform_name}: Visual story with emojis.",
                        f"Competitor example 4 for {platform_name}: Professional insight.",
                        f"Competitor example 5 for {platform_name}: Conversational share."
                    ],
                    "competitor_why": [
                        f"Sparks interaction through direct questions, boosting comments on {platform}.",
                        f"Leverages trending hashtags for wider reach and algorithmic promotion on {platform}.",
                        f"Uses emojis and short paragraphs for better mobile readability on {platform}.",
                        f"Provides value with insights, encouraging shares and professional networking on {platform}.",
                        f"Builds community with relatable language, increasing engagement rates on {platform}."
                    ]
                }

        # Ensure arrays are present and have at least 3 items
        if "improvements" not in analysis_data or not isinstance(analysis_data["improvements"], list):
            analysis_data["improvements"] = ["Add engaging elements.", "Optimize for platform.", "Include CTA."]
        if "competitor_examples" not in analysis_data or not isinstance(analysis_data["competitor_examples"], list):
            analysis_data["competitor_examples"] = ["Example 1.", "Example 2.", "Example 3."]
        if "competitor_why" not in analysis_data or not isinstance(analysis_data["competitor_why"], list):
            analysis_data["competitor_why"] = ["Reason 1.", "Reason 2.", "Reason 3."]

        # Ensure score is integer 0-100
        if "score" in analysis_data:
            analysis_data["score"] = max(0, min(100, int(analysis_data["score"])))

        return jsonify(analysis_data)

    except ValueError as e:
        logger.warning(f"Validation error in analysis: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error analyzing post: {e}")
        return jsonify({"error": "An error occurred while analyzing the post. Please try again."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0')
