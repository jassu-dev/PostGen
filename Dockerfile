FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Stable Diffusion model
RUN python -c "\
import torch; \
from diffusers import StableDiffusionPipeline; \
print('Downloading Stable Diffusion model...'); \
model_id = 'runwayml/stable-diffusion-v1-5'; \
try: \
    pipeline = StableDiffusionPipeline.from_pretrained( \
        model_id, \
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, \
        safety_checker=None, \
        requires_safety_checker=False \
    ); \
    print('Model downloaded successfully'); \
except Exception as e: \
    print(f'Error downloading model: {e}'); \
    try: \
        pipeline = StableDiffusionPipeline.from_pretrained( \
            model_id, \
            torch_dtype=torch.float32, \
            safety_checker=None, \
            requires_safety_checker=False \
        ); \
        print('Model downloaded successfully with CPU fallback'); \
    except Exception as e2: \
        print(f'CPU fallback also failed: {e2}'); \
"

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p static/uploads

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the application
CMD ["python", "-u", "app.py"]