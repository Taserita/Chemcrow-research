FROM python:3.10-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libxrender1 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU inference avoids both ARM emulation and a CUDA runtime dependency.
# Keep this legacy OpenNMT release away from newer incompatible torch defaults.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "torch==2.2.2+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "numpy==1.26.4" "rdkit-pypi==2022.3.1" \
    "OpenNMT-py==2.2.0" "Flask==3.0.3" "waitress==3.0.2" "gdown==5.2.0"

COPY app.py .
# Mount the separately downloaded checkpoint at /app/models.
RUN mkdir -p /app/models
EXPOSE 5000
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "--threads=1", "app:app"]
