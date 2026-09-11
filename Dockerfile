FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# libgomp is needed by LightGBM's wheel. Use the HTTPS mirror: some networks block plain-HTTP apt.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY aeropredict ./aeropredict
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps -e .

# Trained models are committed under artifacts/ so the API is demoable immediately;
# data/ is downloaded on first use of the fleet endpoints (or mount ./data).
COPY artifacts ./artifacts

EXPOSE 8010
CMD ["uvicorn", "aeropredict.api.main:app", "--host", "0.0.0.0", "--port", "8010"]
