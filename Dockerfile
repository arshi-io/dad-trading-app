FROM python:3.12-slim

# MALLOC_ARENA_MAX: glibc otherwise grows a memory arena per thread (yfinance/live polling threads)
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Kolkata MALLOC_ARENA_MAX=2
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code ./code
RUN useradd --create-home app && mkdir -p code/data/snapshots code/data/prices && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % __import__('os').environ.get('PORT', '8000'))"
# Railway: attach ONE volume at /data. The trades/watchlist DB, snapshots and price cache are linked into it
# (code/data also holds Python packages, so the volume can't be mounted there directly).
CMD ["sh", "-c", "if [ -d /data ]; then mkdir -p /data/snapshots /data/prices && for d in snapshots prices; do rm -rf code/data/$d && ln -s /data/$d code/data/$d; done && ln -sf /data/papa.db code/data/papa.db && ln -sf /data/pairs_cointegration_cache.json code/data/pairs_cointegration_cache.json; fi; exec uvicorn code.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-server-header --proxy-headers"]
