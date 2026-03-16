FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

# Install with server + postgres extras (psycopg2-binary needs no build deps)
RUN pip install --no-cache-dir ".[server,postgres,encrypt]"

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV OMEGA_HOME=/data/omega \
    PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["omega", "serve", "--daemon"]
