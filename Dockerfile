FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY src/ src/
COPY server/ server/
COPY dashboard/ dashboard/
COPY data/chunks.json data/chunks.json
COPY 01_Product_Disclosure_Sheet.pdf 02_Policy_Wording.pdf 03_Benefits_Schedule.pdf 04_Claims_Procedure_Guide.pdf ./

RUN python src/index.py

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
