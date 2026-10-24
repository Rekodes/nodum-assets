# Una imagen por algoritmo, construida desde la raíz del repositorio:
# docker build -f shared/Dockerfile --build-arg ALGORITHM=algorithms/generic/line_count .
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ARG ALGORITHM
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY shared/*.py ./
COPY ${ALGORITHM}.py ./main.py

USER 1000:1000
ENTRYPOINT ["python", "/app/main.py"]
