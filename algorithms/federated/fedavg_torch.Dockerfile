# El script de FedAvg con PyTorch sobre CPU. Se construye desde la raíz: make build-torch.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu
COPY algorithms/federated/fedavg_mlp.py algorithms/federated/fedavg_torch.py /app/
RUN useradd --uid 1000 --no-create-home app
USER 1000
ENTRYPOINT ["python", "/app/fedavg_torch.py"]
