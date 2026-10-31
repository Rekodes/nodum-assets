# El script de FedAvg con PyTorch sobre CPU. Se construye desde la raíz: make build-torch.
# Sin emular: pip corre en la plataforma de quien construye y baja las ruedas de la de destino; la imagen final no
# ejecuta nada al construirse.
FROM --platform=$BUILDPLATFORM python:3.12-slim AS deps
ARG TARGETARCH
RUN arch=$([ "$TARGETARCH" = arm64 ] && echo aarch64 || echo x86_64) && \
    pip install --no-cache-dir --target /deps --only-binary=:all: --implementation cp --python-version 3.12 \
      --platform "manylinux_2_28_$arch" --index-url https://download.pytorch.org/whl/cpu numpy torch

FROM python:3.12-slim
COPY --from=deps /deps /usr/local/lib/python3.12/site-packages
COPY algorithms/federated/fedavg_mlp.py algorithms/federated/fedavg_torch.py /app/
USER 1000
ENTRYPOINT ["python", "/app/fedavg_torch.py"]
