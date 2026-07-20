# FlashNode agent image.
# Build from the Zolli-Labs workspace root (needs the sibling flashruntime
# checkout for the protocol package):
#   docker build -f flashnode/Dockerfile -t flashml/node:poc-v1 .
FROM python:3.11-slim

RUN useradd --create-home --uid 10001 flashnode
WORKDIR /app

COPY flashruntime/pyproject.toml flashruntime/README.md flashruntime/LICENSE ./flashruntime/
COPY flashruntime/flashruntime ./flashruntime/flashruntime
COPY flashruntime/flashml_workloads ./flashruntime/flashml_workloads
COPY flashnode/pyproject.toml flashnode/README.md flashnode/LICENSE ./flashnode/
COPY flashnode/flashnode ./flashnode/flashnode

RUN pip install --no-cache-dir ./flashruntime ./flashnode

USER flashnode
CMD ["flashnode", "agent"]
