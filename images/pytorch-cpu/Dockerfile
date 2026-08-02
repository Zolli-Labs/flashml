# Curated image: pytorch-cpu
#
# Alias resolved in apps/api/flashml_cloud_api/images.py -> CURATED["pytorch-cpu"].
# packages = stdlib + {torch, numpy}. Keep this Dockerfile and that set in
# agreement: installing a package here without adding it there makes
# preflight reject code that would actually run, and adding it there without
# installing it here makes preflight approve code that will crash on a
# volunteer's machine.
#
# WHY NOT `FROM pytorch/pytorch` — two reasons, both load-bearing:
#   1. There is no CPU-only tag. `pytorch/pytorch:2.3.1-cpu` does not exist
#      (404 from Docker Hub); every published 2.3.1 tag is a CUDA build.
#      This Dockerfile previously named it and could never have built.
#   2. Those tags are 3.7-9.4 GB because they carry CUDA and cuDNN. Our nodes
#      are volunteers' laptops on home connections and M1 has no GPU support,
#      so that is multiple gigabytes of driver stack downloaded to run a CPU
#      job.
# Installing the CPU wheel on the same python:3.11.9-slim base the other two
# curated images use lands near 1 GB AND shares every layer below torch with
# them — a host that has run any FlashML job already holds most of this image.
#
# SECURITY (load-bearing — see ../README.md and
# flashml-cloud/docs/superpowers/plans/2026-08-01-windows-hosts.md):
# the final USER here is what lets flashnode omit `--user uid:gid` on
# Windows hosts. Do not add a later `USER root`, and keep FLASHML_UID equal
# to the other two curated images (python-slim, sklearn) so /work ownership
# is predictable regardless of which image a task uses.
FROM docker.io/library/python:3.11.9-slim

ARG FLASHML_UID=10001
ARG FLASHML_GID=10001

# --index-url, NOT --extra-index-url: the CPU build must come from PyTorch's
# own index. With --extra-index-url pip is free to resolve `torch` from PyPI
# instead, which is the CUDA build — silently reintroducing the gigabytes
# this image exists to avoid, and only noticeable as a mysteriously slow
# first job on a volunteer's machine.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 \
    && pip install --no-cache-dir numpy==1.26.4 \
    && groupadd --gid "${FLASHML_GID}" flashml \
    && useradd --uid "${FLASHML_UID}" --gid "${FLASHML_GID}" \
       --no-create-home --shell /usr/sbin/nologin flashml \
    && mkdir -p /work \
    && chown "${FLASHML_UID}:${FLASHML_GID}" /work

WORKDIR /work

# Fixed, non-"nobody" uid, matching python-slim and sklearn.
USER 10001:10001
