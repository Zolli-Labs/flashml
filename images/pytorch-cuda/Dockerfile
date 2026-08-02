# Curated image: pytorch-cuda
#
# Alias resolved in apps/api/flashml_cloud_api/images.py -> CURATED["pytorch-cuda"].
# packages = stdlib + {torch, numpy} — the same set as pytorch-cpu, on a CUDA
# build of the same library. Keep this Dockerfile and that set in agreement:
# installing a package here without adding it there makes preflight reject
# code that would actually run, and adding it there without installing it
# here makes preflight approve code that will crash on a volunteer's machine.
# (The CURATED entry lives in the PRIVATE repo and is added with the IMAGE_TAG
# bump that first publishes this image — see ../README.md.)
#
# WHY CUDA 12.4 AND NOT 12.8 — the July 2026 RunPod validation ran 12.8, and
# this deliberately does not follow it. A volunteer's driver version is not
# something we control: a CUDA minor version needs a driver at least as new,
# and 12.4 is the widest currently-supported floor. A host too old even for
# 12.4 fails `flashnode doctor`'s image-pull check, which is a diagnosable
# refusal at enrolment rather than a task that dies on a stranger's machine.
#
# WHY NOT `FROM pytorch/pytorch` — the same reason pytorch-cpu gives, minus
# the size argument: those tags pin CUDA and cuDNN versions we would then be
# inheriting rather than choosing, and the version choice above is the whole
# point of this file.
#
# THIS IMAGE IS BIG — see ../README.md. The base alone is ~1.5 GB compressed
# and the torch cu124 wheel another ~800 MB, before its bundled CUDA runtime
# dependencies. It shares no layer with the other three curated images, which
# all sit on python:3.11.9-slim. It is only ever pulled by a host that
# advertised a GPU and was placed a `gpus: N` task; a CPU-only volunteer
# never sees it.
#
# SECURITY (load-bearing — see ../README.md and
# flashml-cloud/docs/superpowers/plans/2026-08-01-windows-hosts.md):
# the final USER here is what lets flashnode omit `--user uid:gid` on
# Windows hosts. Do not add a later `USER root`, and keep FLASHML_UID equal
# to the other curated images (python-slim, sklearn, pytorch-cpu) so /work
# ownership is predictable regardless of which image a task uses. Note that
# `--gpus` does not weaken any of the other hardening flags: the container
# still runs `--network none --read-only --cap-drop=ALL` as unprivileged
# 10001. The one thing it does open — VRAM is not zeroed between containers
# — is a driver-level problem no flag here can close, and is recorded in
# the GPU spec's §5.1 rather than papered over.
#
# Fully qualified for the same reason the other three are: `nvidia/cuda` is a
# Docker Hub org, not `library`, so this is `docker.io/nvidia/cuda`.
FROM docker.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG FLASHML_UID=10001
ARG FLASHML_GID=10001

# Ubuntu 22.04 carries python 3.10, not the 3.11.9 the other curated images
# use. That difference is real and deliberate: it is the interpreter NVIDIA's
# supported base ships, and pinning a newer one would mean building python
# from source inside the image every host has to pull.
#
# `python-is-python3` is not cosmetic. flashnode's docker runner executes
# `python -m <module>` and a task's own entrypoint is `python /work/...`;
# Ubuntu provides only `python3`, so without this symlink every task in this
# image fails with "executable file not found" — and only on GPU hosts.
#
# DEBIAN_FRONTEND is set for this RUN only, not with ENV: it must not leak
# into the environment a submitted job runs in.
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# --index-url, NOT --extra-index-url: the cu124 build must come from
# PyTorch's own index. With --extra-index-url pip is free to resolve `torch`
# from PyPI instead, which is the default CUDA build for whatever version
# PyPI happens to carry — silently pinning a CUDA version this file exists to
# choose, and only noticeable as a job that fails on some volunteers' drivers
# and not others.
#
# torch 2.4.1, not the 2.3.1 pytorch-cpu pins: 2.4.0 is the earliest release
# with cu124 wheels at all, so the two images cannot share a version until
# pytorch-cpu moves. numpy is pinned to 1.26.4 to match pytorch-cpu, and
# because torch 2.4.x is built against numpy 1.x.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1 \
    && pip install --no-cache-dir numpy==1.26.4 \
    && groupadd --gid "${FLASHML_GID}" flashml \
    && useradd --uid "${FLASHML_UID}" --gid "${FLASHML_GID}" \
       --no-create-home --shell /usr/sbin/nologin flashml \
    && mkdir -p /work \
    && chown "${FLASHML_UID}:${FLASHML_GID}" /work

WORKDIR /work

# Fixed, non-"nobody" uid, matching python-slim, sklearn and pytorch-cpu.
USER 10001:10001
