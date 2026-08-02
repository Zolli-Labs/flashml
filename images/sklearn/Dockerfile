# Curated image: sklearn
#
# Alias resolved in apps/api/flashml_cloud_api/images.py -> CURATED["sklearn"].
# packages = stdlib + {numpy, pandas, sklearn, scipy, joblib, threadpoolctl}.
# The pip installs below must keep exactly that set importable — adding a
# package here without adding it to images.py's `packages` (or vice versa)
# reopens the "preflight validates against a lie" hole the module docstring
# warns about.
#
# SECURITY (load-bearing — see ../README.md and
# flashml-cloud/docs/superpowers/plans/2026-08-01-windows-hosts.md):
# the final USER here is what lets flashnode omit `--user uid:gid` on
# Windows hosts. Do not add a later `USER root`, and keep FLASHML_UID equal
# to the other two curated images (python-slim, pytorch-cpu) so /work
# ownership is predictable regardless of which image a task uses.
FROM docker.io/library/python:3.11.9-slim

ARG FLASHML_UID=10001
ARG FLASHML_GID=10001

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.5.0 \
        scipy==1.13.1 \
        joblib==1.4.2 \
        threadpoolctl==3.5.0 \
    && groupadd --gid "${FLASHML_GID}" flashml \
    && useradd --uid "${FLASHML_UID}" --gid "${FLASHML_GID}" \
       --no-create-home --shell /usr/sbin/nologin flashml \
    && mkdir -p /work \
    && chown "${FLASHML_UID}:${FLASHML_GID}" /work

WORKDIR /work

# Fixed, non-"nobody" uid, matching python-slim and pytorch-cpu.
USER 10001:10001
