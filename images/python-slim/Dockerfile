# Curated image: python-slim
#
# Alias resolved in apps/api/flashml_cloud_api/images.py -> CURATED["python-slim"].
# packages = stdlib only (frozenset(sys.stdlib_module_names)) — this image
# must provide nothing beyond what the base Python installation already
# gives a script, or preflight's unknown-import check would be validating
# against a lie (see images.py module docstring).
#
# SECURITY (load-bearing — see ../README.md and
# flashml-cloud/docs/superpowers/plans/2026-08-01-windows-hosts.md):
# the final USER here is what lets flashnode omit `--user uid:gid` on
# Windows hosts (Docker Desktop cannot map host uids the way the Linux
# daemon can). Without it, every Windows volunteer would run task code as
# root inside the container. Do not add a later `USER root` after this
# line, and do not change FLASHML_UID without updating sklearn/ and
# pytorch-cpu/ to match — the three images must agree on one uid so /work
# ownership is predictable regardless of which image a task uses.
FROM docker.io/library/python:3.11.9-slim

ARG FLASHML_UID=10001
ARG FLASHML_GID=10001

RUN groupadd --gid "${FLASHML_GID}" flashml \
    && useradd --uid "${FLASHML_UID}" --gid "${FLASHML_GID}" \
       --no-create-home --shell /usr/sbin/nologin flashml \
    && mkdir -p /work \
    && chown "${FLASHML_UID}:${FLASHML_GID}" /work

WORKDIR /work

# Fixed, non-"nobody" uid — deliberately not the last line's only job:
# everything above must be root-owned setup, and nothing below this line
# may run as root again.
USER 10001:10001
