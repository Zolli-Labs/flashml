# Curated images

Dockerfiles for the curated images named in
`apps/api/flashml_cloud_api/images.py` (`CURATED`): `python-slim`, `sklearn`,
`pytorch-cpu`, and `pytorch-cuda`. Each `packages` frozenset there must stay
in sync with what its Dockerfile actually installs — a mismatch means
preflight's unknown-import check validates a repo's imports against a lie.

**Not built or pushed yet.** These Dockerfiles exist so the image content
and the non-root `USER` contract can be reviewed and tested (see
`tests/test_image_dockerfiles.py`) now. Building and pushing them
to the registry `images.py`'s pinned `reference` fields point at is Plan 7's
job.

## The images

| alias | base | provides | size |
|---|---|---|---|
| `python-slim` | `python:3.11.9-slim` | stdlib only | ~150 MB |
| `sklearn` | `python:3.11.9-slim` | numpy, pandas, sklearn, scipy, joblib, threadpoolctl | ~1 GB |
| `pytorch-cpu` | `python:3.11.9-slim` | torch (CPU build), numpy | ~1 GB |
| `pytorch-cuda` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | torch (cu124), numpy | **multiple GB — see below** |

The first three share every layer below their own installs, so a host that
has run any FlashML job already holds most of the next one.

## `pytorch-cuda` is the expensive one, and only GPU hosts should ever pull it

It shares **nothing** with the other three — different base, different
distro, python 3.10 instead of 3.11.9. The `nvidia/cuda` runtime base is
about 1.5 GB compressed on its own, the cu124 torch wheel is another ~800 MB,
and torch's bundled CUDA runtime dependencies land on top of that. Call it
**several gigabytes to pull and more than that on disk** — it is an order of
magnitude larger than `pytorch-cpu`, which is exactly why `pytorch-cpu`
exists and stays the default.

It is useful **only to a host with an NVIDIA GPU and a working driver.**
Nothing in the image detects that; placement does, upstream:

- `flashnode` probes `nvidia-smi` and advertises `capabilities.gpus`.
- The scheduler's GPU gate refuses a `gpus: N` task on a node advertising
  fewer than `N`, failing closed.
- Only a task that passed that gate ever names this image.

A CPU-only volunteer therefore never pulls it. A host that *believes* it
contributed a GPU but did not will be told so by `flashnode doctor`'s
(non-gating) GPU check rather than by a task failing.

CUDA **12.4**, not the 12.8 the July 2026 RunPod validation ran: a CUDA minor
version requires a driver at least as new, a volunteer's driver is not
something we control, and 12.4 is the widest currently-supported floor. A
host too old even for 12.4 fails `doctor`'s image-pull check — a diagnosable
refusal at enrolment rather than tasks dying on a stranger's machine.

`torch.cuda.is_available()` is **not** asserted in CI: GitHub-hosted runners
have no device, so it is False there however correct the image is. What CI
does check is that `torch.version.cuda` is 12.4 — the failure `--index-url`
exists to prevent. Proving the image actually runs on a device is the rented-
GPU validation in the GPU plan's Task 11.

## Publishing a new image is a two-repo, one-tag-bump operation

`.github/workflows/images.yml` refuses to overwrite an existing tag, on
purpose — a repushed tag reaches only the hosts that never pulled it. So
adding an alias to the matrix is not, by itself, enough to publish it:

1. Bump `IMAGE_TAG` in `.github/workflows/images.yml`.
2. Bump `IMAGE_TAG` and add the `CuratedImage` entry (alias, reference,
   `packages`) in flashml-cloud's `apps/api/flashml_cloud_api/images.py`.
3. After the first successful publish, make the GHCR package public by hand
   (Danger Zone → Change visibility) — there is no REST endpoint for it, and
   until it is done every volunteer's `docker pull` gets a 401.
4. Run flashml-cloud's
   `test_every_curated_image_is_anonymously_pullable`, which is what actually
   catches steps 1–3 having disagreed.

Running the workflow **without** step 1 fails the three already-published
images on the immutability guard while the new one succeeds — `fail-fast:
false` means the new image still lands, but the run is red and the red is
telling the truth.

## The non-root `USER` is load-bearing for Windows hosts

Every image here ends with `USER 10001:10001` — a fixed, dedicated uid
(not `nobody`, whose id varies by base image and complicates `/work`
ownership), identical across all of them so `/work` ownership is
predictable regardless of which image a task uses.

This is not incidental hardening. `flashnode/flashnode/executor/hardening.py`
passes `--user {uid}:{gid}` to `docker run` on POSIX hosts, but
`os.getuid`/`os.getgid` do not exist on Windows, so Windows hosts omit the
flag entirely (see
`flashml-cloud/docs/superpowers/plans/2026-08-01-windows-hosts.md`, "The
trap at the centre of this plan"). **Omitting `--user` is only safe because
these images declare a non-root `USER`.** If any curated image regresses to
running as root (e.g. someone "simplifies" a Dockerfile and drops the final
`USER` line, or adds a later `USER root` that undoes it), every Windows
volunteer's container silently starts running strangers' code as root —
and nothing in `docker run`'s own flags would catch that, because
`--user` was never passed to override the image's own default.

`--gpus` does not change any of this. A GPU container is still
`--network none --read-only --cap-drop=ALL --security-opt=no-new-privileges`
as uid 10001. The one thing GPUs do open — VRAM is not zeroed between
containers, so one tenant's residue is in principle readable by the next —
is a driver-level problem that no flag in `hardening.py` closes; it is
recorded in the GPU spec's §5.1 and is deliberately out of scope, not
inherited quietly.

If you touch these Dockerfiles:

- Keep a `USER <uid>:<gid>` as the **last** user-affecting instruction.
- Keep the uid non-root (not `0`, not `root`) and identical across all
  four Dockerfiles.
- Keep each image's installed packages in exact sync with the matching
  `CuratedImage.packages` entry in `apps/api/flashml_cloud_api/images.py`.
- Add the new directory to `EXPECTED_ALIASES` in
  `tests/test_image_dockerfiles.py` — that test fails on an image directory
  it does not know about, so an unchecked image cannot slip in.

`tests/test_image_dockerfiles.py` enforces the first two
mechanically. The third (package sync) is enforced by inspection — there is
no build step in CI to catch it yet.
