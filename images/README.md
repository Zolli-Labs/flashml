# Curated images

Dockerfiles for the three curated M1 images named in
`apps/api/flashml_cloud_api/images.py` (`CURATED`): `python-slim`, `sklearn`,
`pytorch-cpu`. Each `packages` frozenset there must stay in sync with what
its Dockerfile actually installs — a mismatch means preflight's
unknown-import check validates a repo's imports against a lie.

**Not built or pushed yet.** These Dockerfiles exist so the image content
and the non-root `USER` contract can be reviewed and tested (see
`apps/api/tests/test_image_dockerfiles.py`) now. Building and pushing them
to the registry `images.py`'s pinned `reference` fields point at is Plan 7's
job.

## The non-root `USER` is load-bearing for Windows hosts

Every image here ends with `USER 10001:10001` — a fixed, dedicated uid
(not `nobody`, whose id varies by base image and complicates `/work`
ownership), identical across all three images so `/work` ownership is
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

If you touch these Dockerfiles:

- Keep a `USER <uid>:<gid>` as the **last** user-affecting instruction.
- Keep the uid non-root (not `0`, not `root`) and identical across all
  three Dockerfiles.
- Keep each image's installed packages in exact sync with the matching
  `CuratedImage.packages` entry in `apps/api/flashml_cloud_api/images.py`.

`apps/api/tests/test_image_dockerfiles.py` enforces the first two
mechanically. The third (package sync) is enforced by inspection — there is
no build step in CI to catch it yet.
