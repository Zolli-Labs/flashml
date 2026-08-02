"""What this host will and will not run, and why an unset env var is fine.

The prefix allowlist is what makes "donate a machine" a one-command setup
and what lets the cloud roll an image forward without touching any host. It
is also, being a prefix, the piece most likely to be widened by accident —
so the near-miss cases below matter as much as the happy path.
"""
from __future__ import annotations

import pytest

from flashnode.executor.argv_runner import ArgvDockerRunner
from flashnode.executor.docker_runner import DockerRunner
from flashnode.executor.images import (
    DEFAULT_ALLOWED_IMAGE_PREFIXES,
    InvalidImageReference,
    image_is_allowed,
    split_reference,
)
from flashnode.executor.runner import TaskExecutionError

CURATED = "ghcr.io/zolli-labs/flashml-pytorch-cpu:2026.08.1"


def test_a_curated_image_needs_no_configuration():
    """The whole point: default arguments, nothing set, it runs."""
    assert image_is_allowed(CURATED) is True


@pytest.mark.parametrize(
    "alias", ["python-slim", "sklearn", "pytorch-cpu"]
)
def test_every_curated_alias_is_allowed_by_default(alias):
    assert image_is_allowed(f"ghcr.io/zolli-labs/flashml-{alias}:2026.08.1")


def test_a_future_tag_is_allowed_without_touching_the_host():
    """A published image must reach existing machines unattended — this is
    the property the old exact-match allowlist did not have."""
    assert image_is_allowed("ghcr.io/zolli-labs/flashml-sklearn:2099.12.31")


def test_a_new_curated_image_is_allowed_without_touching_the_host():
    assert image_is_allowed("ghcr.io/zolli-labs/flashml-jax-cpu:2026.09.1")


# --- what the prefix must NOT let through ---------------------------------


@pytest.mark.parametrize(
    "image",
    [
        # Merely CONTAINING the namespace is not being in it. Without
        # parsing, a naive `prefix in image` would accept this.
        "evil.example.com/ghcr.io/zolli-labs/flashml-x:1",
        # A different org on the same registry.
        "ghcr.io/zolli-labz/flashml-sklearn:1",
        # Our org, but outside the flashml- namespace.
        "ghcr.io/zolli-labs/internal-admin:1",
        # Registry that merely starts with the same letters.
        "ghcr.io.evil.test/zolli-labs/flashml-x:1",
        # Plain public images.
        "alpine:3.20",
        "docker.io/library/python:3.11-slim",
    ],
)
def test_images_outside_the_namespace_are_refused(image):
    assert image_is_allowed(image) is False


def test_a_tag_cannot_smuggle_the_namespace():
    """Authorizing the whole string rather than the parsed name would let a
    crafted tag carry the prefix past the check."""
    assert image_is_allowed("evil.test/x:ghcr.io_zolli-labs_flashml-y") is False


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/zolli-labs/flashml-sklearn",       # no tag: docker would say :latest
        "ghcr.io/zolli-labs/flashml-sklearn:",      # empty tag
        "ghcr.io/zolli-labs/../flashml-sklearn:1",  # traversal-ish
        "ghcr.io/zolli-labs/flashml-sklearn:1 x",   # whitespace
        "ghcr.io/zolli-labs/flashml-sklearn:1\nrm", # newline
        "",
        "   ",
    ],
)
def test_malformed_or_unversioned_references_are_refused(image):
    """Fails closed and returns False — never raises into a caller that might
    read the exception as an unrelated failure."""
    assert image_is_allowed(image) is False


def test_an_untagged_reference_is_refused_by_name():
    """A floating :latest means the image a host runs is not the image the
    cloud validated the job against."""
    with pytest.raises(InvalidImageReference, match="latest"):
        split_reference("ghcr.io/zolli-labs/flashml-sklearn")


def test_digest_references_parse():
    name, version = split_reference(
        "ghcr.io/zolli-labs/flashml-sklearn@sha256:" + "a" * 64
    )
    assert name == "ghcr.io/zolli-labs/flashml-sklearn"
    assert version.startswith("sha256:")


def test_a_registry_port_is_not_mistaken_for_a_tag():
    name, version = split_reference("registry.test:5000/zolli-labs/flashml-x:2.0")
    assert name == "registry.test:5000/zolli-labs/flashml-x"
    assert version == "2.0"


# --- the operator escape hatch --------------------------------------------


def test_an_explicit_exact_image_is_additive():
    """FLASHNODE_ALLOWED_IMAGES is for self-hosting and integration tests.
    Anyone who can set it can already run anything on that machine."""
    assert image_is_allowed("python:3.11-alpine") is False
    assert image_is_allowed(
        "python:3.11-alpine", frozenset({"python:3.11-alpine"})
    ) is True


def test_an_explicit_exact_image_does_not_disable_the_builtin():
    assert image_is_allowed(CURATED, frozenset({"python:3.11-alpine"})) is True


def test_narrowing_the_prefix_set_narrows_what_runs():
    """The shape a host owner's policy.json uses to forbid more."""
    assert image_is_allowed(CURATED, frozenset(), frozenset()) is False


# --- the runners actually consult it --------------------------------------


def test_docker_runner_default_construction_accepts_a_curated_image(tmp_path):
    """Constructed with no arguments — the volunteer's configuration."""
    runner = DockerRunner()
    assert image_is_allowed(CURATED, runner.allowed_images, runner.allowed_image_prefixes)


def test_docker_runner_still_refuses_an_off_namespace_image(tmp_path):
    runner = DockerRunner()
    with pytest.raises(TaskExecutionError, match="not allowlisted"):
        runner.run(
            {"module": next(iter(runner.allowed_modules)), "image": "alpine:3.20"},
            tmp_path,
            {},
        )


def test_argv_runner_still_refuses_an_off_namespace_image(tmp_path):
    runner = ArgvDockerRunner()
    with pytest.raises(TaskExecutionError, match="not allowlisted"):
        runner.run({"argv": ["echo", "hi"], "image": "alpine:3.20"}, tmp_path, {})


def test_argv_runner_default_construction_accepts_a_curated_image():
    runner = ArgvDockerRunner()
    assert image_is_allowed(CURATED, runner.allowed_images, runner.allowed_image_prefixes)


def test_the_builtin_prefix_is_a_namespace_not_a_registry():
    """`ghcr.io/` alone would trust every package on GitHub."""
    for prefix in DEFAULT_ALLOWED_IMAGE_PREFIXES:
        assert prefix.count("/") >= 2, f"{prefix!r} is too broad to be a namespace"
        assert not prefix.endswith("/"), f"{prefix!r} must bind to a name, not a directory"
