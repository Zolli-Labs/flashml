"""Which container images this host will run, and why it is a prefix.

A volunteer installs FlashNode and runs `flashnode work`. They do not build
an image, do not log in to a registry, and do not maintain a list of image
references — the built-in allowlist below is what makes that true. When the
cloud publishes a new curated image, hosts pick it up on their next task
with no action from their owner.

WHY A NAMESPACE PREFIX RATHER THAN EXACT REFERENCES
---------------------------------------------------
Exact matching was the original design and it does not survive contact with
more than a handful of machines: every published image would strand every
host until its owner edited an env var, which in practice means the fleet
runs a mix of versions and security fixes never fully land.

The prefix is a namespace *we* control and publish to. It is genuinely
weaker than three fixed strings, and the honest reason that is acceptable
is this: the coordinator already tells a node which module to run, which
argv to execute, and which inputs to fetch. An attacker who could name an
arbitrary image has, by then, long since been able to name arbitrary work.
The image allowlist defends against a *confused* coordinator naming
something off-namespace — say `alpine` with a mounted socket — not against
a fully compromised one. Pinning exact tags buys nothing against the latter
while costing the whole fleet its ability to be updated.

See M1_DECISIONS.md D16.

WHAT THIS STILL REFUSES
-----------------------
- Anything outside the namespace, including near-misses that merely contain
  it (`evil.example.com/ghcr.io/zolli-labs/flashml-x`).
- References with no explicit tag or digest. Docker would silently append
  `:latest`, which is a floating pointer to whatever was pushed last — the
  exact mutability the curated-image tags are immutable to avoid.
- Malformed references, rather than passing them to `docker run` and hoping
  its parser agrees with ours about where the name ends.

TWO DIFFERENT KNOBS, TWO DIFFERENT DIRECTIONS
---------------------------------------------
- ``FLASHNODE_ALLOWED_IMAGES`` (env, passed here as ``allowed_exact``) is
  ADDITIVE. It is an operator/testing escape hatch — self-hosting the whole
  stack, or an integration test running `python:3.11-alpine`. Anyone who can
  set it can already run anything on that machine, so it grants nothing they
  did not have.
- ``policy.json`` ``allowed_images`` is SUBTRACTIVE. That is the host
  *owner's* safety policy and it intersects with whatever the above permits
  (see flashnode/config): an owner can forbid more, never permit code the
  build does not trust.
"""

from __future__ import annotations

import re

__all__ = [
    "DEFAULT_ALLOWED_IMAGE_PREFIXES",
    "InvalidImageReference",
    "image_is_allowed",
    "split_reference",
]

#: Namespace the cloud publishes curated task images to. Must stay in step
#: with REGISTRY_PREFIX in flashml-cloud's apps/api/flashml_cloud_api/
#: images.py — the e2e suite asserts every reference that API can emit is
#: accepted here, because this seam has broken four times before and neither
#: repo's own tests can see it (M1_DECISIONS.md D12).
DEFAULT_ALLOWED_IMAGE_PREFIXES = frozenset({"ghcr.io/zolli-labs/flashml-"})

# Deliberately stricter than the distribution spec. A reference we cannot
# confidently parse is one we cannot confidently authorize, and every
# reference we actually need is a plain lowercase path — so the safe reading
# is to reject the exotic rest rather than approximate it.
# Structure: an optional `host[:port]/`, then one or more lowercase path
# components.
#
# The leading group exists for the port. A private registry is reached as
# `registry.example:5000/team/img:2.0`, and while split_reference() gets the
# tag boundary right (only a ':' after the final '/' can start a tag), the
# name it hands back still carries that ':5000' — so a name pattern with no
# room for a port would reject every self-hosted registry as malformed.
#
# The trailing path group is `*`, not `+`: a single-component name like
# `python` is valid Docker Hub shorthand, and an owner who explicitly
# allowlists one has named something real. It can never match a namespace
# prefix (those all contain '/'), so permitting the shape costs nothing.
_NAME_RE = re.compile(
    r"^(?:[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]{1,5})?/)?"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$"
)
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


class InvalidImageReference(ValueError):
    """A reference that could not be parsed, so could not be authorized."""


def split_reference(image: str) -> tuple[str, str]:
    """Split ``image`` into ``(name, tag_or_digest)``.

    Raises ``InvalidImageReference`` for anything malformed or lacking an
    explicit version. Splitting is done here, once, rather than at each call
    site: authorizing the *whole* string by prefix would let a crafted tag
    smuggle the namespace past the check, so every caller must be comparing
    the same parsed name this function returns.
    """
    if not isinstance(image, str) or not image or image.strip() != image:
        raise InvalidImageReference(f"not a usable image reference: {image!r}")
    if any(c.isspace() or ord(c) < 32 for c in image):
        raise InvalidImageReference(f"image reference contains whitespace or control characters: {image!r}")

    # Digest form wins: `name@sha256:...` has no tag to strip afterwards.
    if "@" in image:
        name, _, digest = image.partition("@")
        version = digest
        if not _DIGEST_RE.match(digest):
            raise InvalidImageReference(f"unsupported digest in {image!r} (want sha256:<64 hex>)")
    else:
        # A tag cannot contain '/', so only a ':' after the final '/' can be
        # a tag separator — anything earlier is a registry port.
        head, sep, tail = image.rpartition(":")
        if sep and "/" not in tail:
            name, version = head, tail
            if not _TAG_RE.match(version):
                raise InvalidImageReference(f"malformed tag in {image!r}")
        else:
            # No explicit version. Docker would default to `:latest`; we
            # refuse instead, because a floating tag means the image a host
            # runs is not the image the cloud validated the job against.
            raise InvalidImageReference(
                f"image {image!r} has no explicit tag or digest — refusing to "
                "let docker default it to :latest"
            )

    if ".." in name or not _NAME_RE.match(name):
        raise InvalidImageReference(f"malformed image name in {image!r}")
    return name, version


def image_is_allowed(
    image: str,
    allowed_exact: frozenset[str] = frozenset(),
    allowed_prefixes: frozenset[str] = DEFAULT_ALLOWED_IMAGE_PREFIXES,
) -> bool:
    """Whether this host will run ``image``.

    Fails closed: an unparseable reference is not allowed, never an
    exception the caller might treat as an unrelated failure.

    ``allowed_exact`` is the host owner's explicit list (env var or
    policy.json) and is matched against the FULL reference — an owner naming
    one exact pinned image means that image, not its namespace.
    """
    try:
        name, _version = split_reference(image)
    except InvalidImageReference:
        return False
    if image in allowed_exact:
        return True
    return any(name.startswith(prefix) for prefix in allowed_prefixes)
