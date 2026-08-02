# tests/test_viewer_flowmap.py
"""The shared flow-map/KPI component: its JS API surface, its token
discipline (colors only via %%token%%/T — the dashboard reuses these strings
verbatim), and self-containment (no external references)."""

from __future__ import annotations

import re

from flashruntime.viewer.flowmap import FLOWMAP_CSS, FLOWMAP_JS


def test_js_defines_the_component_api():
    for name in ("renderKpiTiles", "renderFlowmap", "renderDetail", "fmStateColor", "fmFmtBytes"):
        assert f"function {name}" in FLOWMAP_JS, f"missing {name}"


def test_kpi_color_is_escaped():
    # renderKpiTiles interpolates t.color into a style="color:..." attribute;
    # per the component's escaping policy every interpolated value passes
    # fmEsc, so a malicious/odd t.color can't break out of the attribute.
    assert "fmEsc(t.color)" in FLOWMAP_JS


def test_css_targets_the_contract_elements():
    for sel in ("#kpis", "#flowmap", "#detail", ".fm-node", ".fm-selected"):
        assert sel in FLOWMAP_CSS, f"missing selector {sel}"


def test_css_uses_token_placeholders():
    assert "%%panel%%" in FLOWMAP_CSS and "%%border%%" in FLOWMAP_CSS
    # no hardcoded hex colors — the single-source-of-truth palette rule
    stripped = re.sub(r"%%[a-z_]+%%", "", FLOWMAP_CSS)
    assert re.search(r"#[0-9a-fA-F]{6}\b", stripped) is None


def test_component_is_self_contained():
    blob = FLOWMAP_CSS + FLOWMAP_JS
    assert "://" not in blob  # no external URLs, ever
    assert "fetch(" not in FLOWMAP_JS  # polling belongs to the page, not the component
