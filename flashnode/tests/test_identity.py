from flashnode.identity.store import load_or_create_node_id


def test_node_id_is_stable_across_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHNODE_STATE_DIR", str(tmp_path))
    first = load_or_create_node_id()
    second = load_or_create_node_id()
    assert first == second
    assert first.startswith("fn-")
    assert (tmp_path / "node-id").read_text().strip() == first


def test_distinct_state_dirs_get_distinct_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHNODE_STATE_DIR", str(tmp_path / "a"))
    a = load_or_create_node_id()
    monkeypatch.setenv("FLASHNODE_STATE_DIR", str(tmp_path / "b"))
    b = load_or_create_node_id()
    assert a != b
