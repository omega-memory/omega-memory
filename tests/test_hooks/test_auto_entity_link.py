"""Test auto entity relationship from file claims."""


def test_extracts_project_from_path():
    from omega.hooks.coord_session_stop import _extract_project_entity

    assert _extract_project_entity("/Users/dev/Projects/omega/src/foo.py") == "omega"
    assert _extract_project_entity("/Users/dev/Projects/acme-app/lib/bar.ts") == "acme-app"
    assert _extract_project_entity("/tmp/test.py") is None


def test_builds_relationships_from_claims():
    from omega.hooks.coord_session_stop import _build_entity_links

    claims = [
        {"file_path": "/Users/dev/Projects/omega/src/bridge.py"},
        {"file_path": "/Users/dev/Projects/omega/website/app/page.tsx"},
    ]
    links = _build_entity_links(claims, current_project="omega")

    # Same project, no cross-project link
    assert len(links) == 0


def test_cross_project_link():
    from omega.hooks.coord_session_stop import _build_entity_links

    claims = [
        {"file_path": "/Users/dev/Projects/omega/src/bridge.py"},
        {"file_path": "/Users/dev/Projects/acme-app/lib/utils.ts"},
    ]
    links = _build_entity_links(claims, current_project="omega")

    assert len(links) == 1
    assert links[0]["from"] == "omega"
    assert links[0]["to"] == "acme-app"
    assert links[0]["relationship"] == "depends_on"
