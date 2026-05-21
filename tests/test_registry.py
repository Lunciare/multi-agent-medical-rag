"""Schema-validation tests for agents.registry.AGENT_REGISTRY.

A misconfigured registry entry would otherwise surface at runtime as an
indistinguishable routing failure. These tests fail-fast on:
  - missing or extra fields per entry,
  - non-existent folder_path,
  - empty name or empty / too-short domain_scope.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "multi-agent_system"))

import pytest


@pytest.fixture(scope="module")
def registry():
    from agents.registry import AGENT_REGISTRY
    return AGENT_REGISTRY


REQUIRED_FIELDS = {"name", "folder_path", "role_prompt", "domain_scope"}


def test_registry_has_at_least_two_specialists(registry):
    assert len(registry) >= 2, (
        f"AGENT_REGISTRY must contain ≥2 specialists, got {len(registry)}"
    )


def test_every_entry_has_required_fields_only(registry):
    for key, cfg in registry.items():
        missing = REQUIRED_FIELDS - set(cfg.keys())
        extra = set(cfg.keys()) - REQUIRED_FIELDS
        assert not missing, f"{key}: missing required fields {missing}"
        assert not extra,   f"{key}: unexpected extra fields {extra}"


def test_every_folder_path_exists(registry):
    for key, cfg in registry.items():
        assert os.path.isdir(cfg["folder_path"]), (
            f"{key}: folder_path {cfg['folder_path']!r} does not exist"
        )


def test_every_name_is_nonempty_string(registry):
    for key, cfg in registry.items():
        assert isinstance(cfg["name"], str) and cfg["name"].strip(), (
            f"{key}: name must be a non-empty string, got {cfg['name']!r}"
        )


def test_every_domain_scope_meets_minimum_length(registry):
    for key, cfg in registry.items():
        scope = cfg["domain_scope"]
        assert isinstance(scope, str) and len(scope.strip()) >= 10, (
            f"{key}: domain_scope must be ≥10 chars of meaningful text, "
            f"got {scope!r}"
        )


def test_every_role_prompt_is_substantial(registry):
    for key, cfg in registry.items():
        prompt = cfg["role_prompt"]
        assert isinstance(prompt, str) and len(prompt) >= 500, (
            f"{key}: role_prompt looks too short ({len(prompt)} chars); "
            f"the canonical prompt with _RULES_AND_FORMAT is ≥1500 chars"
        )
