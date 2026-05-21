"""Schema-validation tests for agents.registry.AGENT_REGISTRY.
Validates all four specialists. Does not require FAISS indices.
"""
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "multi-agent_system"))
import pytest

@pytest.fixture(scope="module")
def registry():
    from agents.registry import AGENT_REGISTRY
    return AGENT_REGISTRY

REQUIRED_FIELDS = {"name", "folder_path", "role_prompt", "domain_scope"}
EXPECTED_KEYS   = {"cardiologist", "endocrinologist",
                   "gastroenterologist", "infectionist"}

def test_registry_has_all_four_specialists(registry):
    assert set(registry.keys()) == EXPECTED_KEYS, \
        f"Expected {EXPECTED_KEYS}, got {set(registry.keys())}"

def test_every_entry_has_required_fields_only(registry):
    for key, cfg in registry.items():
        missing = REQUIRED_FIELDS - set(cfg.keys())
        extra   = set(cfg.keys()) - REQUIRED_FIELDS
        assert not missing, f"{key}: missing {missing}"
        assert not extra,   f"{key}: unexpected {extra}"

def test_every_name_is_nonempty(registry):
    for key, cfg in registry.items():
        assert isinstance(cfg["name"], str) and cfg["name"].strip(), \
            f"{key}: name empty"

def test_every_domain_scope_is_substantial(registry):
    for key, cfg in registry.items():
        assert len(cfg["domain_scope"].strip()) >= 20, \
            f"{key}: domain_scope too short"

def test_every_role_prompt_is_substantial(registry):
    for key, cfg in registry.items():
        assert len(cfg["role_prompt"]) >= 500, \
            f"{key}: role_prompt too short ({len(cfg['role_prompt'])} chars)"

def test_new_specialists_have_correct_keys(registry):
    assert "gastroenterologist" in registry
    assert "infectionist" in registry

def test_new_role_prompts_contain_critical_rule(registry):
    for key in ("gastroenterologist", "infectionist"):
        assert "CRITICAL_RULE" in registry[key]["role_prompt"], \
            f"{key}: CRITICAL_RULE block missing from role_prompt"
        assert "Insufficient evidence" in registry[key]["role_prompt"], \
            f"{key}: fallback phrase missing"
