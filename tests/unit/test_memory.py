"""Tests unitaires pour le loader de mémoire Markdown."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from jvl.core import memory


@pytest.fixture
def patch_memory_paths(tmp_path, monkeypatch):
    """Surcharge les chemins de mémoire et d'état vers un dossier temporaire."""
    tmp_mem_dir = tmp_path / "jvl" / "memory"
    tmp_state_file = tmp_path / "jvl" / "state.json"

    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_mem_dir)
    monkeypatch.setattr(memory, "STATE_FILE", tmp_state_file)

    return tmp_mem_dir, tmp_state_file


def test_initialize_memory_structure(patch_memory_paths):
    mem_dir, _ = patch_memory_paths

    assert not mem_dir.exists()

    memory.initialize_memory_structure()

    assert mem_dir.exists()
    assert (mem_dir / "projects").exists()
    assert (mem_dir / "global.md").exists()
    assert "Mémoire Globale" in (mem_dir / "global.md").read_text(encoding="utf-8")


def test_get_active_project(patch_memory_paths):
    _, state_file = patch_memory_paths

    # Cas 1 : Pas de fichier d'état
    assert memory.get_active_project() is None

    # Cas 2 : Fichier d'état valide
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"active_project": "client-a"}), encoding="utf-8")
    assert memory.get_active_project() == "client-a"

    # Cas 3 : Fichier d'état corrompu
    state_file.write_text("invalid json", encoding="utf-8")
    assert memory.get_active_project() is None


def test_truncate_to_max_tokens():
    # 1 token = 4 char, donc max_tokens = 5 => 20 char max
    text = "line1\nline2\nline3\nline4\n"
    # len(text) est 24 char.
    res = memory.truncate_to_max_tokens(text, max_tokens=5)
    assert "line1\nline2\nline3" in res
    assert "tronqué" in res

    # Sans troncature
    res_short = memory.truncate_to_max_tokens("hello", max_tokens=5)
    assert res_short == "hello"


def test_load_memory_without_project(patch_memory_paths):
    mem_dir, _ = patch_memory_paths
    memory.initialize_memory_structure()

    # Créer wedr-internal.md
    (mem_dir / "wedr-internal.md").write_text("WEDR RULES", encoding="utf-8")

    loaded = memory.load_memory(project=None)
    assert "Mémoire Globale :" in loaded
    assert "WEDR RULES" in loaded
    assert "Contexte Projet" not in loaded


def test_load_memory_with_project(patch_memory_paths):
    mem_dir, _ = patch_memory_paths
    memory.initialize_memory_structure()

    # Configurer un projet
    project_dir = mem_dir / "projects" / "client-a"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "memory.md").write_text("CLIENT-A DECISIONS", encoding="utf-8")
    (project_dir / "pending.md").write_text("CLIENT-A TODOS", encoding="utf-8")

    loaded = memory.load_memory(project="client-a")
    assert "Mémoire Globale :" in loaded
    assert "CLIENT-A DECISIONS" in loaded
    assert "CLIENT-A TODOS" in loaded


def test_strip_memory_tags():
    from jvl.core.memory import strip_memory_tags

    # 1. Aucun tag
    assert strip_memory_tags(None) is None
    assert strip_memory_tags("hello") == "hello"

    # 2. Tag simple
    prompt = "Tu es JVLIVS.\n\n<MEMOIRE_UTILISATEUR>\n### Mémoire Globale :\nSome Prefs\n</MEMOIRE_UTILISATEUR>"
    assert strip_memory_tags(prompt) == "Tu es JVLIVS."

    # 3. Uniquement le tag
    prompt_only_tag = "<MEMOIRE_UTILISATEUR>\nSome Prefs\n</MEMOIRE_UTILISATEUR>"
    assert strip_memory_tags(prompt_only_tag) is None


def test_enrich_system_prompt(patch_memory_paths):
    from jvl.core.memory import enrich_system_prompt
    mem_dir, _ = patch_memory_paths
    memory.initialize_memory_structure()

    # Configurer une mémoire globale
    (mem_dir / "global.md").write_text("My Preferences", encoding="utf-8")

    # 1. Enrichir prompt de base
    res = enrich_system_prompt("Base Prompt")
    assert "Base Prompt" in res
    assert "<MEMOIRE_UTILISATEUR>" in res
    assert "My Preferences" in res

    # 2. Re-enrichir un prompt déjà enrichi (doit nettoyer l'ancien et mettre le nouveau)
    (mem_dir / "global.md").write_text("Updated Preferences", encoding="utf-8")
    res_second = enrich_system_prompt(res)
    assert "Base Prompt" in res_second
    assert "My Preferences" not in res_second
    assert "Updated Preferences" in res_second

