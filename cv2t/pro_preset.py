"""
Professional Mode preset management.

Each preset defines a reusable set of AI text-cleanup instructions:
tone, grammar, punctuation flags, a custom system prompt, domain
vocabulary to preserve, and the OpenAI model to use.

Presets are stored as individual JSON files in the ``config/presets/``
directory.  Five built-in presets are always available even when no
files exist on disk.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ProPreset:
    """A single Professional Mode preset."""

    name: str = "General Professional"
    system_prompt: str = ""
    fix_tone: bool = True
    fix_grammar: bool = True
    fix_punctuation: bool = True
    vocabulary: str = ""
    model: str = "gpt-5.4-mini"

    # ── Serialisation ────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write this preset to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        log.info("Preset saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> ProPreset:
        """Load a preset from a JSON file."""
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        instance = cls(**{k: v for k, v in data.items() if k in known})
        instance.validate()
        return instance

    def validate(self) -> None:
        """Clamp invalid values to safe defaults."""
        if not self.name or not self.name.strip():
            self.name = "Untitled Preset"
        if not self.model or not self.model.strip():
            self.model = "gpt-5.4-mini"


# ── Built-in presets ─────────────────────────────────────────────────────────

_BUILTIN_PRESETS: list[ProPreset] = [
    ProPreset(
        name="General Professional",
        system_prompt=(
            "Rewrite the text in a professional, neutral business tone. "
            "Keep sentences clear and concise."
        ),
        fix_tone=True,
        fix_grammar=True,
        fix_punctuation=True,
    ),
    ProPreset(
        name="Technical / Engineering",
        system_prompt=(
            "Rewrite the text for a technical audience. Preserve all "
            "technical jargon, acronyms, and domain-specific terminology "
            "exactly as dictated. Keep the tone precise and objective."
        ),
        fix_tone=True,
        fix_grammar=True,
        fix_punctuation=True,
    ),
    ProPreset(
        name="Casual / Friendly",
        system_prompt=(
            "Rewrite the text in a warm, approachable, and conversational "
            "tone. Keep it friendly but still clear and easy to read."
        ),
        fix_tone=True,
        fix_grammar=True,
        fix_punctuation=True,
    ),
    ProPreset(
        name="Email / Correspondence",
        system_prompt=(
            "Rewrite the text as professional email correspondence. "
            "Use an appropriate greeting and sign-off tone. Keep "
            "paragraphs short and action items clear."
        ),
        fix_tone=True,
        fix_grammar=True,
        fix_punctuation=True,
    ),
    ProPreset(
        name="Simplified (8th Grade)",
        system_prompt=(
            "Rewrite the text at an 8th-grade reading level. Use short "
            "sentences, common words, and simple sentence structures. "
            "Avoid jargon and complex vocabulary."
        ),
        fix_tone=False,
        fix_grammar=True,
        fix_punctuation=True,
    ),
]

BUILTIN_PRESET_NAMES: frozenset[str] = frozenset(p.name for p in _BUILTIN_PRESETS)


def get_builtin_presets() -> dict[str, ProPreset]:
    """Return a *copy* of the built-in presets keyed by name."""
    return {p.name: ProPreset(**asdict(p)) for p in _BUILTIN_PRESETS}


# ── Preset manager ───────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Convert a preset name to a safe filesystem name."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", name.strip())
    return safe or "preset"


def load_all_presets(presets_dir: Path) -> dict[str, ProPreset]:
    """Load built-in presets plus any user presets from *presets_dir*.

    User presets on disk override built-in presets with the same name.
    """
    presets = get_builtin_presets()

    if presets_dir.is_dir():
        for path in sorted(presets_dir.glob("*.json")):
            try:
                preset = ProPreset.load(path)
                presets[preset.name] = preset
            except Exception:
                log.warning("Failed to load preset: %s", path, exc_info=True)

    return presets


def save_preset(preset: ProPreset, presets_dir: Path) -> Path:
    """Save a preset to *presets_dir* and return the file path."""
    filename = _safe_filename(preset.name) + ".json"
    path = presets_dir / filename
    preset.save(path)
    return path


def delete_preset(name: str, presets_dir: Path) -> bool:
    """Delete a user preset by name.  Returns True if deleted.

    Built-in presets cannot be deleted.
    """
    if name in BUILTIN_PRESET_NAMES:
        log.warning("Cannot delete built-in preset: %s", name)
        return False

    filename = _safe_filename(name) + ".json"
    path = presets_dir / filename
    if path.is_file():
        path.unlink()
        log.info("Preset deleted: %s", path)
        return True

    # Fallback: scan directory for matching name
    if presets_dir.is_dir():
        for p in presets_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8-sig"))
                if data.get("name") == name:
                    p.unlink()
                    log.info("Preset deleted: %s", p)
                    return True
            except Exception:
                pass

    return False


def bootstrap_presets(presets_dir: Path) -> None:
    """Create the presets directory and write built-in preset files if missing."""
    presets_dir.mkdir(parents=True, exist_ok=True)
    for preset in _BUILTIN_PRESETS:
        path = presets_dir / (_safe_filename(preset.name) + ".json")
        if not path.exists():
            preset.save(path)
