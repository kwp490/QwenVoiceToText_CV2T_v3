"""
Professional Mode settings dialog.

Provides a sidebar-navigated dialog for configuring API credentials,
managing presets (built-in + user-created), editing custom system
prompts, and defining domain-specific vocabulary to preserve.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .pro_preset import (
    BUILTIN_PRESET_NAMES,
    ProPreset,
    delete_preset,
    save_preset,
)
from .text_processor import (
    TextProcessor,
    delete_api_key_from_keyring,
    load_api_key_from_keyring,
    save_api_key_to_keyring,
)

log = logging.getLogger(__name__)


class ProSettingsDialog(QDialog):
    """Sidebar-navigated dialog for Professional Mode configuration."""

    def __init__(
        self,
        settings,
        presets: dict[str, ProPreset],
        presets_dir,
        parent: Optional[QWidget] = None,
        api_key: str = "",
    ):
        super().__init__(parent)
        self._settings = settings
        self._presets: dict[str, ProPreset] = {
            k: ProPreset(**asdict(v)) for k, v in presets.items()
        }
        self._presets_dir = presets_dir
        self._api_key = api_key
        self._active_preset_name: str = settings.pro_active_preset

        self.setWindowTitle("Professional Mode Settings")
        self.setMinimumSize(700, 500)
        self._build_ui()
        self._populate()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        splitter = QSplitter()

        # ── Sidebar ──────────────────────────────────────────────────────
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(180)
        self._sidebar.addItem("API Configuration")
        self._sidebar.addItem("Presets")
        self._sidebar.addItem("Custom Instructions")
        self._sidebar.addItem("Vocabulary")
        self._sidebar.currentRowChanged.connect(self._on_page_changed)
        splitter.addWidget(self._sidebar)

        # ── Stacked pages ────────────────────────────────────────────────
        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_api_page())
        self._pages.addWidget(self._build_presets_page())
        self._pages.addWidget(self._build_instructions_page())
        self._pages.addWidget(self._build_vocabulary_page())
        splitter.addWidget(self._pages)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # ── Button box ───────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Page builders ────────────────────────────────────────────────────

    def _build_api_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(12, 12, 12, 12)

        # Model selector
        self._pro_model = QComboBox()
        self._pro_model.setEditable(True)
        self._pro_model.addItems(["gpt-5.4-mini", "gpt-5.4-nano"])
        form.addRow("Default model:", self._pro_model)

        # API key row
        key_row = QHBoxLayout()
        self._pro_api_key = QLineEdit()
        self._pro_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._pro_api_key.setPlaceholderText("sk-…")
        key_row.addWidget(self._pro_api_key)

        self._btn_eye = QPushButton("\U0001f441")
        self._btn_eye.setFixedWidth(32)
        self._btn_eye.setCheckable(True)
        self._btn_eye.setToolTip("Show / hide API key")
        self._btn_eye.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self._btn_eye)
        form.addRow("API key:", key_row)

        self._pro_store_key = QCheckBox(
            "Remember API key (Windows Credential Manager)"
        )
        form.addRow(self._pro_store_key)

        # Validate button
        validate_row = QHBoxLayout()
        self._btn_validate_key = QPushButton("Validate API Key")
        self._btn_validate_key.clicked.connect(self._on_validate_api_key)
        validate_row.addWidget(self._btn_validate_key)
        self._lbl_validate_result = QLabel("")
        validate_row.addWidget(self._lbl_validate_result)
        validate_row.addStretch()
        form.addRow(validate_row)

        return page

    def _build_presets_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        # Preset list
        self._preset_list = QListWidget()
        self._preset_list.currentItemChanged.connect(self._on_preset_selected)
        layout.addWidget(self._preset_list)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_new_preset = QPushButton("New")
        self._btn_new_preset.clicked.connect(self._on_new_preset)
        btn_row.addWidget(self._btn_new_preset)

        self._btn_dup_preset = QPushButton("Duplicate")
        self._btn_dup_preset.clicked.connect(self._on_duplicate_preset)
        btn_row.addWidget(self._btn_dup_preset)

        self._btn_del_preset = QPushButton("Delete")
        self._btn_del_preset.clicked.connect(self._on_delete_preset)
        btn_row.addWidget(self._btn_del_preset)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Preset detail area ───────────────────────────────────────────
        detail_group = QGroupBox("Preset Details")
        detail_form = QFormLayout()

        self._preset_name_edit = QLineEdit()
        self._preset_name_edit.setPlaceholderText("Preset name")
        detail_form.addRow("Name:", self._preset_name_edit)

        self._preset_model = QComboBox()
        self._preset_model.setEditable(True)
        self._preset_model.addItems(["(use default)", "gpt-5.4-mini", "gpt-5.4-nano"])
        detail_form.addRow("Model override:", self._preset_model)

        self._preset_fix_tone = QCheckBox("Fix tone")
        detail_form.addRow(self._preset_fix_tone)

        self._preset_fix_grammar = QCheckBox("Fix grammar")
        detail_form.addRow(self._preset_fix_grammar)

        self._preset_fix_punctuation = QCheckBox(
            "Fix punctuation && capitalization"
        )
        detail_form.addRow(self._preset_fix_punctuation)

        detail_group.setLayout(detail_form)
        layout.addWidget(detail_group)

        return page

    def _build_instructions_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        self._lbl_instructions_preset = QLabel("Select a preset on the Presets page first.")
        layout.addWidget(self._lbl_instructions_preset)

        self._instructions_edit = QPlainTextEdit()
        self._instructions_edit.setPlaceholderText(
            "Enter custom system prompt instructions for the selected preset…\n\n"
            "Example: Always use Oxford comma. Keep paragraphs under 3 sentences."
        )
        layout.addWidget(self._instructions_edit)

        return page

    def _build_vocabulary_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        self._lbl_vocab_preset = QLabel("Select a preset on the Presets page first.")
        layout.addWidget(self._lbl_vocab_preset)

        self._vocab_edit = QPlainTextEdit()
        self._vocab_edit.setPlaceholderText(
            "Enter domain-specific terms to preserve (comma or newline separated)…\n\n"
            "Example:\nKubernetes, gRPC, OAuth2, CI/CD"
        )
        layout.addWidget(self._vocab_edit)

        return page

    # ── Populate / Save ──────────────────────────────────────────────────

    def _populate(self) -> None:
        # API page
        if self._api_key:
            self._pro_api_key.setText(self._api_key)
        elif self._settings.store_api_key:
            stored = load_api_key_from_keyring()
            if stored:
                self._pro_api_key.setText(stored)

        self._pro_store_key.setChecked(self._settings.store_api_key)

        # Populate preset list
        self._refresh_preset_list()

        # Select active preset
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            if item and item.text() == self._active_preset_name:
                self._preset_list.setCurrentItem(item)
                break

        # Default sidebar to first page
        self._sidebar.setCurrentRow(0)

    def _refresh_preset_list(self) -> None:
        current_name = None
        if self._preset_list.currentItem():
            current_name = self._preset_list.currentItem().text()

        self._preset_list.clear()
        for name in sorted(self._presets.keys()):
            item = QListWidgetItem(name)
            if name in BUILTIN_PRESET_NAMES:
                item.setToolTip("Built-in preset")
            self._preset_list.addItem(item)

        # Restore selection
        if current_name:
            for i in range(self._preset_list.count()):
                item = self._preset_list.item(i)
                if item and item.text() == current_name:
                    self._preset_list.setCurrentItem(item)
                    return

    def _save_and_accept(self) -> None:
        # Flush current preset edits
        self._flush_preset_edits()

        # API key
        self._api_key = self._pro_api_key.text().strip()
        self._settings.store_api_key = self._pro_store_key.isChecked()

        if self._settings.store_api_key and self._api_key:
            save_api_key_to_keyring(self._api_key)
        elif not self._settings.store_api_key:
            delete_api_key_from_keyring()

        # Save all user presets to disk
        for name, preset in self._presets.items():
            save_preset(preset, self._presets_dir)

        # Persist active preset choice
        if self._preset_list.currentItem():
            self._settings.pro_active_preset = self._preset_list.currentItem().text()

        self._settings.save()
        log.info("Professional Mode settings saved")
        self.accept()

    # ── Page navigation ──────────────────────────────────────────────────

    def _on_page_changed(self, index: int) -> None:
        self._pages.setCurrentIndex(index)

    # ── Preset management ────────────────────────────────────────────────

    def _current_preset(self) -> ProPreset | None:
        item = self._preset_list.currentItem()
        if item:
            return self._presets.get(item.text())
        return None

    def _flush_preset_edits(self) -> None:
        """Write UI edits back to the current in-memory preset."""
        preset = self._current_preset()
        if preset is None:
            return

        old_name = preset.name
        new_name = self._preset_name_edit.text().strip()

        # Update fields
        if new_name and new_name != old_name:
            # Don't allow renaming built-in presets
            if old_name not in BUILTIN_PRESET_NAMES:
                del self._presets[old_name]
                preset.name = new_name
                self._presets[new_name] = preset

        model_text = self._preset_model.currentText().strip()
        preset.model = "" if model_text == "(use default)" else model_text
        preset.fix_tone = self._preset_fix_tone.isChecked()
        preset.fix_grammar = self._preset_fix_grammar.isChecked()
        preset.fix_punctuation = self._preset_fix_punctuation.isChecked()
        preset.system_prompt = self._instructions_edit.toPlainText()
        preset.vocabulary = self._vocab_edit.toPlainText()

    def _on_preset_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        # Flush edits for the previously selected preset
        if previous and previous.text() in self._presets:
            self._flush_preset_edits_for(previous.text())

        if current is None:
            return

        preset = self._presets.get(current.text())
        if preset is None:
            return

        # Populate detail widgets
        self._preset_name_edit.setText(preset.name)
        is_builtin = preset.name in BUILTIN_PRESET_NAMES
        self._preset_name_edit.setReadOnly(is_builtin)

        # Model
        model = preset.model or "(use default)"
        idx = self._preset_model.findText(model)
        if idx >= 0:
            self._preset_model.setCurrentIndex(idx)
        else:
            self._preset_model.setCurrentText(model)

        self._preset_fix_tone.setChecked(preset.fix_tone)
        self._preset_fix_grammar.setChecked(preset.fix_grammar)
        self._preset_fix_punctuation.setChecked(preset.fix_punctuation)

        # Instructions + Vocabulary
        self._instructions_edit.setPlainText(preset.system_prompt)
        self._lbl_instructions_preset.setText(f"Custom instructions for: {preset.name}")

        self._vocab_edit.setPlainText(preset.vocabulary)
        self._lbl_vocab_preset.setText(f"Vocabulary for: {preset.name}")

        # Disable delete for built-in presets
        self._btn_del_preset.setEnabled(not is_builtin)

    def _flush_preset_edits_for(self, name: str) -> None:
        """Flush edits for a specific preset by name (used when switching)."""
        preset = self._presets.get(name)
        if preset is None:
            return

        model_text = self._preset_model.currentText().strip()
        preset.model = "" if model_text == "(use default)" else model_text
        preset.fix_tone = self._preset_fix_tone.isChecked()
        preset.fix_grammar = self._preset_fix_grammar.isChecked()
        preset.fix_punctuation = self._preset_fix_punctuation.isChecked()
        preset.system_prompt = self._instructions_edit.toPlainText()
        preset.vocabulary = self._vocab_edit.toPlainText()

    def _on_new_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New Preset", "Preset name:"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._presets:
            QMessageBox.warning(
                self, "Duplicate Name",
                f"A preset named '{name}' already exists.",
            )
            return

        preset = ProPreset(name=name)
        self._presets[name] = preset
        self._refresh_preset_list()
        # Select new preset
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            if item and item.text() == name:
                self._preset_list.setCurrentItem(item)
                break

    def _on_duplicate_preset(self) -> None:
        source = self._current_preset()
        if source is None:
            return

        name, ok = QInputDialog.getText(
            self, "Duplicate Preset",
            "Name for the copy:",
            text=f"{source.name} (copy)",
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._presets:
            QMessageBox.warning(
                self, "Duplicate Name",
                f"A preset named '{name}' already exists.",
            )
            return

        dup = ProPreset(**asdict(source))
        dup.name = name
        self._presets[name] = dup
        self._refresh_preset_list()
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            if item and item.text() == name:
                self._preset_list.setCurrentItem(item)
                break

    def _on_delete_preset(self) -> None:
        preset = self._current_preset()
        if preset is None:
            return
        if preset.name in BUILTIN_PRESET_NAMES:
            QMessageBox.information(
                self, "Cannot Delete",
                "Built-in presets cannot be deleted.",
            )
            return

        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Delete preset '{preset.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        delete_preset(preset.name, self._presets_dir)
        del self._presets[preset.name]
        self._refresh_preset_list()

    # ── API key helpers ──────────────────────────────────────────────────

    def _toggle_key_visibility(self, show: bool) -> None:
        if show:
            self._pro_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            self._pro_api_key.setEchoMode(QLineEdit.EchoMode.Password)

    def _on_validate_api_key(self) -> None:
        key = self._pro_api_key.text().strip()
        if not key:
            self._lbl_validate_result.setText("\u274c No API key entered")
            self._lbl_validate_result.setStyleSheet("color: #c62828;")
            return

        self._lbl_validate_result.setText("Validating\u2026")
        self._lbl_validate_result.setStyleSheet("color: #757575;")
        self._btn_validate_key.setEnabled(False)

        model = self._pro_model.currentText()

        def _do_validate():
            processor = TextProcessor(api_key=key, model=model)
            return processor.validate_key()

        from .workers import Worker

        worker = Worker(_do_validate)
        worker.signals.result.connect(self._on_validate_result)
        worker.signals.error.connect(self._on_validate_error)
        QThreadPool.globalInstance().start(worker)

    def _on_validate_result(self, result: tuple) -> None:
        self._btn_validate_key.setEnabled(True)
        ok, msg = result
        if ok:
            self._lbl_validate_result.setText(f"\u2705 {msg}")
            self._lbl_validate_result.setStyleSheet("color: #2e7d32;")
        else:
            self._lbl_validate_result.setText(f"\u274c {msg}")
            self._lbl_validate_result.setStyleSheet("color: #c62828;")

    def _on_validate_error(self, err: str) -> None:
        self._btn_validate_key.setEnabled(True)
        self._lbl_validate_result.setText(f"\u274c {err}")
        self._lbl_validate_result.setStyleSheet("color: #c62828;")

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def api_key(self) -> str:
        """Return the API key entered by the user (in-memory only)."""
        return self._api_key

    @property
    def presets(self) -> dict[str, ProPreset]:
        """Return the (possibly modified) preset dict."""
        return self._presets

    @property
    def active_preset_name(self) -> str:
        return self._settings.pro_active_preset
