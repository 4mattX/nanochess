"""Training screen for Nanochess TUI - Dracula theme."""

import subprocess
import sys
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Footer, Button, Label
from textual.widget import Widget

from nanochess.tui.widgets.dataset_panel import DatasetPanel
from nanochess.tui.widgets.config_panel import ConfigPanel
from nanochess.tui.widgets.preview_panel import PreviewPanel
from nanochess.tui.widgets.status_bar import StatusBar
from nanochess.tui.utils import PRESETS, DEFAULT_CONFIG, build_training_command
from nanochess.tui.icons import DATABASE, SETTINGS, CHART


class FocusablePanel(Container):
    """A container that can receive focus and shows visual feedback."""

    can_focus = True


class TrainingScreen(Screen):
    """The training configuration screen with panel navigation."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("tab", "focus_next_panel", "Next Panel", show=True),
        Binding("shift+tab", "focus_prev_panel", "Prev Panel", show=False),
        Binding("f1", "help", "Help", show=True),
        Binding("f5", "load_preset", "Preset", show=True),
        Binding("f10", "start_training", "Start", show=True, priority=True),
        Binding("1", "focus_dataset", "Dataset", show=False),
        Binding("2", "focus_config", "Config", show=False),
        Binding("3", "focus_preview", "Preview", show=False),
        Binding("j", "focus_next", "Down", show=False),
        Binding("k", "focus_previous", "Up", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._config = DEFAULT_CONFIG.copy()
        self._current_panel = 0
        self._panels = ["dataset-panel", "config-panel", "preview-panel"]

    def compose(self) -> ComposeResult:
        """Compose the training screen."""
        # Header
        yield Horizontal(
            Horizontal(
                Static("[bold $text]Training Configuration[/]", id="header-title"),
                Static("[$text-muted]Configure and launch model training[/]", id="header-subtitle"),
                id="header-left",
            ),
            Static("[$accent]Tab[/][$text-muted]: switch panels[/]  [$accent]ESC[/][$text-muted]: back[/]", id="header-right"),
            id="training-header",
        )

        # Panel navigation hints
        yield Horizontal(
            Static("[$accent]1[/] [$text]Dataset[/]", classes="tab-item"),
            Static("[$accent]2[/] [$text]Config[/]", classes="tab-item"),
            Static("[$accent]3[/] [$text]Preview[/]", classes="tab-item"),
            Static("[$text-muted]|[/]", classes="tab-item"),
            Static("[$accent]j/k[/] [$text-muted]navigate[/]", classes="tab-item"),
            Static("[$accent]Tab[/] [$text-muted]switch panel[/]", classes="tab-item"),
            id="panel-tabs",
        )

        # Main content area with three panels
        dataset_container = Container(
            DatasetPanel(id="dataset-widget"),
            id="dataset-panel",
        )
        dataset_container.border_title = f"{DATABASE} Dataset"

        config_container = Container(
            ConfigPanel(id="config-widget"),
            id="config-panel",
        )
        config_container.border_title = f"{SETTINGS} Configuration"

        preview_container = Container(
            PreviewPanel(id="preview-widget"),
            id="preview-panel",
        )
        preview_container.border_title = f"{CHART} Preview"

        yield Horizontal(
            dataset_container,
            config_container,
            preview_container,
            id="main-content",
        )

        # Status bar
        yield StatusBar()

        # Footer with keybindings
        yield Horizontal(
            Static("[$secondary]F1[/] [$text]Help[/]", classes="footer-item"),
            Static("[$secondary]F5[/] [$text]Preset[/]", classes="footer-item"),
            Static("[$secondary]F10[/] [$accent]Start[/]", classes="footer-item"),
            Static("[$secondary]Tab[/] [$text]Panel[/]", classes="footer-item"),
            Static("[$secondary]ESC[/] [$text]Back[/]", classes="footer-item"),
            id="footer-bar",
        )

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self._update_preview()
        # Focus the first interactive element in the dataset panel
        self._focus_panel(0)

    def _focus_panel(self, index: int) -> None:
        """Focus a specific panel by index."""
        self._current_panel = index % len(self._panels)
        panel_id = self._panels[self._current_panel]

        try:
            panel = self.query_one(f"#{panel_id}", Container)
            # Try to focus the first focusable widget inside the panel
            focusables = list(panel.query("Button, Input, RadioButton, Checkbox").results())
            if focusables:
                focusables[0].focus()
            else:
                panel.focus()
        except Exception:
            pass

    def action_focus_next_panel(self) -> None:
        """Focus the next panel."""
        self._focus_panel(self._current_panel + 1)

    def action_focus_prev_panel(self) -> None:
        """Focus the previous panel."""
        self._focus_panel(self._current_panel - 1)

    def action_focus_dataset(self) -> None:
        """Focus the dataset panel."""
        self._focus_panel(0)

    def action_focus_config(self) -> None:
        """Focus the config panel."""
        self._focus_panel(1)

    def action_focus_preview(self) -> None:
        """Focus the preview panel."""
        self._focus_panel(2)

    def action_focus_next(self) -> None:
        """Focus the next widget within the current panel."""
        self.screen.focus_next()

    def action_focus_previous(self) -> None:
        """Focus the previous widget within the current panel."""
        self.screen.focus_previous()

    def on_dataset_panel_preset_selected(self, event: DatasetPanel.PresetSelected) -> None:
        """Handle preset selection from dataset panel."""
        self._load_preset(event.preset_name)

    def on_config_panel_config_changed(self, event: ConfigPanel.ConfigChanged) -> None:
        """Handle configuration changes."""
        self._config = event.config
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the preview panel with current config."""
        try:
            preview = self.query_one("#preview-widget", PreviewPanel)
            preview.update_config(self._config)
        except Exception:
            pass

    def _load_preset(self, preset_name: str) -> None:
        """Load a preset configuration."""
        if preset_name in PRESETS:
            preset_config = DEFAULT_CONFIG.copy()
            preset_config.update(PRESETS[preset_name])
            self._config = preset_config

            # Update the config panel
            try:
                config_panel = self.query_one("#config-widget", ConfigPanel)
                config_panel.set_config(preset_config)
            except Exception:
                pass

            # Update the preview
            self._update_preview()

            self.app.notify(
                f"[$success]Loaded preset: {preset_name}[/]",
                title="Preset Loaded",
                severity="information",
            )

    def action_back(self) -> None:
        """Go back to the main menu."""
        self.app.pop_screen()

    def action_help(self) -> None:
        """Show help information."""
        help_text = """[bold $text]Training Configuration Help[/]

[$accent]Navigation:[/]
  [$success]Tab[/]      [$text-muted]- Switch between panels[/]
  [$success]1/2/3[/]    [$text-muted]- Jump to Dataset/Config/Preview[/]
  [$success]j/k[/]      [$text-muted]- Navigate up/down within panel[/]
  [$success]arrows[/]   [$text-muted]- Navigate and adjust values[/]

[$accent]Model Architecture:[/]
  [$text]Depth[/]     [$text-muted]- Number of transformer layers[/]
  [$text]Embedding[/] [$text-muted]- Must be divisible by heads[/]
  [$text]Heads[/]     [$text-muted]- Number of attention heads[/]

[$accent]Actions:[/]
  [$success]F1[/]  [$text-muted]- Show this help[/]
  [$success]F5[/]  [$text-muted]- Cycle through presets[/]
  [$success]F10[/] [$text-muted]- Start training[/]
  [$success]ESC[/] [$text-muted]- Go back to main menu[/]
"""
        self.app.notify(help_text, title="Help", timeout=15)

    def action_load_preset(self) -> None:
        """Cycle through presets."""
        preset_names = list(PRESETS.keys())
        current_idx = 0
        for i, name in enumerate(preset_names):
            if PRESETS[name].get("depth") == self._config.get("depth"):
                current_idx = (i + 1) % len(preset_names)
                break

        next_preset = preset_names[current_idx]
        self._load_preset(next_preset)

    def action_start_training(self) -> None:
        """Start the training process by exiting TUI and running in terminal."""
        # Validate configuration
        try:
            config_panel = self.query_one("#config-widget", ConfigPanel)
            is_valid, errors = config_panel.validate_config()
            if not is_valid:
                self.app.notify(
                    "[$error]" + "\n".join(errors) + "[/]",
                    title="Validation Error",
                    severity="error",
                )
                return
        except Exception:
            pass

        # Build command
        cmd = build_training_command(self._config)

        # Store the command in the app so we can run it after TUI exits
        self.app._training_command = cmd

        # Exit the TUI
        self.app.exit()

