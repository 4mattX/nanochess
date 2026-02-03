"""Dataset panel widget for the training screen - Dracula theme."""

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Rule, Button
from textual.widget import Widget
from textual.message import Message

from nanochess.tui.utils import get_dataset_info, format_number, shorten_path, PRESETS
from nanochess.tui.icons import DATABASE, FOLDER, SETTINGS, TRAINING, CIRCLE_FILLED, CIRCLE_EMPTY


class DatasetPanel(Widget):
    """Panel for dataset selection and info display."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._selected_source = "prepared"  # Default selection
        self._selected_preset = None  # No preset selected by default

    class PresetSelected(Message):
        """Message sent when a preset is selected."""

        def __init__(self, preset_name: str) -> None:
            super().__init__()
            self.preset_name = preset_name

    class DataSourceChanged(Message):
        """Message sent when data source changes."""

        def __init__(self, source: str) -> None:
            super().__init__()
            self.source = source


    def compose(self) -> ComposeResult:
        """Compose the dataset panel."""
        with VerticalScroll():
            yield Static(f"[$accent]{DATABASE}  DATA SOURCE[/]", classes="section-title")

            yield Vertical(
                Button(f"{CIRCLE_EMPTY}  Local PGN", id="source-pgn", classes="source-option"),
                Button(f"{CIRCLE_FILLED}  Prepared Data", id="source-prepared", classes="source-option"),
                Button(f"{CIRCLE_EMPTY}  Lichess DB", id="source-lichess", classes="source-option"),
                id="data-source-buttons",
            )

            yield Static("", id="data-path", classes="stat-label")

            yield Vertical(
                Static("", id="data-stats-content"),
                id="data-stats",
                classes="data-stats",
            )

            yield Rule()

            yield Static(f"[$accent]{SETTINGS}  PRESETS[/]", classes="section-title")
            yield Vertical(
                *[
                    Button(
                        f"{CIRCLE_EMPTY}  {name}",
                        id=f"preset-{name.lower().replace(' ', '-').replace('(', '').replace(')', '')}",
                        classes="preset-option",
                    )
                    for name in PRESETS.keys()
                ],
                id="presets-container",
            )

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self._update_data_info()

    def _update_data_info(self) -> None:
        """Update the dataset information display."""
        try:
            info = get_dataset_info()

            path_display = self.query_one("#data-path", Static)
            stats_display = self.query_one("#data-stats-content", Static)

            if info.exists:
                path_display.update(f"[$text-muted]Path:[/] [$text]{shorten_path(info.path, 25)}[/]")
                stats_display.update(
                    f"[$text-muted]Train:[/]  [$success]{format_number(info.num_train_positions)}[/] [$text-muted]positions[/]\n"
                    f"[$text-muted]Val:[/]    [$success]{format_number(info.num_val_positions)}[/] [$text-muted]positions[/]\n"
                    f"[$text-muted]Shards:[/] [$accent]{info.num_shards}[/] [$text-muted]files[/]"
                )
            else:
                path_display.update(f"[$text-muted]Path:[/] [$text]{shorten_path(info.path, 25)}[/]")
                stats_display.update(
                    f"[$warning]No data found[/]\n"
                    f"[$text-muted]Run prepare script first[/]"
                )
        except Exception as e:
            path_display = self.query_one("#data-path", Static)
            stats_display = self.query_one("#data-stats-content", Static)
            path_display.update(f"[$text-muted]Path:[/] [$warning]Unknown[/]")
            stats_display.update(f"[$error]Error: {e}[/]")

    def _update_source_buttons(self) -> None:
        """Update the source button icons based on selection."""
        try:
            pgn_btn = self.query_one("#source-pgn", Button)
            prep_btn = self.query_one("#source-prepared", Button)
            lich_btn = self.query_one("#source-lichess", Button)

            pgn_btn.label = f"{CIRCLE_FILLED if self._selected_source == 'pgn' else CIRCLE_EMPTY}  Local PGN"
            prep_btn.label = f"{CIRCLE_FILLED if self._selected_source == 'prepared' else CIRCLE_EMPTY}  Prepared Data"
            lich_btn.label = f"{CIRCLE_FILLED if self._selected_source == 'lichess' else CIRCLE_EMPTY}  Lichess DB"
        except Exception:
            pass

    def _update_preset_buttons(self) -> None:
        """Update the preset button icons based on selection."""
        try:
            for preset_name in PRESETS.keys():
                preset_id = f"preset-{preset_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}"
                try:
                    btn = self.query_one(f"#{preset_id}", Button)
                    icon = CIRCLE_FILLED if self._selected_preset == preset_name else CIRCLE_EMPTY
                    btn.label = f"{icon}  {preset_name}"
                except Exception:
                    pass
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for both source selection and presets."""
        button_id = event.button.id

        # Handle data source selection
        if button_id in ["source-pgn", "source-prepared", "source-lichess"]:
            source_map = {
                "source-pgn": "pgn",
                "source-prepared": "prepared",
                "source-lichess": "lichess",
            }
            self._selected_source = source_map[button_id]
            self._update_source_buttons()
            self.post_message(self.DataSourceChanged(self._selected_source))

        # Handle preset selection
        elif button_id and button_id.startswith("preset-"):
            # Find the matching preset name
            for preset_name in PRESETS.keys():
                preset_id = f"preset-{preset_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}"
                if preset_id == button_id:
                    self._selected_preset = preset_name
                    self._update_preset_buttons()
                    self.post_message(self.PresetSelected(preset_name))
                    break
