"""Command preview panel widget - Dracula theme."""

from textual.app import ComposeResult
from textual.widgets import Static
from textual.widget import Widget

from nanochess.tui.utils import build_training_command, format_command_preview, DEFAULT_CONFIG
from nanochess.tui.icons import PLAY, CLOCK, DOLLAR, CHART, SUCCESS, ERROR


class PreviewPanel(Widget):
    """Panel showing the command preview and estimates."""


    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = DEFAULT_CONFIG.copy()

    def compose(self) -> ComposeResult:
        """Compose the preview panel."""
        yield Static(f"[$accent]{PLAY}  COMMAND[/]", classes="section-title")
        yield Static(self._get_command_text(), id="command-text", classes="command-preview")

        yield Static(f"[$accent]{CHART}  ESTIMATES[/]", classes="section-title")
        yield Static("", id="time-estimate")
        yield Static("", id="cost-estimate")
        yield Static("", id="throughput-estimate")

        yield Static(f"[$accent]{SUCCESS}  VALIDATION[/]", classes="section-title")
        yield Static("", id="validation-status")

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self._update_display()

    def update_config(self, config: dict) -> None:
        """Update the preview with new configuration."""
        self._config = config.copy()
        self._update_display()

    def _get_command_text(self) -> str:
        """Get the formatted command text."""
        cmd = build_training_command(self._config)
        return format_command_preview(cmd)

    def _update_display(self) -> None:
        """Update the display with current config."""
        try:
            command_widget = self.query_one("#command-text", Static)
            command_widget.update(f"[$text]{self._get_command_text()}[/]")

            time_widget = self.query_one("#time-estimate", Static)
            cost_widget = self.query_one("#cost-estimate", Static)
            throughput_widget = self.query_one("#throughput-estimate", Static)
            validation_widget = self.query_one("#validation-status", Static)

            # Estimate training time (rough approximation)
            iterations = self._config.get("num_iterations", 10000)
            total_batch = self._config.get("total_batch_size", 16384)
            positions_total = iterations * total_batch

            # Assume ~5000 pos/sec average throughput on consumer GPU
            est_seconds = positions_total / 5000
            if est_seconds < 60:
                time_str = f"{est_seconds:.0f} seconds"
            elif est_seconds < 3600:
                time_str = f"{est_seconds/60:.1f} minutes"
            else:
                time_str = f"{est_seconds/3600:.1f} hours"

            time_widget.update(f"[$text-muted]{CLOCK}[/]  [$primary]~{time_str}[/]")

            # Rough cost estimate (assuming cloud GPU pricing)
            # RTX 4070 equivalent ~$0.20/hr
            hours = est_seconds / 3600
            cost = hours * 0.20
            cost_widget.update(f"[$text-muted]{DOLLAR}[/]  [$primary]~${cost:.2f}[/] [$text-muted](cloud GPU)[/]")

            # Throughput estimate
            throughput_widget.update(f"[$text-muted]{CHART}[/]  [$primary]{positions_total:,}[/] [$text-muted]positions[/]")

            # Validation
            errors = self._validate_config()
            if errors:
                validation_widget.update(
                    f"[$error]{ERROR}  " + "\n  ".join(errors) + "[/]"
                )
            else:
                validation_widget.update(f"[$success]{SUCCESS}  Configuration valid[/]")

        except Exception:
            pass

    def _validate_config(self) -> list[str]:
        """Validate the configuration and return error messages."""
        errors = []

        n_embd = self._config.get("n_embd", 192)
        n_head = self._config.get("n_head", 12)
        if n_embd % n_head != 0:
            errors.append(f"n_embd ({n_embd}) not divisible by n_head ({n_head})")

        device_batch = self._config.get("device_batch_size", 256)
        total_batch = self._config.get("total_batch_size", 16384)
        if total_batch % device_batch != 0:
            errors.append(f"total_batch ({total_batch}) not divisible by device_batch ({device_batch})")

        depth = self._config.get("depth", 8)
        if depth <= 0:
            errors.append("depth must be positive")

        iterations = self._config.get("num_iterations", 10000)
        if iterations <= 0:
            errors.append("iterations must be positive")

        return errors
