"""Configuration panel widget for training parameters - Dracula theme."""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.widgets import Static, Input, Button, Rule
from textual.widget import Widget
from textual.message import Message

from nanochess.tui.utils import DEFAULT_CONFIG
from nanochess.tui.icons import MODEL, SETTINGS, CHART, TRAINING, METRICS, TAG, CHECK, SQUARE_FILLED, SQUARE_EMPTY


class ConfigPanel(Widget):
    """Panel for training configuration parameters."""

    class ConfigChanged(Message):
        """Message sent when configuration changes."""

        def __init__(self, config: dict) -> None:
            super().__init__()
            self.config = config


    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = DEFAULT_CONFIG.copy()

    def compose(self) -> ComposeResult:
        """Compose the configuration panel."""
        with VerticalScroll():
            # Model Architecture
            yield Static(f"[$accent]{MODEL}  MODEL ARCHITECTURE[/]", classes="section-title")
            yield self._config_row("Depth", "depth", self._config["depth"], "[$text-muted]layers[/]")
            yield self._config_row("Embedding", "n_embd", self._config["n_embd"], "[$text-muted]dim[/]")
            yield self._config_row("Heads", "n_head", self._config["n_head"], "[$text-muted]attn[/]")

            yield Rule()

            # Optimization
            yield Static(f"[$accent]{SETTINGS}  OPTIMIZATION[/]", classes="section-title")
            yield self._config_row("Iterations", "num_iterations", self._config["num_iterations"])
            yield self._config_row("Device batch", "device_batch_size", self._config["device_batch_size"])
            yield self._config_row("Total batch", "total_batch_size", self._config["total_batch_size"])
            yield self._config_row("Embed LR", "embedding_lr", self._config["embedding_lr"])
            yield self._config_row("Matrix LR", "matrix_lr", self._config["matrix_lr"])
            yield self._config_row("Weight decay", "weight_decay", self._config["weight_decay"])
            yield self._config_row("Adam b1", "adam_beta1", self._config["adam_beta1"])
            yield self._config_row("Adam b2", "adam_beta2", self._config["adam_beta2"])

            yield Rule()

            # Schedule
            yield Static(f"[$accent]{CHART}  SCHEDULE[/]", classes="section-title")
            yield self._config_row("Warmup", "warmup_ratio", self._config["warmup_ratio"], "[$text-muted]ratio[/]")
            yield self._config_row("Warmdown", "warmdown_ratio", self._config["warmdown_ratio"], "[$text-muted]ratio[/]")
            yield self._config_row("Final LR", "final_lr_frac", self._config["final_lr_frac"], "[$text-muted]frac[/]")

            yield Rule()

            # Loss Weights
            yield Static(f"[$accent]{TRAINING}  LOSS WEIGHTS[/]", classes="section-title")
            yield self._config_row("Policy", "policy_weight", self._config["policy_weight"])
            yield self._config_row("Value", "value_weight", self._config["value_weight"])

            yield Rule()

            # Evaluation
            yield Static(f"[$accent]{METRICS}  EVALUATION[/]", classes="section-title")
            yield self._config_row("Eval every", "eval_every", self._config["eval_every"], "[$text-muted]steps[/]")
            yield self._config_row("Eval pos", "eval_positions", self._config["eval_positions"])
            yield self._config_row("Save every", "save_every", self._config["save_every"], "[$text-muted]steps[/]")
            yield self._config_row("Resume", "resume_from_step", self._config["resume_from_step"], "[$text-muted]step[/]")

            yield Rule()

            # Logging
            yield Static(f"[$accent]{TAG}  LOGGING[/]", classes="section-title")
            yield self._config_row("Run name", "run_name", self._config["run_name"])
            yield self._config_row("Model tag", "model_tag", self._config["model_tag"])

            # Wandb toggle with icon (preset-style)
            icon = SQUARE_FILLED if self._config["wandb_enabled"] else SQUARE_EMPTY
            yield Button(
                f"{icon}  Enable Wandb logging",
                id="toggle-wandb",
                classes="config-toggle",
            )

    def _config_row(self, label: str, key: str, value, hint: str = "") -> Horizontal:
        """Create a configuration row with label and input."""
        return Horizontal(
            Static(f"[$text-muted]{label}[/]", classes="config-label"),
            Input(str(value), id=f"input-{key}", classes="config-input"),
            Static(hint) if hint else Static(""),
            classes="config-row",
        )

    def get_config(self) -> dict:
        """Get the current configuration."""
        return self._config.copy()

    def set_config(self, config: dict) -> None:
        """Set the configuration and update inputs."""
        self._config.update(config)
        self._update_inputs()

    def _update_inputs(self) -> None:
        """Update all input fields from the current config."""
        for key, value in self._config.items():
            try:
                if key == "wandb_enabled":
                    self._update_wandb_toggle()
                else:
                    input_widget = self.query_one(f"#input-{key}", Input)
                    input_widget.value = str(value)
            except Exception:
                pass

    def _update_wandb_toggle(self) -> None:
        """Update the wandb toggle button icon."""
        try:
            toggle_btn = self.query_one("#toggle-wandb", Button)
            icon = SQUARE_FILLED if self._config["wandb_enabled"] else SQUARE_EMPTY
            toggle_btn.label = f"{icon}  Enable logging"
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        input_id = event.input.id
        if input_id and input_id.startswith("input-"):
            key = input_id[6:]  # Remove "input-" prefix
            self._update_config_value(key, event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses (toggle buttons)."""
        if event.button.id == "toggle-wandb":
            # Toggle the wandb_enabled value
            self._config["wandb_enabled"] = not self._config.get("wandb_enabled", False)
            self._update_wandb_toggle()
            self._notify_config_changed()

    def _update_config_value(self, key: str, value: str) -> None:
        """Update a config value, converting to the appropriate type."""
        # Get the default type
        default_value = DEFAULT_CONFIG.get(key)

        if default_value is None:
            self._config[key] = value
        elif isinstance(default_value, bool):
            self._config[key] = value.lower() in ("true", "1", "yes")
        elif isinstance(default_value, int):
            try:
                self._config[key] = int(value)
            except ValueError:
                pass  # Keep old value on invalid input
        elif isinstance(default_value, float):
            try:
                self._config[key] = float(value)
            except ValueError:
                pass  # Keep old value on invalid input
        else:
            self._config[key] = value

        self._notify_config_changed()

    def _notify_config_changed(self) -> None:
        """Send config changed message."""
        self.post_message(self.ConfigChanged(self._config.copy()))

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate the current configuration.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check n_embd divisible by n_head
        n_embd = self._config.get("n_embd", 192)
        n_head = self._config.get("n_head", 12)
        if n_embd % n_head != 0:
            errors.append(f"Embedding ({n_embd}) must be divisible by heads ({n_head})")

        # Check batch size constraints
        device_batch = self._config.get("device_batch_size", 256)
        total_batch = self._config.get("total_batch_size", 16384)
        if total_batch % device_batch != 0:
            errors.append(f"Total batch ({total_batch}) must be divisible by device batch ({device_batch})")

        # Check positive values
        if self._config.get("depth", 8) <= 0:
            errors.append("Depth must be positive")
        if self._config.get("num_iterations", 10000) <= 0:
            errors.append("Iterations must be positive")

        return len(errors) == 0, errors
