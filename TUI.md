# Nanochess TUI

The Nanochess TUI provides a terminal-based interface for training, evaluating, and interacting with PRA (Piece-Routed Attention) chess models. It serves as the primary user-facing component of the nanochess project, abstracting away the complexity of command-line arguments and configuration management.

Within the broader nanochess architecture:
- **nanochat/** contains the core ML components (PRA model, dataloaders, optimizers)
- **scripts/** contains the training/evaluation entry points (`pra_train.py`, `pra_eval.py`)
- **nanochess/** provides the TUI that wraps these scripts with a modern terminal interface

---

## Technology Stack

The TUI is built with [Textual](https://textual.textualize.io/) (v0.89.0+), a modern Python TUI framework that provides:
- CSS-like styling via TCSS (Textual CSS)
- Async-first architecture for non-blocking operations
- Rich widget library (inputs, buttons, containers, logs)
- Built-in mouse and keyboard navigation

---

## Architecture

```
nanochess/
├── __init__.py              # Package version
├── __main__.py              # Entry point: python -m nanochess
└── tui/
    ├── __init__.py
    ├── app.py               # NanochessApp - main Textual application
    ├── utils.py             # GPU detection, dataset info, presets, command building
    ├── styles/
    │   └── nanochess.tcss   # Catppuccin Mocha theme
    ├── screens/
    │   ├── __init__.py
    │   ├── main_menu.py     # MainMenuScreen - home screen
    │   ├── training.py      # TrainingScreen - training configuration
    │   ├── evaluation.py    # EvaluationScreen (placeholder)
    │   └── interactive.py   # InteractiveScreen (placeholder)
    └── widgets/
        ├── __init__.py
        ├── dataset_panel.py # Dataset selection, preset buttons
        ├── config_panel.py  # Training parameter inputs
        ├── preview_panel.py # Command preview, time/cost estimates
        └── status_bar.py    # GPU/VRAM/dataset status display
```

---

## Entry Point

The TUI can be launched two ways:

```bash
# Module invocation
python -m nanochess

# Console script (after pip install)
nanochess
```

The entry point is defined in `pyproject.toml`:
```toml
[project.scripts]
nanochess = "nanochess.__main__:main"
```

`nanochess/__main__.py` instantiates and runs the Textual app:
```python
from nanochess.tui.app import NanochessApp

def main():
    app = NanochessApp()
    app.run()
```

---

## Application Class

`NanochessApp` (in `tui/app.py`) extends `textual.app.App`:

```python
class NanochessApp(App):
    CSS_PATH = Path(__file__).parent / "styles" / "nanochess.tcss"
    TITLE = "Nanochess"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back"),
    ]

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())
```

Key responsibilities:
- Loads the TCSS stylesheet
- Defines global key bindings
- Manages screen stack navigation
- Pushes the initial `MainMenuScreen` on mount

---

## Screen System

Screens are full-terminal views managed via a stack. Navigation uses `push_screen()` and `pop_screen()`:

| Screen | Purpose | Key Bindings |
|--------|---------|--------------|
| `MainMenuScreen` | Home menu with Train/Evaluate/Interactive/Quit | T, E, I, Q |
| `TrainingScreen` | Training configuration and launch | F1 (help), F5 (preset), F10 (start), ESC (back) |
| `EvaluationScreen` | Model evaluation (placeholder) | ESC |
| `InteractiveScreen` | Interactive play (placeholder) | ESC |

Each screen extends `textual.screen.Screen` and implements:
- `compose()` - yields child widgets
- `BINDINGS` - screen-specific key bindings
- Action methods (`action_*`) - respond to bindings

---

## Widget Composition

### DatasetPanel (`widgets/dataset_panel.py`)

Displays dataset information and preset selection:

```
DATASET
Source:
○ Local PGN
● Prepared Data
○ Lichess DB

Data path:
~/.cache/nanochat/pra_data

Train: 36,034
Val:   4,003

─────────────

PRESETS
[Quick Test]
[RTX 4070 (16GB)]
[8xH100 Speedrun]
[CPU Debug]
```

Emits messages:
- `PresetSelected(preset_name)` - when a preset button is clicked
- `DataSourceChanged(source)` - when radio selection changes

### ConfigPanel (`widgets/config_panel.py`)

Scrollable form with all training parameters organized by section:

| Section | Parameters |
|---------|------------|
| MODEL ARCHITECTURE | depth, n_embd, n_head |
| OPTIMIZATION | num_iterations, device_batch_size, total_batch_size, embedding_lr, matrix_lr, weight_decay, adam_beta1, adam_beta2 |
| SCHEDULE | warmup_ratio, warmdown_ratio, final_lr_frac |
| LOSS WEIGHTS | policy_weight, value_weight |
| EVALUATION | eval_every, eval_positions, save_every, resume_from_step |
| LOGGING | run_name, model_tag, wandb_enabled |

Key methods:
- `get_config() -> dict` - returns current configuration
- `set_config(config: dict)` - updates all inputs from config
- `validate_config() -> (bool, list[str])` - validates constraints

Emits: `ConfigChanged(config)` on any input change

### PreviewPanel (`widgets/preview_panel.py`)

Shows the generated training command and estimates:

```
COMMAND PREVIEW
python -m scripts.pra_train \
  --depth=8 \
  --n-embd=192 \
  ...

Estimated:
~2.5 hours
~$0.50 compute
```

Updates reactively when `ConfigPanel` emits `ConfigChanged`.

### StatusBar (`widgets/status_bar.py`)

Docked at the bottom, shows system status:

```
GPU: NVIDIA RTX 4070 | VRAM: 1.2/16GB | Positions: 40,037 ready
```

---

## Color Palette

The TUI uses a minimal blue color palette defined **once** in `tui/colors.py`:

| Color | Hex | Usage |
|-------|-----|-------|
| BACKGROUND | `#384959` | Main background |
| TEXT | `white` | Primary text |
| PRIMARY | `#6A89A7` | Muted blue (focus, accents) |
| LIGHT | `#BDDDFC` | Light blue (highlights) |
| MEDIUM | `#88BDF2` | Medium blue (interactive elements) |

**Important**: Colors are defined **only once** in `colors.py`. The stylesheet automatically uses these values via template substitution.

## Styling (TCSS)

The stylesheet (`styles/nanochess.tcss`) uses template placeholders that are automatically replaced with colors from `colors.py`:

```css
/* Global styles - colors injected from colors.py */
Screen {
    background: {BACKGROUND};
    color: {TEXT};
}

Input:focus {
    border: ascii {MEDIUM};
}

Button.-primary {
    background: {MEDIUM};
    color: {BACKGROUND};
}
```

The `NanochessApp` class (in `app.py`) reads the TCSS file and formats it with colors from `colors.py` at runtime. This ensures a single source of truth for the color palette.

All TCSS styling is centralized in `nanochess.tcss`. Individual widgets and screens should not define their own `DEFAULT_CSS` blocks unless absolutely necessary for layout.

### Changing Colors

To change the entire color scheme:
1. Edit `nanochess/tui/colors.py` only
2. The changes automatically apply to both Python code (Rich markup) and TCSS styling

---

## Utilities (`tui/utils.py`)

### GPU Detection

```python
@dataclass
class GPUInfo:
    available: bool
    device_type: str  # cuda, mps, cpu
    device_name: str
    vram_total_gb: float
    vram_used_gb: float
    cuda_version: str
    pytorch_version: str

def get_gpu_info() -> GPUInfo:
    # Uses torch.cuda.* APIs to detect GPU
```

### Dataset Detection

```python
@dataclass
class DatasetInfo:
    path: str
    num_train_positions: int
    num_val_positions: int
    num_shards: int
    exists: bool

def get_dataset_info() -> DatasetInfo:
    # Scans ~/.cache/nanochat/pra_data for parquet shards
```

### Presets

```python
PRESETS = {
    "Quick Test": {
        "depth": 4, "n_embd": 144, "num_iterations": 100, ...
    },
    "RTX 4070 (16GB)": {
        "depth": 8, "n_embd": 192, "num_iterations": 10000, ...
    },
    "8xH100 Speedrun": {
        "depth": 8, "n_embd": 288, "num_iterations": 50000, ...
    },
    "CPU Debug": {
        "depth": 2, "n_embd": 96, "num_iterations": 10, ...
    },
}
```

### Command Building

```python
def build_training_command(config: dict) -> list[str]:
    # Returns: ["python", "-m", "scripts.pra_train", "--depth=8", ...]
```

---

## Training Launch

When the user presses F10, `TrainingScreen.action_start_training()`:

1. Validates configuration via `ConfigPanel.validate_config()`
2. Builds command via `build_training_command(config)`
3. Shows the `RichLog` widget for output
4. Spawns a background worker via `run_worker()`
5. The worker runs `subprocess.Popen()` and streams stdout to the log
6. Output lines are color-coded (errors in red, steps in green)

The training can be cancelled with Ctrl+C or the Cancel button, which calls `process.terminate()`.

---

## Message Flow

```
User clicks preset button
         │
         ▼
DatasetPanel.on_button_pressed()
         │
         ▼
post_message(PresetSelected("RTX 4070 (16GB)"))
         │
         ▼
TrainingScreen.on_dataset_panel_preset_selected()
         │
         ├──► ConfigPanel.set_config(preset_config)
         │              │
         │              ▼
         │    Input widgets update, emit ConfigChanged
         │              │
         ▼              ▼
TrainingScreen.on_config_panel_config_changed()
         │
         ▼
PreviewPanel.update_config(config)
         │
         ▼
Command preview and estimates refresh
```

---

## Testing

The TUI can be tested programmatically using Textual's test mode:

```python
import asyncio
from nanochess.tui.app import NanochessApp

async def test():
    app = NanochessApp()
    async with app.run_test() as pilot:
        await pilot.press("t")  # Go to training
        assert type(app.screen).__name__ == "TrainingScreen"
        await pilot.press("escape")  # Back to menu

asyncio.run(test())
```

---

## Future Work

- **EvaluationScreen**: Load checkpoints, run evaluation, display metrics
- **InteractiveScreen**: Play against the model, visualize board state
- **PGN Processing**: Integrate `prepare_chess_positions.py` into the TUI
- **Training Progress**: Parse training output for live loss/accuracy plots
- **Checkpoint Browser**: List and manage saved checkpoints
