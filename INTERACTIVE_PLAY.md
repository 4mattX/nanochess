# Interactive Play - Quick Start Guide

Watch your trained PRA chess model play against itself in real-time!

## Quick Start

```bash
# Launch the TUI
python -m nanochess

# Or if installed:
nanochess
```

Then:
1. Press `I` to enter Interactive mode
2. Select your trained model from the dropdown (e.g., "pra_d2 (step 5)")
3. Click "Load Model"
4. Click "Start" or press Space
5. Watch your model play chess!

## Controls

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| R | Reset game |
| ESC | Back to menu |

| Button | Action |
|--------|--------|
| Load Model | Load selected checkpoint |
| Start | Begin self-play |
| Pause | Pause the game |
| Reset | Start new game |

## Features

### Chess Board Display

The board shows:
- Unicode chess pieces (♔♕♖♗♘♙ for white, ♚♛♜♝♞♟ for black)
- Checkerboard pattern for empty squares
- Algebraic notation (a-h, 1-8)

### Move History

Scrollable panel on the right shows all moves in standard algebraic notation (SAN):
```
1. e4
1... e5
2. Nf3
2... Nc6
3. Bc4
...
```

### Statistics Panel

Real-time game statistics:
- **Move**: Current move number
- **Model eval**: Position evaluation from model's value head
  - Positive values = white advantage
  - Negative values = black advantage
  - Range: -1.0 to +1.0
- **Stockfish**: Optional engine evaluation (if installed)
- **Status**: Current game state

## Optional: Stockfish Integration

For comparing your model's evaluation with a strong chess engine:

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install stockfish
```

**macOS:**
```bash
brew install stockfish
```

**Windows:**
1. Download from https://stockfishchess.org/download/
2. Extract and add to PATH, or place `stockfish.exe` in your project directory

**Verify Installation:**
```bash
stockfish --help
```

### Usage

1. In the Interactive screen, check "Use Stockfish evaluation"
2. If Stockfish is found, the statistics panel will show:
   - **Model eval**: Your model's assessment
   - **Stockfish**: Engine evaluation in pawns (e.g., +0.5 = half pawn advantage)

3. Watch how your model's evaluation compares to Stockfish's!

## Understanding Evaluations

### Model Evaluation

The model's value head predicts position evaluation:
- **+1.0**: Model thinks white is winning
- **0.0**: Model thinks position is equal
- **-1.0**: Model thinks black is winning

For your current model (trained on 5 steps with ~5% accuracy):
- Evaluations may be noisy or inaccurate
- This is expected for an undertrained model
- As you train longer, evaluations should become more reliable

### Stockfish Evaluation

Stockfish evaluates in centipawns (1 pawn = 100 centipawns):
- **+1.0**: White ahead by one pawn
- **+3.0**: White has significant advantage (3 pawns)
- **±10.0+**: Forced mate detected

**Rough Strength Estimates:**
- ±0.5: Slight advantage
- ±1.0-2.0: Clear advantage
- ±3.0+: Winning position
- ±10.0+: Mate in N moves

## Estimating Model Strength

Based on move accuracy (from training logs):

| Accuracy | Est. Elo | Skill Level |
|----------|----------|-------------|
| 5-10% | ~800 | Random/Beginner |
| 10-20% | ~1000 | Casual player |
| 20-30% | ~1200 | Club player |
| 30-40% | ~1500 | Intermediate |
| 40-50% | ~1800 | Advanced |
| 50-60% | ~2000 | Expert |
| 60-70% | ~2200 | Master |
| 70%+ | ~2400+ | Grandmaster |

Your current model (5.26% accuracy) is at the "learning to move pieces" stage. With more training data and iterations, this will improve significantly!

## Troubleshooting

### "No checkpoints found"

Make sure you've trained a model:
```bash
python -m scripts.pra_train --depth=2 --n-embd=60 --num-iterations=100
```

Checkpoints are saved to: `~/.cache/nanochat/pra_checkpoints/`

### "Stockfish not found"

Stockfish is optional. The interactive screen works without it - you just won't see the Stockfish evaluation column.

### Model loads slowly

- First load may take a few seconds on CPU
- GPU inference is much faster if available
- Model stays loaded between games (no need to reload)

### Moves seem random

At only 5 training steps with 5% accuracy, the model is essentially random. This is expected! Try training longer:

```bash
# Quick test (5-10 minutes)
python -m scripts.pra_train --depth=4 --n-embd=144 --num-iterations=1000

# Serious training (several hours)
python -m scripts.pra_train --depth=8 --n-embd=192 --num-iterations=10000
```

## Next Steps

1. **Train more**: Your current model is undertrained. Run more training iterations with more data.

2. **Compare models**: Train multiple models with different hyperparameters and watch them play.

3. **Analyze divergence**: When Stockfish is enabled, watch where your model's moves diverge from optimal play.

4. **Check training metrics**: Review your training logs for loss curves and accuracy improvements.

## Tips

- **Game speed**: Currently fixed at 1 second per move (future: adjustable slider)
- **Long games**: Some games may go on for 100+ moves. Use Reset to start fresh.
- **Draws**: Games end on checkmate, stalemate, or other draw conditions
- **Evaluation divergence**: Large gaps between model and Stockfish eval indicate learning opportunities

## Example Session

```
$ python -m nanochess
[Main Menu]
> Press I for Interactive

[Interactive Screen]
> Select: "pra_d2 (step 5)"
> Click: "Load Model"
Status: Loaded pra_d2 (step 5)

> Check: "Use Stockfish evaluation"
Status: Stockfish enabled

> Click: "Start"
[Game begins...]

Move: 1    Model: +0.05    Stockfish: +0.2
Move: 5    Model: -0.12    Stockfish: +0.5
Move: 10   Model: +0.34    Stockfish: -0.3
...

[Watch the board update in real-time!]
```

Enjoy watching your model learn to play chess! 🎉
