"""Nerd Font icons for the Nanochess TUI.

All icons use Unicode characters from Nerd Fonts (FontAwesome and Codicons).
Make sure your terminal has a Nerd Font installed.
"""

# Box Drawing - Rounded (LazyVim style)
BOX_ROUNDED_TL = "\u256d"  # ╭ Top Left
BOX_ROUNDED_TR = "\u256e"  # ╮ Top Right
BOX_ROUNDED_BR = "\u256f"  # ╯ Bottom Right
BOX_ROUNDED_BL = "\u2570"  # ╰ Bottom Left
BOX_HORIZONTAL = "\u2500"  # ─ Horizontal
BOX_VERTICAL = "\u2502"    # │ Vertical

# Box Drawing - Single
BOX_SINGLE_TL = "\u250c"   # ┌
BOX_SINGLE_TR = "\u2510"   # ┐
BOX_SINGLE_BR = "\u2518"   # ┘
BOX_SINGLE_BL = "\u2514"   # └
BOX_SINGLE_H = "\u2500"    # ─
BOX_SINGLE_V = "\u2502"    # │

# File & Folder Icons (FontAwesome)
FOLDER = "\uf07b"          #
FOLDER_OPEN = "\uf07c"     #
FILE = "\uea7b"            #

# Data & Database Icons
DATABASE = "\uf1c0"        #
SERVER = "\uf233"          #
HARD_DRIVE = "\uf0a0"      #

# System & Hardware Icons
GPU = "\uf0c5"             #  (server icon, alternative for GPU)
CPU = "\uf0e4"             #
MEMORY = "\uf538"          #
CHIP = "\uf2db"            #

# Status & Info Icons
INFO = "\uf05a"            #
WARNING = "\uf071"         #
ERROR = "\uf057"           #
SUCCESS = "\uf00c"         #
HINT = "\uf0eb"            #

# Action Icons
PLAY = "\uf04b"            #
STOP = "\uf04d"            #
PAUSE = "\uf04c"           #
REFRESH = "\uf021"         #
SETTINGS = "\uf013"        #

# Training & ML Icons
TRAINING = "\uf0e7"        # ⚡ (lightning bolt for training)
MODEL = "\uf2db"           #
CHART = "\uf080"           #
METRICS = "\uf201"         #

# Navigation Icons
ARROW_RIGHT = "\uf054"     #
ARROW_LEFT = "\uf053"      #
CHEVRON_RIGHT = "\uf054"   #
CHEVRON_LEFT = "\uf053"    #

# Misc Icons
CHESS = "\u265f"           # ♟ (chess pawn)
SEARCH = "\uf002"          #
CLOCK = "\uf017"           #
DOLLAR = "\uf155"          #
TAG = "\uf02b"             #

# Separator
SEPARATOR = BOX_VERTICAL   # │

# Unicode Block Elements (for progress bars, etc.)
BLOCK_FULL = "\u2588"      # █
BLOCK_HALF = "\u2592"      # ▒
BLOCK_LIGHT = "\u2591"     # ░

# Selection & Toggle Icons
CHECK = "\uf00c"           # ✓
XMARK = "\uf00d"           # ✗
CIRCLE_FILLED = "\uf111"   # ● (filled circle for selected radio)
CIRCLE_EMPTY = "\uf10c"    # ○ (empty circle for unselected radio)
SQUARE_FILLED = "\uf14a"   # ■ (filled square for checked)
SQUARE_EMPTY = "\uf096"    # □ (empty square for unchecked)
