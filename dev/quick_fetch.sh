#!/bin/bash
# Quick fetch presets for common use cases

set -e

echo "NANOCHESS DATA FETCHER"
echo "====================="
echo ""
echo "Select a preset:"
echo ""
echo "1) Quick test (1K positions, fast)"
echo "2) Small dataset (50K positions, ~10 min)"
echo "3) Medium dataset (500K positions, ~1 hour)"
echo "4) Large dataset (2M positions, ~3 hours)"
echo "5) Magnus Carlsen games only (all available)"
echo "6) Current month Lichess DB (very large, 10+ GB)"
echo "7) Custom (manual configuration)"
echo ""
read -p "Choice [1-7]: " choice

case $choice in
    1)
        echo "Fetching quick test dataset..."
        python dev/fetch_chess_data.py \
            --source lichess-elite \
            --max-games 100 \
            --max-positions 1000 \
            --time-control rapid
        ;;
    2)
        echo "Fetching small dataset..."
        python dev/fetch_chess_data.py \
            --source lichess-elite \
            --max-games 500 \
            --max-positions 50000 \
            --time-control classical
        ;;
    3)
        echo "Fetching medium dataset..."
        python dev/fetch_chess_data.py \
            --source lichess-elite \
            --max-games 2000 \
            --max-positions 500000 \
            --time-control classical
        ;;
    4)
        echo "Fetching large dataset..."
        python dev/fetch_chess_data.py \
            --source lichess-elite \
            --max-games 10000 \
            --max-positions 2000000 \
            --time-control classical
        ;;
    5)
        echo "Fetching Magnus Carlsen games..."
        python dev/fetch_chess_data.py \
            --source lichess-user \
            --username DrNykterstein \
            --max-games 5000 \
            --time-control all
        ;;
    6)
        echo "Downloading current month Lichess database..."
        current_month=$(date +%Y-%m)
        python dev/fetch_chess_data.py \
            --source lichess-db \
            --month "$current_month" \
            --sample-rate 0.01
        ;;
    7)
        echo ""
        echo "Custom configuration:"
        read -p "Source [lichess-elite/lichess-user/chesscom-titled]: " source
        read -p "Max games [10000]: " max_games
        read -p "Max positions [-1 for all]: " max_positions
        read -p "Time control [classical/rapid/blitz/all]: " time_control

        max_games=${max_games:-10000}
        max_positions=${max_positions:--1}
        time_control=${time_control:-classical}

        python dev/fetch_chess_data.py \
            --source "$source" \
            --max-games "$max_games" \
            --max-positions "$max_positions" \
            --time-control "$time_control"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✓ Done! Data ready at ~/.cache/nanochat/pra_data"
echo ""
echo "Next steps:"
echo "  1. Check dataset: ls -lh ~/.cache/nanochat/pra_data"
echo "  2. Start training: python -m nanochess"
