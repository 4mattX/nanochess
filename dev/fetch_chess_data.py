"""
Fetch high-quality chess games from online sources.

Downloads games from top-rated players and automatically converts them
to the parquet format required for PRA training.

Supported sources:
- Lichess API (2500+ rated players)
- Lichess database archives (monthly dumps)
- Chess.com API (requires username)

Usage:
    # Fetch recent games from Lichess titled players (GMs, IMs)
    python dev/fetch_chess_data.py --source lichess-elite --max-games 10000

    # Fetch from specific Lichess user (e.g., Magnus Carlsen)
    python dev/fetch_chess_data.py --source lichess-user --username DrNykterstein --max-games 500

    # Download Lichess database archive (large, pre-annotated with evals)
    python dev/fetch_chess_data.py --source lichess-db --month 2024-01

    # Fetch from Chess.com titled players
    python dev/fetch_chess_data.py --source chesscom-titled --max-games 5000

The script automatically:
- Downloads games in PGN format
- Filters by rating threshold
- Converts to parquet using prepare_chess_positions.py
- Saves to ~/.cache/nanochess/pra_data
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Optional
import random
import tempfile

try:
    import requests
except ImportError:
    print("Error: requests library required")
    print("Install with: pip install requests")
    sys.exit(1)


class LichessAPIFetcher:
    """Fetch games from Lichess API."""

    BASE_URL = "https://lichess.org/api"

    def __init__(self, min_rating: int = 2500):
        self.min_rating = min_rating
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/x-chess-pgn"})

    def get_titled_players(self) -> List[str]:
        """Get list of titled players (GM, IM, etc.)."""
        url = f"{self.BASE_URL}/player"
        response = self.session.get(url)
        if response.status_code != 200:
            print(f"Warning: Could not fetch titled players list")
            return []

        # Parse NDJSON response
        players = []
        for line in response.text.strip().split('\n'):
            if line:
                import json
                player = json.loads(line)
                if 'username' in player:
                    players.append(player['username'])

        return players

    def get_top_players(self, limit: int = 100) -> List[str]:
        """Get top rated players from leaderboard."""
        url = f"{self.BASE_URL}/player/top/{limit}/classical"
        response = self.session.get(url)
        if response.status_code != 200:
            print(f"Warning: Could not fetch top players")
            return []

        import json
        data = json.loads(response.text)
        return [player['username'] for player in data.get('users', [])]

    def fetch_user_games(
        self,
        username: str,
        max_games: int = 100,
        time_control: str = "classical",
        with_evals: bool = True,
    ) -> str:
        """Fetch games for a specific user.

        Args:
            username: Lichess username
            max_games: Maximum number of games to fetch
            time_control: blitz, rapid, classical, or None for all
            with_evals: Include computer analysis

        Returns:
            PGN string of all games
        """
        url = f"{self.BASE_URL}/games/user/{username}"
        params = {
            "max": max_games,
            "perfType": time_control if time_control != "all" else None,
            "evals": "true" if with_evals else "false",
            "clocks": "false",
            "opening": "false",
        }

        print(f"  Fetching games for {username}...")
        response = self.session.get(url, params=params, stream=True)

        if response.status_code != 200:
            print(f"  Warning: Failed to fetch games for {username}")
            return ""

        pgn_data = response.text
        return pgn_data

    def fetch_elite_games(
        self,
        max_games: int = 10000,
        time_control: str = "classical",
        with_evals: bool = True,
    ) -> str:
        """Fetch games from elite players.

        Args:
            max_games: Maximum total games to fetch
            time_control: blitz, rapid, classical, or all
            with_evals: Include computer analysis

        Returns:
            Combined PGN string
        """
        print("Fetching top players...")
        players = self.get_top_players(limit=100)

        if not players:
            print("Warning: Could not get player list, using fallback list")
            # Fallback to known strong players
            players = [
                "DrNykterstein",  # Magnus Carlsen
                "FabianoCaruana",
                "Hikaru",
                "DanielNaroditsky",
                "gmwesleysoso",
                "Duhless",
                "GMHKCHESSIFY",
                "GCAGM",
            ]

        print(f"Found {len(players)} top players")

        # Just fetch from first few players instead of spreading across all
        # Request many games per player since we don't care about diversity
        games_per_player = max_games
        all_pgn = []
        total_games_collected = 0

        for username in players:
            if total_games_collected >= max_games:
                break

            pgn = self.fetch_user_games(
                username,
                max_games=games_per_player,
                time_control=time_control,
                with_evals=with_evals,
            )

            if pgn:
                # Count games in this batch
                num_games = pgn.count('[Event "')
                total_games_collected += num_games
                all_pgn.append(pgn)
                print(f"  → Collected {num_games} games (total so far: {total_games_collected})")

            # Rate limiting
            time.sleep(0.5)

        print(f"\nTotal games collected: {total_games_collected}")
        return "\n\n".join(all_pgn)


class LichessDatabaseFetcher:
    """Fetch games from Lichess database archives."""

    BASE_URL = "https://database.lichess.org"

    def download_month(self, year: int, month: int, output_path: str):
        """Download a monthly database archive.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            output_path: Path to save the downloaded file

        The downloaded file is in .pgn.zst format (zstandard compressed).
        """
        month_str = f"{year:04d}-{month:02d}"
        url = f"{self.BASE_URL}/standard/lichess_db_standard_rated_{month_str}.pgn.zst"

        print(f"Downloading Lichess database for {month_str}...")
        print(f"URL: {url}")
        print("Warning: This can be 10+ GB! Download may take a while.")

        # Check if file already exists
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}")
            return output_path

        # Stream download with progress
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download: HTTP {response.status_code}")

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = 100 * downloaded / total_size
                        print(f"  Progress: {pct:.1f}% ({downloaded // 1024 // 1024} MB)", end="\r")

        print(f"\nDownload complete: {output_path}")
        return output_path


class ChessComAPIFetcher:
    """Fetch games from Chess.com API."""

    BASE_URL = "https://api.chess.com/pub"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "nanochess-data-fetcher/1.0"
        })

    def get_titled_players(self, title: str = "GM") -> List[str]:
        """Get titled players from Chess.com.

        Args:
            title: GM, IM, FM, WGM, WIM, etc.

        Returns:
            List of usernames
        """
        url = f"{self.BASE_URL}/titled/{title}"
        response = self.session.get(url)

        if response.status_code != 200:
            print(f"Warning: Could not fetch {title} list")
            return []

        import json
        data = json.loads(response.text)
        return data.get("players", [])

    def fetch_user_games(
        self,
        username: str,
        year: int,
        month: int,
    ) -> str:
        """Fetch games for a user in a specific month.

        Args:
            username: Chess.com username
            year: Year
            month: Month (1-12)

        Returns:
            PGN string
        """
        url = f"{self.BASE_URL}/player/{username}/games/{year}/{month:02d}/pgn"
        response = self.session.get(url)

        if response.status_code != 200:
            return ""

        return response.text

    def fetch_titled_games(
        self,
        max_games: int = 5000,
        title: str = "GM",
        year: int = 2024,
        month: int = 1,
    ) -> str:
        """Fetch games from titled players.

        Args:
            max_games: Maximum total games
            title: GM, IM, etc.
            year: Year to fetch
            month: Month to fetch

        Returns:
            Combined PGN string
        """
        print(f"Fetching {title} players from Chess.com...")
        players = self.get_titled_players(title)

        if not players:
            print(f"Warning: No {title} players found")
            return ""

        print(f"Found {len(players)} {title} players")

        # Sample players randomly
        random.shuffle(players)
        all_pgn = []
        games_fetched = 0

        for username in players[:50]:  # Limit to 50 players
            if games_fetched >= max_games:
                break

            print(f"  Fetching games for {username}...")
            pgn = self.fetch_user_games(username, year, month)

            if pgn:
                all_pgn.append(pgn)
                # Rough estimate: 50 games per month per player
                games_fetched += 50

            # Rate limiting
            time.sleep(1)

        return "\n\n".join(all_pgn)


def convert_pgn_to_parquet(
    pgn_path: str,
    output_dir: str,
    with_eval: bool = True,
    max_positions: int = -1,
    sample_rate: float = 1.0,
):
    """Convert PGN to parquet using prepare_chess_positions.py.

    Args:
        pgn_path: Path to PGN file
        output_dir: Output directory for parquet files
        with_eval: Extract evaluations from comments
        max_positions: Maximum positions to extract
        sample_rate: Sampling rate (0-1)
    """
    script_path = Path(__file__).parent / "prepare_chess_positions.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input", pgn_path,
        "--output-dir", output_dir,
        "--sample-rate", str(sample_rate),
    ]

    if with_eval:
        cmd.append("--with-eval")
    else:
        # When not using position-specific evals, use game outcomes instead
        cmd.append("--use-game-outcome")

    if max_positions > 0:
        cmd.extend(["--max-positions", str(max_positions)])

    print("\nConverting PGN to parquet...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Warning: Conversion may have failed")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch high-quality chess data from online sources"
    )

    parser.add_argument(
        "--source",
        choices=[
            "lichess-elite",
            "lichess-user",
            "lichess-db",
            "chesscom-titled",
        ],
        default="lichess-elite",
        help="Data source to fetch from",
    )

    parser.add_argument(
        "--username",
        type=str,
        help="Specific username to fetch (for lichess-user)",
    )

    parser.add_argument(
        "--max-games",
        type=int,
        default=10000,
        help="Maximum number of games to fetch",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=-1,
        help="Maximum positions to extract (-1 = all)",
    )

    parser.add_argument(
        "--month",
        type=str,
        help="Month to download (YYYY-MM) for lichess-db",
    )

    parser.add_argument(
        "--time-control",
        choices=["blitz", "rapid", "classical", "all"],
        default="rapid",
        help="Time control to fetch (default: rapid - good balance of speed and quality)",
    )

    parser.add_argument(
        "--min-rating",
        type=int,
        default=2500,
        help="Minimum player rating",
    )

    parser.add_argument(
        "--with-eval",
        action="store_true",
        help="Extract position-specific evaluations from analyzed games (rare). Default: use game outcomes (win/draw/loss)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.cache/nanochat/pra_data"),
        help="Output directory for parquet files",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Sampling rate for positions (0-1)",
    )

    parser.add_argument(
        "--keep-pgn",
        action="store_true",
        help="Keep intermediate PGN file",
    )

    args = parser.parse_args()

    # Create temporary PGN file
    temp_dir = tempfile.mkdtemp()
    pgn_path = os.path.join(temp_dir, "downloaded_games.pgn")

    try:
        # Fetch games based on source
        if args.source == "lichess-elite":
            fetcher = LichessAPIFetcher(min_rating=args.min_rating)
            pgn_data = fetcher.fetch_elite_games(
                max_games=args.max_games,
                time_control=args.time_control,
                with_evals=args.with_eval,
            )

            print(f"\nWriting PGN to {pgn_path}...")
            with open(pgn_path, "w") as f:
                f.write(pgn_data)

        elif args.source == "lichess-user":
            if not args.username:
                print("Error: --username required for lichess-user source")
                sys.exit(1)

            fetcher = LichessAPIFetcher(min_rating=args.min_rating)
            pgn_data = fetcher.fetch_user_games(
                username=args.username,
                max_games=args.max_games,
                time_control=args.time_control,
                with_evals=args.with_eval,
            )

            print(f"\nWriting PGN to {pgn_path}...")
            with open(pgn_path, "w") as f:
                f.write(pgn_data)

        elif args.source == "lichess-db":
            if not args.month:
                print("Error: --month required for lichess-db source (e.g., 2024-01)")
                sys.exit(1)

            year, month = map(int, args.month.split("-"))
            fetcher = LichessDatabaseFetcher()
            pgn_path = os.path.join(temp_dir, f"lichess_db_{args.month}.pgn.zst")
            fetcher.download_month(year, month, pgn_path)

        elif args.source == "chesscom-titled":
            import datetime
            current_date = datetime.datetime.now()

            fetcher = ChessComAPIFetcher()
            pgn_data = fetcher.fetch_titled_games(
                max_games=args.max_games,
                title="GM",
                year=current_date.year,
                month=current_date.month,
            )

            print(f"\nWriting PGN to {pgn_path}...")
            with open(pgn_path, "w") as f:
                f.write(pgn_data)

        # Convert to parquet
        if os.path.exists(pgn_path):
            convert_pgn_to_parquet(
                pgn_path=pgn_path,
                output_dir=args.output_dir,
                with_eval=args.with_eval,
                max_positions=args.max_positions,
                sample_rate=args.sample_rate,
            )

            print(f"\n✓ Data ready in {args.output_dir}")

            if args.keep_pgn:
                final_pgn = os.path.join(args.output_dir, "downloaded_games.pgn")
                os.rename(pgn_path, final_pgn)
                print(f"PGN saved to: {final_pgn}")
        else:
            print("Error: No PGN file created")
            sys.exit(1)

    finally:
        # Cleanup
        if not args.keep_pgn and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
