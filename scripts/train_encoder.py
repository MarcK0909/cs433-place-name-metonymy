"""Run the adapted Group 1 CE / CE+SupCon encoder trainer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metonymy.group1.train_encoder import main

if __name__ == "__main__":
    main()
