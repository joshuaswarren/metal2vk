"""Write uzu's published per-device benchmark tables (from the saved metrics page) under docs/omarchy-linux/data/."""
import pathlib
import subprocess
import sys

here = pathlib.Path(__file__).parent
out = here.parent.parent / "docs/omarchy-linux/data"
out.mkdir(parents=True, exist_ok=True)
page = sys.argv[1]
for dev in ("macos-m1-pool", "macos-m2-pool", "macos-m5-max-pool"):
    text = subprocess.run([sys.executable, str(here / "parse-uzu-metrics.py"), page, dev], capture_output=True, text=True, check=True).stdout
    (out / f"uzu-published-{dev}.txt").write_text(text)
    print(dev, len(text.splitlines()), "lines")
