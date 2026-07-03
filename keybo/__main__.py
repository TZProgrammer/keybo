"""Enable ``python -m keybo`` by delegating to the CLI dispatcher."""

from keybo.cli.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
