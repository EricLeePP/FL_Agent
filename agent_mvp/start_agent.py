"""Start the synthetic real-backend CLI demonstration; secrets stay local."""

import os
import sys
from pathlib import Path

from federated_agent.cli import main


def launcher_arguments(arguments: list[str]) -> list[str]:
    output = Path(__file__).resolve().parents[1] / "integration" / "agent-demo"
    defaults = ["--chat", "--provider", "deepseek", "--agent-runtime", "sdk", "--backend", "vantage6",
                "--confirm-synthetic-unencrypted-sandbox", "--output", str(output)]
    key_file = Path.home() / "Desktop" / "deepseek-api.txt"
    if "DEEPSEEK_API_KEY_FILE" not in os.environ and key_file.is_file():
        defaults.extend(["--deepseek-api-key-file", str(key_file)])
    return [*defaults, *arguments]


if __name__ == "__main__":
    sys.argv[1:] = launcher_arguments(sys.argv[1:])
    main()
