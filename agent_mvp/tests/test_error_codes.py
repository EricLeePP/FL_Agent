"""Every researcher-facing error code used in the source has an explanation."""

import ast
import unittest
from pathlib import Path

from federated_agent.feedback import MESSAGES

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "federated_agent"
CODE_CALLS = {"issue", "BackendExecutionError"}


def codes_in(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name in CODE_CALLS and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                found.add(node.args[0].value)
            for keyword in node.keywords:
                if keyword.arg == "code" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    found.add(keyword.value.value)
        if isinstance(node, ast.Dict) and node.keys and all(isinstance(key, ast.Constant) and isinstance(key.value, int)
                                                            for key in node.keys):
            found.update(value.value for value in node.values
                         if isinstance(value, ast.Constant) and isinstance(value.value, str))
    return found


class ErrorCodeVocabularyTests(unittest.TestCase):
    def test_every_code_used_in_the_source_has_a_message(self):
        used = set().union(*(codes_in(path) for path in PACKAGE.rglob("*.py")))
        self.assertGreater(len(used), 30)
        self.assertEqual(sorted(used - set(MESSAGES)), [])


if __name__ == "__main__":
    unittest.main()
