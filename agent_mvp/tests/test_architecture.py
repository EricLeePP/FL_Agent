"""Architecture ratchet tests.

Each limit records the current state of a structural rule. A refactor may lower
a limit; nothing may raise one. When a limit reaches its target, the rule
becomes a hard invariant.
"""

import ast
import re
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "federated_agent"

# Checks that branch on a collaborator's concrete class instead of a declared capability.
CONCRETE_TYPE_CHECK = re.compile(
    r"isinstance\([a-z_.]+, *\(?[A-Z][A-Za-z0-9]*(Executor|Checker|Planner|Provider)"
    r"|type\([a-z_.]+\) is( not)? [A-Z][A-Za-z0-9]*(Executor|Checker|Planner|Provider)")

LIMITS = {
    "concrete_type_checks": 0,
    "source_tree_paths": 0,
    "write_json_definitions": 1,
}

# Third-party integrations that only their adapter modules may import.
INTEGRATION_OWNERS = {
    "vantage6": {"adapters/vantage6.py"},
    "agents": {"sdk_runtime.py"},
    "openai": {"sdk_runtime.py"},
    "httpx2": {"sdk_runtime.py"},
    "mcp": {"mcp_server.py"},
}


# Transports translate input and output. Study, run and policy rules belong to services/ and the agent.
TRANSPORTS = {"cli.py", "terminal_ui.py"}
RULE_MODULES_FORBIDDEN_IN_CLI = {"runs", "policy", "storage", "run_manifest", "confirmation", "sessions.SessionStore"}


def modules():
    return sorted(PACKAGE.rglob("*.py"))


def relative(path):
    return str(path.relative_to(PACKAGE)).replace("\\", "/")


def package_imports(path):
    """Package-relative imports of ``path`` as dotted names, e.g. "runs.store" and "runs.store.RunStore"."""
    found = set()
    package = relative(path).split("/")[:-1]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            module = [*package[:len(package) - node.level + 1], *(node.module.split(".") if node.module else [])]
            found.add(".".join(module))
            found.update(".".join([*module, alias.name]) for alias in node.names)
    return found


def imported_roots(path):
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class ArchitectureRatchetTests(unittest.TestCase):
    def count(self, predicate):
        return sum(predicate(line) for path in modules() for line in path.read_text(encoding="utf-8").splitlines())

    def test_concrete_type_checks_do_not_grow(self):
        found = self.count(lambda line: bool(CONCRETE_TYPE_CHECK.search(line)))
        self.assertLessEqual(found, LIMITS["concrete_type_checks"],
                             "Declare a capability on the collaborator instead of checking its class.")

    def test_source_tree_relative_paths_do_not_grow(self):
        found = self.count(lambda line: "parents[2]" in line)
        self.assertLessEqual(found, LIMITS["source_tree_paths"],
                             "Resolve resources through federated_agent.resources, not the source tree.")

    def test_source_checkout_is_only_a_compatibility_name(self):
        users = {relative(path) for path in modules() if "SOURCE_CHECKOUT" in path.read_text(encoding="utf-8")}
        self.assertLessEqual(users, {"resources.py", "cli.py"})

    def test_atomic_json_writer_is_not_duplicated(self):
        found = self.count(lambda line: line.startswith("def write_json("))
        self.assertLessEqual(found, LIMITS["write_json_definitions"])

    def test_services_do_not_depend_on_transports(self):
        for path in sorted((PACKAGE / "services").glob("*.py")):
            transports = {name for name in package_imports(path) if name.split(".")[0] + ".py" in TRANSPORTS}
            self.assertFalse(transports, f"{relative(path)} imports a transport: {transports}")

    def test_command_line_holds_no_run_or_policy_rules(self):
        imports = package_imports(PACKAGE / "cli.py")
        offending = {name for name in imports
                     if name.split(".")[0] in RULE_MODULES_FORBIDDEN_IN_CLI or name in RULE_MODULES_FORBIDDEN_IN_CLI}
        self.assertFalse(offending, "Move these rules into services/: " + ", ".join(sorted(offending)))

    def test_application_code_never_names_a_result_type(self):
        """Result handling is chosen by the adapter the catalog names (result_adapters.py)."""
        for path in [PACKAGE / "service.py", PACKAGE / "cli.py", *sorted((PACKAGE / "services").glob("*.py"))]:
            with self.subTest(module=relative(path)):
                self.assertNotIn("descriptive_statistics_v1", path.read_text(encoding="utf-8"))

    def test_model_instructions_live_in_the_prompts_module(self):
        for path in modules():
            if relative(path) == "prompts.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = {target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)}
                if (isinstance(node, ast.Assign) and any(name.endswith(("PROMPT", "INSTRUCTIONS")) for name in names)
                        and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                    self.fail(f"{relative(path)} defines {names}; keep model instructions in prompts.py")

    def test_third_party_integrations_stay_in_their_adapters(self):
        for path in modules():
            for root in imported_roots(path) & set(INTEGRATION_OWNERS):
                with self.subTest(module=relative(path), imports=root):
                    self.assertIn(relative(path), INTEGRATION_OWNERS[root])


if __name__ == "__main__":
    unittest.main()
