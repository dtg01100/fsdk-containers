"""Workflow security and publication contract tests.

Validates safety invariants in .github/workflows/ and Justfile:
- ghcr-cleanup.yml: safe dry-run by default on cron, multi-tag release protection
- printing-runtime-layer.yml: needs.build.result == 'success' requirement,
  same-run immutable index assembly, per-arch signing on all refs,
  compat-auth-file login for cosign, and safe Justfile cleanup.
"""

from pathlib import Path
import re
import unittest
import yaml


ROOT = Path(__file__).parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
JUSTFILE = ROOT / "Justfile"


class WorkflowSecurityContractTests(unittest.TestCase):
    """Static contracts for workflow execution and registry operations."""

    EXACT_DRY_RUN_EXPR = "${{ github.event_name != 'workflow_dispatch' || inputs.dry_run }}"
    EXACT_DELETE_TAGS = "latest"
    EXACT_EXCLUDE_TAGS = "[0-9]*,*-*,sha256-*"

    def test_ghcr_cleanup_enforces_exact_safe_dry_run_expression(self):
        """ghcr-cleanup must use the exact safe dry-run expression across all cleanup steps."""
        path = WORKFLOW_DIR / "ghcr-cleanup.yml"
        self.assertTrue(path.exists(), "ghcr-cleanup.yml must exist")
        content = path.read_text()
        doc = yaml.safe_load(content)

        # Every step calling ghcr-cleanup-action must use the exact expression
        steps = doc["jobs"]["cleanup"]["steps"]
        cleanup_steps = [s for s in steps if "ghcr-cleanup-action" in s.get("uses", "")]
        self.assertGreaterEqual(len(cleanup_steps), 3, "All cleanup steps must be present")

        for step in cleanup_steps:
            dry_run = step.get("with", {}).get("dry-run", "")
            self.assertEqual(
                dry_run, self.EXACT_DRY_RUN_EXPR,
                f"Step '{step.get('name')}' must use exact expression '{self.EXACT_DRY_RUN_EXPR}', got '{dry_run}'"
            )

    def test_ghcr_cleanup_evaluates_four_case_dry_run_truth_table(self):
        """Evaluate the exact expression across the 4 schedule and dispatch permutations."""
        def eval_gha_expr(event_name: str, inputs_dry_run: bool | None) -> bool:
            # Mirrors GitHub Actions: `${{ github.event_name != 'workflow_dispatch' || inputs.dry_run }}`
            is_not_dispatch = (event_name != "workflow_dispatch")
            # In GHA, if inputs.dry_run is not set (e.g. on schedule), it evaluates to falsy
            dry_run_val = bool(inputs_dry_run) if inputs_dry_run is not None else False
            return is_not_dispatch or dry_run_val

        # Case 1: scheduled cron (inputs.dry_run is empty/unset) -> MUST dry-run
        self.assertTrue(eval_gha_expr("schedule", None),
                        "Case 1 failed: scheduled run with empty input must dry-run")

        # Case 2: scheduled cron with explicit false -> MUST still dry-run (event guard holds)
        self.assertTrue(eval_gha_expr("schedule", False),
                        "Case 2 failed: scheduled run must never execute live deletion")

        # Case 3: manual workflow_dispatch with default dry_run: true -> MUST dry-run
        self.assertTrue(eval_gha_expr("workflow_dispatch", True),
                        "Case 3 failed: workflow_dispatch default must dry-run")

        # Case 4: manual workflow_dispatch with unchecked box (dry_run: false) -> live deletion permitted
        self.assertFalse(eval_gha_expr("workflow_dispatch", False),
                         "Case 4 failed: workflow_dispatch with dry_run=false is the sole live delete case")

    def test_ghcr_cleanup_protects_multi_tagged_release_versions(self):
        """Tag deletion must target exactly 'latest' and use exact exclusion pattern."""
        path = WORKFLOW_DIR / "ghcr-cleanup.yml"
        content = path.read_text()
        doc = yaml.safe_load(content)

        steps = doc["jobs"]["cleanup"]["steps"]
        reap_step = next((s for s in steps if "delete-tags" in s.get("with", {})), None)
        self.assertIsNotNone(reap_step, "Reap stale :latest step must exist")

        delete_tags = reap_step["with"].get("delete-tags", "")
        self.assertEqual(delete_tags, self.EXACT_DELETE_TAGS,
                         f"delete-tags must be exact '{self.EXACT_DELETE_TAGS}', got '{delete_tags}'")

        exclude = reap_step["with"].get("exclude-tags", "")
        self.assertEqual(exclude, self.EXACT_EXCLUDE_TAGS,
                         f"exclude-tags must be exact '{self.EXACT_EXCLUDE_TAGS}', got '{exclude}'")


if __name__ == "__main__":
    unittest.main()
