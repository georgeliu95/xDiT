"""Dependency-free contracts for the CUDA 13 FA4 image overlay."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "docker/Dockerfile.fa4-cu13"
PROBE_PATH = ROOT / "docker/verify_fa4_runtime.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class Fa4RuntimeRecipeContractTest(unittest.TestCase):
    def test_probe_help_does_not_require_runtime_dependencies(self) -> None:
        # Given: a host that is only inspecting the image preflight interface.
        command = [sys.executable, str(PROBE_PATH), "--help"]

        # When: the probe help is requested outside a CUDA runtime image.
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        # Then: argparse documents the command without importing torch or FA4.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--skip-device-check", result.stdout)

    def test_cleans_cutlass_distributions_before_installing_fa4(self) -> None:
        # Given: the CUDA 13 overlay used on an existing runtime image.
        recipe = _read(RECIPE_PATH)

        # When: its cleanup and FA4 installation stages are located.
        uninstall_start = recipe.index("python3 -m pip uninstall --yes")
        install_start = recipe.index("python3 -m pip install --no-cache-dir")
        uninstall_command = recipe[uninstall_start:install_start]
        uninstall_tokens = uninstall_command.replace("\\", "").split()

        # Then: every old CUTLASS layout is removed before the fresh install.
        self.assertLess(uninstall_start, install_start)
        for package in (
            "cutlass",
            "nvidia-cutlass",
            "nvidia-cutlass-dsl",
            "nvidia-cutlass-dsl-libs-base",
            "nvidia-cutlass-dsl-libs-cu13",
        ):
            self.assertIn(package, uninstall_tokens)

    def test_pins_fa4_and_preserves_the_existing_fa2_install(self) -> None:
        # Given: the overlay dependency recipe.
        recipe = _read(RECIPE_PATH)

        # When: the cleanup command is separated from the install command.
        uninstall_start = recipe.index("python3 -m pip uninstall --yes")
        install_start = recipe.index("python3 -m pip install --no-cache-dir")
        uninstall_command = recipe[uninstall_start:install_start]

        # Then: FA4 and CUTLASS are exact, while the inherited FA2 stays intact.
        self.assertIn("ARG FLASH_ATTN_4_VERSION=4.0.0b22", recipe)
        self.assertIn("ARG CUTLASS_DSL_VERSION=4.6.0.dev0", recipe)
        self.assertIn(
            '"flash-attn-4[cu13]==${FLASH_ATTN_4_VERSION}"',
            recipe,
        )
        self.assertNotIn("--force-reinstall", recipe)
        self.assertNotIn("flash-attn", uninstall_command)
        self.assertNotIn("flash_attn", uninstall_command)

    def test_build_runs_the_import_probe_without_requiring_a_gpu(self) -> None:
        # Given: a Docker build, where a GPU is normally unavailable.
        recipe = _read(RECIPE_PATH)

        # When/Then: the shared runtime probe is copied and run import-only.
        self.assertIn(
            "COPY docker/verify_fa4_runtime.py /usr/local/bin/verify-xdit-fa4-runtime",
            recipe,
        )
        self.assertIn(
            "python3 /usr/local/bin/verify-xdit-fa4-runtime --skip-device-check",
            recipe,
        )

    def test_runtime_probe_checks_sm120_and_records_both_backends(self) -> None:
        # Given: the preflight program embedded in the overlay image.
        probe = _read(PROBE_PATH)

        # When/Then: default execution fails closed off SM120 and emits evidence.
        self.assertIn("torch.cuda.get_device_capability()", probe)
        self.assertIn("capability != (12, 0)", probe)
        self.assertIn("from flash_attn import flash_attn_func as fa2", probe)
        self.assertIn(
            "from flash_attn.cute.interface import flash_attn_func as fa4",
            probe,
        )
        for field in (
            '"flash_attn_version"',
            '"flash_attn_4_version"',
            '"nvidia_cutlass_dsl_version"',
            '"gpu_capability"',
        ):
            self.assertIn(field, probe)


if __name__ == "__main__":
    unittest.main()
