"""
Compatibility shim for local imports.

When the package is used without installing the project (editable or wheel), Python may find
`agrovision_core` as a namespace package pointing at the project root. However our real
package code lives under `src/agrovision_core`. To make `import agrovision_core.*` work
when the package isn't installed, extend the package __path__ to include the `src` layout.

This keeps the behavior the same when the package *is* installed, and helps running code
from the repository without requiring editable installs in some environments (e.g., notebooks).
"""
from __future__ import annotations

from pathlib import Path

# Ensure the `src/agrovision_core` folder is on the package search path so submodules
# like `agrovision_core.train` load even when the project isn't installed into the env.
__path__.append(str(Path(__file__).resolve().parent.joinpath("src", "agrovision_core")))
