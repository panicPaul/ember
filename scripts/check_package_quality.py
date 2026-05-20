#!/usr/bin/env python3

"""Audit package source for hidden names and missing docstrings."""

from __future__ import annotations

import argparse
import ast
import fnmatch
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "packages"

GENERATED_FILE_NAMES = frozenset({"_version.py"})
GENERATED_PATH_PARTS = frozenset({"__pycache__", "build"})
ALLOWED_PRIVATE_EXACT_NAMES = frozenset({"_", "__all__"})
ALLOWED_PRIVATE_PREFIXES = ("__",)
DEFAULT_IGNORES = (
    "packages/ember-native-powerfoam/src/ember_native_powerfoam/powerfoam/native/warp/*",
    "packages/ember-native-radfoam/src/ember_native_radfoam/radfoam/native/*",
)


@dataclass(frozen=True)
class Finding:
    """One package quality finding with enough context for a direct fix."""

    code: str
    path: Path
    line: int
    name: str
    message: str


def package_source_files() -> list[Path]:
    """Return Python package files that should be checked."""
    return sorted(
        path
        for path in PACKAGE_ROOT.glob("*/src/**/*.py")
        if not GENERATED_PATH_PARTS.intersection(path.parts)
        and path.name not in GENERATED_FILE_NAMES
    )


def relative_path(path: Path) -> str:
    """Return a repository-relative path with POSIX separators."""
    return path.relative_to(REPO_ROOT).as_posix()


def package_name(path: Path) -> str:
    """Return the owning package directory for a package source file."""
    return path.relative_to(PACKAGE_ROOT).parts[0]


def path_is_ignored(path: Path, ignore_patterns: Iterable[str]) -> bool:
    """Return whether the repository-relative path matches an ignore pattern."""
    relative = relative_path(path)
    return any(
        fnmatch.fnmatch(relative, pattern) for pattern in ignore_patterns
    )


def is_private_name(name: str) -> bool:
    """Return whether a name uses a hidden leading-underscore spelling."""
    return name.startswith("_") and name not in ALLOWED_PRIVATE_EXACT_NAMES


def is_allowed_private_name(name: str) -> bool:
    """Return whether a hidden-looking name is a Python protocol spelling."""
    return name.startswith(ALLOWED_PRIVATE_PREFIXES) and name.endswith("__")


def collect_assignment_names(target: ast.expr) -> list[tuple[str, str]]:
    """Collect assigned names and attributes from a Python assignment target."""
    if isinstance(target, ast.Name):
        return [(target.id, "name")]
    if isinstance(target, ast.Attribute):
        base_name = getattr(target.value, "id", "")
        kind = "self attribute" if base_name == "self" else "attribute"
        return [(target.attr, kind)]
    if isinstance(target, ast.Starred):
        return collect_assignment_names(target.value)
    if isinstance(target, ast.Tuple | ast.List):
        names: list[tuple[str, str]] = []
        for element in target.elts:
            names.extend(collect_assignment_names(element))
        return names
    return []


def check_docstring(
    findings: list[Finding],
    path: Path,
    node: ast.AST,
    name: str,
    kind: str,
) -> None:
    """Append a finding when a function or class has no docstring."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        if any(
            isinstance(decorator, ast.Name) and decorator.id == "overload"
            for decorator in node.decorator_list
        ):
            return
    if ast.get_docstring(node) is not None:
        return
    findings.append(
        Finding(
            code="DOC001",
            path=path,
            line=getattr(node, "lineno", 1),
            name=name,
            message=f"{kind} `{name}` is missing a docstring",
        )
    )


def check_private_name(
    findings: list[Finding],
    path: Path,
    line: int,
    name: str,
    kind: str,
) -> None:
    """Append a finding when a name uses a hidden leading underscore."""
    if not is_private_name(name) or is_allowed_private_name(name):
        return
    findings.append(
        Finding(
            code="NAM001",
            path=path,
            line=line,
            name=name,
            message=f"{kind} `{name}` uses a hidden leading underscore",
        )
    )


def check_dynamic_access(
    findings: list[Finding],
    path: Path,
    node: ast.Call,
) -> None:
    """Append a finding for reflection that should stay localized and named."""
    parent: ast.AST | None = node
    while parent is not None:
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
            if parent.name == "__getattr__":
                return
            break
        parent = getattr(parent, "parent", None)

    function = node.func
    if not isinstance(function, ast.Name):
        return
    if function.id not in {
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
    }:
        return
    findings.append(
        Finding(
            code="DYN001",
            path=path,
            line=node.lineno,
            name=function.id,
            message=f"`{function.id}` should be isolated behind a typed helper",
        )
    )


def check_file(path: Path) -> list[Finding]:
    """Return quality findings for one package source file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            check_docstring(findings, path, node, node.name, "class")
            check_private_name(findings, path, node.lineno, node.name, "class")
            continue

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            check_docstring(findings, path, node, node.name, "function")
            check_private_name(
                findings, path, node.lineno, node.name, "function"
            )
            continue

        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name, kind in collect_assignment_names(target):
                    check_private_name(findings, path, node.lineno, name, kind)
            continue

        if isinstance(node, ast.AnnAssign | ast.AugAssign):
            for name, kind in collect_assignment_names(node.target):
                check_private_name(findings, path, node.lineno, name, kind)
            continue

        if isinstance(node, ast.Call):
            check_dynamic_access(findings, path, node)

    return findings


def sorted_findings(findings: list[Finding]) -> list[Finding]:
    """Return findings in stable path, line, code order."""
    return sorted(
        findings,
        key=lambda finding: (
            relative_path(finding.path),
            finding.line,
            finding.code,
            finding.name,
        ),
    )


def print_summary(findings: list[Finding], checked_file_count: int) -> None:
    """Print finding counts by code and package."""
    by_code = Counter(finding.code for finding in findings)
    by_package = Counter(package_name(finding.path) for finding in findings)

    print("Package quality audit summary")
    print(f"files checked: {checked_file_count}")
    print(f"findings: {len(findings)}")
    for code, count in sorted(by_code.items()):
        print(f"{code}: {count}")
    print()
    print("Findings by package")
    for package, count in by_package.most_common():
        print(f"{package}: {count}")


def print_findings(findings: list[Finding], limit: int | None) -> None:
    """Print individual findings in a grep-friendly format."""
    for index, finding in enumerate(sorted_findings(findings)):
        if limit is not None and index >= limit:
            remaining = len(findings) - limit
            print(f"... {remaining} more findings")
            return
        print(
            f"{relative_path(finding.path)}:{finding.line}: "
            f"{finding.code} {finding.message}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the package quality audit."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only aggregate counts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="maximum number of findings to print; use -1 for no limit",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=list(DEFAULT_IGNORES),
        help="repository-relative fnmatch pattern to ignore",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit with a non-zero status when findings remain",
    )
    return parser.parse_args()


def main() -> int:
    """Run the package quality audit."""
    args = parse_args()
    findings: list[Finding] = []
    checked_file_count = 0
    for path in package_source_files():
        if path_is_ignored(path, args.ignore):
            continue
        checked_file_count += 1
        try:
            findings.extend(check_file(path))
        except SyntaxError as error:
            findings.append(
                Finding(
                    code="SYN001",
                    path=path,
                    line=error.lineno or 1,
                    name=path.name,
                    message=f"failed to parse file: {error.msg}",
                )
            )

    print_summary(findings, checked_file_count)
    if findings and not args.summary_only:
        print()
        print_findings(findings, None if args.limit < 0 else args.limit)

    return 1 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
