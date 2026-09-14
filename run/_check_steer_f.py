#!/usr/bin/env python3
"""Check that the `steer_f` on this box has the symbols `verl` calls by name.

WHY THIS EXISTS

    This repository is split across two branches with *unrelated* git
    histories, and both of them carry a ``steer_f/``.  They are not two
    versions of one package -- they are two lineages, and only the
    ``origin/paper`` one matches the ``verl/`` that ships beside it:

        verl/workers/actor/dp_actor.py:407
            from steer_f.verl_integration import forecast_h_togo

    ``forecast_h_togo`` does not exist in this branch's ``steer_f``.  Nor do
    ``compute_a_h``, ``sibling_support``, ``oracle_h_togo``,
    ``token_weight_distribution`` or ``first_divergence``, and
    ``compute_token_weights_steerf`` lives in a different module than verl
    imports it from.  A box assembled with this branch's copy trains nothing:
    every tree and steer arm dies inside worker init.

    It was not caught because every gate in the repo checks ``import
    steer_f.tree_rollout`` -- the one file the two lineages happen to share
    byte for byte.  So the gate passed on a box where the treatment could not
    run.  This script checks what verl actually imports instead of what we
    happened to grep for: it reads the import statements out of ``verl/``
    itself, so it stays right when verl changes.

Usage:  python3 run/_check_steer_f.py [repo-root]
        exit 0 = every symbol resolves (or there is no verl/ here to serve)
        exit 1 = something verl imports is missing; the fix is printed
"""
import ast
import pathlib
import sys

GREEN, YELLOW, RED, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def import_sites(verl_root: pathlib.Path):
    """Every `from steer_f.x import a, b` verl contains -> {module: {names}}."""
    want: dict[str, set[str]] = {}
    for path in verl_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.split(".")[0] == "steer_f":
                want.setdefault(node.module, set()).update(
                    a.name for a in node.names if a.name != "*")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "steer_f":
                        want.setdefault(a.name, set())
    return want


def defined_names(mod_path: pathlib.Path) -> set[str]:
    """Top-level names a module source binds -- defs, classes, assignments and
    re-exports.  Used only when the module cannot be imported (no torch on a
    CI box); an import is the real test, this is the honest fallback."""
    names: set[str] = set()
    try:
        tree = ast.parse(mod_path.read_text(errors="replace"))
    except (OSError, SyntaxError):
        return names
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    verl_root, sf_root = root / "verl", root / "steer_f"
    if not verl_root.is_dir():
        print(f"  {GREEN}OK{OFF}    no verl/ here -- nothing calls into steer_f",
              file=sys.stderr)
        return 0
    if not sf_root.is_dir():
        print(f"  {RED}FAIL{OFF}  verl/ is here but steer_f/ is not", file=sys.stderr)
        return 1
    # A checkout of this branch alone has an empty verl/trainer/ skeleton and no
    # trainer in it. Finding no import sites there means there is nothing to
    # serve yet, not that steer_f is complete -- say which.
    if not (verl_root / "workers" / "actor" / "dp_actor.py").is_file():
        print(f"  {YELLOW}WARN{OFF}  verl/ here has no workers/actor/dp_actor.py -- "
              f"this is not a trainer checkout, so there is nothing to check yet",
              file=sys.stderr)
        return 0

    sys.path.insert(0, str(root))
    missing, checked, by_import = [], 0, True
    for module, names in sorted(import_sites(verl_root).items()):
        try:
            obj = __import__(module, fromlist=["_"])
            have = set(dir(obj))
        except Exception:
            by_import = False
            have = defined_names(sf_root / (module.split(".", 1)[1] + ".py")) \
                if "." in module else defined_names(sf_root / "__init__.py")
            if not have:
                missing.append((module, "<module>"))
                continue
        for name in sorted(names):
            checked += 1
            if name not in have:
                missing.append((module, name))

    how = "imported" if by_import else "read from source (no import possible here)"
    if not missing:
        print(f"  {GREEN}OK{OFF}    steer_f has all {checked} symbol(s) verl imports "
              f"({how})", file=sys.stderr)
        return 0

    print(f"  {RED}FAIL{OFF}  steer_f is missing {len(missing)} symbol(s) that verl "
          f"imports by name ({how}):", file=sys.stderr)
    for module, name in missing:
        print(f"          {module}.{name}", file=sys.stderr)
    print(f"""
        This is the two-lineage split, not a half-finished install. The verl/
        on this box comes from origin/paper and calls the origin/paper
        steer_f; this checkout has the other one. Training cannot start.

        fix:  git checkout origin/paper -- steer_f
              python3 run/_check_steer_f.py
""", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
