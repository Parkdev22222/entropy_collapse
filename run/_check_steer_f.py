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


def steer_f_signatures(sf_root: pathlib.Path) -> dict:
    """{function name: (positional names, kw-only names, n required, **kwargs?)}
    read from steer_f's own sources. Classes map to None and are skipped."""
    sigs: dict = {}
    for path in sorted(sf_root.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                sigs[node.name] = None
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                pos = [x.arg for x in a.posonlyargs + a.args]
                sigs[node.name] = (pos, [x.arg for x in a.kwonlyargs],
                                   len(pos) - len(a.defaults), a.kwarg is not None)
    return sigs


def queue_scripts(root: pathlib.Path) -> set[str]:
    """Scripts the queues in run/ actually invoke. A mismatch in one of those
    fails a stage; a mismatch anywhere else is worth saying but not refusing."""
    import re
    names: set[str] = set()
    for sh in (root / "run").glob("*.sh"):
        names.update(re.findall(r"scripts/[A-Za-z0-9_]+\.py", sh.read_text(errors="replace")))
    return names


def check_call_sites(root: pathlib.Path, sf_root: pathlib.Path) -> list[str]:
    """Names existing is not enough: the two lineages also disagree on
    SIGNATURES. measure_ah_support.py imported entropy_advantage successfully
    and then died on `unexpected keyword argument 'response_ids'`, because the
    donor's takes (h_togo_vals, group_index, mask, responses=...) and returns a
    tensor where this branch's takes response_ids=/group_size= and returns a
    2-tuple. Returns the fatal problems; warnings are printed as it goes."""
    sigs = steer_f_signatures(sf_root)
    queued, fatal = queue_scripts(root), []
    for path in sorted((root / "scripts").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except (OSError, SyntaxError):
            continue
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    and n.module and n.module.split(".")[0] == "steer_f"
                    for a in n.names}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in imported):
                continue
            sig = sigs.get(node.func.id)
            if sig is None:
                continue
            pos, kwonly, n_req, has_kwargs = sig
            used = [k.arg for k in node.keywords if k.arg]
            unknown = [k for k in used if k not in pos + kwonly and not has_kwargs]
            rel = path.relative_to(root).as_posix()
            level = fatal if rel in queued else None
            if unknown:
                msg = (f"{rel}: {node.func.id}(...) passes {unknown}, but this "
                       f"steer_f takes {pos}")
            elif len(node.args) + len(used) < n_req and not any(
                    k.arg is None for k in node.keywords):
                msg = (f"{rel}: {node.func.id}(...) gets "
                       f"{len(node.args) + len(used)} argument(s), this steer_f "
                       f"requires {n_req}")
            else:
                continue
            if level is None:
                print(f"  {YELLOW}WARN{OFF}  {msg} (no queue runs it)", file=sys.stderr)
            else:
                fatal.append(msg)
    return fatal


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
        bad_calls = check_call_sites(root, sf_root) if (root / "scripts").is_dir() else []
        if bad_calls:
            print(f"  {RED}FAIL{OFF}  a script a queue runs calls steer_f with the "
                  f"other lineage's signature:", file=sys.stderr)
            for msg in bad_calls:
                print(f"          {msg}", file=sys.stderr)
            print("""
        Importing the name worked; calling it will not. Take the donor's copy
        of that script -- it ships one that matches its own steer_f.

        fix:  bash run/bootstrap_pod.sh
""", file=sys.stderr)
            return 1
        print(f"  {GREEN}OK{OFF}    every queue-run script calls it compatibly",
              file=sys.stderr)
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
