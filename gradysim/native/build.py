"""
Compilation of native protocol modules.
"""
import hashlib
import importlib.machinery
import importlib.util
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

INCLUDE_DIRECTORY = Path(__file__).parent / "include"
FLAGS = ["-O3", "-std=c++17", "-shared", "-fPIC",
         # Keep floating point results identical to Python's: Python squares with libm pow and never fuses
         # multiplications and additions
         "-fno-builtin-pow", "-ffp-contract=off"]

_loaded = {}


def default_cache_directory() -> Path:
    return Path(os.environ.get("GRADYSIM_NATIVE_CACHE", Path.home() / ".cache" / "gradysim" / "native"))


def build(source: Union[str, Path], cache_directory: Optional[Union[str, Path]] = None,
          compiler: Optional[str] = None, extra_flags: Sequence[str] = ()):
    """
    Compiles a C++ source file declaring native protocols into a Python extension module and imports it.

    Compiled modules are cached by the content of the source, the engine headers and the compiler flags, so building
    an unchanged source again, in this or any other process, just imports the cached module. Build once before
    starting worker processes, so they don't all compile at the same time.

    Args:
        source: Path of the C++ source. It includes `gradysim_native/module.hpp` and declares its protocols with
            `GRADYSIM_PROTOCOLS`
        cache_directory: Where compiled modules are kept. Defaults to $GRADYSIM_NATIVE_CACHE or
            ~/.cache/gradysim/native
        compiler: C++ compiler. Defaults to $CXX or c++
        extra_flags: Additional compiler flags

    Returns:
        The imported module
    """
    import pybind11

    source = Path(source).resolve()
    compiler = compiler or os.environ.get("CXX", "c++")
    cache_directory = Path(cache_directory) if cache_directory is not None else default_cache_directory()
    flags = FLAGS + list(extra_flags)

    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    for header in sorted(INCLUDE_DIRECTORY.rglob("*.hpp")):
        digest.update(header.read_bytes())
    digest.update(" ".join([compiler, *flags, sys.version, pybind11.__version__]).encode())
    name = f"{source.stem}_{digest.hexdigest()[:16]}"

    if name in _loaded:
        return _loaded[name]

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    library = cache_directory / f"{name}{suffix}"
    if not library.exists():
        cache_directory.mkdir(parents=True, exist_ok=True)
        includes = [f"-I{INCLUDE_DIRECTORY}", f"-I{pybind11.get_include()}",
                    f"-I{sysconfig.get_paths()['include']}"]
        with tempfile.TemporaryDirectory(dir=cache_directory) as temporary:
            output = Path(temporary) / library.name
            command = [compiler, *flags, *includes, f"-DGRADYSIM_MODULE_NAME={name}", str(source), "-o", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f"Compilation of {source} failed:\n{shlex.join(command)}\n{completed.stderr}")
            # Atomic, so processes building the same module concurrently don't see a partial file
            os.replace(output, library)

    loader = importlib.machinery.ExtensionFileLoader(name, str(library))
    spec = importlib.util.spec_from_file_location(name, str(library), loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    _loaded[name] = module
    return module
