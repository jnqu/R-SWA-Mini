from __future__ import annotations

import platform
import sys


def _safe_import_version(module_name: str) -> str:
    """Import a module and return its __version__, or a readable error string.

    We do this defensively so a single missing package produces a clear
    message instead of crashing the whole report.
    """
    try:
        module = __import__(module_name)
    except Exception as exc: 
        return f"NOT INSTALLED ({type(exc).__name__}: {exc})"
    return getattr(module, "__version__", "unknown")


def main() -> None:
    print("=" * 60)
    print("R-SWA Mini — environment check (Milestone 0)")
    print("=" * 60)

    # --- Interpreter / OS
    print("\n[interpreter / OS]")
    print(f"  Python           : {platform.python_version()}")
    print(f"  Executable       : {sys.executable}")
    print(f"  Platform         : {platform.platform()}")
    print(f"  Machine          : {platform.machine()}")  # e.g. arm64 on Apple Silicon

    # --- Core libraries
    print("\n[libraries]")
    print(f"  torch            : {_safe_import_version('torch')}")
    print(f"  transformers     : {_safe_import_version('transformers')}")
    print(f"  numpy            : {_safe_import_version('numpy')}")
    print(f"  matplotlib       : {_safe_import_version('matplotlib')}")
    print(f"  pandas           : {_safe_import_version('pandas')}")

    print("\n[compute device]")
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        mps_ok = torch.backends.mps.is_available()
        print(f"  CUDA available   : {cuda_ok}")
        if cuda_ok:
            print(f"  CUDA device      : {torch.cuda.get_device_name(0)}")
        print(f"  MPS available    : {mps_ok}  (Apple GPU / Metal backend)")

        if cuda_ok:
            recommended = "cuda"
        elif mps_ok:
            recommended = "mps  (but use 'cpu' for deterministic correctness tests)"
        else:
            recommended = "cpu"
        print(f"  Recommended dev  : {recommended}")
        print(f"  Default dtype    : {torch.get_default_dtype()}")

        # --- Check tensors work 

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        product = a @ b
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        matches = torch.allclose(product, expected)
        print("\n[sanity op]")
        print(f"  2x2 matmul works : {matches}  (a @ b == expected)")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch unavailable: {exc}")



if __name__ == "__main__":
    main()
