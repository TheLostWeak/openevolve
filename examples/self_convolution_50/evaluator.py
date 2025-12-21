import importlib.util
import sys
import traceback
from typing import Dict
import numpy as np

def _load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = mod
    spec.loader.exec_module(mod)
    return mod


def evaluate(program_path: str) -> Dict[str, float]:
    try:
        mod = _load_candidate(program_path)
        if hasattr(mod, "get_steps"):
            a = np.asarray(mod.get_steps(), dtype=float)
        elif hasattr(mod, "steps"):
            a = np.asarray(mod.steps, dtype=float)
        else:
            raise ValueError("No steps defined")

        if np.min(a) < 0.0:
            raise ValueError("Steps must be nonnegative")

        n = 50
        if a.shape != (n,):
            raise ValueError(f"Expected {n} steps")

        conv_len = 2*n
        slopes = np.zeros(conv_len)
        intercepts = np.zeros(conv_len)

        Delta = 1 / (2 * n)

        for i in range(n):
            ai = a[i]
            if ai == 0.0:
                continue
            for j in range(n):
                aj = a[j]
                if aj == 0.0:
                    continue
                slopes[i+j] += ai * aj
                slopes[i+j+1] -= ai * aj
                intercepts[i+j+1] += ai * aj

        L1 = 0.0
        L2sq = 0.0
        Linf = 0.0

        for k in range(conv_len):
            m = slopes[k]
            b = intercepts[k]
            L1 += 0.5 * m + b
            L2sq += (m ** 2 / 3) + m * b + b ** 2
            Linf = max(Linf, max(abs(m + b), abs(b)))
        
        ratio = L2sq / (L1 * Linf) if (L1 > 0.0 and Linf > 0.0) else 0.0

        return {
            "combined_score": float(ratio),
            "size": float(np.count_nonzero(a)),
            "max_norm": float(np.max(a)),
            "self_convolution_L1": float(L1) / Delta,
            "self_convolution_L2sq": float(L2sq) / Delta / Delta,
            "self_convolution_Linf": float(Linf),
        }

    except Exception as e:
        # Print traceback to stdout/stderr for real-time logs
        traceback.print_exc()

        # Attempt to persist the problematic candidate source for offline analysis
        try:
            import os
            import time
            import uuid

            base_dir = os.path.dirname(__file__)
            failures_dir = os.path.join(base_dir, "openevolve_output", "llm_candidate_errors")
            os.makedirs(failures_dir, exist_ok=True)

            # Read program source if available
            candidate_source = None
            try:
                with open(program_path, "r", encoding="utf-8") as pf:
                    candidate_source = pf.read()
            except Exception:
                candidate_source = None

            file_name = f"candidate_err_{int(time.time())}_{uuid.uuid4().hex}.txt"
            file_path = os.path.join(failures_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(f"error: {str(e)}\n")
                fh.write("--- traceback ---\n")
                fh.write(''.join(traceback.format_exception(None, e, e.__traceback__)))
                fh.write("\n\n--- candidate source (if available) ---\n")
                fh.write(candidate_source or "<unavailable>")

            print(f"Saved candidate failure to: {file_path}")
        except Exception:
            # Don't let persistence errors mask the original evaluation error
            pass

        return {
            "combined_score": 0.0,
            "size": 0,
            "max_norm": 0.0,
            "self_convolution_L1": 0.0,
            "self_convolution_L2sq": 0.0,
            "self_convolution_Linf": 0.0,
            "error": str(e),
        }

if __name__ == "__main__":
    class Dummy:
        steps = [1.0] * 50

    import tempfile
    import os

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write("steps = [1.0]*50\n")
        path = f.name

    print(evaluate(path))
    os.remove(path)
