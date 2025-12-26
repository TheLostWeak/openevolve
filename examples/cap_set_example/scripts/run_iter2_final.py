import importlib.util
import time
import traceback

MODULE_PATH = "examples/cap_set_example/exports/iter2/final_code.py"

def load_module(path: str):
    spec = importlib.util.spec_from_file_location("final_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    print(f"Loading module: {MODULE_PATH}")
    try:
        mod = load_module(MODULE_PATH)
    except Exception as e:
        print("Failed to import module:")
        traceback.print_exc()
        return 1

    if not hasattr(mod, "generate_set"):
        print("Module does not define generate_set(n). Nothing to run.")
        return 1

    try:
        start = time.time()
        result = mod.generate_set(8)
        elapsed = time.time() - start
        print(f"generate_set returned type: {type(result)}")
        try:
            length = len(result)
        except Exception:
            length = "<no len>"
        print(f"Result length: {length}")
        print(f"Elapsed time: {elapsed:.2f} s")
        # Print a short sample of the result
        if isinstance(result, (list, tuple)):
            print("Sample items:")
            for i, item in enumerate(result[:5]):
                print(f"  [{i}] {item}")
        return 0
    except Exception:
        print("generate_set raised an exception:")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
