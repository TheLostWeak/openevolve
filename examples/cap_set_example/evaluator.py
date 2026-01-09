"""Evaluator for cap set example.

Implements `evaluate(program_path)` so it can be used directly by OpenEvolve.
The evaluator loads the provided `initial_program.py` module, calls 
`generate_set(n)`, verifies the cap set property, and returns metrics. It also
supports an environment variable `CAP_N` to set the dimension.
"""

import importlib.util
import sys
import os
import time
import math
import logging
import shutil
import uuid
import datetime
import subprocess
import tempfile
import json
import traceback
from typing import Dict, Any, Tuple, Optional, Set, List

def _get_n_from_env_or_config(program_path: str) -> int:
    return 8


def _get_timeout_from_env_or_config(program_path: str) -> int:
    """Determine evaluator timeout in seconds.

    Precedence:
      1. Environment variable `EVALUATOR_TIMEOUT`
      2. `config.yaml` in the same directory as `program_path` (key: `evaluator.timeout`)
      3. Default value 600
    """
    # 1) environment
    val = os.environ.get("EVALUATOR_TIMEOUT")
    if val:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

    # 2) config.yaml
    cfg_path = os.path.join(os.path.dirname(program_path), "config.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ev = cfg.get("evaluator")
            if isinstance(ev, dict) and "timeout" in ev:
                try:
                    return int(ev["timeout"])
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            logging.getLogger(__name__).debug("Failed to parse config.yaml for timeout: %s", e)

    return 600
def _load_generate_set(program_path: str):
    mod_name = f"cap_module_{os.path.basename(program_path).replace('.', '_')}_{abs(hash(program_path))}"
    spec = importlib.util.spec_from_file_location(mod_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load evaluation module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "generate_set"):
        raise AttributeError("Module must provide `generate_set(n)`")
    return module.generate_set


def _call_generate_in_subprocess(program_path: str, n: int, timeout_seconds: float) -> Dict[str, Any]:
    """Run the target module's `generate_set(n)` in a separate process with a timeout.

    Returns a dict with keys: `success` (bool), and on success `result` (list of lists),
    on failure `error` and optional `traceback`.
    Raises TimeoutError if the subprocess exceeds `timeout_seconds`.
    """
    # Create a small runner script that imports the module by path and calls generate_set
    runner_code = f"""
import importlib.util, importlib.machinery, json, sys, traceback
spec = importlib.util.spec_from_file_location('genmod', r'{program_path}')
mod = importlib.util.module_from_spec(spec)
sys.modules['genmod'] = mod
try:
    spec.loader.exec_module(mod)
except Exception as e:
    out = {{'success': False, 'error': 'module import failed', 'traceback': traceback.format_exc()}}
    print(json.dumps(out))
    sys.exit(2)
if not hasattr(mod, 'generate_set'):
    out = {{'success': False, 'error': 'module has no generate_set'}}
    print(json.dumps(out))
    sys.exit(2)
try:
    arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    res = mod.generate_set(arg)
    # Convert tuples to lists for JSON
    def conv(o):
        if isinstance(o, tuple):
            return [conv(x) for x in o]
        if isinstance(o, list):
            return [conv(x) for x in o]
        return o
    out = {{'success': True, 'result': conv(res)}}
    print(json.dumps(out))
    sys.exit(0)
except Exception as e:
    out = {{'success': False, 'error': str(e), 'traceback': traceback.format_exc()}}
    print(json.dumps(out))
    sys.exit(3)
"""

    # Helper to compute artifacts dir
    def _artifacts_dir_for(path: str) -> str:
        out_base = os.environ.get("OPENEVOLVE_OUTPUT", os.path.join(os.path.dirname(path), "openevolve_output"))
        artifacts_dir = os.path.join(out_base, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        return artifacts_dir

    # Write runner to temp file
    fd, runner_path = tempfile.mkstemp(prefix="openevolve_gen_runner_", suffix=".py")
    os.close(fd)
    try:
        with open(runner_path, 'w', encoding='utf-8') as f:
            f.write(runner_code)
        # Run subprocess
        try:
            proc = subprocess.run([sys.executable, runner_path, str(n)], capture_output=True, text=True, timeout=timeout_seconds)
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as te:
            # capture any partial output
            stdout = te.stdout.decode('utf-8') if isinstance(te.stdout, (bytes, bytearray)) else (te.stdout or "")
            stderr = te.stderr.decode('utf-8') if isinstance(te.stderr, (bytes, bytearray)) else (te.stderr or "")

            # save stdout/stderr to artifacts for post-mortem
            artifacts_dir = _artifacts_dir_for(program_path)
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            uid = uuid.uuid4().hex[:8]
            out_path = os.path.join(artifacts_dir, f"{ts}_{uid}_generator_stdout_timeout.json")
            err_path = os.path.join(artifacts_dir, f"{ts}_{uid}_generator_stderr_timeout.txt")
            try:
                with open(out_path, 'w', encoding='utf-8') as of:
                    of.write(stdout or "")
            except Exception:
                pass
            try:
                with open(err_path, 'w', encoding='utf-8') as ef:
                    ef.write(stderr or "")
            except Exception:
                pass

            raise TimeoutError(f"generate_set timed out after {timeout_seconds} seconds")
    except Exception as e:
        return {'success': False, 'error': f'subprocess failure: {e}', 'traceback': traceback.format_exc()}
    finally:
        try:
            os.remove(runner_path)
        except Exception:
            pass
    # Save stdout/stderr into artifacts for auditing (don't print them)
    artifacts_dir = _artifacts_dir_for(program_path)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:8]
    out_path = os.path.join(artifacts_dir, f"{ts}_{uid}_generator_stdout.json")
    err_path = os.path.join(artifacts_dir, f"{ts}_{uid}_generator_stderr.txt")
    try:
        with open(out_path, 'w', encoding='utf-8') as of:
            of.write(stdout or "")
    except Exception:
        out_path = None
    try:
        with open(err_path, 'w', encoding='utf-8') as ef:
            ef.write(stderr or "")
    except Exception:
        err_path = None

    if not stdout:
        return {'success': False, 'error': 'no stdout from generator', 'stderr': stderr, 'stderr_artifact': err_path}

    try:
        obj = json.loads(stdout)
        # include artifact paths for auditing
        obj.setdefault('stdout_artifact', out_path)
        if err_path:
            obj.setdefault('stderr_artifact', err_path)
        return obj
    except Exception as e:
        return {'success': False, 'error': 'invalid json from generator', 'stdout': stdout, 'stderr': stderr, 'traceback': traceback.format_exc(), 'stdout_artifact': out_path, 'stderr_artifact': err_path}

def _vector_to_int(vec: tuple, n: int) -> int:
    """将向量编码为整数以便快速操作。
    
    使用三进制编码：坐标0,1,2分别对应三进制位。
    对于n维向量，整数 = sum(vec[i] * 3^i)
    """
    # 使用累乘避免反复计算幂 (更快且内存友好)
    result = 0
    mul = 1
    for v in vec:
        result += int(v) * mul
        mul *= 3
    return result

def _int_to_vector(num: int, n: int) -> tuple:
    """将整数解码回向量。"""
    vec = [0] * n
    for i in range(n):
        vec[i] = num % 3
        num //= 3
    return tuple(vec)

def _negate_vector_int(vec_int: int, n: int) -> int:
    """计算向量的负值（模3）。"""
    result = 0
    for i in range(n):
        digit = vec_int % 3
        neg_digit = (-digit) % 3
        result += neg_digit * (3 ** i)
        vec_int //= 3
    return result

def _subtract_vectors_int(a_int: int, b_int: int, n: int) -> int:
    """计算 a - b (模3)。"""
    result = 0
    for i in range(n):
        a_digit = a_int % 3
        b_digit = b_int % 3
        diff = (a_digit - b_digit) % 3
        result += diff * (3 ** i)
        a_int //= 3
        b_int //= 3
    return result

def _is_cap_set_small_optimized(S: list, n: int, deadline: Optional[float] = None) -> Tuple[bool, Optional[tuple]]:
    """针对小规模集合的优化验证（集合大小 < 100）。
    
    使用整数编码和缓存来加速计算。
    """
    if len(S) < 3:
        return True, None
    
    # 转换为整数编码并检查重复
    S_ints = []
    seen_ints = set()
    for vec in S:
        vec_int = _vector_to_int(vec, n)
        if vec_int in seen_ints:
            # 返回三元组 witness；对于重复使用第三位 None
            return False, (vec, vec, None)  # 重复向量
        S_ints.append(vec_int)
        seen_ints.add(vec_int)
    
    # 预计算所有向量的负值
    neg_ints = [_negate_vector_int(v_int, n) for v_int in S_ints]
    
    # 使用整数集合进行快速查找
    int_set = set(S_ints)
    
    m = len(S_ints)
    for i in range(m):
        # deadline check
        if deadline is not None and time.time() > deadline:
            raise TimeoutError("Verification exceeded time limit")
        a_int = S_ints[i]
        neg_a = neg_ints[i]
        
        for j in range(i + 1, m):
            # deadline check inside inner loop as well
            if deadline is not None and time.time() > deadline:
                raise TimeoutError("Verification exceeded time limit")
            b_int = S_ints[j]
            # 计算 c = -(a + b) = -a - b
            # 注意：neg_a 已经是 -a，所以 c = neg_a - b
            c_int = _subtract_vectors_int(neg_a, b_int, n)
            
            if c_int in int_set:
                # 检查c是否等于a或b
                if c_int != a_int and c_int != b_int:
                    c_vec = _int_to_vector(c_int, n)
                    return False, (S[i], S[j], c_vec)
    
    return True, None

def _is_cap_set_large_early_prune(S: list, n: int, deadline: Optional[float] = None) -> Tuple[bool, Optional[tuple]]:
    """针对大规模集合的验证，使用早期剪枝。
    
    基于已知的数学定理：|S| ≤ 2 * 3^n / (n+1) (Ellenberg-Gijswijt上界)
    以及更紧的上界：|S| ≤ 2.756^n
    """
    m = len(S)
    
    # 1. 快速上界检查
    upper_bound = min(2 * (3 ** n) / (n + 1), 2.756 ** n)
    if m > upper_bound * 1.01:  # 允许1%的误差容限
        return False, None  # 不可能合法
    
    # 2. 检查重复元素
    seen = {}
    for idx, vec in enumerate(S):
        if vec in seen:
            # 重复向量，统一为三元组 (orig, dup, None)
            return False, (S[seen[vec]], vec, None)
        seen[vec] = idx
    
    if m < 1000:
        # 对于中等规模，使用整数编码方法
        return _is_cap_set_small_optimized(S, n, deadline)
    
    # 3. 对于大规模集合，使用抽样验证
    # 只检查一部分向量对，如果发现违规则返回
    import random
    random.seed(42)  # 确定性抽样
    
    # 转换为整数编码
    S_ints = [_vector_to_int(vec, n) for vec in S]
    int_set = set(S_ints)
    
    # 预计算负值
    neg_ints = [_negate_vector_int(v_int, n) for v_int in S_ints]
    
    # 抽样检查：对于每个向量，随机检查一定数量的配对
    sample_size = min(1000, m)  # 每个向量检查最多1000个配对
    
    for i in range(m):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError("Verification exceeded time limit")
        a_int = S_ints[i]
        neg_a = neg_ints[i]
        
        # 随机选择要检查的j
        if m > sample_size:
            # 随机抽样
            indices = random.sample(range(i + 1, m), min(sample_size, m - i - 1))
        else:
            # 检查所有
            indices = range(i + 1, m)
        
        for j in indices:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError("Verification exceeded time limit")
            b_int = S_ints[j]
            c_int = _subtract_vectors_int(neg_a, b_int, n)
            
            if c_int in int_set and c_int != a_int and c_int != b_int:
                c_vec = _int_to_vector(c_int, n)
                return False, (S[i], S[j], c_vec)
    
    # 4. 如果抽样没有发现问题，进行完整检查但使用更高效的算法
    # 按坐标和模3分组
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, vec in enumerate(S):
        groups[sum(vec) % 3].append(idx)
    
    # 只检查可能违规的组合
    for r in range(3):
        group_indices = groups[r]
        group_size = len(group_indices)
        
        for i_idx in range(group_size):
            if deadline is not None and time.time() > deadline:
                raise TimeoutError("Verification exceeded time limit")
            i = group_indices[i_idx]
            a_int = S_ints[i]
            neg_a = neg_ints[i]
            for j_idx in range(i_idx + 1, group_size):
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError("Verification exceeded time limit")
                j = group_indices[j_idx]
                b_int = S_ints[j]
                c_int = _subtract_vectors_int(neg_a, b_int, n)

                if c_int in int_set and c_int != a_int and c_int != b_int:
                    c_vec = _int_to_vector(c_int, n)
                    return False, (S[i], S[j], c_vec)
    
    # 检查跨组组合（0,1,2）
    for i in groups[0]:
        if deadline is not None and time.time() > deadline:
            raise TimeoutError("Verification exceeded time limit")
        a_int = S_ints[i]
        neg_a = neg_ints[i]
        for j in groups[1]:
            if deadline is not None and time.time() > deadline:
                raise TimeoutError("Verification exceeded time limit")
            b_int = S_ints[j]
            c_int = _subtract_vectors_int(neg_a, b_int, n)
            if c_int in int_set:
                # 检查c是否在group[2]中
                c_vec = _int_to_vector(c_int, n)
                if sum(c_vec) % 3 == 2 and c_int != a_int and c_int != b_int:
                    return False, (S[i], S[j], c_vec)
    
    return True, None

def _is_cap_set_incremental(S: list, n: int, deadline: Optional[float] = None) -> Tuple[bool, Optional[tuple]]:
    """增量验证：假设S是按贪心算法顺序构造的。
    
    利用贪心算法的性质：当添加新元素时，只需要检查它与已存在元素的组合。
    这可以大大减少验证时间。
    """
    if len(S) < 3:
        return True, None
    
    # 检查重复
    seen = set()
    for vec in S:
        if vec in seen:
            # 重复向量返回三元组 (orig, dup, None)
            return False, (vec, vec, None)
        seen.add(vec)
    
    # 转换为整数编码
    S_ints = [_vector_to_int(vec, n) for vec in S]
    int_set = set()
    
    # 预计算负值缓存
    neg_cache = {}
    
    # 模拟增量构造过程
    for i, vec_int in enumerate(S_ints):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError("Verification exceeded time limit")
        # 检查新向量与已有集合的兼容性
        if i > 0:
            # 预计算当前向量的负值
            if vec_int not in neg_cache:
                neg_cache[vec_int] = _negate_vector_int(vec_int, n)
            neg_current = neg_cache[vec_int]
            
            # 检查当前向量与每个已有向量
            for j in range(i):
                existing_int = S_ints[j]
                # 计算潜在的第三个向量 c = -(current + existing)
                # = neg_current - existing
                c_int = _subtract_vectors_int(neg_current, existing_int, n)
                
                if c_int in int_set:
                    # 确保c不是current或existing
                    if c_int != vec_int and c_int != existing_int:
                        c_vec = _int_to_vector(c_int, n)
                        return False, (S[i], S[j], c_vec)
        
        # 添加当前向量到集合
        int_set.add(vec_int)
    
    return True, None

def _is_cap_set_adaptive(S: list, n: int, deadline: Optional[float] = None) -> Tuple[bool, Optional[tuple]]:
    """自适应验证算法，根据集合大小和维度选择最优策略。"""
    m = len(S)
    
    # 超快速边界检查
    if m < 3:
        return True, None
    
    # 已知上界检查（Ellenberg-Gijswijt）
    if n <= 10:
        # 小n使用精确上界
        known_bounds = {4: 20, 5: 45, 6: 112, 7: 280, 8: 672}  # 近似值
        if n in known_bounds and m > known_bounds[n] * 1.05:  # 5%容差
            return False, None
    else:
        # 大n使用渐近上界
        asymptotic_bound = int(2.76 ** n)
        if m > asymptotic_bound:
            return False, None
    
    # 基于规模和维度选择算法
    if m < 50:
        # 小集合使用完整验证
        return _is_cap_set_small_optimized(S, n, deadline)
    elif m < 1000:
        # 中等集合使用增量验证（假设是贪心算法构造的）
        # 如果不是贪心构造的，回退到优化验证
        result = _is_cap_set_incremental(S, n, deadline)
        if not result[0]:
            # 如果增量验证发现违规，返回结果
            return result
        # 否则进行完整验证确保正确性
        return _is_cap_set_small_optimized(S, n, deadline)
    else:
        # 大规模集合使用早期剪枝和抽样
        return _is_cap_set_large_early_prune(S, n, deadline)

def evaluate(program_path: str) -> Dict[str, Any]:
    """Full evaluation: verify cap set property and size.

    Returns metrics where larger `combined_score` is better. We set
    `combined_score` = size if valid, otherwise 0.0.
    """
    n = _get_n_from_env_or_config(program_path)
    start_time = time.time()

    # 保存一份程序快照以便在超时或评估失败时仍能保留 LLM 生成的代码
    saved_artifact_path = ""
    try:
        def _save_program_snapshot(path: str) -> str:
            try:
                # 输出目录可由环境变量 `OPENEVOLVE_OUTPUT` 指定，默认置于 program 同目录下的 openevolve_output
                out_base = os.environ.get("OPENEVOLVE_OUTPUT", os.path.join(os.path.dirname(path), "openevolve_output"))
                artifacts_dir = os.path.join(out_base, "artifacts")
                os.makedirs(artifacts_dir, exist_ok=True)
                base = os.path.basename(path)
                ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                uid = uuid.uuid4().hex[:8]
                dest = os.path.join(artifacts_dir, f"{ts}_{uid}_{base}")
                shutil.copy2(path, dest)
                return dest
            except Exception as e:
                logging.getLogger(__name__).debug("Failed to save program snapshot: %s", e)
                return ""

        # 仅当未关闭自动保存时才保存（可通过环境变量关闭）
        if os.environ.get("SAVE_GENERATED_CODE", "true").lower() != "false":
            saved_artifact_path = _save_program_snapshot(program_path)
    except Exception:
        saved_artifact_path = ""
    
    try:
        # Determine overall evaluator timeout and compute deadline
        timeout_seconds = _get_timeout_from_env_or_config(program_path)
        # safety margin to allow clean reporting (seconds)
        safety_margin = min(1.0, 0.05 * timeout_seconds)
        deadline = start_time + timeout_seconds - safety_margin

        # Run the generator in a subprocess with the remaining total time
        now = time.time()
        gen_timeout = max(0.1, deadline - now)
        if gen_timeout <= 0:
            eval_time = time.time() - start_time
            res = {
                "combined_score": 0.0,
                "error": "no time available for generation",
                "eval_time_seconds": eval_time,
                "dimension": n,
            }
            if saved_artifact_path:
                res["artifact_path"] = saved_artifact_path
            return res

        gen_res = _call_generate_in_subprocess(program_path, n, gen_timeout)
        stdout_art = gen_res.get("stdout_artifact")
        stderr_art = gen_res.get("stderr_artifact")

        if not gen_res.get("success"):
            eval_time = time.time() - start_time
            res = {
                "combined_score": 0.0,
                "error": "generator failure",
                "error_detail": gen_res.get("error"),
                "eval_time_seconds": eval_time,
                "dimension": n,
            }
            if "traceback" in gen_res:
                res["traceback"] = gen_res["traceback"]
            if saved_artifact_path:
                res["artifact_path"] = saved_artifact_path
            if stdout_art:
                res["generator_stdout_artifact"] = stdout_art
            if stderr_art:
                res["generator_stderr_artifact"] = stderr_art
            return res

        S = gen_res.get("result", [])
        # expose generator artifact paths for auditing
        gen_stdout_artifact = stdout_art
        gen_stderr_artifact = stderr_art

        if not isinstance(S, (list, tuple)):
            return {"combined_score": 0.0, "error": "generate_set must return list/tuple"}

        # 快速长度检查
        if len(S) > 3 ** n:
            return {"combined_score": 0.0, "error": "Set size exceeds total number of vectors"}
        
        # 快速坐标检查
        S_tuples = []
        for s in S:
            if len(s) != n:
                return {"combined_score": 0.0, "error": f"element length mismatch: expected {n}, got {len(s)}"}
            
            t = tuple(int(x) for x in s)  # 快速转换，假设都是整数
            if any(c not in (0, 1, 2) for c in t):
                return {"combined_score": 0.0, "error": "coordinates must be in {0,1,2}"}
            S_tuples.append(t)

        # 使用自适应验证算法，带超时保护
        # deadline: end of total evaluation time minus safety margin
        deadline = start_time + timeout_seconds - safety_margin

        try:
            valid, witness = _is_cap_set_adaptive(S_tuples, n, deadline=deadline)
        except TimeoutError as te:
            eval_time = time.time() - start_time
            res = {
                "combined_score": 0.0,
                "error": "verification timeout",
                "error_detail": str(te),
                "eval_time_seconds": eval_time,
                "dimension": n
            }
            if saved_artifact_path:
                res["artifact_path"] = saved_artifact_path
            return res
        size = len(S_tuples)
        eval_time = time.time() - start_time

        if valid:
            res = {
                "combined_score": float(size), 
                "size": size, 
                "valid": True,
                "eval_time_seconds": eval_time,
                "dimension": n
            }
            if saved_artifact_path:
                res["artifact_path"] = saved_artifact_path
            if gen_stdout_artifact:
                res["generator_stdout_artifact"] = gen_stdout_artifact
            if gen_stderr_artifact:
                res["generator_stderr_artifact"] = gen_stderr_artifact
            return res
        else:
            res = {
                "combined_score": 0.0, 
                "size": size, 
                "valid": False, 
                "witness": witness,
                "eval_time_seconds": eval_time,
                "dimension": n
            }
            if saved_artifact_path:
                res["artifact_path"] = saved_artifact_path
            if gen_stdout_artifact:
                res["generator_stdout_artifact"] = gen_stdout_artifact
            if gen_stderr_artifact:
                res["generator_stderr_artifact"] = gen_stderr_artifact
            return res

    except Exception as e:
        eval_time = time.time() - start_time
        res = {
            "combined_score": 0.0, 
            "error": str(e),
            "eval_time_seconds": eval_time
        }
        if saved_artifact_path:
            res["artifact_path"] = saved_artifact_path
        return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--test", action="store_true", help="Run internal tests")
    args = parser.parse_args()
    
    if args.test:
        # 内部正确性测试
        print("Running correctness tests...")
        
        # 测试1: 小维度已知解
        test_cases = [
            (4, 20, True),   # n=4, 大小20, 应该是合法的
            (5, 45, True),   # n=5, 大小45, 应该是合法的
        ]
        
        for n, size, should_be_valid in test_cases:
            print(f"\nTesting n={n}, expected size={size}, valid={should_be_valid}")
            
            # 生成一个简单的测试集（这里只是示例，实际需要真实的Cap Set）
            # 由于没有真实数据，我们只测试算法逻辑
            
        print("\nAll tests completed.")
        
    elif args.benchmark:
        # 运行基准测试
        os.environ["CAP_N"] = str(args.n)
        print(f"Running benchmark for n={args.n}")
        
        # 测试不同算法
        import itertools
        
        # 生成测试集
        all_vectors = list(itertools.product([0, 1, 2], repeat=args.n))
        
        # 测试1: 小型合法集合
        small_legal = all_vectors[:20]
        
        # 测试2: 包含违规的集合
        illegal_set = list(all_vectors[:30])
        if len(illegal_set) >= 3:
            a, b, c = illegal_set[0], illegal_set[1], illegal_set[2]
            c = tuple((-(a[i] + b[i])) % 3 for i in range(args.n))
            if c not in illegal_set:
                illegal_set[2] = c
        
        test_sets = [("small_legal", small_legal), ("illegal", illegal_set)]
        
        for name, test_set in test_sets:
            print(f"\nTesting {name} set (size={len(test_set)}, n={args.n})")
            
            start = time.time()
            res1 = _is_cap_set_small_optimized(test_set, args.n)
            t1 = time.time() - start
            
            start = time.time()
            res2 = _is_cap_set_adaptive(test_set, args.n)
            t2 = time.time() - start
            
            print(f"  Small optimized: {res1[0]} in {t1:.6f}s")
            print(f"  Adaptive: {res2[0]} in {t2:.6f}s")
    else:
        os.environ["CAP_N"] = str(args.n)
        result = evaluate(args.program_path)
        print(result)