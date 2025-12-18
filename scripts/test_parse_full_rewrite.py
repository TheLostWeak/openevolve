from openevolve.utils.code_utils import parse_full_rewrite

cases = {
    "fenced_lf": "```python\ndef f():\n    return 1\n```",
    "fenced_crlf": "```python\r\ndef g():\r\n    return 2\r\n```",
    "bare_fenced": "```\nprint(\"hello\")\n```",
    "no_fence": "def h():\n    return 3",
    "leading_ticks_only": "```python\n# comment\nprint(\"x\")\n",  # missing closing fence
    "embedded_text": "Some explanation\n```python\nimport math\nprint(math.pi)\n```\nMore text",
}

for name, text in cases.items():
    print(f"--- CASE: {name} ---")
    try:
        code = parse_full_rewrite(text, language='python')
        print("EXTRACTED:\n" + code)
        if '```' in code:
            print("RESULT: Contains backticks -> BAD")
        else:
            try:
                compile(code, '<string>', 'exec')
                print("RESULT: Compiles -> OK")
            except SyntaxError as se:
                print(f"RESULT: SyntaxError when compiling: {se}")
    except Exception as e:
        print(f"ERROR during parsing: {e}")

print('TEST COMPLETE')
