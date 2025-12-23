import traceback
try:
    from examples.cap_set_example import evaluator
    print('imported evaluator OK')
except Exception as e:
    traceback.print_exc()
    print('FAILED IMPORT')
    raise
