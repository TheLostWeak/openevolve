from openevolve.config import load_config
from openevolve.prompt.sampler import PromptSampler
cfg = load_config('examples/self_convolution_50/config.yaml')
ps = PromptSampler(cfg.prompt)
user = ps.build_prompt(current_program='def f():\n    return 0', parent_program='def f():\n    return 0', program_metrics={'combined_score':0.5}, previous_programs=[], top_programs=[], inspirations=[], language='python', evolution_round=1, diff_based_evolution=False)
print('---USER PROMPT START---')
print(user['user'])
print('---USER PROMPT END---')
