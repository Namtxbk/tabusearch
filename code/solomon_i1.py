# Backward-compat wrapper
from construction import build_initial_solution

def solomon_i1_construction(inst, **kwargs):
    return build_initial_solution(inst)

def multi_start_i1(inst, n_starts=1):
    return build_initial_solution(inst)
