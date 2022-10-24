import os
import sys
import pulp as pl

warm_start = False

def update_probabilities (sim, arrival_rates, serv_time, serv_time_cloud, init_time, offload_time):
    F = sim.functions
    C = sim.classes
    F_C = [(f,c) for f in F for c in f.get_invoking_classes()]

    invoked_functions = {c: [] for c in C}
    for f,c in F_C:
        invoked_functions[c].append(f)


    prob = pl.LpProblem("MyProblem", pl.LpMaximize)
    x = pl.LpVariable.dicts("Share", (F, C), 0, None, pl.LpContinuous)
    pE = pl.LpVariable.dicts("ProbExec", (F, C), 0, 1, pl.LpContinuous)
    pO = pl.LpVariable.dicts("ProbOffl", (F, C), 0, 1, pl.LpContinuous)
    pD = pl.LpVariable.dicts("ProbDrop", (F, C), 0, 1, pl.LpContinuous)
    pCold = pl.LpVariable.dicts("ProbCold", F, 0, 1, pl.LpContinuous)

    prob += (pl.lpSum([c.utility*arrival_rates[(f,c)]*(pE[f][c]+pO[f][c]) for f,c in F_C]) + 
             pl.lpSum([pCold[f] for f in F])
             , "objUtil")

    # Probability
    for f,c in F_C:
        prob += (pE[f][c] + pO[f][c] + pD[f][c] >= 1)
        prob += (pE[f][c] + pO[f][c] + pD[f][c] <= 1)

    # Memory
    prob += (pl.lpSum([f.memory*x[f][c] for f in F for c in C]) <= sim.edge.total_memory)

    # Share
    for f,c in F_C:
        prob += (pE[f][c]*arrival_rates[(f,c)]*serv_time[f] <= x[f][c])

    # Resp Time
    for f,c in F_C:
        prob += (pE[f][c]*serv_time[f] +  
                 pO[f][c]*(serv_time_cloud[f] + offload_time) +
                 pCold[f]*init_time
                 <= c.max_rt)

    # Min completion
    for c in C:
        if c.min_completion_percentage > 0.0:
            prob += (pl.lpSum([pD[f][c]*arrival_rates[(f,c)] for f in invoked_functions[c] if c in f.get_invoking_classes()])/sum([arrival_rates[(f,c)] for f in F if c in f.get_invoking_classes()])
                     <= 1 - c.min_completion_percentage)


    status = solve(prob)
    assert(status == "Optimal") # TODO

    print("Obj = ", pl.value(prob.objective))
    for f in F:
        print(f"Pcold[{f}]={pl.value(pCold[f])}")
    shares = {(f,c): pl.value(x[f][c]) for f,c in F_C}
    print(f"Shares: {shares}")

    probs = {(f,c): [pl.value(pE[f][c]),
                     pl.value(pO[f][c]),
                     max(0.0,1.0-pl.value(pE[f][c])-pl.value(pO[f][c]))] for f,c in F_C}
    print(probs)
    return probs

def solve (problem):
    global warm_start
    solver_name = os.environ.get("PULP_SOLVER", "GLPK_CMD")
    
    if not solver_name in ["CPLEX_CMD", "GUROBI_CMD", "PULP_CBC_CMD", "CBC_CMD", "CPLEX_PY", "GUROBI"] and warm_start:
        print("WARNING: warmStart not supported by solver {}".format(solver_name))
        warm_start = False

    if not warm_start:
        if solver_name == "GUROBI_CMD":
            solver = pl.getSolver(solver_name, gapRel=0.02, timeLimit=900)
        else:
            solver = pl.getSolver(solver_name, msg=False)
    else:
        solver = pl.getSolver(solver_name, warmStart=warm_start)

    problem.solve(solver)
    return pl.LpStatus[problem.status]


