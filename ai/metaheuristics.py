"""
ai/metaheuristics.py
═══════════════════════════════════════════════════════════════════════════════
Metaheuristic Optimization Module
  1. Genetic Algorithm (GA) — for route/assignment optimization
  2. Particle Swarm Optimization (PSO) — for continuous cost optimization
 
Both methods solve the same problem and compare with classical results.
Problem: Minimize total transportation/assignment cost (combinatorial).
═══════════════════════════════════════════════════════════════════════════════
"""
 
import numpy as np
import pandas as pd
import random
import copy
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — GENETIC ALGORITHM (Assignment Problem)
# ─────────────────────────────────────────────────────────────────────────────
 
class GeneticAlgorithm:
    """
    Genetic Algorithm for Assignment Problem (minimize cost).
 
    Chromosome: permutation of [0..n-1]  (col assigned to each row)
    Fitness: total cost of the assignment (lower = better)
 
    Operators:
      - Selection:  Tournament selection
      - Crossover:  Order Crossover (OX)
      - Mutation:   Swap mutation
    """
 
    def __init__(self, cost_matrix, population_size=50,
                 max_generations=200, mutation_rate=0.1,
                 tournament_size=5, random_seed=42):
        self.cost      = np.array(cost_matrix, dtype=float)
        self.n         = self.cost.shape[0]
        self.pop_size  = population_size
        self.max_gen   = max_generations
        self.mut_rate  = mutation_rate
        self.tourn_k   = tournament_size
        random.seed(random_seed)
        np.random.seed(random_seed)
 
    # ── Fitness ───────────────────────────────────────────────────────────────
    def fitness(self, chromosome):
        """Total assignment cost for a permutation chromosome."""
        return sum(self.cost[i][chromosome[i]] for i in range(self.n))
 
    # ── Population ────────────────────────────────────────────────────────────
    def init_population(self):
        """Create initial population of random permutations."""
        return [list(np.random.permutation(self.n)) for _ in range(self.pop_size)]
 
    # ── Tournament Selection ──────────────────────────────────────────────────
    def tournament_select(self, population):
        """Select the best individual from a random tournament."""
        contestants = random.sample(population, self.tourn_k)
        return min(contestants, key=self.fitness)
 
    # ── Order Crossover (OX) ──────────────────────────────────────────────────
    def crossover(self, parent1, parent2):
        """
        Order Crossover: preserves relative order from parent2
        for positions not copied from parent1's central segment.
        """
        n = self.n
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b+1] = parent1[a:b+1]     # copy central segment from parent1
        remaining = [x for x in parent2 if x not in child]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = remaining[idx]
                idx += 1
        return child
 
    # ── Swap Mutation ─────────────────────────────────────────────────────────
    def mutate(self, chromosome):
        """Randomly swap two positions with probability mut_rate."""
        if random.random() < self.mut_rate:
            i, j = random.sample(range(self.n), 2)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome
 
    # ── Main GA Loop ──────────────────────────────────────────────────────────
    def solve(self):
        """
        Run the Genetic Algorithm.
 
        Returns
        -------
        best_chromosome : list  — optimal assignment permutation
        best_cost       : float — minimum total cost found
        history         : list  — best cost per generation (convergence curve)
        """
        population = self.init_population()
        history    = []
 
        best_chromosome = min(population, key=self.fitness)
        best_cost       = self.fitness(best_chromosome)
 
        for generation in range(self.max_gen):
            new_population = []
 
            # Elitism: carry best individual forward
            elite = min(population, key=self.fitness)
            new_population.append(copy.copy(elite))
 
            # Generate rest of new population
            while len(new_population) < self.pop_size:
                p1 = self.tournament_select(population)
                p2 = self.tournament_select(population)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
 
            population = new_population
 
            # Track best
            gen_best  = min(population, key=self.fitness)
            gen_cost  = self.fitness(gen_best)
            history.append(round(gen_cost, 4))
 
            if gen_cost < best_cost:
                best_cost       = gen_cost
                best_chromosome = copy.copy(gen_best)
 
        assignment = [(i, best_chromosome[i]) for i in range(self.n)]
        return assignment, round(best_cost, 4), history
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PARTICLE SWARM OPTIMIZATION (Continuous LP-style)
# ─────────────────────────────────────────────────────────────────────────────
 
class ParticleSwarmOptimizer:
    """
    PSO for continuous optimization problems.
    Applied to transportation cost minimization.
 
    Each particle represents a continuous allocation matrix;
    nearest feasible integer allocation extracted at evaluation.
 
    Parameters (standard PSO):
      w   = inertia weight       (controls momentum)
      c1  = cognitive coefficient (personal best attraction)
      c2  = social coefficient    (global best attraction)
    """
 
    def __init__(self, cost_matrix, supply, demand,
                 n_particles=40, max_iter=200,
                 w=0.7, c1=1.5, c2=1.5, random_seed=42):
        self.cost    = np.array(cost_matrix, dtype=float)
        self.supply  = np.array(supply, dtype=float)
        self.demand  = np.array(demand, dtype=float)
        self.m, self.n = self.cost.shape
        self.n_par   = n_particles
        self.max_iter= max_iter
        self.w       = w
        self.c1      = c1
        self.c2      = c2
        np.random.seed(random_seed)
 
    def _decode(self, particle):
        """
        Decode a continuous particle into a feasible allocation.
        Uses a greedy proportional approach.
        """
        m, n   = self.m, self.n
        alloc  = np.zeros((m, n))
        supply = self.supply.copy()
        demand = self.demand.copy()
 
        # Sort cells by (normalized particle value × cost) — prefer low cost + high priority
        cells  = [(i, j) for i in range(m) for j in range(n)]
        values = [(particle[i*n+j], i, j) for i,j in cells]
        values.sort(reverse=True)   # higher particle value = allocate first
 
        for _, i, j in values:
            amt = min(supply[i], demand[j])
            alloc[i][j] = amt
            supply[i]  -= amt
            demand[j]  -= amt
 
        return alloc
 
    def _fitness(self, particle):
        """Total transportation cost for a decoded particle."""
        alloc = self._decode(particle)
        return float(np.sum(self.cost * alloc))
 
    def solve(self):
        """
        Run PSO.
 
        Returns
        -------
        best_allocation : np.ndarray
        best_cost       : float
        history         : list of best cost per iteration
        """
        dim = self.m * self.n
 
        # Initialize particles and velocities
        particles = np.random.uniform(0, 1, (self.n_par, dim))
        velocities= np.random.uniform(-0.1, 0.1, (self.n_par, dim))
 
        personal_best      = particles.copy()
        personal_best_cost = np.array([self._fitness(p) for p in particles])
 
        global_best_idx  = np.argmin(personal_best_cost)
        global_best      = personal_best[global_best_idx].copy()
        global_best_cost = personal_best_cost[global_best_idx]
 
        history = []
 
        for iteration in range(self.max_iter):
            r1 = np.random.uniform(0, 1, (self.n_par, dim))
            r2 = np.random.uniform(0, 1, (self.n_par, dim))
 
            # Velocity update (classic PSO)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best - particles) +
                          self.c2 * r2 * (global_best   - particles))
 
            # Position update
            particles += velocities
            particles  = np.clip(particles, 0, 1)   # keep in [0,1]
 
            # Evaluate fitness
            costs = np.array([self._fitness(p) for p in particles])
 
            # Update personal bests
            improved = costs < personal_best_cost
            personal_best[improved]      = particles[improved]
            personal_best_cost[improved] = costs[improved]
 
            # Update global best
            best_idx = np.argmin(personal_best_cost)
            if personal_best_cost[best_idx] < global_best_cost:
                global_best      = personal_best[best_idx].copy()
                global_best_cost = personal_best_cost[best_idx]
 
            history.append(round(global_best_cost, 4))
 
        best_alloc = self._decode(global_best)
        return best_alloc, round(global_best_cost, 4), history
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MASTER SOLVER (public API)
# ─────────────────────────────────────────────────────────────────────────────
 
def solve_metaheuristics(cost_matrix, supply=None, demand=None,
                          problem_type="assignment",
                          ga_params=None, pso_params=None):
    """
    Run GA and/or PSO on the given problem and return comparison.
 
    Parameters
    ----------
    cost_matrix  : 2-D list/array
    supply, demand: lists (required for transportation PSO)
    problem_type : 'assignment' or 'transportation'
    ga_params    : dict of GA hyperparameters (optional)
    pso_params   : dict of PSO hyperparameters (optional)
 
    Returns
    -------
    dict with ga_result, pso_result, comparison_df, convergence data
    """
    cost = np.array(cost_matrix, dtype=float)
    ga_params  = ga_params  or {}
    pso_params = pso_params or {}
 
    results = {}
 
    # ── Genetic Algorithm ─────────────────────────────────────────────────────
    ga = GeneticAlgorithm(cost, **ga_params)
    ga_assign, ga_cost, ga_history = ga.solve()
 
    results["ga"] = {
        "assignment":  ga_assign,
        "cost":        ga_cost,
        "history":     ga_history,
        "method":      "Genetic Algorithm",
        "convergence": pd.DataFrame({"generation": range(len(ga_history)),
                                      "best_cost": ga_history}),
    }
 
    # ── PSO (for transportation) ───────────────────────────────────────────────
    if problem_type == "transportation" and supply is not None and demand is not None:
        pso = ParticleSwarmOptimizer(cost, supply, demand, **pso_params)
        pso_alloc, pso_cost, pso_history = pso.solve()
 
        results["pso"] = {
            "allocation": pso_alloc,
            "cost":       pso_cost,
            "history":    pso_history,
            "method":     "Particle Swarm Optimization",
            "convergence":pd.DataFrame({"iteration": range(len(pso_history)),
                                         "best_cost": pso_history}),
        }
    else:
        # PSO on assignment (treat as continuous relaxation)
        flat_supply = [1] * cost.shape[0]
        flat_demand = [1] * cost.shape[1]
        pso = ParticleSwarmOptimizer(cost, flat_supply, flat_demand, **pso_params)
        pso_alloc, pso_cost, pso_history = pso.solve()
        results["pso"] = {
            "allocation": pso_alloc,
            "cost":       pso_cost,
            "history":    pso_history,
            "method":     "Particle Swarm Optimization",
            "convergence":pd.DataFrame({"iteration": range(len(pso_history)),
                                         "best_cost": pso_history}),
        }
 
    # ── Comparison ────────────────────────────────────────────────────────────
    comparison = pd.DataFrame([
        {"Method": "Genetic Algorithm",          "Best Cost": ga_cost,
         "Final Generation": len(ga_history)},
        {"Method": "Particle Swarm Optimization","Best Cost": pso_cost,
         "Final Generation": len(pso_history)},
    ])
 
    return {
        "ga":         results["ga"],
        "pso":        results["pso"],
        "comparison": comparison,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    cost = [[9, 2, 7, 8],
            [6, 4, 3, 7],
            [5, 8, 1, 8],
            [7, 6, 9, 4]]
 
    result = solve_metaheuristics(cost, problem_type="assignment")
 
    print("═"*50)
    print("  METAHEURISTICS COMPARISON")
    print("═"*50)
    print(result["comparison"].to_string(index=False))
 
    print(f"\nGA  Best Assignment: {result['ga']['assignment']}")
    print(f"GA  Best Cost      : {result['ga']['cost']}")
    print(f"PSO Best Cost      : {result['pso']['cost']}")
    print(f"\n(Hungarian optimal is 13.0 — compare!)")
 