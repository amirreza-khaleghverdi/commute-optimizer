"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""
import random
from typing import Tuple, Dict, List
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover, random_state


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)
        self.tournament_size: int = self.params.get("tournament_size", 3)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        problem =self.problem
        self.history=[]
        cost_cache: Dict[State, float]={}

        def cost(s:State)->float:
            if s not in cost_cache:
                cost_cache[s] = problem.evaluate(s)
            return cost_cache[s]
        
        def fitness(s:State)->float:
            return 1.0 /(1.0+ cost(s))
        def tournament(pop: List[State]) -> State:
            contenders = random.sample(pop, min(self.tournament_size, len(pop)))
            return max(contenders, key=fitness)

        pop: List[State] = [problem.generate_initial_state()]
        while len(pop) < self.population_size:
            pop.append(random_state(problem))

        best = min(pop, key=cost)
        self.history.append(cost(best))

        n_elite = max(0, min(self.elitism, self.population_size))
        for _ in range(self.generations):
            ranked = sorted(pop, key=lambda s: (cost(s), s.path, s.departure_time))
            next_pop: List[State] = ranked[:n_elite]

            while len(next_pop) < self.population_size:
                p1, p2 = tournament(pop), tournament(pop)
                child = crossover(p1, p2, problem) if random.random() < self.crossover_rate else p1
                if random.random() < self.mutation_rate:
                    options = get_neighbors(child, problem)
                    if options:
                        child = random.choice(options)
                if not problem.is_valid_state(child):
                    child = p1
                next_pop.append(child)

            pop = next_pop
            gen_best = min(pop, key=cost)
            if cost(gen_best) < cost(best):
                best = gen_best
            self.history.append(cost(best))

        return best, cost(best)