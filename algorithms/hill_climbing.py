"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state

_EPS = 1e-9

class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)


    def _climb(self, start: State, best_state: State, best_cost: float) -> Tuple[State, float]:
        problem = self.problem
        current = start
        current_cost = problem.evaluate(current)
        visited = {current}
        sideways = 0

        if current_cost < best_cost:
            best_state, best_cost = current, current_cost
        self.history.append(best_cost)

        while True:
            scored = [(problem.evaluate(n), n) for n in get_neighbors(current, problem)]
            if not scored:
                break
            best_nb_cost = min(c for c, _ in scored)

            if best_nb_cost < current_cost - _EPS:
                ties = [n for c, n in scored if abs(c - best_nb_cost) <= _EPS]
                current = min(ties, key=lambda s: (s.path, s.departure_time)) 
                current_cost = best_nb_cost
                sideways = 0
            elif abs(best_nb_cost - current_cost) <= _EPS and sideways < self.max_sideways:
                ties = [n for c, n in scored
                        if abs(c - current_cost) <= _EPS and n not in visited]
                if not ties:
                    break
                current = random.choice(ties)
                sideways += 1
            else:
                break     

            visited.add(current)
            if current_cost < best_cost:
                best_state, best_cost = current, current_cost
            self.history.append(best_cost)

        return best_state, best_cost


    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        self.history = []
        best_state = self.problem.generate_initial_state()
        best_cost = self.problem.evaluate(best_state)

        start = best_state
        for run in range(self.max_restarts + 1):
            best_state, best_cost = self._climb(start, best_state, best_cost)
            start = random_state(self.problem)
        return best_state, best_cost
