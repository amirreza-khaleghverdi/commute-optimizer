"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        problem=self.problem
        K=self.beam_width
        self.history=[]
        cost_cache: dict[State,float]= {}

        def cost(s: State)-> float:
            if s not in cost_cache:
                cost_cache[s]=problem.evaluate(s)
                return cost_cache[s]
        def rank(s: State):
            return (cost(s), s.path, s.departure_time)


        beam = {problem.generate_initial_state()}
        tries = 0
        while len(beam) < K and tries < 20* K:
            beam.add(random_state(problem))
            tries +=1
        beam=sorted(beam, key=rank)[:K]
        self.history.append(cost(beam[0]))

        for empty in range(self.max_iterations):
            pool=set(beam)
            for s in beam:
                pool.update(get_neighbors(s, problem))
            new_beam = sorted(pool, key=rank)[:K]
            self.history.append(cost(new_beam[0]))
            if set(new_beam)==set(beam):
                break
            beam = new_beam
        best= beam[0]
        return best,cost(best)
