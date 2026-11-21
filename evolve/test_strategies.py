import pytest

from evolve.strategies import EvolutionStrategy, GeneticAlgorithm, run
from nn.mlp import MLP, param_count


def neg_sphere(g):
    return -sum((v - 1.0) ** 2 for v in g)


@pytest.mark.parametrize("cls,kw", [
    (GeneticAlgorithm, dict(pop_size=32, sigma=0.1)),
    (EvolutionStrategy, dict(pop_size=32, sigma=0.1, lr=0.3)),
])
def test_optimises_shifted_sphere(cls, kw):
    x0 = [0.0] * 8
    strat = cls(x0, seed=3, **kw)
    best, f, hist = run(strat, neg_sphere, generations=60)
    assert neg_sphere(x0) == -8.0
    assert f > -0.3
    assert hist[-1].best > hist[0].best


def test_ga_elitism_never_loses_the_best():
    strat = GeneticAlgorithm([0.0] * 5, seed=1, pop_size=16, elite=2, parents=4)
    _, _, hist = run(strat, neg_sphere, generations=25)
    bests = [h.best for h in hist]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))


@pytest.mark.parametrize("cls", [GeneticAlgorithm, EvolutionStrategy])
def test_runs_are_reproducible_from_seed(cls):
    def go(seed):
        return run(cls([0.0] * 4, seed=seed, pop_size=8), neg_sphere, generations=5)[1]
    assert go(11) == go(11)
    assert go(11) != go(12)


def test_es_antithetic_pairs():
    es = EvolutionStrategy([0.5, -0.5], seed=0, pop_size=6, sigma=0.2)
    pop = es.ask()
    for a, b in zip(pop[:3], pop[3:]):
        assert [x + y for x, y in zip(a, b)] == pytest.approx([1.0, -1.0])


def test_evolves_an_mlp_to_solve_xor():
    sizes = [2, 4, 1]
    data = [([0, 0], -1), ([0, 1], 1), ([1, 0], 1), ([1, 1], -1)]

    def fitness(g):
        net = MLP(sizes, g)
        return -sum((net(x)[0] - y) ** 2 for x, y in data)

    strat = GeneticAlgorithm([0.0] * param_count(sizes), seed=5, pop_size=40,
                             sigma=0.3, init_sigma=1.0)
    best, f, _ = run(strat, fitness, generations=80)
    net = MLP(sizes, best)
    assert all((net(x)[0] > 0) == (y > 0) for x, y in data), f


def test_on_generation_callback_and_validation():
    seen = []
    run(GeneticAlgorithm([0.0], seed=0, pop_size=4, elite=1, parents=2), neg_sphere,
        generations=3, on_generation=seen.append)
    assert [g.index for g in seen] == [0, 1, 2]
    with pytest.raises(ValueError):
        GeneticAlgorithm([0.0], seed=0, pop_size=4, elite=5, parents=2)
    with pytest.raises(ValueError):
        EvolutionStrategy([0.0], seed=0, pop_size=5)


@pytest.mark.parametrize("pop", [2, 4, 6, 32])
def test_ga_defaults_scale_with_population(pop):
    ga = GeneticAlgorithm([0.0], seed=0, pop_size=pop)
    assert 0 < ga.elite <= ga.parents <= pop
    ga.tell([0.0] * pop)
    assert len(ga.ask()) == pop
