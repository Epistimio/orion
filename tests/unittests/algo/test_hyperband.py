#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.hyperband`."""

import hashlib

import numpy as np
import pytest

from orion.algo.hyperband import Bracket, compute_budgets, Hyperband
from orion.algo.space import Fidelity, Real, Space


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real('lr', 'uniform', 0, 1))
    space.register(Fidelity('epoch', 1, 9, 3))
    return space


@pytest.fixture
def budgets():
    """Return a configuration for a bracket."""
    return [(27, 1), (9, 3), (3, 9)]


@pytest.fixture
def hyperband(space):
    """Return an instance of Hyperband."""
    return Hyperband(space)


@pytest.fixture
def hyperband_repeat_one(space):
    """Return an instance of Hyperband."""
    return Hyperband(space, repeat=1)


@pytest.fixture
def bracket(budgets, hyperband):
    """Return a `Bracket` instance configured with `b_config`."""
    return Bracket(hyperband, budgets)


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    points = np.linspace(0, 26, 27)
    return dict(
        n_trials=27,
        resources=1,
        results={hashlib.md5(str([point]).encode('utf-8')).hexdigest(): (point, (1, point))
                 for point in points})


@pytest.fixture
def rung_1(rung_0):
    """Create fake points and objectives for rung 1."""
    values = map(lambda v: (v[0], (3, v[0])), list(sorted(rung_0['results'].values()))[:9])
    return dict(
        n_trials=9,
        resources=3,
        results={hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value
                 for value in values})


@pytest.fixture
def rung_2(rung_1):
    """Create fake points and objectives for rung 1."""
    values = map(lambda v: (v[0], (9, v[0])), list(sorted(rung_1['results'].values()))[:3])
    return dict(
        n_trials=3,
        resources=9,
        results={hashlib.md5(str([value[0]]).encode('utf-8')).hexdigest(): value
                 for value in values})


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(16, 4) == [[(48, 1), (12, 4), (3, 16)], [(12, 4), (3, 16)], [(3, 16)]]
    assert compute_budgets(16, 5) == [[(10, 4), (2, 16)], [(2, 16)]]


class TestBracket():
    """Tests for the `Bracket` class."""

    def test_rungs_creation(self, bracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0] == dict(n_trials=27, resources=1, results=dict())
        assert bracket.rungs[1] == dict(n_trials=9, resources=3, results=dict())
        assert bracket.rungs[2] == dict(n_trials=3, resources=9, results=dict())

    def test_register(self, hyperband, bracket):
        """Check that a point is correctly registered inside a bracket."""
        bracket.hyperband = hyperband
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        bracket.register(point, 0.0)

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]['results']
        assert (0.0, point) == bracket.rungs[0]['results'][point_hash]

    def test_bad_register(self, hyperband, bracket):
        """Check that a non-valid point is not registered."""
        bracket.hyperband = hyperband

        with pytest.raises(IndexError) as ex:
            bracket.register((55, 0.0), 0.0)

        assert 'Bad fidelity level 55' in str(ex.value)

    def test_candidate_promotion(self, hyperband, bracket, rung_0):
        """Test that correct point is promoted."""
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        points = bracket.get_candidates(0)

        assert points[0] == (1, 0.0)

    def test_promotion_with_rung_1_hit(self, hyperband, bracket, rung_0):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0
        bracket.rungs[1]['results'][point_hash] = (0.0, point)

        points = bracket.get_candidates(0)

        assert points[0] == (1, 1)

    def test_no_promotion_when_rung_full(self, hyperband, bracket, rung_0, rung_1):
        """Test that get_candidate returns `None` if rung 1 is full."""
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1

        points = bracket.get_candidates(0)

        assert points == []

    def test_no_promotion_if_not_completed(self, hyperband, bracket, rung_0):
        """Test the get_candidate return None if trials are not completed."""
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0
        rung = bracket.rungs[0]['results']

        # points = bracket.get_candidates(0)

        for p_id in rung.keys():
            rung[p_id] = (None, rung[p_id][1])

        with pytest.raises(AssertionError):
            bracket.get_candidates(0)

    def test_is_done(self, bracket, rung_0):
        """Test that the `is_done` property works."""
        assert not bracket.is_done

        # Actual value of the point is not important here
        bracket.rungs[2]['results'] = {'1': (1, 0.0), '2': (1, 0.0), '3': (1, 0.0)}

        assert bracket.is_done

    def test_update_rungs_return_candidate(self, hyperband, bracket, rung_1):
        """Check if a valid modified candidate is returned by update_rungs."""
        bracket.hyperband = hyperband
        bracket.rungs[1] = rung_1
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        candidates = bracket.promote()

        assert point_hash in bracket.rungs[1]['results']
        assert bracket.rungs[1]['results'][point_hash] == (0.0, (3, 0.0))
        assert candidates[0][0] == 9

    def test_update_rungs_return_no_candidate(self, hyperband, bracket, rung_1):
        """Check if no candidate is returned by update_rungs."""
        bracket.hyperband = hyperband

        candidate = bracket.promote()

        assert candidate is None

    def test_repr(self, bracket, rung_0, rung_1, rung_2):
        """Test the string representation of Bracket"""
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        assert str(bracket) == 'Bracket([1, 3, 9])'


class TestHyperband():
    """Tests for the algo Hyperband."""

    def test_register(self, hyperband, bracket, rung_0, rung_1):
        """Check that a point is registered inside the bracket."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs = [rung_0, rung_1]
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode('utf-8')).hexdigest()

        hyperband.observe([point], [{'objective': 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]['results']
        assert (0.0, point) == bracket.rungs[0]['results'][point_hash]

    def test_register_bracket_multi_fidelity(self, space):
        """Check that a point is registered inside the same bracket for diff fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 1
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        hyperband.observe([point], [{'objective': 0.0}])

        bracket = hyperband.brackets[0]

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]['results']
        assert (0.0, point) == bracket.rungs[0]['results'][point_hash]

        fidelity = 3
        point = [fidelity, value]
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        hyperband.observe([point], [{'objective': 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[1]['results']
        assert (0.0, point) != bracket.rungs[0]['results'][point_hash]
        assert (0.0, point) == bracket.rungs[1]['results'][point_hash]

    def test_register_next_bracket(self, space):
        """Check that a point is registered inside the good bracket when higher fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 3
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        hyperband.observe([point], [{'objective': 0.0}])

        assert sum(len(rung['results']) for rung in hyperband.brackets[0].rungs) == 0
        assert sum(len(rung['results']) for rung in hyperband.brackets[1].rungs) == 1
        assert sum(len(rung['results']) for rung in hyperband.brackets[2].rungs) == 0
        assert point_hash in hyperband.brackets[1].rungs[0]['results']
        assert (0.0, point) == hyperband.brackets[1].rungs[0]['results'][point_hash]

        value = 51
        fidelity = 9
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode('utf-8')).hexdigest()

        hyperband.observe([point], [{'objective': 0.0}])

        assert sum(len(rung['results']) for rung in hyperband.brackets[0].rungs) == 0
        assert sum(len(rung['results']) for rung in hyperband.brackets[1].rungs) == 1
        assert sum(len(rung['results']) for rung in hyperband.brackets[2].rungs) == 1
        assert point_hash in hyperband.brackets[2].rungs[0]['results']
        assert (0.0, point) == hyperband.brackets[2].rungs[0]['results'][point_hash]

    def test_register_invalid_fidelity(self, space):
        """Check that a point cannot registered if fidelity is invalid."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 2
        point = (fidelity, value)

        with pytest.raises(ValueError) as ex:
            hyperband.observe([point], [{'objective': 0.0}])

        assert 'No bracket found for point' in str(ex.value)

    def test_register_corrupted_db(self, caplog, space):
        """Check that a point cannot registered if passed in order diff than fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 3
        point = (fidelity, value)

        hyperband.observe([point], [{'objective': 0.0}])
        assert 'Point registered to wrong bracket' not in caplog.text

        fidelity = 1
        point = [fidelity, value]

        caplog.clear()
        hyperband.observe([point], [{'objective': 0.0}])
        assert 'Point registered to wrong bracket' in caplog.text

    def test_get_id(self, space):
        """Test valid id of points"""
        hyperband = Hyperband(space)

        assert hyperband.get_id(['whatever', 1]) == hyperband.get_id(['is here', 1])
        assert hyperband.get_id(['whatever', 1]) != hyperband.get_id(['is here', 2])

    def test_get_id_multidim(self):
        """Test valid id for points with dim of shape > 1"""
        space = Space()
        space.register(Fidelity('epoch', 1, 9, 3))
        space.register(Real('lr', 'uniform', 0, 1, shape=2))

        hyperband = Hyperband(space)

        assert hyperband.get_id(['whatever', [1, 1]]) == hyperband.get_id(['is here', [1, 1]])
        assert hyperband.get_id(['whatever', [1, 1]]) != hyperband.get_id(['is here', [2, 2]])

    def test_suggest_new(self, monkeypatch, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that a new point is sampled."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        def sample(num=1, seed=None):
            return [('fidelity', i) for i in range(num)]

        monkeypatch.setattr(hyperband.space, 'sample', sample)

        points = hyperband.suggest()

        assert points[0] == (1.0, 0)
        assert points[1] == (1.0, 1)

    def test_suggest_duplicates(self, monkeypatch, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that sampling collisions are handled."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        duplicate_point = ('fidelity', 0.0)
        new_point = ('fidelity', 0.5)

        duplicate_id = hashlib.md5(str([duplicate_point]).encode('utf-8')).hexdigest()
        bracket.rungs[0]['results'] = {duplicate_id: (0.0, duplicate_point)}

        hyperband.trial_info[hyperband.get_id(duplicate_point)] = bracket

        points = [duplicate_point, new_point]

        def sample(num=1, seed=None):
            return points + [('fidelity', i) for i in range(num - 2)]

        monkeypatch.setattr(hyperband.space, 'sample', sample)

        assert hyperband.suggest()[0][1] == new_point[1]

    def test_suggest_inf_duplicates(self, monkeypatch, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that sampling inf collisions raises runtime error."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        zhe_point = ('fidelity', 0.0)
        hyperband.trial_info[hyperband.get_id(zhe_point)] = bracket

        def sample(num=1, seed=None):
            return [zhe_point] * num

        monkeypatch.setattr(hyperband.space, 'sample', sample)

        with pytest.raises(RuntimeError) as exc:
            hyperband.suggest()

        assert 'Hyperband keeps sampling already existing points.' in str(exc.value)

    def test_suggest_promote(self, hyperband, bracket, rung_0):
        """Test that correct point is promoted and returned."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        points = hyperband.suggest()

        assert points == [(3, i) for i in range(9)]

    def test_is_filled(self, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that Hyperband bracket detects when rung is filled."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        rung = bracket.rungs[0]['results']
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert not bracket.is_filled
        assert not bracket.has_rung_filled(0)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled
        assert bracket.has_rung_filled(0)
        assert not bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        bracket.rungs[1] = rung_1

        rung = bracket.rungs[1]['results']
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert not bracket.has_rung_filled(1)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        bracket.rungs[2] = rung_2

        rung = bracket.rungs[2]['results']
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert bracket.has_rung_filled(2)

    def test_is_ready(self, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that Hyperband bracket detects when rung is ready."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        rung = bracket.rungs[0]['results']
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()
        assert not bracket.is_ready(0)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()
        assert bracket.is_ready(0)
        assert not bracket.is_ready(1)
        assert not bracket.is_ready(2)

        bracket.rungs[1] = rung_1

        rung = bracket.rungs[1]['results']
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert not bracket.is_ready(1)
        assert not bracket.is_ready(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert not bracket.is_ready(2)

        bracket.rungs[2] = rung_2

        rung = bracket.rungs[2]['results']
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert not bracket.is_ready(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert bracket.is_ready(2)

    def test_suggest_opt_out(self, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that Hyperband opts out when rungs are not ready."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        bracket.rungs[0] = rung_0

        trial_id = next(iter(rung_0['results'].keys()))
        objective, point = rung_0['results'][trial_id]
        rung_0['results'][trial_id] = (None, point)

        points = hyperband.suggest()

        assert points is None

    def test_seed_rng(self, hyperband):
        """Test that algo is seeded properly"""
        hyperband.seed_rng(1)
        a = hyperband.suggest(1)
        # Hyperband will always return the full first rung
        assert np.allclose(a, hyperband.suggest(1))

        hyperband.seed_rng(2)
        assert not np.allclose(a, hyperband.suggest(1))

    def test_set_state(self, hyperband):
        """Test that state is reset properly"""
        hyperband.seed_rng(1)
        state = hyperband.state_dict
        points = hyperband.suggest(1)
        # Hyperband will always return the full first rung
        assert np.allclose(points, hyperband.suggest(1))

        hyperband.seed_rng(2)
        assert not np.allclose(points, hyperband.suggest(1))

        hyperband.set_state(state)
        assert np.allclose(points, hyperband.suggest(1))

    def test_full_process_repeat_one(self, monkeypatch, hyperband_repeat_one):
        """Test Hyperband full process with execution repeat one time."""
        points = [('fidelity', i) for i in range(400)]

        def sample(num=1, seed=None):
            return points[:num]

        monkeypatch.setattr(hyperband_repeat_one.space, 'sample', sample)

        # Fill all brackets' first rung

        for i in range(3):
            point = hyperband_repeat_one.suggest()[0]
            assert point == (9, i)
            hyperband_repeat_one.observe([point], [{'objective': None}])

        for i in range(9):
            point = hyperband_repeat_one.suggest()[0]
            assert point == (3, i + 3)
            hyperband_repeat_one.observe([point], [{'objective': None}])

        for i in range(27):
            point = hyperband_repeat_one.suggest()[0]
            assert point == (1, i + 3 + 9)
            hyperband_repeat_one.observe([point], [{'objective': None}])

        assert hyperband_repeat_one.brackets[0].has_rung_filled(0)
        assert not hyperband_repeat_one.brackets[0].is_ready()
        assert hyperband_repeat_one.suggest() is None
        assert hyperband_repeat_one.suggest() is None

        # Observe first bracket first rung

        for i in range(27):
            hyperband_repeat_one.observe([(1, i + 3 + 9)], [{'objective': 40 - i}])

        assert hyperband_repeat_one.brackets[0].is_ready()
        assert not hyperband_repeat_one.brackets[1].is_ready()
        assert not hyperband_repeat_one.brackets[2].is_ready()

        # Promote first bracket first rung

        for i in range(9):
            point = hyperband_repeat_one.suggest()[0]
            assert point == (3, 40 - 2 - i)
            hyperband_repeat_one.observe([point], [{'objective': None}])

        assert hyperband_repeat_one.brackets[0].has_rung_filled(1)
        assert not hyperband_repeat_one.brackets[0].is_ready()
        assert not hyperband_repeat_one.brackets[1].is_ready()
        assert not hyperband_repeat_one.brackets[2].is_ready()

        # Observe first bracket second rung

        for i in range(9):
            hyperband_repeat_one.observe([(3, 40 - 2 - i)], [{'objective': 20 - i}])

        assert hyperband_repeat_one.brackets[0].is_ready()
        assert not hyperband_repeat_one.brackets[1].is_ready()
        assert not hyperband_repeat_one.brackets[2].is_ready()

        # Observe second bracket first rung

        for i in range(9):
            hyperband_repeat_one.observe([(3, i + 3)], [{'objective': 20 - i}])

        assert hyperband_repeat_one.brackets[0].is_ready()
        assert hyperband_repeat_one.brackets[1].is_ready()
        assert not hyperband_repeat_one.brackets[2].is_ready()

        # Promote second bracket first rung

        for i in range(3):
            point = hyperband_repeat_one.suggest()[0]
            assert point == (9, 9 + 2 - i)
            hyperband_repeat_one.observe([point], [{'objective': None}])

        assert hyperband_repeat_one.brackets[0].is_ready()
        assert hyperband_repeat_one.brackets[1].has_rung_filled(1)
        assert not hyperband_repeat_one.brackets[1].is_ready()
        assert not hyperband_repeat_one.brackets[2].is_ready()

        # Observe third bracket first rung

        for i in range(3):
            hyperband_repeat_one.observe([(9, i)], [{'objective': 3 - i}])

        assert not hyperband_repeat_one.brackets[0].is_ready(2)
        assert not hyperband_repeat_one.brackets[1].is_ready(1)
        assert hyperband_repeat_one.brackets[2].is_ready(0)
        assert hyperband_repeat_one.brackets[2].is_done

        # Observe second bracket second rung

        for i in range(3):
            hyperband_repeat_one.observe([(9, 9 + 2 - i)], [{'objective': 5 - i}])

        assert not hyperband_repeat_one.brackets[0].is_ready(2)
        assert hyperband_repeat_one.brackets[1].is_ready(1)
        assert hyperband_repeat_one.brackets[1].is_done

        # Observe first bracket third rung

        for i in range(3):
            point = hyperband_repeat_one.suggest()[0]
            assert point == (9, 30 + i)
            hyperband_repeat_one.observe([(9, 30 + i)], [{'objective': 7 - i}])

        assert hyperband_repeat_one.is_done
        assert hyperband_repeat_one.brackets[0].is_ready(2)
        assert hyperband_repeat_one.brackets[0].is_done

        points = [('fidelity', i + 400) for i in range(400)]

        def samplemore(num=1, seed=None):
            return points[:num]

        monkeypatch.setattr(hyperband_repeat_one.space, 'sample', samplemore)
        assert hyperband_repeat_one.suggest() is None
        assert hyperband_repeat_one.suggest() is None

    def test_full_process(self, monkeypatch, hyperband):
        """Test Hyperband full process."""
        points = [('fidelity', i) for i in range(400)]

        def sample(num=1, seed=None):
            return points[:num]

        monkeypatch.setattr(hyperband.space, 'sample', sample)

        # Fill all brackets' first rung

        for i in range(3):
            point = hyperband.suggest()[0]
            assert point == (9, i)
            hyperband.observe([point], [{'objective': None}])

        for i in range(9):
            point = hyperband.suggest()[0]
            assert point == (3, i + 3)
            hyperband.observe([point], [{'objective': None}])

        for i in range(27):
            point = hyperband.suggest()[0]
            assert point == (1, i + 3 + 9)
            hyperband.observe([point], [{'objective': None}])

        assert hyperband.brackets[0].has_rung_filled(0)
        assert not hyperband.brackets[0].is_ready()
        assert hyperband.suggest() is None
        assert hyperband.suggest() is None

        # Observe first bracket first rung

        for i in range(27):
            hyperband.observe([(1, i + 3 + 9)], [{'objective': 40 - i}])

        assert hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote first bracket first rung

        for i in range(9):
            point = hyperband.suggest()[0]
            assert point == (3, 40 - 2 - i)
            hyperband.observe([point], [{'objective': None}])

        assert hyperband.brackets[0].has_rung_filled(1)
        assert not hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe first bracket second rung

        for i in range(9):
            hyperband.observe([(3, 40 - 2 - i)], [{'objective': 20 - i}])

        assert hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe second bracket first rung

        for i in range(9):
            hyperband.observe([(3, i + 3)], [{'objective': 20 - i}])

        assert hyperband.brackets[0].is_ready()
        assert hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote second bracket first rung

        for i in range(3):
            point = hyperband.suggest()[0]
            assert point == (9, 9 + 2 - i)
            hyperband.observe([point], [{'objective': None}])

        assert hyperband.brackets[0].is_ready()
        assert hyperband.brackets[1].has_rung_filled(1)
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe third bracket first rung

        for i in range(3):
            hyperband.observe([(9, i)], [{'objective': 3 - i}])

        assert not hyperband.brackets[0].is_ready(2)
        assert not hyperband.brackets[1].is_ready(1)
        assert hyperband.brackets[2].is_ready(0)
        assert hyperband.brackets[2].is_done

        # Observe second bracket second rung

        for i in range(3):
            hyperband.observe([(9, 9 + 2 - i)], [{'objective': 5 - i}])

        assert not hyperband.brackets[0].is_ready(2)
        assert hyperband.brackets[1].is_ready(1)
        assert hyperband.brackets[1].is_done

        # Observe first bracket third rung

        for i in range(3):
            point = hyperband.suggest()[0]
            assert point == (9, 30 + i)
            hyperband.observe([(9, 30 + i)], [{'objective': 7 - i}])

        assert not hyperband.is_done
        assert not hyperband.brackets[0].is_ready(2)
        assert not hyperband.brackets[0].is_done

        points = [('fidelity', i + 400) for i in range(400)]

        def samplemore(num=1, seed=None):
            return points[:num]

        monkeypatch.setattr(hyperband.space, 'sample', samplemore)
        assert hyperband.suggest() is not None
        assert hyperband.suggest() is not None
