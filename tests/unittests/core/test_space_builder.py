#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`metaopt.core.io.space_builder`."""

import copy
import sys

import pytest
from scipy.stats import distributions as dists

from metaopt.algo.space import (Categorical, Integer, Real)
from metaopt.core.io.space_builder import (DimensionBuilder, SpaceBuilder)
from metaopt.core.worker.trial import Trial


@pytest.fixture(scope='module')
def dimbuilder():
    """Return a `DimensionBuilder` instance."""
    return DimensionBuilder()


@pytest.fixture(scope='module')
def spacebuilder():
    """Return a `SpaceBuilder` instance."""
    return SpaceBuilder()


class TestDimensionBuilder(object):
    """Ways of Dimensions builder."""

    def test_build_loguniform(self, dimbuilder):
        """Check that loguniform is built into reciprocal correctly."""
        dim = dimbuilder.build('yolo', 'loguniform(0.001, 10)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'reciprocal'
        assert 3.3 in dim and 11.1 not in dim
        assert isinstance(dim.prior, dists.reciprocal_gen)

        dim = dimbuilder.build('yolo2', 'loguniform(1, 1000, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'reciprocal'
        assert 3 in dim and 0 not in dim and 3.3 not in dim
        assert isinstance(dim.prior, dists.reciprocal_gen)

    def test_eval_nonono(self, dimbuilder):
        """Make malevolent/naive eval access more difficult. I think."""
        with pytest.raises(RuntimeError):
            dimbuilder.build('la', "__class__")

    def test_build_a_good_real(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build('yolo2', 'alpha(0.9, low=0, high=10, shape=2)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'alpha'
        assert 3.3 not in dim and (3.3, 11.1) not in dim and (3.3, 6) in dim
        assert isinstance(dim.prior, dists.alpha_gen)

    def test_build_a_good_integer(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build('yolo3', 'poisson(5)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo3'
        assert dim._prior_name == 'poisson'
        assert isinstance(dim.prior, dists.poisson_gen)

    def test_build_a_good_real_discrete(self, dimbuilder):
        """Check that non registered names are good, as long as they are in
        `scipy.stats.distributions`.
        """
        dim = dimbuilder.build('yolo3', 'alpha(1.1, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo3'
        assert dim._prior_name == 'alpha'
        assert isinstance(dim.prior, dists.alpha_gen)

    def test_build_fails_because_of_name(self, dimbuilder):
        """Build fails because distribution name is not supported..."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo3', 'lalala(1.1, discrete=True)')
        assert 'Parameter' in str(exc.value)
        assert 'supported' in str(exc.value)

    def test_build_fails_because_of_unexpected_args(self, dimbuilder):
        """Build fails because argument is not supported..."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo3', 'alpha(1.1, whatisthis=5, discrete=True)')
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'unexpected' in str(exc.value.__cause__)

    def test_build_fails_because_of_ValueError_on_run(self, dimbuilder):
        """Build fails because ValueError happens on init."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', 'alpha(0.9, low=4, high=6, shape=2)')
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'Improbable bounds' in str(exc.value.__cause__)

    def test_build_fails_because_of_ValueError_on_init(self, dimbuilder):
        """Build fails because ValueError happens on init."""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', 'alpha(0.9, low=4, high=10, size=2)')
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'size' in str(exc.value.__cause__)

    def test_build_gaussian(self, dimbuilder):
        """Check that gaussian/normal/norm is built into reciprocal correctly."""
        dim = dimbuilder.build('yolo', 'gaussian(3, 5)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

        dim = dimbuilder.build('yolo2', 'gaussian(1, 0.5, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

    def test_build_normal(self, dimbuilder):
        """Check that gaussian/normal/norm is built into reciprocal correctly."""
        dim = dimbuilder.build('yolo', 'normal(0.001, 10)')
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

        dim = dimbuilder.build('yolo2', 'normal(1, 0.5, discrete=True)')
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'norm'
        assert isinstance(dim.prior, dists.norm_gen)

    def test_build_enum(self, dimbuilder):
        """Create correctly a `Categorical` dimension."""
        dim = dimbuilder.build('yolo', "enum('adfa', 1, 0.3, 'asaga', shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "enum(['adfa', 1])")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo2', "enum({'adfa': 0.1, 3: 0.4, 5: 0.5})")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', "enum({'adfa': 0.1, 3: 0.4})")
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'sum' in str(exc.value.__cause__)

    def test_build_fails_because_empty_args(self, dimbuilder):
        """What happens if somebody 'forgets' stuff?"""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo', "enum()")
        assert 'Parameter' in str(exc.value)
        assert 'categories' in str(exc.value)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build('what', "alpha()")
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'positional' in str(exc.value.__cause__)

    def test_build_fails_because_troll(self, dimbuilder):
        """What happens if somebody does not fit regular expression expected?"""
        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo', "lalalala")
        assert 'Parameter' in str(exc.value)
        assert 'form for prior' in str(exc.value)

    def test_build_random(self, dimbuilder):
        """Things built by random keyword."""
        dim = dimbuilder.build('yolo', "random('adfa', 1, 0.3, 'asaga', shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(-3, 1, 8, 50, shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(['adfa', 1])")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(('adfa', 1))")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)
        assert dim.interval() == (('adfa', 1),)

        dim = dimbuilder.build('yolo2', "random({'adfa': 0.1, 3: 0.4, 5: 0.5})")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo2'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        with pytest.raises(TypeError) as exc:
            dimbuilder.build('yolo2', "random({'adfa': 0.1, 3: 0.4})")
        assert 'Parameter' in str(exc.value)
        if sys.version_info[0] == 3:
            assert 'sum' in str(exc.value.__cause__)

        dim = dimbuilder.build('yolo', "random('adfa', 1, shape=4)")
        assert isinstance(dim, Categorical)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'Distribution'
        assert isinstance(dim.prior, dists.rv_discrete)

        dim = dimbuilder.build('yolo', "random(-3, 5.4, shape=4)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (-3, 5.4)

        dim = dimbuilder.build('yolo', "random(4)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (4.0, 5.0)

        dim = dimbuilder.build('yolo', "random()")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (0.0, 1.0)

        dim = dimbuilder.build('yolo', "random(-3, 5, shape=4, discrete=True)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (-3, 5)

        # These two below, do not make any sense for integers.. meh
        dim = dimbuilder.build('yolo', "random(4, discrete=True)")
        assert isinstance(dim, Real)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (4.0, 5.0)

        dim = dimbuilder.build('yolo', "random(discrete=True)")
        assert isinstance(dim, Integer)
        assert dim.name == 'yolo'
        assert dim._prior_name == 'uniform'
        assert isinstance(dim.prior, dists.uniform_gen)
        assert dim.interval() == (0.0, 1.0)


class TestSpaceBuilder(object):
    """Check whether space definition from various input format is successful."""

    def test_build_from_yaml_config(self, spacebuilder, yaml_sample_path):
        """Build space from a yaml config only."""
        space = spacebuilder.build_from([yaml_sample_path])
        print(space)
        assert spacebuilder.userconfig == yaml_sample_path
        assert len(space) == 6
        assert '/layers/1/width' in space
        assert '/layers/1/type' in space
        assert '/layers/2/type' in space
        assert '/training/lr0' in space
        assert '/training/mbs' in space
        assert '/something-same' in space

    def test_build_from_json_config(self, spacebuilder, json_sample_path):
        """Build space from a json config only."""
        space = spacebuilder.build_from([json_sample_path])
        print(space)
        assert spacebuilder.userconfig == json_sample_path
        assert len(space) == 6
        assert '/layers/1/width' in space
        assert '/layers/1/type' in space
        assert '/layers/2/type' in space
        assert '/training/lr0' in space
        assert '/training/mbs' in space
        assert '/something-same' in space

    def test_parse_equivalency(self, spacebuilder, yaml_sample_path, json_sample_path):
        """Templates found from json and yaml are the same."""
        spacebuilder.build_from([yaml_sample_path])
        dict_from_yaml = copy.deepcopy(spacebuilder.userconfig_tmpl)
        spacebuilder.build_from([json_sample_path])
        dict_from_json = copy.deepcopy(spacebuilder.userconfig_tmpl)
        assert dict_from_json == dict_from_yaml

    def test_build_from_args_only(self, spacebuilder):
        """Build a space using only args."""
        cmd_args = ["--seed=555",
                    "-yolo~random(-3, 1)",
                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                    "--arch2~random({'lala': 0.2, 'yolo': 0.8})"]
        space = spacebuilder.build_from(cmd_args)
        print(space)
        assert spacebuilder.userconfig is None
        assert len(space) == 2
        assert '/yolo' in space
        assert '/arch2' in space
        assert len(spacebuilder.userargs_tmpl) == 3
        assert spacebuilder.userargs_tmpl[None] == ["--seed=555",
                                                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})"]
        assert spacebuilder.userargs_tmpl['/yolo'] == '-yolo='
        assert spacebuilder.userargs_tmpl['/arch2'] == '--arch2='

    def test_build_from_args_and_config1(self, spacebuilder, yaml_sample_path):
        """Build a space using both args and config file!"""
        cmd_args = [yaml_sample_path, "--seed=555",
                    "-yolo~random(-3, 1)",
                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                    "--arch2~random({'lala': 0.2, 'yolo': 0.8})"]
        space = spacebuilder.build_from(cmd_args)
        print(space)
        assert len(space) == 8
        assert '/yolo' in space
        assert '/arch2' in space
        assert '/layers/1/width' in space
        assert '/layers/1/type' in space
        assert '/layers/2/type' in space
        assert '/training/lr0' in space
        assert '/training/mbs' in space
        assert '/something-same' in space
        assert len(spacebuilder.userargs_tmpl) == 3
        assert spacebuilder.userargs_tmpl[None] == ["--seed=555",
                                                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})"]
        assert spacebuilder.userargs_tmpl['/yolo'] == '-yolo='
        assert spacebuilder.userargs_tmpl['/arch2'] == '--arch2='

    def test_build_from_args_and_config2(self, spacebuilder, yaml_sample_path):
        """Build a space using both args and config file!"""
        cmd_args = ["--seed=555",
                    "-yolo~random(-3, 1)",
                    "--config=" + yaml_sample_path,
                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                    "--arch2~random({'lala': 0.2, 'yolo': 0.8})"]
        space = spacebuilder.build_from(cmd_args)
        print(space)
        assert len(space) == 8
        assert '/yolo' in space
        assert '/arch2' in space
        assert '/layers/1/width' in space
        assert '/layers/1/type' in space
        assert '/layers/2/type' in space
        assert '/training/lr0' in space
        assert '/training/mbs' in space
        assert '/something-same' in space
        assert len(spacebuilder.userargs_tmpl) == 3
        assert spacebuilder.userargs_tmpl[None] == ["--seed=555",
                                                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})"]
        assert spacebuilder.userargs_tmpl['/yolo'] == '-yolo='
        assert spacebuilder.userargs_tmpl['/arch2'] == '--arch2='

    def test_build_finds_conflict(self, spacebuilder, yaml_sample_path):
        """Conflicting definition in args and config~ raise an error!"""
        cmd_args = ["--seed=555",
                    "--config=" + yaml_sample_path,
                    "-yolo~random(-3, 1)",
                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                    "--something-same~random({'lala': 0.2, 'yolo': 0.8})"]
        with pytest.raises(ValueError) as exc:
            spacebuilder.build_from(cmd_args)
        assert 'Conflict' in str(exc.value)

    def test_build_finds_two_configs(self, spacebuilder, yaml_sample_path):
        """There are two explicit definitions of config paths."""
        cmd_args = ["--seed=555",
                    "--config=" + yaml_sample_path,
                    "-yolo~random(-3, 1)",
                    "--config=" + yaml_sample_path,
                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                    "--something-same~random({'lala': 0.2, 'yolo': 0.8})"]
        with pytest.raises(ValueError) as exc:
            spacebuilder.build_from(cmd_args)
        assert 'Already' in str(exc.value)

    def test_build_with_nothing(self, spacebuilder):
        """Return an empty space if nothing is provided."""
        space = spacebuilder.build_from([])
        assert not space

        space = spacebuilder.build_from(["--seed=555", "--naedw"])
        assert not space

    def test_generate_without_config(self, spacebuilder):
        """Build a space using only args."""
        cmd_args = ["--seed=555",
                    "-yolo~random(-3, 1)",
                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                    "--arch2~random({'lala': 0.2, 'yolo': 0.8})"]
        spacebuilder.build_from(cmd_args)
        trial = Trial(params=[
            {'name': '/yolo', 'type': 'real', 'value': -2.4},
            {'name': '/arch2', 'type': 'categorical', 'value': 'yolo'}])

        cmd_inst = spacebuilder.build_to(None, trial)
        assert cmd_inst == ["--seed=555",
                            "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                            "-yolo=-2.4",
                            "--arch2=yolo"]

    def test_generate_only_with_yaml_config(self, spacebuilder,
                                            yaml_sample_path, tmpdir, yaml_converter):
        """Build a space using only a yaml config."""
        spacebuilder.build_from([yaml_sample_path])
        trial = Trial(params=[
            {'name': '/layers/1/width', 'type': 'integer', 'value': 100},
            {'name': '/layers/1/type', 'type': 'categorical', 'value': 'relu'},
            {'name': '/layers/2/type', 'type': 'categorical', 'value': 'sigmoid'},
            {'name': '/training/lr0', 'type': 'real', 'value': 0.032},
            {'name': '/training/mbs', 'type': 'integer', 'value': 64},
            {'name': '/something-same', 'type': 'categorical', 'value': '3'}])
        output_file = str(tmpdir.join("output.yml"))
        cmd_inst = spacebuilder.build_to(output_file, trial)
        assert cmd_inst == [output_file]
        output_data = yaml_converter.parse(output_file)
        assert output_data == {'yo': 5, 'training': {'lr0': 0.032, 'mbs': 64},
                               'layers': [{'width': 64, 'type': 'relu'},
                                          {'width': 100, 'type': 'relu'},
                                          {'width': 16, 'type': 'sigmoid'}],
                               'something-same': '3'}

    def test_generate_only_with_json_config(self, spacebuilder,
                                            json_sample_path, tmpdir, json_converter):
        """Build a space using only a json config."""
        spacebuilder.build_from(['--config=' + json_sample_path])
        trial = Trial(params=[
            {'name': '/layers/1/width', 'type': 'integer', 'value': 100},
            {'name': '/layers/1/type', 'type': 'categorical', 'value': 'relu'},
            {'name': '/layers/2/type', 'type': 'categorical', 'value': 'sigmoid'},
            {'name': '/training/lr0', 'type': 'real', 'value': 0.032},
            {'name': '/training/mbs', 'type': 'integer', 'value': 64},
            {'name': '/something-same', 'type': 'categorical', 'value': '3'}])
        output_file = str(tmpdir.join("output.json"))
        cmd_inst = spacebuilder.build_to(output_file, trial)
        assert cmd_inst == ['--config=' + output_file]
        output_data = json_converter.parse(output_file)
        assert output_data == {'yo': 5, 'training': {'lr0': 0.032, 'mbs': 64},
                               'layers': [{'width': 64, 'type': 'relu'},
                                          {'width': 100, 'type': 'relu'},
                                          {'width': 16, 'type': 'sigmoid'}],
                               'something-same': '3'}

    def test_generate_from_args_and_config(self, spacebuilder,
                                           json_sample_path, tmpdir, json_converter):
        """Build a space using only a json config."""
        cmd_args = ["--seed=555",
                    "-yolo~random(-3, 1)",
                    '--config=' + json_sample_path,
                    "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
                    "--arch2~random({'lala': 0.2, 'yolo': 0.8})"]
        spacebuilder.build_from(cmd_args)
        trial = Trial(params=[
            {'name': '/yolo', 'type': 'real', 'value': -2.4},
            {'name': '/arch2', 'type': 'categorical', 'value': 'yolo'},
            {'name': '/layers/1/width', 'type': 'integer', 'value': 100},
            {'name': '/layers/1/type', 'type': 'categorical', 'value': 'relu'},
            {'name': '/layers/2/type', 'type': 'categorical', 'value': 'sigmoid'},
            {'name': '/training/lr0', 'type': 'real', 'value': 0.032},
            {'name': '/training/mbs', 'type': 'integer', 'value': 64},
            {'name': '/something-same', 'type': 'categorical', 'value': '3'}])
        output_file = str(tmpdir.join("output.json"))
        cmd_inst = spacebuilder.build_to(output_file, trial)
        assert cmd_inst == ['--config=' + output_file] + [
            "--seed=555",
            "--arch1=random({'lala': 0.2, 'yolo': 0.8})",
            "-yolo=-2.4",
            "--arch2=yolo"]
        output_data = json_converter.parse(output_file)
        assert output_data == {'yo': 5, 'training': {'lr0': 0.032, 'mbs': 64},
                               'layers': [{'width': 64, 'type': 'relu'},
                                          {'width': 100, 'type': 'relu'},
                                          {'width': 16, 'type': 'sigmoid'}],
                               'something-same': '3'}
