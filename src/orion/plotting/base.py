"""
Provides public plotting API
=============================
"""
import orion.plotting.backend_plotly as backend


def lpi(
    experiment,
    with_evc_tree=True,
    model="RandomForestRegressor",
    model_kwargs=None,
    n_points=20,
    n_runs=10,
    **kwargs,
):
    """
    Make a bar plot to visualize the local parameter importance metric.

    For more information on the metric, see original paper at
    https://ml.informatik.uni-freiburg.de/papers/18-LION12-CAVE.pdf.

    Biedenkapp, André, et al. "Cave: Configuration assessment, visualization and evaluation."
    International Conference on Learning and Intelligent Optimization. Springer, Cham, 2018.

    Parameters
    ----------
    experiment: ExperimentClient or Experiment
        The orion object containing the experiment data

    with_evc_tree: bool, optional
        Fetch all trials from the EVC tree.
        Default: True

    model: str
        Name of the regression model to use. Can be one of
        - AdaBoostRegressor
        - BaggingRegressor
        - ExtraTreesRegressor
        - GradientBoostingRegressor
        - RandomForestRegressor (Default)

        Arguments for the regressor model.
    model_kwargs: dict
        Arguments for the regressor model.
    n: int
        Number of points to compute the variances. Default is 20.
    kwargs: dict
        All other plotting keyword arguments to be passed to
        :plotly:`express.line`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided or if regressor name is invalid.

    """
    return backend.lpi(
        experiment,
        with_evc_tree=with_evc_tree,
        model=model,
        model_kwargs=model_kwargs,
        n_points=n_points,
        n_runs=n_runs,
        **kwargs,
    )


def parallel_coordinates(experiment, with_evc_tree=True, order=None, **kwargs):
    """
    Make a Parallel Coordinates Plot to visualize the effect of the hyperparameters
    on the objective.

    Parameters
    ----------
    experiment: ExperimentClient or Experiment
        The orion object containing the experiment data

    with_evc_tree: bool, optional
        Fetch all trials from the EVC tree.
        Default: True

    order: list of str or None
        Indicates the order of columns in the parallel coordinate plot. By default
        the columns are sorted alphabetically with the exception of the first column
        which is reserved for a fidelity dimension is there is one in the search space.

    kwargs: dict
        All other plotting keyword arguments to be passed to
        :plotly:`express.line`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided.

    """
    return backend.parallel_coordinates(
        experiment, with_evc_tree=with_evc_tree, order=order, **kwargs
    )


def partial_dependencies(
    experiment,
    with_evc_tree=True,
    params=None,
    smoothing=0.85,
    verbose_hover=True,
    n_grid_points=10,
    n_samples=50,
    colorscale="Blues",
    model="RandomForestRegressor",
    model_kwargs=None,
):
    """
    Make countour plots to visualize the search space of each combination of params.

    Parameters
    ----------
    experiment: ExperimentClient or Experiment
        The orion object containing the experiment data

    with_evc_tree: bool, optional
        Fetch all trials from the EVC tree.
        Default: True

    params: list of str, optional
        Indicates the parameters to include in the plots. All parameters are included by default.

    smoothing: float, optional
        Smoothing applied to the countor plot. 0 corresponds to no smoothing. Default is 0.85.

    verbose_hover: bool
        Indicates whether to display the hyperparameter in hover tooltips. True by default.

    colorscale: str, optional
        The colorscale used for the contour plots. Supported values depends on the backend.
        Default is 'Blues'.

    n_grid_points: int, optional
        Number of points in the grid to compute partial dependency. Default is 10.

    n_samples: int, optinal
        Number of samples to randomly generate the grid used to compute the partial dependency.
        Default is 50.

    model: str
        Name of the regression model to use. Can be one of
        - AdaBoostRegressor
        - BaggingRegressor
        - ExtraTreesRegressor
        - GradientBoostingRegressor
        - RandomForestRegressor (Default)

    model_kwargs: dict, optional
        Arguments for the regressor model.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided.

    """

    return backend.partial_dependencies(
        experiment,
        with_evc_tree=with_evc_tree,
        params=params,
        smoothing=smoothing,
        verbose_hover=verbose_hover,
        n_grid_points=n_grid_points,
        n_samples=n_samples,
        colorscale=colorscale,
        model=model,
        model_kwargs=model_kwargs,
    )


def rankings(experiments, with_evc_tree=True, order_by="suggested", **kwargs):
    """
    Make a plot to visually compare the ranking of different hyper-optimization processes.

    The x-axis contain the trials and the y-axis their respective ranking.

    4 formats are supported for the experiments:

        1. List of experiments. The names of the experiments will be used for the figure labels.
        2. Dictionary of experiments. The keys of the dictionary will be used for the figure labels.
        3. List of dictionary of experiments. The keys of the dictionary will be used for the figure \
        labels. The ranking will be averaged across the dictionaries.
        4. Dictionary of list of experiments. The keys of the dictionary will be used for the figure \
        labels. A dictionary of experiments will be build grouping the i-th experiments of each \
        list to result in a list of dictionary of experiments. Behavior will be same as format 3.

    Parameters
    ----------
    experiments: list or dict
        List or dictionary of experiments.

    with_evc_tree: bool, optional
        Fetch all trials from the EVC tree.
        Default: True

    order_by: str
        Indicates how the trials should be ordered. Acceptable options are below.
        See attributes of ``Trial`` for more details.

        * 'suggested': Sort by trial suggested time (default).
        * 'reserved': Sort by trial reserved time.
        * 'completed': Sort by trial completed time.

    kwargs: dict
        All other plotting keyword arguments to be passed to
        :plotly:`express.line`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided or order_by is invalid.

    """
    return backend.rankings(
        experiments, with_evc_tree=with_evc_tree, order_by=order_by, **kwargs
    )


def regret(
    experiment, with_evc_tree=True, order_by="suggested", verbose_hover=True, **kwargs
):
    """
    Make a plot to visualize the performance of the hyper-optimization process.

    The x-axis contain the trials and the y-axis their respective best performance.

    Parameters
    ----------
    experiment: ExperimentClient or Experiment
        The orion object containing the experiment data

    with_evc_tree: bool, optional
        Fetch all trials from the EVC tree.
        Default: True

    order_by: str
        Indicates how the trials should be ordered. Acceptable options are below.
        See attributes of ``Trial`` for more details.

        * 'suggested': Sort by trial suggested time (default).
        * 'reserved': Sort by trial reserved time.
        * 'completed': Sort by trial completed time.

    verbose_hover: bool
        Indicates whether to display the hyperparameter in hover tooltips. True by default.

    kwargs: dict
        All other plotting keyword arguments to be passed to
        :plotly:`express.line`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided.

    """
    return backend.regret(
        experiment,
        with_evc_tree=with_evc_tree,
        order_by=order_by,
        verbose_hover=verbose_hover,
        **kwargs,
    )


def regrets(experiments, with_evc_tree=True, order_by="suggested", **kwargs):
    """
    Make a plot to visually compare the performance of different hyper-optimization processes.

    The x-axis contain the trials and the y-axis their respective best performance.

    3 formats are supported for the experiments:

        1. List of experiments. The names of the experiments will be used for the figure labels.
        2. Dictionary of experiments. The keys of the dictionary will be used for the figure labels.
        3. Dictionary of list of experiments. The keys of the dictionary will be used for the figure \
        labels. The objective of the experiments in each list will be averaged at every time step \
        (ex: across all first trials for a given list of experiments.)

    Parameters
    ----------
    experiments: list or dict
        List or dictionary of experiments.

    with_evc_tree: bool, optional
        Fetch all trials from the EVC tree.
        Default: True

    order_by: str
        Indicates how the trials should be ordered. Acceptable options are below.
        See attributes of ``Trial`` for more details.

        * 'suggested': Sort by trial suggested time (default).
        * 'reserved': Sort by trial reserved time.
        * 'completed': Sort by trial completed time.

    kwargs: dict
        All other plotting keyword arguments to be passed to
        :plotly:`express.line`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided or order_by is invalid.

    """
    return backend.regrets(
        experiments, with_evc_tree=with_evc_tree, order_by=order_by, **kwargs
    )


PLOT_METHODS = {
    "lpi": lpi,
    "parallel_coordinates": parallel_coordinates,
    "partial_dependencies": partial_dependencies,
    "regret": regret,
    "regrets": regrets,
    "rankings": rankings,
}


SINGLE_EXPERIMENT_PLOTS = {
    "lpi": lpi,
    "parallel_coordinates": parallel_coordinates,
    "partial_dependencies": partial_dependencies,
    "regret": regret,
}


class PlotAccessor:
    """
    Make plots of ExperimentClient.

    Parameters
    ----------
    experiment : ExperimentClient
        The object for which the method is called.

    Raises
    ------
    ValueError
        If no experiment is provided.
    """

    def __init__(self, experiment):
        if not experiment:
            raise ValueError("Parameter 'experiment' is None")
        self._experiment = experiment

    def __call__(self, **kwargs):
        """
        Make different kinds of plots of ExperimentClient.

        Parameters
        ----------
        kind : str
            The kind of plot to produce:

            - 'regret' : Regret plot (default)
        """
        kind = kwargs.pop("kind", "regret")

        if kind not in PLOT_METHODS.keys():
            raise ValueError(
                f"Plot of kind '{kind}' is not one of {list(PLOT_METHODS.keys())}"
            )

        return PLOT_METHODS[kind](self._experiment, **kwargs)

    def lpi(self, **kwargs):
        """Make a bar plot of the local parameter importance metrics."""
        __doc__ = lpi.__doc__
        return self(kind="lpi", **kwargs)

    def parallel_coordinates(self, **kwargs):
        """Make a parallel coordinates plot to visualize the performance of the
        hyper-optimization process.
        """
        __doc__ = parallel_coordinates.__doc__
        return self(kind="parallel_coordinates", **kwargs)

    def partial_dependencies(self, **kwargs):
        """Make countour plots to visualize the search space of each combination of params."""
        __doc__ = partial_dependencies.__doc__
        return self(kind="partial_dependencies", **kwargs)

    def regret(self, **kwargs):
        """Make a plot to visualize the performance of the hyper-optimization process."""
        __doc__ = regret.__doc__
        return self(kind="regret", **kwargs)
