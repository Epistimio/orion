def learning(experiment, **kwargs):
    """
    Make a plot to visualize the learning performance of the hyper-optimization process.

    The x-axis contain the trials and the y-axis their respective best performance.

    Parameters
    ----------
        experiment: ExperimentClient
            The orion object containing the data

        **kwargs
            All other plotting keyword arguments to be passed to
            :meth:`plotly.express.line`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If no experiment is provided.

    """
    if not experiment:
        raise ValueError("Parameter 'experiment' is None")
    
    raise NotImplementedError
    # TODO: Make a plot using Plotly

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


    def __call__(self, *args, **kwargs):
        """
        Convenience method to call different types of plots or ExperimentClient.
        """
        raise NotImplementedError()
        # TODO: Delegate call to correct plotting functions.


    def learning(self):
        """ Convenience method for :meth:`orion.plotting.learning`"""
        raise NotImplementedError()
        # TODO: Call self(kind='learning')

