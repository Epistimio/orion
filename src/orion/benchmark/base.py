from abc import abstractmethod


class BaseAssess():

    def __init__(self, task_num, **kwargs):
        """
        :param task_num: number of tasks that need to run for the assessment
        :param kwargs:
        """
        self.task_number = task_num
        self._param_names = list(kwargs.keys())

    @property
    def task_num(self):
        return self.task_number

    @abstractmethod
    def display(self, task, experiments, notebook):
        """
        - define the visual charts of the assess, based on the task performance output
        :return:
        """
        pass

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        """
        dict_form = dict()
        for attrname in self._param_names:
            if attrname.startswith('_'):  # Do not log _space or others in conf
                continue
            attr = getattr(self, attrname)
            dict_form[attrname] = attr
        dict_form['task_num'] = self.task_num

        mod = self.__class__.__module__
        fullname = mod + '.' + self.__class__.__qualname__
        fullname = fullname.replace('.', '-')
        return {fullname: dict_form}


class BaseTask():

    def __init__(self, **kwargs):
        """
        - build orion experiment
        """
        self._param_names = list(kwargs.keys())

    @abstractmethod
    def get_blackbox_function(self):
        """
        The black box function to optimize, the function will expect hyper-parameters to search and return
        objective values of trial with the hyper-parameters.
        :return:
        """
        pass

    @abstractmethod
    def get_max_trials(self):
        """
        The max number of trials to run for the task during search
        :return:
        """
        pass

    @abstractmethod
    def get_search_space(self):
        """
        The search space for the hyper-parameters of the black box function
        :return:
        """
        pass

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        """
        dict_form = dict()
        for attrname in self._param_names:
            if attrname.startswith('_'):  # Do not log _space or others in conf
                continue
            attr = getattr(self, attrname)
            dict_form[attrname] = attr

        mod = self.__class__.__module__
        fullname = mod + '.' + self.__class__.__qualname__
        fullname = fullname.replace('.', '-')
        return {fullname: dict_form}