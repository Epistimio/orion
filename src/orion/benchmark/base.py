from abc import abstractmethod


class BaseAssess():

    def __init__(self, task_num, **kwargs):
        """
        :param task_num: number of tasks that need to run for the assessment
        :param kwargs:
        """
        self.task_number = task_num

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

    def register(self):
        """
        register assess object into db
        :return:
        """
        pass


class BaseTask():

    def __init__(self, **kwargs):
        """
        - build orion experiment
        """
        pass

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
