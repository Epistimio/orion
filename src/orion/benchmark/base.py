class BaseAssess():

    def __init__(self, task):
        """
        - build assess object
        - build task object (from db[existing], or from config[new])
        """
        pass

    def execute(self):
        """
        - run the tasks
        - there may be needs to run the task multiple times (such as when assess average performance)
        :return:
        """
        pass

    def status(self):
        """
        - get the overall status of the assess, like how many tasks to run and the status of each task(experiment)
        :return:
        """
        pass

    def result(self):
        """
        -  json format of the result
        :return:
        """
        pass

    def display(self):
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

    # assessments that the particular task supports
    assessments = []

    def __init__(self, algorithm):
        """
        - build orion experiment
        """
        pass

    def run(self):
        """
        - run the orion experiment
        :return:
        """
        pass

    def status(self):
        """
        - status of the orion experiment
        """
        pass

    def performance(self):
        """
        - formatted the experiment result for the particular assess
        :return:
        """
        pass