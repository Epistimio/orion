from orion.benchmark.base import BaseAssess
from tabulate import tabulate
from orion.benchmark import Benchmark
import pandas as pd
import plotly.graph_objects as go


class TimeToResult(BaseAssess):

    def __init__(self, task_num=1):
        super(TimeToResult, self).__init__(task_num)

    def display(self, task, experiments, notebook=True):
        """
        - define the visual charts of the assess, based on the task performance output
        :return:
        """
        time_to_result = []
        algorithm_exp_trials = {}

        for _, exp in experiments:
            exp_column = dict()
            stats = exp.stats
            trial_id = stats['best_trials_id']
            best_trial = exp.get_trial(uid=trial_id)
            exp_column['Algorithm'] = list(exp.configuration['algorithms'].keys())[0]
            exp_column['Assessment'] = self.__class__.__name__
            exp_column['Experiment Name'] = exp.name
            exp_column['Best Trial'] = trial_id  # TODO use sequential id
            exp_column['Best Evaluation'] = stats['best_evaluation']
            for param_name, param_value in best_trial.params.items():
                exp_column[param_name] = param_value
            time_to_result.append(exp_column)

            algo = list(exp.configuration['algorithms'].keys())[0]
            trials = list(filter(lambda trial: trial.status == 'completed', exp.fetch_trials()))

            algorithm_exp_trials[algo] = self._build_frame(trials, algo)

        if notebook:
            from IPython.display import HTML, display
            display(HTML(tabulate(time_to_result, headers='keys', tablefmt='html', stralign='center', numalign='center')))
        else:
            table = tabulate(time_to_result, headers='keys', tablefmt='grid', stralign='center', numalign='center')
            print(table)

        fig = go.Figure()
        for algo, df in algorithm_exp_trials.items():
            fig.add_scatter(y=df['objective'],
                            mode='lines',
                            name=algo)
        title = 'Assessment {} over Task {}'.format(self.__class__.__name__, task)
        fig.update_layout(title=title,
                          xaxis_title="trial_seq",
                          yaxis_title='objective')
        fig.show()

    def _build_frame(self, trials, algorithm, order_by='suggested'):
        """Builds the dataframe for the plot"""
        data = [(trial.status,
                trial.submit_time,
                trial.objective.value) for trial in trials]

        df = pd.DataFrame(data, columns=['status', 'suggested', 'objective'])

        df = df.sort_values(order_by)

        del df['status']
        del df['suggested']

        df.index.name = 'seq'
        return df

    def register(self):
        """
        register assess object into db
        :return:
        """
        pass
