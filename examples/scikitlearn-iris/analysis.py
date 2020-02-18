import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from orion.client import create_experiment

experiment = create_experiment(name="scitkit-iris-tutorial")
data = [(1 - trial.objective.value, trial.params['/_pos_0']) for trial in experiment.fetch_trials_by_status('completed')]

df = pd.DataFrame(data, columns=['accuracy', 'epsilon']).sort_values('accuracy')

sns.set_style("whitegrid")
ax = sns.scatterplot(x='epsilon', y='accuracy', data=df, color='black', alpha=0.3)
ax.set_xscale('log')
ax.set_xticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])

plt.savefig("hyperparameter-optimization.pdf", format='pdf')
# plt.show()  # If tkinter is installed (sudo apt-get install python3-tk)
