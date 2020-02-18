# Scikit-learn example on the iris dataset
This folder contains a simple example script (main.py) used to showcase the simplicity of integrating
 Oríon into an existing workflow. We encourage reading the example on the [documentation](https://orion.readthedocs.io/en/latest/examples/scikit-learn.html).

## Pre-requisites
- Install the dependencies `$ pip install -r requirements.txt`
- Configure Oríon database ([documentation](https://orion.readthedocs.io/en/latest/install/database.html)) 
- _main.py_ and _analysis.py_ are executable files (`$ chmod +x <file>`)

## Misc   
- Generate a graph from the data produced by Orion: `./analysis.py`
- View the graph using `xdg-open hyperparameter-optimization.pdf`