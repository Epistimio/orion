# Basic sklearn example on iris dataset

```
pip install -r requirements.txt
python main.py
```

Prerequisites:
- Orion is installed
- The database is setup (you can test with `orion db test`)
- _main.py_ and _analysis.py_ are executable files (`chmod +x <file>`)

Using the commands: 

- Normally would be called `./main.py <epsilon>`
- `orion hunt -n scitkit-iris-tutorial --max-trials 500 ./main.py 'orion~loguniform(1e-5, 1.0)'`
    
    `--max-trials` specifies the budget of the hyper-parameter optimization.
   
- Generate a graph from the data produced by Orion: `./analysis.py`
- View the graph using `xdg-open hyperparameter-optimization.pdf`