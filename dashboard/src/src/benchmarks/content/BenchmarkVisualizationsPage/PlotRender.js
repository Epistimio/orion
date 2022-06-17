import React from 'react';
import Plot from 'react-plotly.js';
import { DEFAULT_BACKEND } from '../../../utils/queryServer';
import { Backend } from '../../../utils/queryServer';

class PlotData {
  constructor(name, data, layout) {
    this.name = name;
    this.data = data;
    this.layout = layout;
  }

  get(algorithms) {
    const data = [];
    for (let entry of this.data) {
      for (let algorithm of algorithms) {
        if (
          entry.name.indexOf(algorithm) === 0 ||
          entry.name.indexOf(`${algorithm}_`) === 0
        ) {
          data.push(entry);
        }
      }
    }
    return { data, layout: this.layout, name: this.name };
  }
}

class PlotsProvider {
  constructor(address) {
    this.backend = new Backend(address);
    this.plots = {};
  }
  async get(benchmark, assessment, task, algorithms) {
    /**@type {Array<PlotData>}*/
    let plots = ((this.plots[benchmark] || {})[assessment] || {})[task] || [];
    if (!plots.length) {
      const query = `benchmarks/${benchmark}?assessment=${assessment}&task=${task}`;
      console.log(`Loading: ${this.backend.baseURL}/${query}`);
      const response = await this.backend.query(query);
      const rawPlots = response.analysis[assessment][task];
      const plotNames = Object.keys(rawPlots);
      plotNames.sort();
      for (let name of plotNames) {
        const parsed = JSON.parse(rawPlots[name]);
        plots.push(new PlotData(name, parsed.data, parsed.layout));
      }
      if (this.plots[benchmark] === undefined) this.plots[benchmark] = {};
      if (this.plots[benchmark][assessment] === undefined)
        this.plots[benchmark][assessment] = {};
      if (this.plots[benchmark][assessment][task] === undefined)
        this.plots[benchmark][assessment][task] = plots;
    }
    return plots.map(plotData => plotData.get(algorithms));
  }
}
const PLOTS_PROVIDER = new PlotsProvider(DEFAULT_BACKEND);

export class PlotRender extends React.Component {
  // Control variable to avoid setting state if component was unmounted before an asynchronous API call finished.
  _isMounted = false;
  constructor(props) {
    // props:
    // benchmark: str
    // assessment: str
    // task: str
    // algorithms: list of strings
    super(props);
    this.state = {
      plots: null,
    };
  }
  render() {
    if (this.state.plots === null)
      return `Loading plots for assessment: ${this.props.assessment}, task: ${
        this.props.task
      }, algorithms: ${this.props.algorithms.join(', ')}`;
    if (this.state.plots === false)
      return (
        <strong>{`Unable to load plots for assessment: ${
          this.props.assessment
        }, task: ${this.props.task}, algorithms: ${this.props.algorithms.join(
          ', '
        )}`}</strong>
      );
    return (
      <div className="orion-plots">
        {this.state.plots.map((plot, plotIndex) => {
          const id = `plot-${this.props.benchmark}-${this.props.assessment}-${
            this.props.task
          }-${plot.name}-${this.props.algorithms.join('-')}`;
          return (
            <Plot
              className="orion-plot"
              key={id}
              divId={id}
              data={plot.data}
              layout={plot.layout}
              config={{ responsive: true }}
              useResizeHandler={true}
            />
          );
        })}
      </div>
    );
  }
  componentDidMount() {
    this._isMounted = true;
    PLOTS_PROVIDER.get(
      this.props.benchmark,
      this.props.assessment,
      this.props.task,
      this.props.algorithms
    )
      .then(plots => {
        if (this._isMounted) {
          this.setState({ plots });
        }
      })
      .catch(error => {
        console.error(error);
        if (this._isMounted) this.setState({ plots: false });
      });
  }
  componentWillUnmount() {
    this._isMounted = false;
  }
  componentDidUpdate(prevProps, prevState, snapshot) {
    // Trigger a window resize event to force displayed plots to resize correctly.
    // Reference (2022/03/24): https://stackoverflow.com/a/43846782
    if (typeof Event === 'function') {
      // modern browsers
      window.dispatchEvent(new Event('resize'));
    } else {
      // for IE and other old browsers
      // causes deprecation warning on modern browsers
      var evt = window.document.createEvent('UIEvents');
      evt.initUIEvent('resize', true, false, window, 0);
      window.dispatchEvent(evt);
    }
  }
}
