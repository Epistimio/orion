import React from 'react';
import { PlotRender } from './PlotRender';
import { Tile, Grid, Row, Column } from 'carbon-components-react';

export default class BenchmarkVisualizationsPage extends React.Component {
  /**
   * Props:
   * benchmark: JSON object representing a benchmark
   * algorithms: set of strings
   * tasks: set of strings
   * assessments: set of strings
   */
  constructor(props) {
    super(props);
    this.onResize = this.onResize.bind(this);
  }
  render() {
    if (this.props.benchmark === null) {
      return (
        <div>
          <h4 className="title-visualizations">No benchmark selected</h4>
        </div>
      );
    }
    if (!this.props.assessments.size) {
      return (
        <div>
          <h4 className="title-visualizations">No assessment selected</h4>
        </div>
      );
    }
    if (!this.props.tasks.size) {
      return (
        <div>
          <h4 className="title-visualizations">No task selected</h4>
        </div>
      );
    }
    if (!this.props.algorithms.size) {
      return (
        <div>
          <h4 className="title-visualizations">No algorithm selected</h4>
        </div>
      );
    }
    const assessments = Array.from(this.props.assessments);
    const tasks = Array.from(this.props.tasks);
    const algorithms = Array.from(this.props.algorithms);
    assessments.sort();
    tasks.sort();
    algorithms.sort();
    /**
     * Key to hash current VisualizationPage properties.
     * Used to force re-rendering of all plots each time any option is (de)selected,
     * and then make sure each plot is entirely redrawn, preventing any graphical bug.
     * As plots are cached after first API call, forcing a redraw is not so-much time-consuming.
     * @type {string}
     */
    const prefix = `viz-${this.props.benchmark.name}-${assessments.join(
      '-'
    )}-${tasks.join('-')}-${algorithms.join('-')}`;
    return (
      <div>
        <h4 className="title-visualizations">Assessments</h4>
        <div className="assessments" id="assessments">
          {assessments.map((assessment, indexAssessment) => (
            <Grid
              fullWidth
              className="assessment"
              key={`assessment-${assessment}`}>
              <Row>
                <Column>
                  <Tile className="plot-tile">
                    <strong>
                      <em>{assessment}</em>
                    </strong>
                  </Tile>
                </Column>
              </Row>
              {tasks.map((task, indexTask) => (
                <Row key={`task-${task}`}>
                  <Column
                    key={`task-${task}-assessment-${assessment}`}
                    className="orion-column">
                    <Tile className="plot-tile">
                      <PlotRender
                        key={`${prefix}-plots-${
                          this.props.benchmark.name
                        }-${assessment}-${task}-${algorithms.join('-')}`}
                        benchmark={this.props.benchmark.name}
                        assessment={assessment}
                        task={task}
                        algorithms={algorithms}
                      />
                    </Tile>
                  </Column>
                </Row>
              ))}
            </Grid>
          ))}
        </div>
      </div>
    );
  }
  componentDidMount() {
    // Make sure to resize grids when page is mounted.
    this.onResize();
    // Make sure to resize grids when window is resized.
    window.addEventListener('resize', this.onResize);
  }
  componentDidUpdate(prevProps, prevState, snapshot) {
    // Make sure to resize grids everytime page is updated.
    this.onResize();
  }
  componentWillUnmount() {
    // Remove ebent listener.
    window.removeEventListener('resize', this.onResize);
  }
  onResize() {
    // Get grids div.
    const divAssessments = document.getElementById('assessments');
    if (!divAssessments) return;
    const width = divAssessments.offsetWidth;
    // Set plot width to grids div / nb. grids
    const plotWidth = width / this.props.assessments.size;
    // Set grid width for each grid
    const grids = divAssessments.getElementsByClassName('assessment');
    for (let i = 0; i < grids.length; ++i) {
      const grid = grids[i];
      // Grid width must be plot width * max nb. of plots displayed in a column.
      // We must then get max number of plots in a column.
      let nbMaxPlots = 1;
      const columns = grid.getElementsByClassName('orion-column');
      for (let iCol = 0; iCol < columns.length; ++iCol) {
        const column = columns[iCol];
        const plots = column.getElementsByClassName('orion-plot');
        if (nbMaxPlots < plots.length) nbMaxPlots = plots.length;
      }
      const gridWidth = plotWidth * nbMaxPlots;
      grid.style.width = `${gridWidth}px`;
    }
  }
}
