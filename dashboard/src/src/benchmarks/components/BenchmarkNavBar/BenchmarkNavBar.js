import React from 'react';
import {
  SideNav,
  Checkbox,
  StructuredListWrapper,
  StructuredListRow,
  StructuredListCell,
  ComboBox,
} from 'carbon-components-react';

function getAlgorithmName(algoDef) {
  if (typeof algoDef === 'string') return algoDef;
  const keys = Object.keys(algoDef);
  if (keys.length === 1) return keys[0];
  else
    throw new Error(
      `Cannot get algorithm name from object: ${JSON.stringify(algoDef)}`
    );
}

export class BenchmarkNavBar extends React.Component {
  constructor(props) {
    // props:
    // benchmarks: list of JSON objects representing benchmarks
    // benchmark: JSON object representing a benchmark
    // algorithms: set of strings
    // tasks: set of strings
    // assessments: set of strings
    super(props);
    this.onChangeComboBox = this.onChangeComboBox.bind(this);
    this.onSelectAlgo = this.onSelectAlgo.bind(this);
    this.onSelectTask = this.onSelectTask.bind(this);
    this.onSelectAssessment = this.onSelectAssessment.bind(this);
  }
  render() {
    return this.props.benchmarks === null ? (
      ''
    ) : (
      <SideNav
        className="benchmark-navbar"
        isFixedNav
        expanded={true}
        isChildOfHeader={false}
        aria-label="Side navigation">
        <ComboBox
          onChange={this.onChangeComboBox}
          id={'combobox-benchmark'}
          items={this.props.benchmarks}
          itemToString={item => (item === null ? null : item.name)}
          placeholder={'Search a benchmark ...'}
        />
        {this.props.benchmark === null ? (
          ''
        ) : (
          <StructuredListWrapper>
            {this.renderAssessments()}
            {this.renderTasks()}
            {this.renderAlgorithms()}
          </StructuredListWrapper>
        )}
      </SideNav>
    );
  }
  renderAssessments() {
    const benchmark = this.props.benchmark;
    const assessments = Object.keys(benchmark.assessments);
    assessments.sort();
    return (
      <>
        <StructuredListRow>
          <StructuredListCell>
            <strong>Assessments</strong>
          </StructuredListCell>
        </StructuredListRow>
        {assessments.map((assessment, indexAssessment) => (
          <StructuredListRow key={indexAssessment}>
            <StructuredListCell>
              <Checkbox
                labelText={assessment}
                id={`assessment-${indexAssessment}`}
                checked={this.props.assessments.has(assessment)}
                onChange={(checked, id, event) =>
                  this.onSelectAssessment(assessment, checked)
                }
              />
            </StructuredListCell>
          </StructuredListRow>
        ))}
      </>
    );
  }
  renderTasks() {
    const benchmark = this.props.benchmark;
    const tasks = Object.keys(benchmark.tasks);
    tasks.sort();
    return (
      <>
        <StructuredListRow>
          <StructuredListCell>
            <strong>Tasks</strong>
          </StructuredListCell>
        </StructuredListRow>
        {tasks.map((task, indexTask) => (
          <StructuredListRow key={indexTask}>
            <StructuredListCell>
              <Checkbox
                labelText={task}
                id={`task-${indexTask}`}
                checked={this.props.tasks.has(task)}
                onChange={(checked, id, event) =>
                  this.onSelectTask(task, checked)
                }
              />
            </StructuredListCell>
          </StructuredListRow>
        ))}
      </>
    );
  }
  renderAlgorithms() {
    const benchmark = this.props.benchmark;
    const algorithms = benchmark.algorithms.map(algo => getAlgorithmName(algo));
    algorithms.sort();
    return (
      <>
        <StructuredListRow>
          <StructuredListCell>
            <strong>Algorithms</strong>
          </StructuredListCell>
        </StructuredListRow>
        {algorithms.map((algorithm, indexAlgorithm) => (
          <StructuredListRow key={indexAlgorithm}>
            <StructuredListCell>
              <Checkbox
                labelText={algorithm}
                id={`algorithm-${indexAlgorithm}`}
                checked={this.props.algorithms.has(algorithm)}
                onChange={(checked, id, event) =>
                  this.onSelectAlgo(algorithm, checked)
                }
              />
            </StructuredListCell>
          </StructuredListRow>
        ))}
      </>
    );
  }
  onChangeComboBox(event) {
    const benchmark = event.selectedItem;
    if (benchmark === null) {
      this.props.onSelectBenchmark(benchmark, new Set(), new Set(), new Set());
    } else {
      const algorithms = benchmark.algorithms.map(algo =>
        getAlgorithmName(algo)
      );
      this.props.onSelectBenchmark(
        benchmark,
        new Set(algorithms),
        new Set(Object.keys(benchmark.tasks)),
        new Set(Object.keys(benchmark.assessments))
      );
    }
  }
  onSelectAlgo(algorithm, checked) {
    const algorithms = new Set(this.props.algorithms);
    if (checked) algorithms.add(algorithm);
    else algorithms.delete(algorithm);
    this.props.onSelectBenchmark(
      this.props.benchmark,
      algorithms,
      this.props.tasks,
      this.props.assessments
    );
  }
  onSelectTask(task, checked) {
    const tasks = new Set(this.props.tasks);
    if (checked) tasks.add(task);
    else tasks.delete(task);
    this.props.onSelectBenchmark(
      this.props.benchmark,
      this.props.algorithms,
      tasks,
      this.props.assessments
    );
  }
  onSelectAssessment(assessment, checked) {
    const assessments = new Set(this.props.assessments);
    if (checked) assessments.add(assessment);
    else assessments.delete(assessment);
    this.props.onSelectBenchmark(
      this.props.benchmark,
      this.props.algorithms,
      this.props.tasks,
      assessments
    );
  }
}

export default BenchmarkNavBar;
