import React from 'react';
import App from '../App';
import { HashRouter as Router } from 'react-router-dom';
import {
  render,
  waitFor,
  queryByText,
  findByText,
  screen,
  fireEvent,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
/* Use MemoryRouter to isolate history for each test */
import { MemoryRouter } from 'react-router-dom';

// Since I updated dependencies in package.json, this seems necessary.
beforeEach(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(), // deprecated
      removeListener: jest.fn(), // deprecated
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    })),
  });
});

/**
 * Return true if given DOM plot element contains all given texts.
 * @param plot - DOM element containing a plot
 * @param texts - texts (strings) to search
 * @return {boolean} - true if plot contains all texts
 */
function plotHasTexts(plot, texts) {
  for (let text of texts) {
    if (queryByText(plot, text) === null) return false;
  }
  return true;
}

/**
 * Check immediately (no async) that we find only 1 plot containing all given texts.
 * @param texts - texts to search
 * @return {boolean} - true if plot is found
 */
function hasPlotImmediately(...texts) {
  const plots = document.querySelectorAll('.orion-plot');
  const filtered = [];
  for (let plot of plots.values()) {
    if (plotHasTexts(plot, texts)) {
      filtered.push(plot);
    }
  }
  return filtered.length === 1;
}

/**
 * Check that we find only 1 plot containing all given texts.
 * @param texts - texts to search
 */
async function lookupPlot(...texts) {
  await waitFor(
    () => {
      expect(hasPlotImmediately(...texts)).toBe(true);
    },
    { interval: 1000, timeout: 20000 }
  );
}

/**
 * Wait for given amount of time
 * @param milliseonds - time to wait
 */
async function sleep(milliseonds) {
  let value = 0;
  await new Promise(r => setTimeout(r, milliseonds)).then(() => {
    value = 1;
  });
  expect(value).toBe(1);
}

// Texts to find plots for benchmark all_assessments_webapi_2
const AAWA2Plots = {
  // 10 full plots
  average_rank_branin: [
    'Average Rankings',
    'Ranking based on branin',
    'Trials ordered by suggested time',
    'random',
    'tpe',
  ],
  average_rank_rosenbrock: [
    'Average Rankings',
    'Ranking based on rosenbrock',
    'Trials ordered by suggested time',
    'random',
    'tpe',
  ],
  average_result_branin: [
    'Average Regret',
    'branin',
    'Trials ordered by suggested time',
    'random',
    'tpe',
  ],
  average_result_rosenbrock: [
    'Average Regret',
    'rosenbrock',
    'Trials ordered by suggested time',
    'random',
    'tpe',
  ],
  parallel_assessment_time_branin: [
    'Time to result',
    'branin',
    'Experiment duration by second(s)',
    'random_workers_1',
    'tpe_workers_1',
  ],
  parallel_assessment_time_rosenbrock: [
    'Time to result',
    'rosenbrock',
    'Experiment duration by second(s)',
    'random_workers_1',
    'tpe_workers_1',
  ],
  parallel_assessment_pa_branin: [
    'Parallel Assessment',
    'branin',
    'Number of workers',
    'random',
    'tpe',
  ],
  parallel_assessment_pa_rosenbrock: [
    'Parallel Assessment',
    'rosenbrock',
    'Number of workers',
    'random',
    'tpe',
  ],
  parallel_assessment_regret_branin: [
    'Average Regret',
    'branin',
    'Trials ordered by suggested time',
    'random_workers_1',
    'tpe_workers_1',
  ],
  parallel_assessment_regret_rosenbrock: [
    'Average Regret',
    'rosenbrock',
    'Trials ordered by suggested time',
    'random_workers_1',
    'tpe_workers_1',
  ],
};
const AAWA2PlotsNoRandom = {
  // 10 full plots
  average_rank_branin: [
    'Average Rankings',
    'Ranking based on branin',
    'Trials ordered by suggested time',
    'tpe',
  ],
  average_rank_rosenbrock: [
    'Average Rankings',
    'Ranking based on rosenbrock',
    'Trials ordered by suggested time',
    'tpe',
  ],
  average_result_branin: [
    'Average Regret',
    'branin',
    'Trials ordered by suggested time',
    'tpe',
  ],
  average_result_rosenbrock: [
    'Average Regret',
    'rosenbrock',
    'Trials ordered by suggested time',
    'tpe',
  ],
  parallel_assessment_time_branin: [
    'Time to result',
    'branin',
    'Experiment duration by second(s)',
    'tpe_workers_1',
  ],
  parallel_assessment_time_rosenbrock: [
    'Time to result',
    'rosenbrock',
    'Experiment duration by second(s)',
    'tpe_workers_1',
  ],
  parallel_assessment_pa_branin: [
    'Parallel Assessment',
    'branin',
    'Number of workers',
    // 'tpe', // NB: When only 1 algorithm selected, algorithms legend is not displayed in this plot
  ],
  parallel_assessment_pa_rosenbrock: [
    'Parallel Assessment',
    'rosenbrock',
    'Number of workers',
    // 'tpe', // NB: When only 1 algorithm selected, algorithms legend is not displayed in this plot
  ],
  parallel_assessment_regret_branin: [
    'Average Regret',
    'branin',
    'Trials ordered by suggested time',
    'tpe_workers_1',
  ],
  parallel_assessment_regret_rosenbrock: [
    'Average Regret',
    'rosenbrock',
    'Trials ordered by suggested time',
    'tpe_workers_1',
  ],
};

const AAWA2_all = [
  AAWA2Plots.average_rank_branin,
  AAWA2Plots.average_rank_rosenbrock,
  AAWA2Plots.average_result_branin,
  AAWA2Plots.average_result_rosenbrock,
  AAWA2Plots.parallel_assessment_time_branin,
  AAWA2Plots.parallel_assessment_time_rosenbrock,
  AAWA2Plots.parallel_assessment_pa_branin,
  AAWA2Plots.parallel_assessment_pa_rosenbrock,
  AAWA2Plots.parallel_assessment_regret_branin,
  AAWA2Plots.parallel_assessment_regret_rosenbrock,
];
const AAWA2_average_rank = [
  AAWA2Plots.average_rank_branin,
  AAWA2Plots.average_rank_rosenbrock,
];
const AAWA2_no_average_rank = [
  AAWA2Plots.average_result_branin,
  AAWA2Plots.average_result_rosenbrock,
  AAWA2Plots.parallel_assessment_time_branin,
  AAWA2Plots.parallel_assessment_time_rosenbrock,
  AAWA2Plots.parallel_assessment_pa_branin,
  AAWA2Plots.parallel_assessment_pa_rosenbrock,
  AAWA2Plots.parallel_assessment_regret_branin,
  AAWA2Plots.parallel_assessment_regret_rosenbrock,
];
const AAWA2_branin = [
  AAWA2Plots.average_rank_branin,
  AAWA2Plots.average_result_branin,
  AAWA2Plots.parallel_assessment_time_branin,
  AAWA2Plots.parallel_assessment_pa_branin,
  AAWA2Plots.parallel_assessment_regret_branin,
];
const AAWA2_no_branin = [
  AAWA2Plots.average_rank_rosenbrock,
  AAWA2Plots.average_result_rosenbrock,
  AAWA2Plots.parallel_assessment_time_rosenbrock,
  AAWA2Plots.parallel_assessment_pa_rosenbrock,
  AAWA2Plots.parallel_assessment_regret_rosenbrock,
];
const AAWA2_no_random = [
  AAWA2PlotsNoRandom.average_rank_branin,
  AAWA2PlotsNoRandom.average_rank_rosenbrock,
  AAWA2PlotsNoRandom.average_result_branin,
  AAWA2PlotsNoRandom.average_result_rosenbrock,
  AAWA2PlotsNoRandom.parallel_assessment_time_branin,
  AAWA2PlotsNoRandom.parallel_assessment_time_rosenbrock,
  AAWA2PlotsNoRandom.parallel_assessment_pa_branin,
  AAWA2PlotsNoRandom.parallel_assessment_pa_rosenbrock,
  AAWA2PlotsNoRandom.parallel_assessment_regret_branin,
  AAWA2PlotsNoRandom.parallel_assessment_regret_rosenbrock,
];

test('Test select benchmark', async () => {
  render(<App />, { wrapper: MemoryRouter });

  // Switch to benchmarks page
  const menu = await screen.findByTitle(/Go to benchmarks visualizations/);
  fireEvent.click(menu);
  await sleep(1000);

  expect(await screen.findByText(/No benchmark selected/)).toBeInTheDocument();
  // Get benchmark search field
  const benchmarkField = await screen.findByPlaceholderText(
    'Search a benchmark ...'
  );
  expect(benchmarkField).toBeInTheDocument();

  // Select branin_baselines_webapi benchmark
  userEvent.type(benchmarkField, 'branin');
  userEvent.keyboard('{enter}');
  expect(benchmarkField.value).toBe('branin_baselines_webapi');
  const leftMenu = document.querySelector('.bx--structured-list');
  expect(leftMenu).toBeInTheDocument();
  expect(await findByText(leftMenu, /AverageResult/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /Branin/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /gridsearch/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /random/)).toBeInTheDocument();
  expect(screen.queryByText(/No benchmark selected/)).toBeNull();
  // Check plot
  await lookupPlot(
    'Average Regret',
    'branin',
    'Trials ordered by suggested time'
  );

  // Select all_algos_webapi benchmark
  // Select text in search bar before typing a new text, so that old text is cleared
  benchmarkField.setSelectionRange(0, benchmarkField.value.length);
  userEvent.type(benchmarkField, 'all_algos');
  userEvent.keyboard('{enter}');
  expect(benchmarkField.value).toBe('all_algos_webapi');
  expect(await findByText(leftMenu, /AverageResult/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /Branin/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /EggHolder/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /RosenBrock/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /gridsearch/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /random/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /tpe/)).toBeInTheDocument();
  expect(screen.queryByText(/No benchmark selected/)).toBeNull();
  // Check plots
  await lookupPlot('Average Regret', 'branin');
  await lookupPlot('Average Regret', 'eggholder');
  await lookupPlot('Average Regret', 'rosenbrock');

  // Select all_assessments_webapi_2
  benchmarkField.setSelectionRange(0, benchmarkField.value.length);
  userEvent.type(benchmarkField, 'all_asses');
  userEvent.keyboard('{enter}');
  expect(benchmarkField.value).toBe('all_assessments_webapi_2');
  expect(await findByText(leftMenu, /AverageRank/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /AverageResult/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /ParallelAssessment/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /Branin/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /RosenBrock/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /random/)).toBeInTheDocument();
  expect(await findByText(leftMenu, /tpe/)).toBeInTheDocument();
  expect(screen.queryByText(/No benchmark selected/)).toBeNull();
  // Check plots
  // Assessment AverageRank
  await lookupPlot(...AAWA2Plots.average_rank_branin);
  await lookupPlot(...AAWA2Plots.average_rank_rosenbrock);
  // Assessment AverageResult
  await lookupPlot(...AAWA2Plots.average_result_branin);
  await lookupPlot(...AAWA2Plots.average_result_rosenbrock);
  // Assessment ParallelAssessment (which also have regret plots as AverageResult)
  await lookupPlot(...AAWA2Plots.parallel_assessment_time_branin);
  await lookupPlot(...AAWA2Plots.parallel_assessment_time_rosenbrock);
  await lookupPlot(...AAWA2Plots.parallel_assessment_pa_branin);
  await lookupPlot(...AAWA2Plots.parallel_assessment_pa_rosenbrock);
  await lookupPlot(...AAWA2Plots.parallel_assessment_regret_branin);
  await lookupPlot(...AAWA2Plots.parallel_assessment_regret_rosenbrock);
});

test('Test (de)select assessments', async () => {
  render(<App />, { wrapper: MemoryRouter });

  // Switch to benchmarks page
  const menu = await screen.findByTitle(/Go to benchmarks visualizations/);
  fireEvent.click(menu);
  await sleep(1000);

  expect(await screen.findByText(/No benchmark selected/)).toBeInTheDocument();
  // Get benchmark search field
  const benchmarkField = await screen.findByPlaceholderText(
    'Search a benchmark ...'
  );
  expect(benchmarkField).toBeInTheDocument();
  // Select all_assessments_webapi_2
  userEvent.type(benchmarkField, 'all_asses');
  userEvent.keyboard('{enter}');
  expect(benchmarkField.value).toBe('all_assessments_webapi_2');

  // Make sure all plots are there (10 plots)
  for (let texts of AAWA2_all) {
    await lookupPlot(...texts);
  }

  // Select 1 assessment.
  const inputAssessmentAverageRank = document.getElementById('assessment-0');
  expect(inputAssessmentAverageRank).toBeInTheDocument();
  expect(inputAssessmentAverageRank.checked).toBe(true);

  // Deselect assessment
  userEvent.click(inputAssessmentAverageRank);
  expect(inputAssessmentAverageRank.checked).toBe(false);
  await sleep(100);
  for (let texts of AAWA2_average_rank) {
    expect(hasPlotImmediately(...texts)).toBe(false);
  }
  for (let texts of AAWA2_no_average_rank) {
    expect(hasPlotImmediately(...texts)).toBe(true);
  }
  // Reselect assessment.
  userEvent.click(inputAssessmentAverageRank);
  expect(inputAssessmentAverageRank.checked).toBe(true);
  await sleep(100);
  for (let texts of AAWA2_all) {
    expect(hasPlotImmediately(...texts)).toBe(true);
  }
});

test('Test (de)select tasks', async () => {
  render(<App />, { wrapper: MemoryRouter });

  // Switch to benchmarks page
  const menu = await screen.findByTitle(/Go to benchmarks visualizations/);
  fireEvent.click(menu);
  await sleep(1000);

  expect(await screen.findByText(/No benchmark selected/)).toBeInTheDocument();
  // Get benchmark search field
  const benchmarkField = await screen.findByPlaceholderText(
    'Search a benchmark ...'
  );
  expect(benchmarkField).toBeInTheDocument();
  // Select all_assessments_webapi_2
  userEvent.type(benchmarkField, 'all_asses');
  userEvent.keyboard('{enter}');
  expect(benchmarkField.value).toBe('all_assessments_webapi_2');

  // Make sure all plots are there (10 plots)
  for (let texts of AAWA2_all) {
    await lookupPlot(...texts);
  }

  // Select 1 task.
  const inputTaskBranin = document.getElementById('task-0');
  expect(inputTaskBranin).toBeInTheDocument();
  expect(inputTaskBranin.checked).toBe(true);

  // Deselect task.
  userEvent.click(inputTaskBranin);
  expect(inputTaskBranin.checked).toBe(false);
  await sleep(100);
  for (let texts of AAWA2_branin) {
    expect(hasPlotImmediately(...texts)).toBe(false);
  }
  for (let texts of AAWA2_no_branin) {
    expect(hasPlotImmediately(...texts)).toBe(true);
  }
  // Reselect task.
  userEvent.click(inputTaskBranin);
  expect(inputTaskBranin.checked).toBe(true);
  await sleep(100);
  for (let texts of AAWA2_all) {
    expect(hasPlotImmediately(...texts)).toBe(true);
  }
});

test('Test (de)select algorithms', async () => {
  render(<App />, { wrapper: MemoryRouter });

  // Switch to benchmarks page
  const menu = await screen.findByTitle(/Go to benchmarks visualizations/);
  fireEvent.click(menu);
  await sleep(1000);

  expect(await screen.findByText(/No benchmark selected/)).toBeInTheDocument();
  // Get benchmark search field
  const benchmarkField = await screen.findByPlaceholderText(
    'Search a benchmark ...'
  );
  expect(benchmarkField).toBeInTheDocument();
  // Select all_assessments_webapi_2
  userEvent.type(benchmarkField, 'all_asses');
  userEvent.keyboard('{enter}');
  expect(benchmarkField.value).toBe('all_assessments_webapi_2');

  // Make sure all plots are there (10 plots)
  for (let texts of AAWA2_all) {
    await lookupPlot(...texts);
  }

  // Select 1 algorithm.
  const inputAlgorithmRandom = document.getElementById('algorithm-0');
  expect(inputAlgorithmRandom).toBeInTheDocument();
  expect(inputAlgorithmRandom.checked).toBe(true);

  // Deselect algorithm.
  userEvent.click(inputAlgorithmRandom);
  expect(inputAlgorithmRandom.checked).toBe(false);
  await sleep(100);
  for (let texts of AAWA2_all) {
    expect(hasPlotImmediately(...texts)).toBe(false);
  }
  for (let textsNoRandom of AAWA2_no_random) {
    expect(hasPlotImmediately(...textsNoRandom)).toBe(true);
  }
  // Reselect algorithm.
  userEvent.click(inputAlgorithmRandom);
  expect(inputAlgorithmRandom.checked).toBe(true);
  await sleep(100);
  for (let texts of AAWA2_all) {
    expect(hasPlotImmediately(...texts)).toBe(true);
  }
});
