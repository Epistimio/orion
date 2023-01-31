import { test, expect } from '@playwright/test';

/*
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
*/

/**
 * Return true if given DOM plot element contains all given texts.
 * @param plot - locator of DOM element containing a plot
 * @param texts - texts (strings) to search
 * @return {boolean} - true if plot contains all texts
 */
async function plotHasTexts(plot, texts) {
  for (let text of texts) {
    const textFinder = await plot.getByText(text);
    const count = await textFinder.count();
    if (count === 0) return false;
  }
  return true;
}

/**
 * Check that we find only 1 plot containing all given texts.
 * @param page - page object
 * @param plotId - plot ID
 * @param texts - texts to search
 */
async function lookupPlotById(page, plotId, ...texts) {
  const plot = await page.locator(`#${plotId}`);
  await plot.waitFor({ timeout: 120000 });
  expect(await plotHasTexts(plot, texts)).toBe(true);
}

/**
 * Check immediately (no waiting) that we find a plot
 * @param page - page object
 * @param plotId - plot ID
 * @param texts - texts to search
 * @return {boolean} - true if plot is found
 */
async function hasPlotByIdAndTextsImmediately(page, plotId, ...texts) {
  const plot = await page.locator(`#${plotId}`);
  const count = await plot.count();
  if (count === 0) return false;
  return await plotHasTexts(plot, texts);
}

/**
 * Wait for given amount of time
 * @param milliseconds - time to wait
 */
async function sleep(milliseconds) {
  let value = 0;
  await new Promise(r => setTimeout(r, milliseconds)).then(() => {
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

const AAWA2_all_with_id = [
  [
    'plot-all_assessments_webapi_2-AverageRank-Branin-rankings-random-tpe',
    AAWA2Plots.average_rank_branin,
  ],
  [
    'plot-all_assessments_webapi_2-AverageRank-RosenBrock-rankings-random-tpe',
    AAWA2Plots.average_rank_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-AverageResult-Branin-regrets-random-tpe',
    AAWA2Plots.average_result_branin,
  ],
  [
    'plot-all_assessments_webapi_2-AverageResult-RosenBrock-regrets-random-tpe',
    AAWA2Plots.average_result_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-durations-random-tpe',
    AAWA2Plots.parallel_assessment_time_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-durations-random-tpe',
    AAWA2Plots.parallel_assessment_time_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-parallel_assessment-random-tpe',
    AAWA2Plots.parallel_assessment_pa_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-parallel_assessment-random-tpe',
    AAWA2Plots.parallel_assessment_pa_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-regrets-random-tpe',
    AAWA2Plots.parallel_assessment_regret_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-regrets-random-tpe',
    AAWA2Plots.parallel_assessment_regret_rosenbrock,
  ],
];
const AAWA2_average_rank_with_id = [
  [
    'plot-all_assessments_webapi_2-AverageRank-Branin-rankings-random-tpe',
    AAWA2Plots.average_rank_branin,
  ],
  [
    'plot-all_assessments_webapi_2-AverageRank-RosenBrock-rankings-random-tpe',
    AAWA2Plots.average_rank_rosenbrock,
  ],
];
const AAWA2_no_average_rank_with_id = [
  [
    'plot-all_assessments_webapi_2-AverageResult-Branin-regrets-random-tpe',
    AAWA2Plots.average_result_branin,
  ],
  [
    'plot-all_assessments_webapi_2-AverageResult-RosenBrock-regrets-random-tpe',
    AAWA2Plots.average_result_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-durations-random-tpe',
    AAWA2Plots.parallel_assessment_time_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-durations-random-tpe',
    AAWA2Plots.parallel_assessment_time_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-parallel_assessment-random-tpe',
    AAWA2Plots.parallel_assessment_pa_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-parallel_assessment-random-tpe',
    AAWA2Plots.parallel_assessment_pa_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-regrets-random-tpe',
    AAWA2Plots.parallel_assessment_regret_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-regrets-random-tpe',
    AAWA2Plots.parallel_assessment_regret_rosenbrock,
  ],
];
const AAWA2_branin_with_id = [
  [
    'plot-all_assessments_webapi_2-AverageRank-Branin-rankings-random-tpe',
    AAWA2Plots.average_rank_branin,
  ],
  [
    'plot-all_assessments_webapi_2-AverageResult-Branin-regrets-random-tpe',
    AAWA2Plots.average_result_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-durations-random-tpe',
    AAWA2Plots.parallel_assessment_time_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-parallel_assessment-random-tpe',
    AAWA2Plots.parallel_assessment_pa_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-regrets-random-tpe',
    AAWA2Plots.parallel_assessment_regret_branin,
  ],
];
const AAWA2_no_branin_with_id = [
  [
    'plot-all_assessments_webapi_2-AverageRank-RosenBrock-rankings-random-tpe',
    AAWA2Plots.average_rank_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-AverageResult-RosenBrock-regrets-random-tpe',
    AAWA2Plots.average_result_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-durations-random-tpe',
    AAWA2Plots.parallel_assessment_time_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-parallel_assessment-random-tpe',
    AAWA2Plots.parallel_assessment_pa_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-regrets-random-tpe',
    AAWA2Plots.parallel_assessment_regret_rosenbrock,
  ],
];
const AAWA2_no_random_with_id = [
  [
    'plot-all_assessments_webapi_2-AverageRank-Branin-rankings-tpe',
    AAWA2PlotsNoRandom.average_rank_branin,
  ],
  [
    'plot-all_assessments_webapi_2-AverageRank-RosenBrock-rankings-tpe',
    AAWA2PlotsNoRandom.average_rank_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-AverageResult-Branin-regrets-tpe',
    AAWA2PlotsNoRandom.average_result_branin,
  ],
  [
    'plot-all_assessments_webapi_2-AverageResult-RosenBrock-regrets-tpe',
    AAWA2PlotsNoRandom.average_result_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-durations-tpe',
    AAWA2PlotsNoRandom.parallel_assessment_time_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-durations-tpe',
    AAWA2PlotsNoRandom.parallel_assessment_time_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-parallel_assessment-tpe',
    AAWA2PlotsNoRandom.parallel_assessment_pa_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-parallel_assessment-tpe',
    AAWA2PlotsNoRandom.parallel_assessment_pa_rosenbrock,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-Branin-regrets-tpe',
    AAWA2PlotsNoRandom.parallel_assessment_regret_branin,
  ],
  [
    'plot-all_assessments_webapi_2-ParallelAssessment-RosenBrock-regrets-tpe',
    AAWA2PlotsNoRandom.parallel_assessment_regret_rosenbrock,
  ],
];

function _test() {}

test('Test sleep', async () => {
  let start = performance.now();
  await sleep(1000);
  let end = performance.now();
  let diff = end - start;
  diff = Math.round(diff);
  console.log(diff);
  expect(diff).toBeGreaterThanOrEqual(1000);
  expect(diff).toBeLessThan(2000);

  start = performance.now();
  await sleep(2000);
  end = performance.now();
  diff = end - start;
  // NB: got 1999.5901880000001 on a run.
  diff = Math.round(diff);
  console.log(diff);
  expect(diff).toBeGreaterThanOrEqual(2000);
  expect(diff).toBeLessThan(3000);

  start = performance.now();
  await sleep(20000);
  end = performance.now();
  diff = end - start;
  diff = Math.round(diff);
  console.log(diff);
  expect(diff).toBeGreaterThanOrEqual(20000);
  expect(diff).toBeLessThan(21000);
});

test.describe('Test benchmark dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Set a hardcoded page size.
    await page.setViewportSize({ width: 1920, height: 1080 });
    // Open Dashboard page.
    await page.goto('localhost:3000');
  });

  test('Test select benchmark', async ({ page }) => {
    // Check we are on home page and not benchmarks page
    await expect(await page.getByText(/Landing Page/)).toHaveCount(1);
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(0);

    // Switch to benchmarks page
    const menuBenchmark = await page.locator('nav > ul > li:nth-child(2)');
    await expect(menuBenchmark).toHaveCount(1);
    await expect(menuBenchmark).toBeVisible();
    await menuBenchmark.click();
    const menu = await menuBenchmark.getByTitle(
      /Go to benchmarks visualizations/
    );
    await expect(menu).toHaveCount(1);
    await expect(menu).toBeVisible();
    await menu.click();

    // Check we are in benchmarks page and not to home page anymore
    await expect(await page.getByText(/Landing Page/)).toHaveCount(0);
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(1);

    // Get benchmark search field
    const benchmarkField = await page.getByPlaceholder(
      'Search a benchmark ...'
    );
    await expect(benchmarkField).toHaveCount(1);

    // Select branin_baselines_webapi benchmark
    await benchmarkField.type('branin');
    await benchmarkField.press('Enter');
    expect(await benchmarkField.getAttribute('value')).toBe(
      'branin_baselines_webapi'
    );
    const leftMenu = await page.locator('.bx--structured-list');
    await expect(leftMenu).toHaveCount(1);
    await expect(await leftMenu.getByText(/AverageResult/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/Branin/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/gridsearch/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/random/)).toHaveCount(1);
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(0);
    // Check plot
    await lookupPlotById(
      page,
      'plot-branin_baselines_webapi-AverageResult-Branin-regrets-gridsearch-random',
      'Average Regret',
      'branin',
      'Trials ordered by suggested time'
    );

    // Select all_algos_webapi benchmark
    // Use Ctrl+A then backspace to clear field before typing new hint
    await benchmarkField.press('Control+A');
    await benchmarkField.press('Backspace');
    expect(await benchmarkField.getAttribute('value')).toBe('');
    await benchmarkField.type('all_algos');
    await benchmarkField.press('Enter');
    expect(await benchmarkField.getAttribute('value')).toBe('all_algos_webapi');
    await expect(await leftMenu.getByText(/AverageResult/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/Branin/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/EggHolder/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/RosenBrock/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/gridsearch/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/random/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/tpe/)).toHaveCount(1);
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(0);
    // Check plots
    await lookupPlotById(
      page,
      'plot-all_algos_webapi-AverageResult-Branin-regrets-gridsearch-random-tpe',
      'Average Regret',
      'branin'
    );
    await lookupPlotById(
      page,
      'plot-all_algos_webapi-AverageResult-EggHolder-regrets-gridsearch-random-tpe',
      'Average Regret',
      'eggholder'
    );
    await lookupPlotById(
      page,
      'plot-all_algos_webapi-AverageResult-RosenBrock-regrets-gridsearch-random-tpe',
      'Average Regret',
      'rosenbrock'
    );

    // Select all_assessments_webapi_2
    await benchmarkField.press('Control+A');
    await benchmarkField.press('Backspace');
    expect(await benchmarkField.getAttribute('value')).toBe('');
    await benchmarkField.type('all_asses');
    await benchmarkField.press('Enter');
    expect(await benchmarkField.getAttribute('value')).toBe(
      'all_assessments_webapi_2'
    );
    await expect(await leftMenu.getByText(/AverageRank/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/AverageResult/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/ParallelAssessment/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/Branin/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/RosenBrock/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/random/)).toHaveCount(1);
    await expect(await leftMenu.getByText(/tpe/)).toHaveCount(1);
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(0);
    // Check plots
    for (let [plotId, texts] of AAWA2_all_with_id) {
      await lookupPlotById(page, plotId, ...texts);
    }
  });

  test('Test (de)select assessments', async ({ page }) => {
    // Switch to benchmarks page
    const menuBenchmark = await page.locator('nav > ul > li:nth-child(2)');
    await menuBenchmark.click();
    const menu = await menuBenchmark.getByTitle(
      /Go to benchmarks visualizations/
    );
    await menu.click();
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(1);

    // Get benchmark search field
    const benchmarkField = await page.getByPlaceholder(
      'Search a benchmark ...'
    );
    // Select all_assessments_webapi_2
    await benchmarkField.type('all_asses');
    await benchmarkField.press('Enter');
    expect(await benchmarkField.getAttribute('value')).toBe(
      'all_assessments_webapi_2'
    );

    // Make sure all plots are there (10 plots)
    for (let [plotId, texts] of AAWA2_all_with_id) {
      await lookupPlotById(page, plotId, ...texts);
    }

    // Select 1 assessment.
    const inputAssessmentAverageRank = await page.locator('#assessment-0');
    await expect(inputAssessmentAverageRank).toHaveCount(1);
    await expect(inputAssessmentAverageRank).toBeChecked({ checked: true });

    // Deselect assessment
    await inputAssessmentAverageRank.uncheck({ force: true });
    await expect(inputAssessmentAverageRank).toBeChecked({ checked: false });

    await sleep(1000);
    for (let [plotId, texts] of AAWA2_average_rank_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        false
      );
    }
    for (let [plotId, texts] of AAWA2_no_average_rank_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        true
      );
    }

    // Reselect assessment.
    await inputAssessmentAverageRank.check({ force: true });
    await expect(inputAssessmentAverageRank).toBeChecked({ checked: true });
    await sleep(1000);
    for (let [plotId, texts] of AAWA2_all_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        true
      );
    }
  });

  test('Test (de)select tasks', async ({ page }) => {
    // Switch to benchmarks page
    const menuBenchmark = await page.locator('nav > ul > li:nth-child(2)');
    await menuBenchmark.click();
    const menu = await menuBenchmark.getByTitle(
      /Go to benchmarks visualizations/
    );
    await menu.click();
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(1);

    // Get benchmark search field
    const benchmarkField = await page.getByPlaceholder(
      'Search a benchmark ...'
    );
    // Select all_assessments_webapi_2
    await benchmarkField.type('all_asses');
    await benchmarkField.press('Enter');
    expect(await benchmarkField.getAttribute('value')).toBe(
      'all_assessments_webapi_2'
    );

    // Make sure all plots are there (10 plots)
    for (let [plotId, texts] of AAWA2_all_with_id) {
      await lookupPlotById(page, plotId, ...texts);
    }

    // Select 1 task.
    const inputTaskBranin = await page.locator('#task-0');
    await expect(inputTaskBranin).toHaveCount(1);
    await expect(inputTaskBranin).toBeChecked({ checked: true });

    // Deselect task.
    await inputTaskBranin.uncheck({ force: true });
    await expect(inputTaskBranin).toBeChecked({ checked: false });
    await sleep(1000);
    for (let [plotId, texts] of AAWA2_branin_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        false
      );
    }
    for (let [plotId, texts] of AAWA2_no_branin_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        true
      );
    }
    // Reselect task.
    await inputTaskBranin.check({ force: true });
    await expect(inputTaskBranin).toBeChecked({ checked: true });
    await sleep(1000);
    for (let [plotId, texts] of AAWA2_all_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        true
      );
    }
  });

  test('Test (de)select algorithms', async ({ page }) => {
    // Switch to benchmarks page
    const menuBenchmark = await page.locator('nav > ul > li:nth-child(2)');
    await menuBenchmark.click();
    const menu = await menuBenchmark.getByTitle(
      /Go to benchmarks visualizations/
    );
    await menu.click();
    await expect(await page.getByText(/No benchmark selected/)).toHaveCount(1);

    // Get benchmark search field
    const benchmarkField = await page.getByPlaceholder(
      'Search a benchmark ...'
    );
    // Select all_assessments_webapi_2
    await benchmarkField.type('all_asses');
    await benchmarkField.press('Enter');
    expect(await benchmarkField.getAttribute('value')).toBe(
      'all_assessments_webapi_2'
    );

    // Make sure all plots are there (10 plots)
    for (let [plotId, texts] of AAWA2_all_with_id) {
      await lookupPlotById(page, plotId, ...texts);
    }

    // Select 1 algorithm.
    const inputAlgorithmRandom = await page.locator('#algorithm-0');
    await expect(inputAlgorithmRandom).toHaveCount(1);
    await expect(inputAlgorithmRandom).toBeChecked({ checked: true });

    // Deselect algorithm.
    await inputAlgorithmRandom.uncheck({ force: true });
    await expect(inputAlgorithmRandom).toBeChecked({ checked: false });
    await sleep(1000);
    for (let [plotId, texts] of AAWA2_all_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        false
      );
    }
    for (let [plotId, textsNoRandom] of AAWA2_no_random_with_id) {
      expect(
        await hasPlotByIdAndTextsImmediately(page, plotId, ...textsNoRandom)
      ).toBe(true);
    }
    // Reselect algorithm.
    await inputAlgorithmRandom.check({ force: true });
    await expect(inputAlgorithmRandom).toBeChecked({ checked: true });
    await sleep(1000);
    for (let [plotId, texts] of AAWA2_all_with_id) {
      expect(await hasPlotByIdAndTextsImmediately(page, plotId, ...texts)).toBe(
        true
      );
    }
  });
});
