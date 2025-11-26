import { test, expect } from '@playwright/test';
import { StatusToProgress } from '../experiments/components/ExperimentStatusBar/ExperimentStatusBar';

const PROGRESS_BAR_NAMES = [
  'success',
  'suspended',
  'warning',
  'danger',
  'info',
];

/**
 * Check if a normal progress bar has expected non-null sub-bars
 * in order given by statusDescendingOrder from longest to smallest.
 *
 * Any sub-bar non-mentioned in statusDescendingOrder
 * is expected to have width zero.
 *
 * A "normal" progress bar is a bar for an experiment
 * whose max_trials is non-infinite.
 *
 * @param row - container to find experiment name and bar
 * @param name {string} - experiment name
 * @param statusDescendingOrder {Array} - list of expected bar names
 * (success, suspended, warning, danger or info).
 * Expected bar names should have non-null width.
 * Order of bar names should be from the longest to the smallest bar.
 * @returns {Promise<void>}
 */
async function checkNormalBars(row, name, statusDescendingOrder) {
  // Check we find experiment name in row.
  await expect(await row.getByText(name)).toHaveCount(1);
  // Get progress bar.
  const bar = await row.locator('.progress');
  const barWidths = {};
  const expectedNUllBars = [];
  for (let barStatus of PROGRESS_BAR_NAMES) {
    const subBar = await bar.locator(`.bg-${barStatus}`);
    // Sub-war may have width 0, thus be considered as non-visible.
    // So, we wait for element to be attached instead of visible.
    await subBar.waitFor({ state: 'attached' });
    await expect(subBar).toHaveCount(1);
    // Collect bar width.
    barWidths[barStatus] = (await subBar.boundingBox()).width;
    // Collect if bar must have width 0.
    if (statusDescendingOrder.indexOf(barStatus) < 0)
      expectedNUllBars.push(barStatus);
  }
  console.log(
    `Testing bars: ${name}\nexpected: ${statusDescendingOrder.join(' >= ') ||
      'no non-null bars'}, null bars: ${expectedNUllBars.join(', ') ||
      'none'}, collected widths: ${PROGRESS_BAR_NAMES.map(
      status => status + ': ' + barWidths[status]
    ).join(', ')}`
  );
  // Check bar widths.
  if (statusDescendingOrder.length) {
    // First expected bar must have width > 0.
    expect(barWidths[statusDescendingOrder[0]]).toBeGreaterThan(0);
    for (let i = 1; i < statusDescendingOrder.length; ++i) {
      // Each following bar must be <= previous.
      expect(barWidths[statusDescendingOrder[i - 1]]).toBeGreaterThanOrEqual(
        barWidths[statusDescendingOrder[i]]
      );
      expect(barWidths[statusDescendingOrder[i]]).toBeGreaterThan(0);
    }
  }
  // Other bars must have width == 0.
  for (let zeroStatus of expectedNUllBars) {
    expect(barWidths[zeroStatus]).toBe(0);
  }
}

test.describe('Test experiment nav bar', () => {
  test.beforeEach(async ({ page }) => {
    // Set a hardcoded page size.
    await page.setViewportSize({ width: 1920, height: 1080 });
    // Open Dashboard page.
    await page.goto('localhost:3000');
  });

  test('Test nav bar scrolling', async ({ page }) => {
    // Get nav bar container and bounding box (x, y, width, height).
    const navBar = await page.locator('.experiment-navbar');
    await expect(navBar).toHaveCount(1);
    const navBarBox = await navBar.boundingBox();

    // Get scrollable container and bounding box inside nav bar.
    const scrollableContainer = await navBar.locator('.experiments-wrapper');
    const scrollableBox = await scrollableContainer.boundingBox();

    // Check default loaded experiments.
    const firstLoadedExperiments = await navBar.locator(
      '.experiments-wrapper .experiment-cell span[title]'
    );
    await firstLoadedExperiments.first().waitFor();
    // For given hardcoded page size, we should have 16 default loaded experiments.
    await expect(firstLoadedExperiments).toHaveCount(16);

    // Get and check first and last of default loaded experiments.
    const currentFirstLoadedExperiment = firstLoadedExperiments.first();
    const currentLastLoadedExperiment = firstLoadedExperiments.last();
    await expect(currentFirstLoadedExperiment).toHaveText('2-dim-exp');
    await expect(currentLastLoadedExperiment).toHaveText(
      'all_algos_webapi_AverageResult_EggHolder_1_2'
    );
    // Get bounding boxes for first and last default loaded experiments.
    const firstBox = await currentFirstLoadedExperiment.boundingBox();
    const lastBox = await currentLastLoadedExperiment.boundingBox();

    // Check some values of collected bounding boxes.
    console.log(navBarBox);
    console.log(scrollableBox);
    console.log(firstBox);
    console.log(lastBox);
    expect(navBarBox.y).toBe(48);
    expect(navBarBox.height).toBe(1032);
    expect(scrollableBox.y).toBe(48);
    expect(scrollableBox.height).toBe(984);
    expect(lastBox.y).toBeGreaterThan(1053);
    /**
     * We expect scrollable container to not be high enough to display
     * all default loaded experiments. So, last default loaded experiment
     * should be positioned after the end of scrollable bounding box
     * vertically.
     */
    expect(scrollableBox.y + scrollableBox.height).toBeLessThan(lastBox.y);

    /**
     * Now, we want to scroll into scrollable container to trigger
     * infinite scroll that should load supplementary experiments.
     * To check that, we prepare a locator for next experiment to be loaded ...
     */
    let nextExperiment = await navBar.getByText(
      'all_algos_webapi_AverageResult_EggHolder_2_1'
    );
    // ... And we don't expect this experiment to be yet in the document.
    await expect(nextExperiment).toHaveCount(0);

    // Then we scroll to the last default loaded experiment.
    await currentLastLoadedExperiment.scrollIntoViewIfNeeded();

    // We wait for next experiment to be loaded to appear.
    await nextExperiment.waitFor();
    // And we check that this newly loaded experiment is indeed in document.
    await expect(nextExperiment).toHaveCount(1);

    // Finally, we check number of loaded experiments after scrolling.
    const newLoadedExperiments = await navBar.locator(
      '.experiments-wrapper .experiment-cell span[title]'
    );
    // For given hardcoded page size, we should not have 18 (16 + 2) experiments.
    await expect(newLoadedExperiments).toHaveCount(18);
  });

  test('Check if experiments are loaded', async ({ page }) => {
    const navBar = await page.locator('.experiment-navbar');
    // Wait for first experiment to appear.
    // This let time for experiments to be loaded.
    const firstExperiment = await navBar.getByText(/2-dim-shape-exp/);
    await firstExperiment.waitFor();
    await expect(firstExperiment).toHaveCount(1);
    // Then, other experiments should be already loaded.
    // NB: Due to scrolling, not all experiments are yet loaded.
    await expect(await navBar.getByText(/4-dim-cat-shape-exp/)).toHaveCount(1);
    await expect(await navBar.getByText(/2-dim-exp/)).toHaveCount(1);
    await expect(await navBar.getByText(/3-dim-cat-shape-exp/)).toHaveCount(1);
    await expect(
      await navBar.getByText(/all_algos_webapi_AverageResult_Branin_0_0/)
    ).toHaveCount(1);
    await expect(
      await navBar.getByText(/all_algos_webapi_AverageResult_Branin_2_1/)
    ).toHaveCount(1);
    await expect(
      await navBar.getByText(/all_algos_webapi_AverageResult_EggHolder_1_2/)
    ).toHaveCount(1);
  });

  test('Check filter experiments with search field', async ({ page }) => {
    const experiments = [
      /2-dim-shape-exp/,
      /4-dim-cat-shape-exp/,
      /2-dim-exp/,
      /3-dim-cat-shape-exp/,
      /random-rosenbrock/,
      /all_algos_webapi_AverageResult_RosenBrock_0_1/,
      /hyperband-cifar10/,
    ];
    const checkExpectations = async (navBar, presences) => {
      for (let i = 0; i < presences.length; ++i) {
        const domElement = await navBar.getByText(experiments[i]);
        await expect(domElement).toHaveCount(presences[i]);
      }
    };

    // Get nav bar and wait for default experiments to be loaded.
    const navBar = await page.locator('.experiment-navbar');
    const firstExperiment = await navBar.getByText(/2-dim-shape-exp/);
    await firstExperiment.waitFor();

    const searchField = await page.getByPlaceholder('Search experiment');
    await expect(searchField).toHaveCount(1);

    let waiter;

    await searchField.type('random');
    await checkExpectations(navBar, [0, 0, 0, 0, 1, 0, 0]);

    await searchField.press('Control+A');
    await searchField.press('Backspace');
    await searchField.type('rosenbrock');
    // NB: random-rosenbrock won't be visible because
    // it's in last loaded experiments, so we need to scroll a lot
    // before seeing it.
    await checkExpectations(navBar, [0, 0, 0, 0, 0, 1, 0]);
    // Scroll until we find random-rosenbrock
    while (true) {
      let loadedExperiments = await navBar.locator(
        '.experiments-wrapper .experiment-cell span[title]'
      );
      await loadedExperiments.first().waitFor();
      await loadedExperiments.last().scrollIntoViewIfNeeded();
      let exp = await navBar.getByText(/random-rosenbrock/);
      if ((await exp.count()) === 1) break;
    }
    // Noe we must find both
    // random-rosenbrock and all_algos_webapi_AverageResult_RosenBrock_0_1
    await checkExpectations(navBar, [0, 0, 0, 0, 1, 1, 0]);

    await searchField.press('Control+A');
    await searchField.press('Backspace');
    await searchField.type('dim-cat');
    await checkExpectations(navBar, [0, 1, 0, 1, 0, 0, 0]);

    await searchField.press('Control+A');
    await searchField.press('Backspace');
    await searchField.type('unknown experiment');
    waiter = await navBar.getByText('No matching experiment');
    await waiter.waitFor();
    await expect(waiter).toHaveCount(1);
    await checkExpectations(navBar, [0, 0, 0, 0, 0, 0, 0]);
  });

  test('Test small progress bar for experiment 2-dim-shape-exp', async ({
    page,
  }) => {
    const navBar = await page.locator('.experiment-navbar');
    // Wait for first experiment to appear.
    // This let time for experiments to be loaded.
    const firstExperiment = await navBar.getByText(/2-dim-shape-exp/);
    await firstExperiment.waitFor();
    await expect(firstExperiment).toHaveCount(1);

    /** Locate progress bar related to this experiment **/
    // Just check experiment element is indeed a span with experiment name as title
    expect(
      await firstExperiment.evaluate(node => node.tagName.toLowerCase())
    ).toBe('span');
    expect(await firstExperiment.evaluate(node => node.title)).toBe(
      '2-dim-shape-exp'
    );
    // Get experiment row. Span parent is cell, span parent's parent is row
    const parent = await firstExperiment.locator('xpath=../..');
    expect(await parent.getAttribute('class')).toBe('bx--structured-list-row');
    // Get progress bar
    const bar = await parent.locator('.progress');
    await expect(bar).toHaveCount(1);
    // Make sure it's a small progress bar, not complete progress bar.
    // Complete progress bar comes with grids to display supplementary info.
    // Small progress bar does not have grid around.
    // So, we must not find a grid inside experiment row.
    await expect(await parent.locator('.bx--grid')).toHaveCount(0);
    // Check sub-bars in progress bar. Experiment 2-dim-shape-exp should be fully completed.
    // So, only success bar should have a width > 0.
    const barSuccess = await bar.locator('.bg-success');
    const barSuspended = await bar.locator('.bg-suspended');
    const barWarning = await bar.locator('.bg-warning');
    const barDanger = await bar.locator('.bg-danger');
    const barInfo = await bar.locator('.bg-info');
    await expect(barSuccess).toHaveCount(1);
    await expect(barSuspended).toHaveCount(1);
    await expect(barWarning).toHaveCount(1);
    await expect(barDanger).toHaveCount(1);
    await expect(barInfo).toHaveCount(1);
    await expect((await barSuccess.boundingBox()).width).toBeGreaterThan(40);
    await expect((await barSuspended.boundingBox()).width).toBe(0);
    await expect((await barWarning.boundingBox()).width).toBe(0);
    await expect((await barDanger.boundingBox()).width).toBe(0);
    await expect((await barInfo.boundingBox()).width).toBe(0);
  });

  test('Test small progress bar for uncompleted experiments', async ({
    page,
  }) => {
    // Get nav bar and wait for default experiments to be loaded.
    const navBar = await page.locator('.experiment-navbar');
    const firstExperiment = await navBar.getByText(/2-dim-shape-exp/);
    await firstExperiment.waitFor();
    // Search uncompleted experiments
    const searchField = await page.getByPlaceholder('Search experiment');
    await expect(searchField).toHaveCount(1);
    await searchField.type('uncompleted');
    // Check we got expected experiments
    const uncompletedExperiments = await navBar.locator(
      '.bx--structured-list-tbody .bx--structured-list-row'
    );
    await uncompletedExperiments.first().waitFor();
    await expect(uncompletedExperiments).toHaveCount(5);
    const expectedNames = [
      'uncompleted_experiment',
      'uncompleted_max_trials_0',
      'uncompleted_max_trials_infinite',
      'uncompleted_max_trials_lt_completed_trials',
      'uncompleted_no_completed_trials',
    ];
    for (let i = 0; i < expectedNames.length; ++i) {
      const row = uncompletedExperiments.nth(i);
      await expect(await row.getByText(expectedNames[i])).toHaveCount(1);
    }
    // Check expected bars.
    await checkNormalBars(
      uncompletedExperiments.nth(0),
      'uncompleted_experiment',
      [
        StatusToProgress.completed,
        StatusToProgress.reserved,
        StatusToProgress.suspended,
        StatusToProgress.interrupted,
        StatusToProgress.broken,
      ]
    );
    await checkNormalBars(
      uncompletedExperiments.nth(1),
      'uncompleted_max_trials_0',
      [
        StatusToProgress.completed,
        StatusToProgress.reserved,
        StatusToProgress.suspended,
        StatusToProgress.interrupted,
        StatusToProgress.broken,
      ]
    );
    await checkNormalBars(
      uncompletedExperiments.nth(3),
      'uncompleted_max_trials_lt_completed_trials',
      [
        StatusToProgress.completed,
        StatusToProgress.reserved,
        StatusToProgress.suspended,
        StatusToProgress.interrupted,
        StatusToProgress.broken,
      ]
    );
    await checkNormalBars(
      uncompletedExperiments.nth(4),
      'uncompleted_no_completed_trials',
      [
        StatusToProgress.reserved,
        StatusToProgress.suspended,
        StatusToProgress.interrupted,
        StatusToProgress.broken,
      ]
    );
    // Check bar for experiment with max_trials infinite
    const rowInfinite = uncompletedExperiments.nth(2);
    await expect(
      await rowInfinite.getByText('uncompleted_max_trials_infinite')
    ).toHaveCount(1);
    const barInfinite = await rowInfinite.locator('.progress');
    await expect(barInfinite).toHaveCount(1);
    expect(await barInfinite.evaluate(node => node.title)).toBe(
      'N/A (max trials ∞)'
    );
    await expect(await barInfinite.locator('.bg-success')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-suspended')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-warning')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-danger')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-info')).toHaveCount(0);
    const subBarRunning = await barInfinite.locator('.bg-running');
    await expect(subBarRunning).toHaveCount(1);
    await expect(subBarRunning).toHaveText(/^N\/A$/);
  });
});
