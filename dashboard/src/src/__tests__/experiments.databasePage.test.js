import { test, expect } from '@playwright/test';
import { StatusToProgress } from '../experiments/components/ExperimentStatusBar/ExperimentStatusBar';

const PROGRESS_BAR_NAMES = [
  'success',
  'suspended',
  'warning',
  'danger',
  'info',
];

async function waitForExperimentToBeLoaded(page, experiment) {
  /**  Wait for trials table to be loaded for given experiment name. */
  // Check if trials are loaded
  const regex = new RegExp(`Experiment Trials for "${experiment}"`);
  const exp = await page.getByTitle(regex);
  await exp.waitFor();
  await expect(exp).toHaveCount(1);
}

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
 * @param bar - progress bar locator
 * @param statusDescendingOrder {Array} - list of expected bar names
 * (success, suspended, warning, danger or info).
 * Expected bar names should have non-null width.
 * Order of bar names should be from the longest to the smallest bar.
 * @returns {Promise<void>}
 */
async function checkNormalBars(bar, statusDescendingOrder) {
  const barWidths = {};
  const expectedNUllBars = [];
  for (let barStatus of PROGRESS_BAR_NAMES) {
    const subBar = await bar.locator(`.bg-${barStatus}`);
    await expect(subBar).toHaveCount(1);
    // Collect bar width.
    barWidths[barStatus] = (await subBar.boundingBox()).width;
    // Collect if bar must have width 0.
    if (statusDescendingOrder.indexOf(barStatus) < 0)
      expectedNUllBars.push(barStatus);
  }
  console.log(
    `Testing bars: expected: ${statusDescendingOrder.join(' >= ') ||
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

test.describe('Test trials table in database page', () => {
  test.beforeEach(async ({ page }) => {
    // Set a hardcoded page size.
    await page.setViewportSize({ width: 1920, height: 1080 });
    // Open Dashboard page.
    await page.goto('localhost:3000');
    // Switch to database page
    const menuExperiments = await page.locator('nav > ul > li:nth-child(1)');
    await menuExperiments.click();
    const menu = await menuExperiments.getByTitle(/Go to experiments database/);
    await menu.click();
    await expect(
      await page.getByText(
        /No trials to display, please select an experiment\./
      )
    ).toHaveCount(1);
  });

  test('Test if experiment trials are loaded', async ({ page }) => {
    const experiment = await page.getByText(/2-dim-shape-exp/);
    await experiment.waitFor();
    await expect(experiment).toHaveCount(1);

    // Select an experiment
    await experiment.click();
    await expect(
      await page.getByText(
        `Loading trials for experiment "2-dim-shape-exp" ...`
      )
    ).toHaveCount(1);

    // Check if trials are loaded
    await waitForExperimentToBeLoaded(page, '2-dim-shape-exp');
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);

    // Unselect experiment
    const row = await page.getByTitle(/unselect experiment '2-dim-shape-exp'/);
    await row.click();

    await expect(
      await page.getByText(
        /No trials to display, please select an experiment\./
      )
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/Experiment Trials for "2-dim-shape-exp"/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(0);

    // re-select experiment and check if trials are loaded
    await experiment.click();
    await expect(
      await page.getByTitle(/Experiment Trials for "2-dim-shape-exp"/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);

    // Select another experiment and check if trials are loaded
    const searchField = await page.getByPlaceholder('Search experiment');
    await searchField.type('tpe-rosenbrock');
    const anotherExperiment = await page.getByText(/tpe-rosenbrock/);
    await expect(anotherExperiment).toHaveCount(1);
    await anotherExperiment.click();
    await waitForExperimentToBeLoaded(page, 'tpe-rosenbrock');
    await expect(
      await page.getByTitle(/15f4ed436861d25de9be04db9837a70c/)
    ).toHaveCount(1);
  });

  test('Test pagination - select items per page', async ({ page }) => {
    // Select an experiment
    const experiment = await page.getByText(/2-dim-shape-exp/);
    await experiment.waitFor();
    await experiment.click();
    await waitForExperimentToBeLoaded(page, '2-dim-shape-exp');
    // Items per page is 10 by default. Check expected trials.
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/13c04ed294010cecf4491b84837d8402/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/1dec3f2f7b72bc707500258d829a7762/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/6479b23d62db27f4563295e68f7aefe1/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
    ).toHaveCount(1);
    // Change items per page to 5 and check expected trials.
    const selectItemsPerPage = await page.locator(
      '#bx-pagination-select-trials-pagination'
    );
    await expect(selectItemsPerPage).toHaveCount(1);
    await selectItemsPerPage.selectOption('5');
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/13c04ed294010cecf4491b84837d8402/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/1dec3f2f7b72bc707500258d829a7762/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/6479b23d62db27f4563295e68f7aefe1/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
    ).toHaveCount(0);
  });

  test('Test pagination - change page', async ({ page }) => {
    // Select an experiment
    const experiment = await page.getByText(/2-dim-shape-exp/);
    await experiment.waitFor();
    await experiment.click();
    await waitForExperimentToBeLoaded(page, '2-dim-shape-exp');
    // We are in first page by default. Check expected trials.
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/13c04ed294010cecf4491b84837d8402/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/1dec3f2f7b72bc707500258d829a7762/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/6479b23d62db27f4563295e68f7aefe1/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
    ).toHaveCount(1);
    // Select 2nd page and check expected trials.
    const selectPage = await page.locator(
      '#bx-pagination-select-trials-pagination-right'
    );
    await expect(selectPage).toHaveCount(1);
    await selectPage.selectOption('2');
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/13c04ed294010cecf4491b84837d8402/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/1dec3f2f7b72bc707500258d829a7762/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/6479b23d62db27f4563295e68f7aefe1/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/b551c6ff4c4d816cdf93b844007eb707/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/c315d0d996290d5d5342cfce3e6d6c9e/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/d2bc2590825ca06cb88e4c54c1142530/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/d669de51fe55d524decf50bf5f5819df/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/f1f350224ae041550658149b55f6c72a/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/f584840e70e38f0cd0cfc4ff1b0e5f2b/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/fac3d17812d82ebd17bd771eae2802bb/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/fced71d7a9bc1b4fe7c0a4029fe73875/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/fd5104909823804b299548acbd089ca6/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/ff3adfecab4d01a5e8cb1550cc74b695/)
    ).toHaveCount(1);
    // Click to previous page button and check expected trials.
    const buttonsPreviousPage = await page.locator(
      '.bx--pagination__button--backward'
    );
    await expect(buttonsPreviousPage).toHaveCount(1);
    await buttonsPreviousPage.click();
    await expect(
      await page.getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/13c04ed294010cecf4491b84837d8402/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/1dec3f2f7b72bc707500258d829a7762/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/6479b23d62db27f4563295e68f7aefe1/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/b551c6ff4c4d816cdf93b844007eb707/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/c315d0d996290d5d5342cfce3e6d6c9e/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/d2bc2590825ca06cb88e4c54c1142530/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/d669de51fe55d524decf50bf5f5819df/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/f1f350224ae041550658149b55f6c72a/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/f584840e70e38f0cd0cfc4ff1b0e5f2b/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/fac3d17812d82ebd17bd771eae2802bb/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/fced71d7a9bc1b4fe7c0a4029fe73875/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/fd5104909823804b299548acbd089ca6/)
    ).toHaveCount(0);
    await expect(
      await page.getByTitle(/ff3adfecab4d01a5e8cb1550cc74b695/)
    ).toHaveCount(0);
  });

  test('Test (de)select columns', async ({ page }) => {
    // Select an experiment
    const experiment = await page.getByText(/2-dim-shape-exp/);
    await experiment.waitFor();
    await experiment.click();
    await waitForExperimentToBeLoaded(page, '2-dim-shape-exp');

    // Define values for first row
    const valID = /0f886905874af10a6db412885341ae0b/;
    const valSubmitTime = '2019-11-19 15:41:16.985000';
    const valStartTime = '2019-11-19 15:41:16.996000';
    const valEndTime = '2019-11-19 21:58:02.820000';
    // Next values must be used with PickledDB
    // const valSubmitTime = '2019-11-19 15:41:16.985075';
    // const valStartTime = '2019-11-19 15:41:16.996319';
    // const valEndTime = '2019-11-19 21:58:02.820319';
    const valStatus = 'completed';
    const valObjective = '-0.7881121864177159';

    const table = await page.locator('.bx--data-table-container');
    await expect(table).toHaveCount(1);

    // Check first row
    await expect(await table.getByTitle(valID)).toHaveCount(1);
    await expect(await table.getByText(valSubmitTime)).toHaveCount(1);
    await expect(await table.getByText(valStartTime)).toHaveCount(1);
    await expect(await table.getByText(valEndTime)).toHaveCount(1);
    // NB: All trials have status 'completed', so queryByText() will complain
    // returning many elements, so we'd better use queryAllByText().
    await expect(await table.getByText(valStatus)).toHaveCount(10);
    await expect(await table.getByText(valObjective)).toHaveCount(1);

    // Locate options
    const multiSelect = await page.locator('#multiselect-columns');
    await (await multiSelect.locator('button.bx--list-box__field')).click();
    const optionID = await multiSelect.getByText('ID');
    const optionSubmitTime = await multiSelect.getByText('Submit time');
    const optionStartTime = await multiSelect.getByText('Start time');
    const optionEndTime = await multiSelect.getByText('End time');
    const optionStatus = await multiSelect.getByText('Status');
    const optionObjective = await multiSelect.getByText('Objective');
    const optionSelectAll = await multiSelect.getByText('(select all)');
    await expect(optionID).toHaveCount(1);
    await expect(optionSubmitTime).toHaveCount(1);
    await expect(optionStartTime).toHaveCount(1);
    await expect(optionEndTime).toHaveCount(1);
    await expect(optionStatus).toHaveCount(1);
    await expect(optionObjective).toHaveCount(1);
    await expect(optionSelectAll).toHaveCount(1);

    // Deselect column Submit time and check first row
    await optionSubmitTime.click();
    await expect(await table.getByTitle(valID)).toHaveCount(1);
    await expect(await table.getByText(valSubmitTime)).toHaveCount(0);
    await expect(await table.getByText(valStartTime)).toHaveCount(1);
    await expect(await table.getByText(valEndTime)).toHaveCount(1);
    await expect(await table.getByText(valStatus)).toHaveCount(10);
    await expect(await table.getByText(valObjective)).toHaveCount(1);

    // Deselect column objective and check first row
    await optionObjective.click();
    await expect(await table.getByTitle(valID)).toHaveCount(1);
    await expect(await table.getByText(valSubmitTime)).toHaveCount(0);
    await expect(await table.getByText(valStartTime)).toHaveCount(1);
    await expect(await table.getByText(valEndTime)).toHaveCount(1);
    await expect(await table.getByText(valStatus)).toHaveCount(10);
    await expect(await table.getByText(valObjective)).toHaveCount(0);

    // Deselect columns ID, start time, end time, status, and check first row
    await optionID.click();
    await optionStartTime.click();
    await optionEndTime.click();
    await optionStatus.click();
    await expect(await table.getByTitle(valID)).toHaveCount(0);
    await expect(await table.getByText(valSubmitTime)).toHaveCount(0);
    await expect(await table.getByText(valStartTime)).toHaveCount(0);
    await expect(await table.getByText(valEndTime)).toHaveCount(0);
    await expect(await table.getByText(valStatus)).toHaveCount(0);
    await expect(await table.getByText(valObjective)).toHaveCount(0);

    // Click to 'select all' and check that all columns are now visible in first row
    await optionSelectAll.click();
    await expect(await table.getByTitle(valID)).toHaveCount(1);
    await expect(await table.getByText(valSubmitTime)).toHaveCount(1);
    await expect(await table.getByText(valStartTime)).toHaveCount(1);
    await expect(await table.getByText(valEndTime)).toHaveCount(1);
    await expect(await table.getByText(valStatus)).toHaveCount(10);
    await expect(await table.getByText(valObjective)).toHaveCount(1);
  });

  test('Test sort columns', async ({ page }) => {
    // Select an experiment
    const experiment = await page.getByText(/2-dim-shape-exp/);
    await experiment.waitFor();
    await experiment.click();
    await waitForExperimentToBeLoaded(page, '2-dim-shape-exp');
    // Get sort button from ID column header
    // ID column header is first column from second tr element in table
    // (first tr contains Parameters column and placeholders)
    const sortButton = await page.locator(
      '.bx--data-table-content thead tr:nth-child(2) th:nth-child(1) button.bx--table-sort'
    );
    await expect(sortButton).toHaveCount(1);
    // const sortButton = document
    //   .querySelectorAll('.bx--data-table-content thead tr')[1]
    //   .querySelector('th button.bx--table-sort');

    // Click once to activate sorting (sort ascending)
    await sortButton.click();
    // Click again to sort descending
    await sortButton.click();
    // Check expected rows
    let rows = await page.locator('.bx--data-table-content tbody tr');
    await expect(rows).toHaveCount(10);
    await expect(
      await rows.nth(0).getByTitle(/ff3adfecab4d01a5e8cb1550cc74b695/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(1).getByTitle(/fd5104909823804b299548acbd089ca6/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(2).getByTitle(/fced71d7a9bc1b4fe7c0a4029fe73875/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(3).getByTitle(/fac3d17812d82ebd17bd771eae2802bb/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(4).getByTitle(/f584840e70e38f0cd0cfc4ff1b0e5f2b/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(5).getByTitle(/f1f350224ae041550658149b55f6c72a/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(6).getByTitle(/d669de51fe55d524decf50bf5f5819df/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(7).getByTitle(/d2bc2590825ca06cb88e4c54c1142530/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(8).getByTitle(/c315d0d996290d5d5342cfce3e6d6c9e/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(9).getByTitle(/b551c6ff4c4d816cdf93b844007eb707/)
    ).toHaveCount(1);
    // Click again to deactivate sorting (back to default order)
    await sortButton.click();
    // Click again to sort ascending
    await sortButton.click();
    // Check expected rows
    rows = await page.locator('.bx--data-table-content tbody tr');
    await expect(rows).toHaveCount(10);
    await expect(
      await rows.nth(0).getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(1).getByTitle(/13c04ed294010cecf4491b84837d8402/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(2).getByTitle(/1dec3f2f7b72bc707500258d829a7762/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(3).getByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(4).getByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(5).getByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(6).getByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(7).getByTitle(/6479b23d62db27f4563295e68f7aefe1/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(8).getByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
    ).toHaveCount(1);
    await expect(
      await rows.nth(9).getByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
    ).toHaveCount(1);
  });

  test('Test drag-and-drop columns', async ({ page }) => {
    // Select an experiment
    const experiment = await page.getByText(/2-dim-shape-exp/);
    await experiment.waitFor();
    await experiment.click();
    await waitForExperimentToBeLoaded(page, '2-dim-shape-exp');

    // Get column ID to drag.
    const draggableColumnID = await page.locator(
      '.bx--data-table-content thead tr:nth-child(2) th:nth-child(1) .header-dnd'
    );
    await expect(draggableColumnID).toHaveCount(1);

    // Get column /dropout to drop into.
    const droppableColumnDropout = await page.locator(
      '.bx--data-table-content thead tr:nth-child(2) th:nth-child(2)'
    );
    await expect(droppableColumnDropout).toHaveCount(1);

    // Check default first row.
    // ID in first column, /dropout in second column.
    let firstRowCols = await page.locator(
      '.bx--data-table-content tbody tr:nth-child(1) td'
    );
    await expect(
      await firstRowCols.nth(0).getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);
    await expect(await firstRowCols.nth(1).getByText('0.2')).toHaveCount(1);

    // Drag-and-drop column ID to column /dropout.
    await draggableColumnID.dragTo(droppableColumnDropout);

    // Check first row after drag-and-drop.
    // /dropout in first column, ID in second column.
    firstRowCols = await page.locator(
      '.bx--data-table-content tbody tr:nth-child(1) td'
    );
    await expect(
      await firstRowCols.nth(0).getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(0);
    await expect(await firstRowCols.nth(1).getByText('0.2')).toHaveCount(0);
    await expect(await firstRowCols.nth(0).getByText('0.2')).toHaveCount(1);
    await expect(
      await firstRowCols.nth(1).getByTitle(/0f886905874af10a6db412885341ae0b/)
    ).toHaveCount(1);
  });

  test('Test display trial info in a dialog', async ({ page }) => {
    // Select an experiment
    const experiment = await page.getByText(/2-dim-shape-exp/);
    await experiment.waitFor();
    await experiment.click();
    await waitForExperimentToBeLoaded(page, '2-dim-shape-exp');

    const trial = await page.getByTitle(/0f886905874af10a6db412885341ae0b/);
    await expect(trial).toHaveCount(1);
    // Dialogs are pre-rendered but not visible.

    // Check if there is no dialog visible.
    await expect(await page.locator('.bx--modal.is-visible')).toHaveCount(0);
    // Click on trial.
    await trial.click();
    // Check if a dialog is visible now.
    const dialog = await page.locator('.bx--modal.is-visible');
    await expect(dialog).toHaveCount(1);
    // Check if visible dialog is the one related to this trial.
    const dialogTitle = await dialog.getByText(/Trial info/);
    const dialogHeader = await dialog.getByText(
      /2-dim-shape-exp \/ 0f886905874af10a6db412885341ae0b/
    );
    await expect(dialogTitle).toHaveCount(1);
    await expect(dialogHeader).toHaveCount(1);
    // Close dialog
    const closButton = await dialog.locator('.bx--modal-close');
    await expect(closButton).toHaveCount(1);
    await closButton.click();
    // Check if there is no dialog visible.
    await expect(await page.locator('.bx--modal.is-visible')).toHaveCount(0);
  });
});

test.describe('Test progress bars above trials table in database page', () => {
  test.beforeEach(async ({ page }) => {
    // Set a hardcoded page size.
    await page.setViewportSize({ width: 1920, height: 1080 });
    // Open Dashboard page.
    await page.goto('localhost:3000');
    // Switch to database page
    const menuExperiments = await page.locator('nav > ul > li:nth-child(1)');
    await menuExperiments.click();
    const menu = await menuExperiments.getByTitle(/Go to experiments database/);
    await menu.click();
    await expect(
      await page.getByText(
        /No trials to display, please select an experiment\./
      )
    ).toHaveCount(1);
    // Get nav bar and wait for default experiments to be loaded.
    const navBar = await page.locator('.experiment-navbar');
    const firstExperiment = await navBar.getByText(/2-dim-shape-exp/);
    await firstExperiment.waitFor();
    // Search uncompleted experiments
    const searchField = await page.getByPlaceholder('Search experiment');
    await expect(searchField).toHaveCount(1);
    await searchField.type('uncompleted');
    await (await navBar.getByText('uncompleted_experiment')).waitFor();
  });

  test('Test uncompleted_experiment and all displayed info', async ({
    page,
  }) => {
    const experiment = await page.getByText('uncompleted_experiment');
    await experiment.click();
    await waitForExperimentToBeLoaded(page, 'uncompleted_experiment');

    // Check that additional experiment info are displayed in the page
    const expectedInfo = [
      ['Best trial ID', '1c31bf5fb0d20680a631ae01c2abc92b'],
      ['Best evaluation', '0.9999999999999999'],
      ['Start time', '2000-01-01 10:00:00.123000'],
      ['Finish time', '2000-01-01 10:41:00.123000'],
      ['Trials', '140'],
      ['Max trials', '200'],
    ];
    const infoRows = await page.locator('.experiment-info .bx--row');
    await expect(infoRows).toHaveCount(expectedInfo.length);
    for (let i = 0; i < 0; ++i) {
      const row = infoRows.nth(i);
      const [infoName, infoValue] = expectedInfo[i];
      await expect(row.getByText(infoName)).toHaveCount(1);
      await expect(row.getByText(infoValue)).toHaveCount(1);
    }

    // Check info displayed around the experiment bar
    const elapsedTime = await page.getByText('Elapsed time: 0:41:00');
    const sumOfTrialsTime = await page.getByText('Sum of trials time: 1:20:00');
    const eta = await page.getByText('ETA: 2:44:00 @ ');
    const progress = await page.getByText('Progress: 21 %');
    await expect(elapsedTime).toHaveCount(1);
    await expect(sumOfTrialsTime).toHaveCount(1);
    await expect(eta).toHaveCount(1);
    await expect(progress).toHaveCount(1);

    // Check tooltips
    const tooltipElapsedTime = await elapsedTime.locator('.bx--tooltip__label');
    await expect(tooltipElapsedTime).toHaveCount(1);
    await expect(
      await page.getByText(
        'Time elapsed since the beginning of the HPO execution'
      )
    ).toHaveCount(0);
    await tooltipElapsedTime.click();
    await expect(
      await page.getByText(
        'Time elapsed since the beginning of the HPO execution'
      )
    ).toHaveCount(1);
    await tooltipElapsedTime.click();
    await expect(
      await page.getByText(
        'Time elapsed since the beginning of the HPO execution'
      )
    ).toHaveCount(0);

    const tooltipTrialsTime = await sumOfTrialsTime.locator(
      '.bx--tooltip__label'
    );
    await expect(tooltipTrialsTime).toHaveCount(1);
    await tooltipTrialsTime.click();
    await expect(
      await page.getByText('Sum of trials execution time')
    ).toHaveCount(1);
    await tooltipTrialsTime.click();

    const tooltipETA = await eta.locator('.bx--tooltip__label');
    await expect(tooltipETA).toHaveCount(1);
    await tooltipETA.click();
    await expect(
      await page.getByText('Estimated time for experiment to finish')
    ).toHaveCount(1);
    await tooltipETA.click();

    const tooltipProgress = await progress.locator('.bx--tooltip__label');
    await expect(tooltipProgress).toHaveCount(1);
    await tooltipProgress.click();
    await expect(
      await page.getByText('Experiment progression percentage')
    ).toHaveCount(1);
    await tooltipProgress.click();

    // Check progress bar legend
    const legend = await page.locator('.experiment-legend');
    await expect(await legend.getByText('Completed (40)')).toHaveCount(1);
    await expect(await legend.getByText('Suspended (20)')).toHaveCount(1);
    await expect(await legend.getByText('Interrupted (15)')).toHaveCount(1);
    await expect(await legend.getByText('Broken (10)')).toHaveCount(1);
    await expect(await legend.getByText('Reserved (25)')).toHaveCount(1);

    // Check progress bar itself
    const bar = await page.locator(
      '.database-container .experiment-progress-bar .main-bar .progress'
    );
    await expect(bar).toHaveCount(1);
    await checkNormalBars(bar, [
      StatusToProgress.completed,
      StatusToProgress.reserved,
      StatusToProgress.suspended,
      StatusToProgress.interrupted,
      StatusToProgress.broken,
    ]);

    // Check table filtering
    const subBarCompleted = await bar.locator('.bg-success');
    const subBarSuspended = await bar.locator('.bg-suspended');
    const table = await page.locator('.bx--data-table-content');
    await expect(await subBarCompleted).toHaveCount(1);
    await expect(await subBarSuspended).toHaveCount(1);
    await expect(table).toHaveCount(1);

    // Check default vue
    await expect(await table.getByText('completed')).toHaveCount(5);
    await expect(await table.getByText('reserved')).toHaveCount(2);
    await expect(await table.getByText('new')).toHaveCount(2);
    await expect(await table.getByText('suspended')).toHaveCount(1);

    // Click on completed bar
    await subBarCompleted.click();
    await expect(await table.getByText('completed')).toHaveCount(10);
    await expect(await table.getByText('reserved')).toHaveCount(0);
    await expect(await table.getByText('new')).toHaveCount(0);
    await expect(await table.getByText('suspended')).toHaveCount(0);

    // Re-click to de-select
    await subBarCompleted.click();
    await expect(await table.getByText('completed')).toHaveCount(5);
    await expect(await table.getByText('reserved')).toHaveCount(2);
    await expect(await table.getByText('new')).toHaveCount(2);
    await expect(await table.getByText('suspended')).toHaveCount(1);

    // Click on suspended bar
    await subBarSuspended.click();
    await expect(await table.getByText('completed')).toHaveCount(0);
    await expect(await table.getByText('reserved')).toHaveCount(0);
    await expect(await table.getByText('new')).toHaveCount(0);
    await expect(await table.getByText('suspended')).toHaveCount(10);

    // Check we switch to completed vue if we immediately click to completed bar
    await subBarCompleted.click();
    await expect(await table.getByText('completed')).toHaveCount(10);
    await expect(await table.getByText('reserved')).toHaveCount(0);
    await expect(await table.getByText('new')).toHaveCount(0);
    await expect(await table.getByText('suspended')).toHaveCount(0);
  });

  test('Test uncompleted_max_trials_0', async ({ page }) => {
    const experiment = await page.getByText('uncompleted_max_trials_0');
    await experiment.click();
    await waitForExperimentToBeLoaded(page, 'uncompleted_max_trials_0');

    // Check max trials == 0
    const infoRows = await page.locator('.experiment-info .bx--row');
    const row = await page.locator('.experiment-info .bx--row:nth-child(6)');
    await expect(row.getByText('Max trials')).toHaveCount(1);
    await expect(row.getByText('0')).toHaveCount(1);

    // Check info displayed around the experiment bar
    const eta = await page.getByText('ETA: (unknown)');
    const progress = await page.getByText('Progress: 100 %');
    await expect(eta).toHaveCount(1);
    await expect(progress).toHaveCount(1);

    // Check progress bar itself
    const bar = await page.locator(
      '.database-container .experiment-progress-bar .main-bar .progress'
    );
    await expect(bar).toHaveCount(1);
    await checkNormalBars(bar, [
      StatusToProgress.completed,
      StatusToProgress.reserved,
      StatusToProgress.suspended,
      StatusToProgress.interrupted,
      StatusToProgress.broken,
    ]);
  });

  test('Test uncompleted_max_trials_infinite', async ({ page }) => {
    const experiment = await page.getByText('uncompleted_max_trials_infinite');
    await experiment.click();
    await waitForExperimentToBeLoaded(page, 'uncompleted_max_trials_infinite');

    // Check info displayed around the experiment bar
    const eta = await page.getByText('ETA: (unknown)');
    const progress = await page.getByText('Progress: (unknown)');
    await expect(eta).toHaveCount(1);
    await expect(progress).toHaveCount(1);

    // Check progress bar itself
    const barInfinite = await page.locator(
      '.database-container .experiment-progress-bar .main-bar .progress'
    );
    await expect(barInfinite).toHaveCount(1);
    await expect(await barInfinite.locator('.bg-success')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-suspended')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-warning')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-danger')).toHaveCount(0);
    await expect(await barInfinite.locator('.bg-info')).toHaveCount(0);
    const subBarRunning = await barInfinite.locator('.bg-running');
    await expect(subBarRunning).toHaveCount(1);
    await expect(subBarRunning).toHaveText(/^N\/A$/);
  });

  test('Test uncompleted_max_trials_lt_completed_trials', async ({ page }) => {
    const experiment = await page.getByText(
      'uncompleted_max_trials_lt_completed_trials'
    );
    await experiment.click();
    await waitForExperimentToBeLoaded(
      page,
      'uncompleted_max_trials_lt_completed_trials'
    );

    // Check max trials
    const infoRows = await page.locator('.experiment-info .bx--row');
    const row = await page.locator('.experiment-info .bx--row:nth-child(6)');
    console.log(await row.innerText());
    await expect(row.getByText('Max trials')).toHaveCount(1);
    await expect(row.getByText('10')).toHaveCount(1);

    // Check number of completed
    const legend = await page.locator('.experiment-legend');
    await expect(await legend.getByText('Completed (20)')).toHaveCount(1);

    // Check info displayed around the experiment bar
    const progress = await page.getByText('Progress: 100 %');
    await expect(progress).toHaveCount(1);

    // Check progress bar itself
    const bar = await page.locator(
      '.database-container .experiment-progress-bar .main-bar .progress'
    );
    await expect(bar).toHaveCount(1);
    await checkNormalBars(bar, [
      StatusToProgress.completed,
      StatusToProgress.reserved,
      StatusToProgress.suspended,
      StatusToProgress.interrupted,
      StatusToProgress.broken,
    ]);
  });

  test('Test uncompleted_no_completed_trials', async ({ page }) => {
    const experiment = await page.getByText('uncompleted_no_completed_trials');
    await experiment.click();
    await waitForExperimentToBeLoaded(page, 'uncompleted_no_completed_trials');

    // Check number of completed
    const legend = await page.locator('.experiment-legend');
    await expect(await legend.getByText('Completed (0)')).toHaveCount(1);

    // Check info displayed around the experiment bar
    const elapsedTime = await page.getByText('Elapsed time: 0:00:00');
    const sumOfTrialsTime = await page.getByText('Sum of trials time: 0:00:00');
    const eta = await page.getByText('ETA: ∞');
    const progress = await page.getByText('Progress: 0 %');
    await expect(elapsedTime).toHaveCount(1);
    await expect(sumOfTrialsTime).toHaveCount(1);
    await expect(eta).toHaveCount(1);
    await expect(progress).toHaveCount(1);

    // Check progress bar itself
    const bar = await page.locator(
      '.database-container .experiment-progress-bar .main-bar .progress'
    );
    await expect(bar).toHaveCount(1);
    await checkNormalBars(bar, [
      StatusToProgress.reserved,
      StatusToProgress.suspended,
      StatusToProgress.interrupted,
      StatusToProgress.broken,
    ]);
  });
});
