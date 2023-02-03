import { test, expect } from '@playwright/test';

async function waitForExperimentToBeLoaded(page, experiment) {
  /**  Wait for trials table to be loaded for given experiment name. */
  // Check if trials are loaded
  const regex = new RegExp(`Experiment Trials for "${experiment}"`);
  const exp = await page.getByTitle(regex);
  await exp.waitFor();
  await expect(exp).toHaveCount(1);
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
    await expect(
      await page.getByTitle(/Experiment Trials for "tpe-rosenbrock"/)
    ).toHaveCount(1);
    await expect(
      await page.getByTitle(/Experiment Trials for "tpe-rosenbrock"/)
    ).toHaveCount(1);
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
