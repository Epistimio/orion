import React from 'react';
import App from '../App';
import {
  render,
  screen,
  queryByText,
  queryByTitle,
  fireEvent,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
/* Use MemoryRouter to isolate history for each test */
import { MemoryRouter } from 'react-router-dom';

async function waitForExperimentToBeLoaded(experiment) {
  /**  Wait for trials table to be loaded for given experiment name. */
  // Check if trials are loaded
  const regex = new RegExp(`Experiment Trials for "${experiment}"`);
  expect(
    await screen.findByTitle(regex, {}, global.CONFIG_WAIT_FOR_LONG)
  ).toBeInTheDocument();
}

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

test('Test if experiment trials are loaded', async () => {
  const user = userEvent.setup();
  // Load page
  render(<App />, { wrapper: MemoryRouter });
  const experiment = await screen.findByText(
    /2-dim-shape-exp/,
    {},
    global.CONFIG_WAIT_FOR_LONG
  );
  expect(experiment).toBeInTheDocument();

  // Switch to database page
  const menu = screen.queryByTitle(/Go to experiments database/);
  await user.click(menu);
  expect(
    await screen.findByText(
      /No trials to display, please select an experiment\./
    )
  ).toBeInTheDocument();

  // Select an experiment
  expect(experiment).toBeInTheDocument();
  await user.click(experiment);

  // Check if trials are loaded
  await waitForExperimentToBeLoaded('2-dim-shape-exp');
  expect(
    screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();

  // Unselect experiment
  const row = screen.queryByTitle(/unselect experiment '2-dim-shape-exp'/);
  expect(row).toBeInTheDocument();
  expect(row.tagName.toLowerCase()).toBe('label');
  await user.click(row);
  expect(
    await screen.findByText(
      /No trials to display, please select an experiment\./
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(
      /Experiment Trials for "2-dim-shape-exp"/,
      {},
      global.CONFIG_WAIT_FOR_LONG
    )
  ).toBeNull();
  expect(screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)).toBeNull();

  // re-select experiment and check if trials are loaded
  await user.click(experiment);
  expect(
    await screen.findByTitle(
      /Experiment Trials for "2-dim-shape-exp"/,
      {},
      global.CONFIG_WAIT_FOR_LONG
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();

  // Select another experiment and check if trials are loaded
  const anotherExperiment = await screen.findByText(/tpe-rosenbrock/);
  expect(anotherExperiment).toBeInTheDocument();
  await user.click(anotherExperiment);
  expect(
    await screen.findByTitle(
      /Experiment Trials for "tpe-rosenbrock"/,
      {},
      global.CONFIG_WAIT_FOR_LONG
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/Experiment Trials for "tpe-rosenbrock"/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/15f4ed436861d25de9be04db9837a70c/)
  ).toBeInTheDocument();
});

test('Test pagination - select items per page', async () => {
  const user = userEvent.setup();
  // Load page
  render(<App />, { wrapper: MemoryRouter });
  // Switch to database page
  await user.click(screen.queryByTitle(/Go to experiments database/));
  // Select an experiment
  await user.click(
    await screen.findByText(/2-dim-shape-exp/, {}, global.CONFIG_WAIT_FOR_LONG)
  );
  await waitForExperimentToBeLoaded('2-dim-shape-exp');
  // Items per page is 10 by default. Check expected trials.
  expect(
    screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/13c04ed294010cecf4491b84837d8402/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/1dec3f2f7b72bc707500258d829a7762/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/6479b23d62db27f4563295e68f7aefe1/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
  ).toBeInTheDocument();
  // Change items per page to 5 and check expected trials.
  const selectItemsPerPage = document.getElementById(
    'bx-pagination-select-trials-pagination'
  );
  expect(selectItemsPerPage).toBeInTheDocument();
  await user.selectOptions(selectItemsPerPage, '5');
  expect(
    screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/13c04ed294010cecf4491b84837d8402/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/1dec3f2f7b72bc707500258d829a7762/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
  ).toBeInTheDocument();
  expect(screen.queryByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)).toBeNull();
  expect(screen.queryByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)).toBeNull();
  expect(screen.queryByTitle(/6479b23d62db27f4563295e68f7aefe1/)).toBeNull();
  expect(screen.queryByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)).toBeNull();
  expect(screen.queryByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)).toBeNull();
});

test('Test pagination - change page', async () => {
  const user = userEvent.setup();
  // Load page
  render(<App />, { wrapper: MemoryRouter });
  // Switch to database page
  await user.click(screen.queryByTitle(/Go to experiments database/));
  // Select an experiment
  await user.click(
    await screen.findByText(/2-dim-shape-exp/, {}, global.CONFIG_WAIT_FOR_LONG)
  );
  await waitForExperimentToBeLoaded('2-dim-shape-exp');
  // We are in first page by default. Check expected trials.
  expect(
    screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/13c04ed294010cecf4491b84837d8402/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/1dec3f2f7b72bc707500258d829a7762/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/6479b23d62db27f4563295e68f7aefe1/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
  ).toBeInTheDocument();
  // Select 2nd page and check expected trials.
  const selectPage = document.getElementById(
    'bx-pagination-select-trials-pagination-right'
  );
  expect(selectPage).toBeInTheDocument();
  await user.selectOptions(selectPage, '2');
  expect(screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)).toBeNull();
  expect(screen.queryByTitle(/13c04ed294010cecf4491b84837d8402/)).toBeNull();
  expect(screen.queryByTitle(/1dec3f2f7b72bc707500258d829a7762/)).toBeNull();
  expect(screen.queryByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)).toBeNull();
  expect(screen.queryByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)).toBeNull();
  expect(screen.queryByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)).toBeNull();
  expect(screen.queryByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)).toBeNull();
  expect(screen.queryByTitle(/6479b23d62db27f4563295e68f7aefe1/)).toBeNull();
  expect(screen.queryByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)).toBeNull();
  expect(screen.queryByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)).toBeNull();
  expect(
    screen.queryByTitle(/b551c6ff4c4d816cdf93b844007eb707/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/c315d0d996290d5d5342cfce3e6d6c9e/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/d2bc2590825ca06cb88e4c54c1142530/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/d669de51fe55d524decf50bf5f5819df/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/f1f350224ae041550658149b55f6c72a/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/f584840e70e38f0cd0cfc4ff1b0e5f2b/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/fac3d17812d82ebd17bd771eae2802bb/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/fced71d7a9bc1b4fe7c0a4029fe73875/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/fd5104909823804b299548acbd089ca6/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/ff3adfecab4d01a5e8cb1550cc74b695/)
  ).toBeInTheDocument();
  // Click to previous page button and check expected trials.
  const buttonsPreviousPage = document.getElementsByClassName(
    'bx--pagination__button--backward'
  );
  expect(buttonsPreviousPage).toHaveLength(1);
  await user.click(buttonsPreviousPage[0]);
  expect(
    screen.queryByTitle(/0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/13c04ed294010cecf4491b84837d8402/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/1dec3f2f7b72bc707500258d829a7762/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/227a7b2e5e9520d577b4c69c64a212c0/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/4262e3b56f7974e46c5ff5d40c4dc1a6/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/582ba78a94a7fbc3e632a0fc40dc99eb/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/5fa4a08bdbafd9a9b57753569b369c62/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/6479b23d62db27f4563295e68f7aefe1/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/85aa9dcbf825d3fcc90ca11c01fe90e4/)
  ).toBeInTheDocument();
  expect(
    screen.queryByTitle(/a7d1729852ebc0bebdbc2db1e9396fc1/)
  ).toBeInTheDocument();
  expect(screen.queryByTitle(/b551c6ff4c4d816cdf93b844007eb707/)).toBeNull();
  expect(screen.queryByTitle(/c315d0d996290d5d5342cfce3e6d6c9e/)).toBeNull();
  expect(screen.queryByTitle(/d2bc2590825ca06cb88e4c54c1142530/)).toBeNull();
  expect(screen.queryByTitle(/d669de51fe55d524decf50bf5f5819df/)).toBeNull();
  expect(screen.queryByTitle(/f1f350224ae041550658149b55f6c72a/)).toBeNull();
  expect(screen.queryByTitle(/f584840e70e38f0cd0cfc4ff1b0e5f2b/)).toBeNull();
  expect(screen.queryByTitle(/fac3d17812d82ebd17bd771eae2802bb/)).toBeNull();
  expect(screen.queryByTitle(/fced71d7a9bc1b4fe7c0a4029fe73875/)).toBeNull();
  expect(screen.queryByTitle(/fd5104909823804b299548acbd089ca6/)).toBeNull();
  expect(screen.queryByTitle(/ff3adfecab4d01a5e8cb1550cc74b695/)).toBeNull();
});

test('Test (de)select columns', async () => {
  const user = userEvent.setup();

  // Load page
  const { container } = render(<App />, { wrapper: MemoryRouter });

  // Switch to database page
  await user.click(screen.queryByTitle(/Go to experiments database/));

  // Select an experiment
  await user.click(
    await screen.findByText(/2-dim-shape-exp/, {}, global.CONFIG_WAIT_FOR_LONG)
  );
  await waitForExperimentToBeLoaded('2-dim-shape-exp');

  // Define values for first row
  const valID = /0f886905874af10a6db412885341ae0b/;
  const valSubmitTime = '2019-11-19 15:41:16.985000';
  const valStartTime = '2019-11-19 15:41:16.996000';
  const valEndTime = '2019-11-19 21:58:02.820000';
  const valObjective = '-0.7881121864177159';

  // Check first row for ID, submit time, start time, end time and objective
  expect(screen.queryByTitle(valID)).toBeInTheDocument();
  expect(queryByText(container, valSubmitTime)).toBeInTheDocument();
  expect(queryByText(container, valStartTime)).toBeInTheDocument();
  expect(queryByText(container, valEndTime)).toBeInTheDocument();
  expect(queryByText(container, valObjective)).toBeInTheDocument();

  // Locate options
  const multiSelect = document.getElementById('multiselect-columns');
  await user.click(multiSelect.querySelector('button.bx--list-box__field'));
  const optionID = queryByText(multiSelect, 'ID');
  const optionSubmitTime = queryByText(multiSelect, 'Submit time');
  const optionStartTime = queryByText(multiSelect, 'Start time');
  const optionEndTime = queryByText(multiSelect, 'End time');
  const optionObjective = queryByText(multiSelect, 'Objective');
  const optionSelectAll = queryByText(multiSelect, '(select all)');
  expect(optionID).toBeInTheDocument();
  expect(optionSubmitTime).toBeInTheDocument();
  expect(optionStartTime).toBeInTheDocument();
  expect(optionEndTime).toBeInTheDocument();
  expect(optionObjective).toBeInTheDocument();
  expect(optionSelectAll).toBeInTheDocument();

  // Deselect column Submit time and check first row
  await user.click(optionSubmitTime);
  expect(screen.queryByTitle(valID)).toBeInTheDocument();
  expect(queryByText(container, valSubmitTime)).toBeNull();
  expect(queryByText(container, valStartTime)).toBeInTheDocument();
  expect(queryByText(container, valEndTime)).toBeInTheDocument();
  expect(queryByText(container, valObjective)).toBeInTheDocument();

  // Deselect column objective and check first row
  await user.click(optionObjective);
  expect(screen.queryByTitle(valID)).toBeInTheDocument();
  expect(queryByText(container, valSubmitTime)).toBeNull();
  expect(queryByText(container, valStartTime)).toBeInTheDocument();
  expect(queryByText(container, valEndTime)).toBeInTheDocument();
  expect(queryByText(container, valObjective)).toBeNull();

  // Deselect columns ID, start time, end time, and check first row
  await user.click(optionID);
  await user.click(optionStartTime);
  await user.click(optionEndTime);
  expect(screen.queryByTitle(valID)).toBeNull();
  expect(queryByText(container, valSubmitTime)).toBeNull();
  expect(queryByText(container, valStartTime)).toBeNull();
  expect(queryByText(container, valEndTime)).toBeNull();
  expect(queryByText(container, valObjective)).toBeNull();

  // Click to 'select all' and check that all columns are now visible in first row
  await user.click(optionSelectAll);
  expect(screen.queryByTitle(valID)).toBeInTheDocument();
  expect(queryByText(container, valSubmitTime)).toBeInTheDocument();
  expect(queryByText(container, valStartTime)).toBeInTheDocument();
  expect(queryByText(container, valEndTime)).toBeInTheDocument();
  expect(queryByText(container, valObjective)).toBeInTheDocument();
});

test('Test sort columns', async () => {
  const user = userEvent.setup();
  // Load page
  render(<App />, { wrapper: MemoryRouter });
  // Switch to database page
  await user.click(screen.queryByTitle(/Go to experiments database/));
  // Select an experiment
  await user.click(
    await screen.findByText(/2-dim-shape-exp/, {}, global.CONFIG_WAIT_FOR_LONG)
  );
  await waitForExperimentToBeLoaded('2-dim-shape-exp');
  // Get sort button from ID column header
  // ID column header is first column from second tr element in table
  // (first tr contains Parameters column and placeholders)
  const sortButton = document
    .querySelectorAll('.bx--data-table-content thead tr')[1]
    .querySelector('th button.bx--table-sort');
  // Click once to activate sorting (sort ascending)
  await user.click(sortButton);
  // Click again to sort descending
  await user.click(sortButton);
  // Check expected rows
  let rows = document.querySelectorAll('.bx--data-table-content tbody tr');
  expect(rows).toHaveLength(10);
  expect(
    queryByTitle(rows[0], /ff3adfecab4d01a5e8cb1550cc74b695/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[1], /fd5104909823804b299548acbd089ca6/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[2], /fced71d7a9bc1b4fe7c0a4029fe73875/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[3], /fac3d17812d82ebd17bd771eae2802bb/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[4], /f584840e70e38f0cd0cfc4ff1b0e5f2b/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[5], /f1f350224ae041550658149b55f6c72a/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[6], /d669de51fe55d524decf50bf5f5819df/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[7], /d2bc2590825ca06cb88e4c54c1142530/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[8], /c315d0d996290d5d5342cfce3e6d6c9e/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[9], /b551c6ff4c4d816cdf93b844007eb707/)
  ).toBeInTheDocument();
  // Click again to deactivate sorting (back to default order)
  await user.click(sortButton);
  // Click again to sort ascending
  await user.click(sortButton);
  // Check expected rows
  rows = document.querySelectorAll('.bx--data-table-content tbody tr');
  expect(rows).toHaveLength(10);
  expect(
    queryByTitle(rows[0], /0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[1], /13c04ed294010cecf4491b84837d8402/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[2], /1dec3f2f7b72bc707500258d829a7762/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[3], /227a7b2e5e9520d577b4c69c64a212c0/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[4], /4262e3b56f7974e46c5ff5d40c4dc1a6/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[5], /582ba78a94a7fbc3e632a0fc40dc99eb/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[6], /5fa4a08bdbafd9a9b57753569b369c62/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[7], /6479b23d62db27f4563295e68f7aefe1/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[8], /85aa9dcbf825d3fcc90ca11c01fe90e4/)
  ).toBeInTheDocument();
  expect(
    queryByTitle(rows[9], /a7d1729852ebc0bebdbc2db1e9396fc1/)
  ).toBeInTheDocument();
});

test('Test drag-and-drop columns', async () => {
  const user = userEvent.setup();
  // Load page
  render(<App />, { wrapper: MemoryRouter });
  // Switch to database page
  await user.click(screen.queryByTitle(/Go to experiments database/));
  // Select an experiment
  await user.click(
    await screen.findByText(/2-dim-shape-exp/, {}, global.CONFIG_WAIT_FOR_LONG)
  );
  await waitForExperimentToBeLoaded('2-dim-shape-exp');
  // Get column ID to drag.
  const draggableColumnID = document
    .querySelectorAll('.bx--data-table-content thead tr')[1]
    .querySelector('th .header-dnd');
  // Get column /dropout to drop into.
  const droppableColumnDropout = document
    .querySelectorAll('.bx--data-table-content thead tr')[1]
    .querySelectorAll('th')[1];
  // Check default first row.
  // ID in first column, /dropout in second column.
  let firstRowCols = document.querySelectorAll(
    '.bx--data-table-content tbody tr td'
  );
  expect(
    queryByTitle(firstRowCols[0], /0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();
  expect(queryByText(firstRowCols[1], '0.2')).toBeInTheDocument();
  // Drag-and-drop column ID to column /dropout.
  fireEvent.dragStart(draggableColumnID);
  fireEvent.dragEnter(droppableColumnDropout);
  fireEvent.dragOver(droppableColumnDropout);
  fireEvent.drop(droppableColumnDropout);
  // Check first row after drag-and-drop.
  // /dropout in first column, ID in second column.
  firstRowCols = document.querySelectorAll(
    '.bx--data-table-content tbody tr td'
  );
  expect(
    queryByTitle(firstRowCols[0], /0f886905874af10a6db412885341ae0b/)
  ).toBeNull();
  expect(queryByText(firstRowCols[1], '0.2')).toBeNull();
  expect(queryByText(firstRowCols[0], '0.2')).toBeInTheDocument();
  expect(
    queryByTitle(firstRowCols[1], /0f886905874af10a6db412885341ae0b/)
  ).toBeInTheDocument();
});

test('Test display trial info in a dialog', async () => {
  const user = userEvent.setup();
  // Load page
  render(<App />, { wrapper: MemoryRouter });
  // Switch to database page
  await user.click(screen.queryByTitle(/Go to experiments database/));
  // Select an experiment
  await user.click(
    await screen.findByText(/2-dim-shape-exp/, {}, global.CONFIG_WAIT_FOR_LONG)
  );
  await waitForExperimentToBeLoaded('2-dim-shape-exp');
  const trial = screen.queryByTitle(/0f886905874af10a6db412885341ae0b/);
  expect(trial).toBeInTheDocument();
  // Dialogs are pre-rendered but not visible.
  // Check if there is no dialog visible.
  expect(document.getElementsByClassName('bx--modal is-visible')).toHaveLength(
    0
  );
  // Click on trial.
  await user.click(trial);
  // Check if a dialog is visible now.
  const dialogs = document.getElementsByClassName('bx--modal is-visible');
  expect(dialogs).toHaveLength(1);
  // Check if visible dialog is the one related to this trial.
  const dialog = dialogs[0];
  const dialogTitle = queryByText(dialog, /Trial info/);
  const dialogHeader = queryByText(
    dialog,
    /2-dim-shape-exp \/ 0f886905874af10a6db412885341ae0b/
  );
  expect(dialogTitle).toBeInTheDocument();
  expect(dialogHeader).toBeInTheDocument();
  // Close dialog
  const closButtons = dialog.getElementsByClassName('bx--modal-close');
  expect(closButtons).toHaveLength(1);
  await user.click(closButtons[0]);
  // Check if there is no dialog visible.
  expect(document.getElementsByClassName('bx--modal is-visible')).toHaveLength(
    0
  );
});
