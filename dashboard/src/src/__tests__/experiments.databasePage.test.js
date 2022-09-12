import React from 'react';
import App from '../App';
import { render, screen } from '@testing-library/react';
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
