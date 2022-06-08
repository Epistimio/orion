import React from 'react';
import App from '../App';
import { render, fireEvent, screen } from '@testing-library/react';
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
  // Load page
  render(<App />, { wrapper: MemoryRouter });
  const experiment = await screen.findByText(
    /2-dim-shape-exp/,
    {},
    { interval: 1000, timeout: 120000 }
  );
  expect(experiment).toBeInTheDocument();

  // Switch to database page
  const menu = screen.queryByTitle(/Go to experiments database/);
  fireEvent.click(menu);
  expect(
    await screen.findByText(
      /No trials to display, please select an experiment\./
    )
  ).toBeInTheDocument();

  // Select an experiment
  expect(experiment).toBeInTheDocument();
  fireEvent.click(experiment);

  // Check if trials are loaded
  expect(
    await screen.findByText(
      /Experiment Trials for "2-dim-shape-exp"/,
      {},
      { interval: 1000, timeout: 120000 }
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByText(/0915da146c84975df9bdf4c3ee9376dc/)
  ).toBeInTheDocument();

  // Unselect experiment
  const span = screen.queryByTitle(/unselect experiment '2-dim-shape-exp'/);
  expect(span).toBeInTheDocument();
  expect(span.tagName.toLowerCase()).toBe('span');
  fireEvent.click(span);
  expect(
    await screen.findByText(
      /No trials to display, please select an experiment\./
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByText(
      /Experiment Trials for "2-dim-shape-exp"/,
      {},
      { interval: 1000, timeout: 120000 }
    )
  ).toBeNull();
  expect(screen.queryByText(/0915da146c84975df9bdf4c3ee9376dc/)).toBeNull();

  // re-select experiment and check if trials are loaded
  fireEvent.click(experiment);
  expect(
    await screen.findByText(
      /Experiment Trials for "2-dim-shape-exp"/,
      {},
      { interval: 1000, timeout: 120000 }
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByText(/0915da146c84975df9bdf4c3ee9376dc/)
  ).toBeInTheDocument();

  // Select another experiment and check if plots are loaded
  const anotherExperiment = await screen.findByText(/tpe-rosenbrock/);
  expect(anotherExperiment).toBeInTheDocument();
  fireEvent.click(anotherExperiment);
  expect(
    await screen.findByText(
      /Experiment Trials for "tpe-rosenbrock"/,
      {},
      { interval: 1000, timeout: 120000 }
    )
  ).toBeInTheDocument();
  expect(
    screen.queryByText(/20 trial\(s\) for experiment "tpe-rosenbrock"/)
  ).toBeInTheDocument();
  expect(
    screen.queryByText(/081134e84076d4c4aba210e88d0dce81/)
  ).toBeInTheDocument();
});
