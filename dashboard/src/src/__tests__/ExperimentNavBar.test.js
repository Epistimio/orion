import React from 'react';
import ExperimentNavBar from '../experiments/components/ExperimentNavBar';
import { render, waitFor, screen } from '@testing-library/react';
import { BackendContext } from '../experiments/BackendContext';
import userEvent from '@testing-library/user-event';

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

test('Check if experiments are loaded', async () => {
  // ExperimentNavBar will be rendered with default value associated to BackendContext.
  render(<ExperimentNavBar />);
  // `findByText` will wait for text to appear.
  // This let time for ExperimentNavBar to load experiments.
  expect(await screen.findByText(/2-dim-shape-exp/)).toBeInTheDocument();
  // Then, all other experiments should be already loaded.
  expect(screen.queryByText(/4-dim-cat-shape-exp/)).toBeInTheDocument();
  expect(screen.queryByText(/2-dim-exp/)).toBeInTheDocument();
  expect(screen.queryByText(/3-dim-cat-shape-exp/)).toBeInTheDocument();
  expect(screen.queryByText(/random-rosenbrock/)).toBeInTheDocument();
  expect(screen.queryByText(/tpe-rosenbrock/)).toBeInTheDocument();
  expect(screen.queryByText(/hyperband-cifar10/)).toBeInTheDocument();
});

test('Check experiments display when backend call fails', async () => {
  render(
    <BackendContext.Provider value={{ address: 'http://localhost:1' }}>
      <ExperimentNavBar />
    </BackendContext.Provider>
  );
  // `findByText` will wait for text to appear.
  // This let time for ExperimentNavBar to try backend call.
  expect(
    await screen.findByText(/No experiment available/)
  ).toBeInTheDocument();
  // Then, we must not find any of expected experiments.
  expect(screen.queryByText(/2-dim-shape-exp/)).toBeNull();
  expect(screen.queryByText(/4-dim-cat-shape-exp/)).toBeNull();
  expect(screen.queryByText(/2-dim-exp/)).toBeNull();
  expect(screen.queryByText(/3-dim-cat-shape-exp/)).toBeNull();
  expect(screen.queryByText(/random-rosenbrock/)).toBeNull();
  expect(screen.queryByText(/tpe-rosenbrock/)).toBeNull();
  expect(screen.queryByText(/hyperband-cifar10/)).toBeNull();
});

test('Check filter experiments with search field', async () => {
  const experiments = [
    /2-dim-shape-exp/,
    /4-dim-cat-shape-exp/,
    /2-dim-exp/,
    /3-dim-cat-shape-exp/,
    /random-rosenbrock/,
    /tpe-rosenbrock/,
    /hyperband-cifar10/,
  ];
  const checkExpectations = presences => {
    for (let i = 0; i < presences.length; ++i) {
      const shouldBePresent = presences[i];
      const domElement = screen.queryByText(experiments[i]);
      if (shouldBePresent) expect(domElement).toBeInTheDocument();
      else expect(domElement).toBeNull();
    }
  };
  render(<ExperimentNavBar />);
  const searchField = await screen.findByPlaceholderText('Search experiment');
  expect(searchField).toBeInTheDocument();
  userEvent.type(searchField, 'random');
  await waitFor(() => checkExpectations([0, 0, 0, 0, 1, 0, 0]), {
    interval: 100,
    timeout: 2000,
  });
  userEvent.clear(searchField);
  userEvent.type(searchField, 'rosenbrock');
  await waitFor(() => checkExpectations([0, 0, 0, 0, 1, 1, 0]), {
    interval: 100,
    timeout: 2000,
  });
  userEvent.clear(searchField);
  userEvent.type(searchField, 'dim-cat');
  await waitFor(() => checkExpectations([0, 1, 0, 1, 0, 0, 0]), {
    interval: 100,
    timeout: 2000,
  });
  userEvent.clear(searchField);
  userEvent.type(searchField, 'unknown experiment');
  await waitFor(() => checkExpectations([0, 0, 0, 0, 0, 0, 0]), {
    interval: 100,
    timeout: 2000,
  });
});
