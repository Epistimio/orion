import ExperimentNavBar from '../experiments/components/ExperimentNavBar';
import { render, queryByText, fireEvent, screen } from '@testing-library/react';
import React from 'react';

test('Test nav bar dimensions', async () => {
  console.log(navigator.userAgent);
  expect(navigator.userAgent).toContain('jsdom');
  // ExperimentNavBar will be rendered with default value associated to BackendContext.
  render(<ExperimentNavBar />);
  const element = document.querySelector('.experiment-navbar');
  expect(element).toBeInTheDocument();
  // Windows seems to have dimensions.
  expect(window.innerWidth).toBe(1024);
  expect(window.innerHeight).toBe(768);
  // Element is indeed loaded with default content ...
  expect(queryByText(element, 'Experiment')).toBeInTheDocument();
  expect(queryByText(element, 'Status')).toBeInTheDocument();
  // ... but does not have dimensions.
  // https://stackoverflow.com/a/28398467
  // https://stackoverflow.com/a/19916624
  expect(element.offsetWidth).toBeGreaterThan(0);
  expect(element.offsetHeight).toBeGreaterThan(0);
});

test('Test nav bar dimensions', async () => {
  console.log(navigator.userAgent);
  expect(navigator.userAgent).toContain('jsdom');
  // ExperimentNavBar will be rendered with default value associated to BackendContext.
  render(<ExperimentNavBar />);
  const scrollableElement = document.querySelector('.experiments-wrapper');
  expect(scrollableElement).toBeInTheDocument();
  fireEvent.scroll(scrollableElement, { target: { scrollY: 1 } });
  fireEvent.scroll(scrollableElement, { target: { scrollY: 5 } });
  fireEvent.scroll(scrollableElement, { target: { scrollY: 10 } });
  expect(await screen.findByText(/2-dim-shape-exp/)).toBeInTheDocument();
});
