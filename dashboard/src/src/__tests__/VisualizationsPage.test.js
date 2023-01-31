import { test, expect } from '@playwright/test';

function _test() {}

test.describe('Test experiments visualization page', () => {
  test.beforeEach(async ({ page }) => {
    // Set a hardcoded page size.
    await page.setViewportSize({ width: 1920, height: 1080 });
    // Open Dashboard page.
    await page.goto('localhost:3000');
  });

  test('Test if we switch to visualization page', async ({ page }) => {
    // Let time for ExperimentNavBar to load experiments
    const firstExperiment = await page.getByText(/2-dim-shape-exp/);
    await firstExperiment.waitFor();
    await expect(firstExperiment).toHaveCount(1);

    // Make sure we are on default (landing) page
    await expect(await page.getByText(/Landing Page/)).toHaveCount(1);

    // Make sure we are not on visualizations page
    await expect(await page.getByText(/Nothing to display/)).toHaveCount(0);

    // Go to visualization page

    const menuExperiments = await page.locator('nav > ul > li:nth-child(1)');
    await expect(menuExperiments).toHaveCount(1);
    await expect(menuExperiments).toBeVisible();
    await menuExperiments.click();
    const menu = await menuExperiments.getByTitle(
      /Go to experiments visualizations/
    );
    await expect(menu).toHaveCount(1);
    await expect(menu).toBeVisible();
    await menu.click();

    // Check we are on visualizations page
    const elements = await page.getByText(/Nothing to display/);
    await expect(elements).toHaveCount(3);
  });

  test('Test if we can select and unselect experiments', async ({ page }) => {
    // Go to visualization page
    const firstExperiment = await page.getByText(/2-dim-shape-exp/);
    await firstExperiment.waitFor();
    const menuExperiments = await page.locator('nav > ul > li:nth-child(1)');
    await menuExperiments.click();
    const menu = await menuExperiments.getByTitle(
      /Go to experiments visualizations/
    );
    await menu.click();
    // Check we are on visualizations page
    await expect(await page.getByText(/Nothing to display/)).toHaveCount(3);

    // Select an experiment
    await firstExperiment.click();

    // Check if plots are loaded
    for (let plotTitle of [
      /Regret for experiment '2-dim-shape-exp'/,
      /Parallel Coordinates PLot for experiment '2-dim-shape-exp'/i,
      /LPI for experiment '2-dim-shape-exp'/,
    ]) {
      const plot = await page.getByText(plotTitle);
      await plot.waitFor();
      await expect(plot).toHaveCount(1);
    }

    // Unselect experiment
    const row = await page.getByTitle(/unselect experiment '2-dim-shape-exp'/);
    await expect(row).toHaveCount(1);
    expect(await row.evaluate(node => node.tagName.toLowerCase())).toBe(
      'label'
    );
    await row.click();

    await expect(await page.getByText(/Nothing to display/)).toHaveCount(3);
    for (let plotTitle of [
      /Regret for experiment '2-dim-shape-exp'/,
      /Parallel Coordinates PLot for experiment '2-dim-shape-exp'/i,
      /LPI for experiment '2-dim-shape-exp'/,
    ]) {
      const plot = await page.getByText(plotTitle);
      await expect(plot).toHaveCount(0);
    }

    // re-select experiment and check if plots are loaded
    await firstExperiment.click();
    for (let plotTitle of [
      /Regret for experiment '2-dim-shape-exp'/,
      /Parallel Coordinates PLot for experiment '2-dim-shape-exp'/i,
      /LPI for experiment '2-dim-shape-exp'/,
    ]) {
      const plot = await page.getByText(plotTitle);
      await plot.waitFor();
      await expect(plot).toHaveCount(1);
    }

    // Select another experiment and check if plots are loaded
    const searchField = await page.getByPlaceholder('Search experiment');
    await searchField.type('tpe-rosenbrock');
    const anotherExperiment = await page.getByText(/tpe-rosenbrock/);
    await expect(anotherExperiment).toHaveCount(1);
    await anotherExperiment.click();
    for (let plotTitle of [
      /Regret for experiment 'tpe-rosenbrock'/,
      /Parallel Coordinates PLot for experiment 'tpe-rosenbrock'/i,
      /LPI for experiment 'tpe-rosenbrock'/,
    ]) {
      const plot = await page.getByText(plotTitle);
      await plot.waitFor();
      await expect(plot).toHaveCount(1);
    }
  });
});
