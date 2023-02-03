import { test, expect } from '@playwright/test';

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
});
