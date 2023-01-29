const { test, expect } = require('@playwright/test');

test('Test experiment nav bar scrolling', async ({ page }) => {
  // Set a hardcoded page size.
  await page.setViewportSize({ width: 1920, height: 1080 });
  // Open Dashboard page.
  await page.goto('localhost:3000');

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
  expect(lastBox.height).toBe(16);
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
