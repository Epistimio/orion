// @ts-check
const { test, expect } = require('@playwright/test');

test('Test experiment nav bar scrolling', async ({ page }) => {
  await page.setViewportSize({width: 1920, height: 1080});
  await page.goto('localhost:3000');

  const navBar = await page.locator('.experiment-navbar');
  await expect(navBar).toHaveCount(1);
  const navBarBox = await navBar.boundingBox();

  const scrollableContainer = await navBar.locator('.experiments-wrapper');
  const scrollableBox = await scrollableContainer.boundingBox();

  const firstLoadedExperiments = await navBar.locator('.experiments-wrapper .experiment-cell span[title]');
  await expect(firstLoadedExperiments).toHaveCount(16);
  const currentFirstLoadedExperiment = firstLoadedExperiments.first();
  const currentLastLoadedExperiment = firstLoadedExperiments.last();

  await expect(currentFirstLoadedExperiment).toHaveText('2-dim-exp');
  await expect(currentLastLoadedExperiment).toHaveText('all_algos_webapi_AverageResult_EggHolder_1_2');

  const firstBox = await currentFirstLoadedExperiment.boundingBox();
  const lastBox = await currentLastLoadedExperiment.boundingBox();
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
  expect(scrollableBox.y + scrollableBox.height).toBeLessThan(lastBox.y);

  let nextExperiment = await navBar.getByText('all_algos_webapi_AverageResult_EggHolder_2_1');
  await expect(nextExperiment).toHaveCount(0);

  await currentLastLoadedExperiment.scrollIntoViewIfNeeded();

  await nextExperiment.waitFor();
  await expect(nextExperiment).toHaveCount(1);

  const newLoadedExperiments = await navBar.locator('.experiments-wrapper .experiment-cell span[title]');
  await expect(newLoadedExperiments).toHaveCount(18);
});
