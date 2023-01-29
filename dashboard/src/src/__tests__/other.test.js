import puppeteer from 'puppeteer';

test('Something', async () => {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  page.emulate({
    viewport: {
      width: 1024,
      height: 768,
    },
    userAgent: '',
  });
  try {
    await page.goto('http://localhost:3000/');
    await page.waitForSelector('.experiments-wrapper');
    await page.waitForSelector('.experiment-cell');
    const dimensions = await page.$eval('.experiments-wrapper', el => {
      return { width: el.offsetWidth, height: el.offsetHeight };
    });
    expect(dimensions.width).toBeGreaterThan(0);
    expect(dimensions.height).toBeGreaterThan(0);
    console.log(dimensions);
    const navBarScroller = await page.$('.experiments-wrapper');
    const data = await navBarScroller.$$(
      "xpath///span[contains(text(), '2-dim-shape-exp')]"
    );
    expect(data.length).toBe(1);
    const things = await navBarScroller.$$('.experiment-cell span[title]');
    const currLen = things.length;
    expect(currLen).toBeGreaterThan(0);
    console.log(currLen);
    let t = await page.$eval(
      '.bx--structured-list-tbody .bx--structured-list-row:last-child',
      el => {
        const c = document.querySelector('.experiments-wrapper');
        c.scrollTop = 1000;
        return { text: el.textContent };
      }
    );
    console.log(t);
    const newThings = await navBarScroller.$$('.experiment-cell span[title]');
    console.log(newThings.length);
  } finally {
    // await browser.close();
  }
}, 10000);
