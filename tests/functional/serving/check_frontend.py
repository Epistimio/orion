"""Script to test orion frontend.
See .github/workflows/test-dashboard-build.yml about how to run this script
Needs backend and frontend to be loaded first.
Selenium needs geckodriver to run Firefox browser (2022/06/06):
https://selenium-python.readthedocs.io/installation.html#drivers
"""

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options


def main():
    # Add options to hide browser
    options = Options()
    options.headless = True
    # Load driver with options
    driver = webdriver.Firefox(options=options)
    driver.get("http://127.0.0.1:3000")
    # Check page title
    assert driver.title == "Oríon Dashboard"
    # Wait to let experiments navbar load
    time.sleep(5)
    # Check we are in landing page (default)
    assert "Landing Page" in driver.page_source
    assert "Nothing to display" not in driver.page_source
    # Move to experiments visualizations page
    menu_experiments = driver.find_element(
        By.XPATH, "//a[@aria-label='experiments (selected)']"
    )
    assert menu_experiments
    menu_experiments.click()
    sub_menu_visualizations = driver.find_element(
        By.XPATH, "//a[@title='Go to experiments visualizations']"
    )
    assert sub_menu_visualizations
    sub_menu_visualizations.click()
    # Check we moved to visualizations page
    assert "Landing Page" not in driver.page_source
    assert driver.page_source.count("Nothing to display") == 3
    # Select an experiment
    experiment = driver.find_element(By.XPATH, "//span[@title='2-dim-shape-exp']")
    assert experiment
    experiment.click()
    # Wait to let plots load
    time.sleep(10)
    # Check plots are loaded
    assert driver.page_source.count("Nothing to display") == 0
    assert "Regret for experiment '2-dim-shape-exp'" in driver.page_source
    assert (
        "Parallel Coordinates Plot for experiment '2-dim-shape-exp'"
        in driver.page_source
    )
    assert "LPI for experiment '2-dim-shape-exp'" in driver.page_source
    # Close driver
    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
