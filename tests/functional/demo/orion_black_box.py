import os
import sys

import orion
import orion.core.cli
from orion.testing import OrionState


def main():
    """This is needed because the user script runs in a separate process, so python
    cannot capture its output.

    Here we run the entire orion script in its own process so we can capture all of its output.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Start main process")
    os.environ["ORION_PRINT"] = "True"
    with OrionState():
        orion.core.cli.main(
            [
                "hunt",
                "-n",
                "default_algo",
                "--exp-max-trials",
                "5",
                "./black_box.py",
                "-x~uniform(0, 5, discrete=True)",
            ]
        )

    print("Main process Error", file=sys.stderr)
    print("End main process")


if __name__ == "__main__":
    main()
