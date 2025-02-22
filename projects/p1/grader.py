"""
This module contains the code to grade the p1.
"""

# Standard libraries
import os
import shutil
import subprocess

# 3pp
import torch
from tqdm.auto import tqdm

# static variables
DATA_PATH: str = "p1-submission"

# set default device
torch.set_default_device("cpu")


def main() -> None:
    # get all names
    files_names: list[str] = os.listdir(DATA_PATH)

    # iterate over files
    for file_name in tqdm(files_names):
        # delte pychache
        if os.path.isdir(f"{DATA_PATH}/{file_name}/src/__pycache__"):
            shutil.rmtree(f"{DATA_PATH}/{file_name}/src/__pycache__")

        # copy test
        if os.path.isdir(f"{DATA_PATH}/{file_name}/tests"):
            shutil.rmtree(f"{DATA_PATH}/{file_name}/tests")
        shutil.copytree("tests", f"{DATA_PATH}/{file_name}/tests")

        # write results
        result = subprocess.run(
            ["pytest", "."], stdout=subprocess.PIPE, cwd=f"{DATA_PATH}/{file_name}"
        )
        output_file = open(f"{DATA_PATH}/{file_name}/summary.txt", "wb")
        output_file.write(result.stdout)
        output_file.close()

    return None


if __name__ == "__main__":
    main()
