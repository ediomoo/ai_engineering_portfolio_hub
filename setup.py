from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> list[str]:
    """ This function returns a list of requirements from a text file
    while excluding the '-e .' triggers. """

    requirements = []
    with open(file_path, "r") as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements





setup(
    name = "ai-engineering-portfolio-hub",
    version = "0.0.1",
    author = "Ediomoabasi_Udoaka",
    author_email = "counseleudoakar@gmail.com",
    packages = find_packages(),
    install_requires = [] #[get_requirements("requirements.txt")]
)