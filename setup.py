from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent

VERSION = '0.0.4'
DESCRIPTION = 'A hyperparameter tuning library for machine learning models based on evolutionary algorithms.'
LONG_DESCRIPTION = (this_directory / "README.md").read_text()


# Setting up
setup(
    name="evolutune",
    version=VERSION,
    author="torchd3v",
    author_email="<burak96egeli@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['numpy', 'joblib', 'scikit-learn'],
    keywords=['python', 'hyperparameter', 'tuning', 'genetic-algorithm', 'model', 'search', 'CV', 'evolutionary-algorithm', 'particle-swarm-optimization'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
