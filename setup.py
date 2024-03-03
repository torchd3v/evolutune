from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent

VERSION = '0.0.1'
DESCRIPTION = 'A Genetic Algorithm-based hyperparameter tuner for machine learning models.'
LONG_DESCRIPTION = (this_directory / "evolutune/README.md").read_text()


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
    install_requires=['numpy==1.23.5', 'joblib==1.2.0', 'scikit-learn==1.2.2'],
    keywords=['python', 'hyperparameter', 'tuning', 'genetic-algorithm', 'model', 'search', 'CV'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
