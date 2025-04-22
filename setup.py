from setuptools import setup, find_packages

setup(
    name="backtest-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pathlib"
    ],
    entry_points={
        "console_scripts": [
            "backtest-engine=backtester.cli:main"
        ]
    },
    author="Hongyi Wang",
    description="Short description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harryo583/prosperity-backtesting-engine",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)