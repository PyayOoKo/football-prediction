"""
Setup script for the football prediction package.
"""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="football-prediction",
    version="0.1.0",
    description="A machine learning pipeline for predicting football match outcomes.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/football-prediction",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.3.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=8.0.0",
            "black>=24.0.0",
            "ruff>=0.3.0",
            "mypy>=1.8.0",
        ],
        "deep": [
            "torch>=2.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Sports",
    ],
)
