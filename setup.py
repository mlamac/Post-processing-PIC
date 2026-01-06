from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pic-postprocessing",
    version="0.1.0",
    author="PIC Post-Processing Team",
    description="Data processing pipeline for PIC simulations (LWFA)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Post-processing-PIC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "h5py>=3.0.0",
        "matplotlib>=3.4.0",
        "plotly>=5.0.0",
        "sdf>=0.2.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9"],
        "notebooks": ["jupyter>=1.0.0", "ipywidgets>=7.6.0"],
    },
)
