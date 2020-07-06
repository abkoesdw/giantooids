from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("LICENSE", "r") as fh:
    long_description = fh.read()

setup(
    name="giantooids",
    version="0.0.1",
    author="Arief Koesdwiady",
    author_email="ariefbarkah@gmail.com",
    description="This a python porting of MATLAB codes to produce some result in: https://www.sciencedirect.com/science/article/abs/pii/S0012821X17301838",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abkoesdw/giantooids",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.2.2",
        "scipy>=1.5.1",
        "tqdm>=4.47.0",
    ],
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
)
