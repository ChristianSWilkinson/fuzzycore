from setuptools import setup, find_packages

# Read the contents of your README file for the package description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fuzzycore",
    version="0.1.0",
    author="C. Wilkinson",
    author_email="christian.s.wilkinson@gmail.com", # Your GitHub email
    description="A robust numerical framework for multi-phase giant planet interior modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChristianSWilkinson/fuzzycore",
    # This tells setuptools to look for your code inside the 'src' directory
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)