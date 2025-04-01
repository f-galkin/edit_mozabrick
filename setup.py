from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mozabrick",
    version="0.1.0",
    author="Fedor Galkin",
    author_email="f.a.galkin@gmail.com",
    description="A tool for editing pixel art mosaics from Mozabrick",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/f-galkin/edit_mozabrick",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "Pillow>=9.0.0",
        "numpy>=1.20.0",
        "pycairo>=1.20.0",
    ],
)
