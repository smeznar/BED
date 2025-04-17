import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BED",
    version="0.1.0",
    author="Sebastian Mežnar",
    author_email="sebastian.meznar@ijs.si",
    description="Behaviour-Aware Expression Distance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smeznar/BED",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires = ["editdistance",
                        "matplotlib",
                        "numpy",
                        "pandas",
                        "pyarrow",
                        "scipy",
                        "seaborn",
                        "zss"
                        ],
)
