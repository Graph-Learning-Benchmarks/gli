"""Installing script."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fp:
    desc = fp.read()

install_requires = []

setup(name="glb",
      version="0.1",
      packages=find_packages(where="glb"),
      install_requires=[
        "numpy",
        "scipy",
        "torch",
        "dgl-cu102"
      ],
      description=desc)
