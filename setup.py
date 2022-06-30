"""Installing script."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fp:
    desc = fp.read()

install_requires = []

setup(name="glb",
      version="0.1",
      packages=find_packages(where="glb"),
      install_requires=[
          "numpy>=1.19",
          "scipy>=1.5",
          "torch>=1.10",
          "dgl>=0.6",
      ],
      python_requires=">=3.6",
      description=desc)

print("The default setup configuration only support CPU computing."
      "If you want to use GPU, please install corresponding dgl."
      "See https://www.dgl.ai/pages/start.html for installation.")
