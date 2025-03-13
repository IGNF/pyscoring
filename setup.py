import io
import re
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "Fiona==1.10b2",
    "geopandas==1.0.1",
    "networkx==3.1",
    "numpy==1.25.2",
    "rasterio==1.3.8",
    "Rtree==1.0.1",
    "scipy==1.11.1",
    "scikit-learn==1.5.0",
    "setuptools==70.0.0",
    "shapely==2.0.1",
    "six==1.16.0",
    "threadpoolctl==3.2.0",
    "tqdm==4.66.3",
    "configparser==6.0.1",
    "cjio==0.9.0",
]

with io.open("pyscoring/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read(), re.M).group(1)

setup(
    name="pyscoring",
    version=str(version),
    packages=find_packages(exclude=["tests", "tests.*", "benchmark"]),
    description="pyscoring library : metrics for comparing vector data",
    include_package_data=True,
    author="Etienne Le Bihan, Emmanuel Séguin, Bruno Vallet",
    license="GPL-3.0-or-later",
    install_requires=REQUIRED_PACKAGES,
    zip_safe=False,
)
