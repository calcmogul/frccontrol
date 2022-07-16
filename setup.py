#!/usr/bin/env python3

from datetime import date
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import exists
from os.path import join
from os.path import splitext
from setuptools import find_packages
from setuptools import setup
import subprocess
import sys

setup_dir = dirname(__file__)
git_dir = join(setup_dir, ".git")
base_package = "frccontrol"
version_file = join(setup_dir, base_package, "version.py")

# Automatically generate a version.py based on the git version
if exists(git_dir):
    proc = subprocess.run(
        [
            "git",
            "rev-list",
            "--count",
            # Includes previous year's commits in case one was merged after the
            # year incremented. Otherwise, the version wouldn't increment.
            '--after="main@{' + str(date.today().year - 1) + '-01-01}"',
            "main",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # If there is no main branch, the commit count defaults to 0
    if proc.returncode:
        commit_count = "0"
    else:
        commit_count = proc.stdout.decode("utf-8")

    # Version number: <year>.<# commits on main>
    version = str(date.today().year) + "." + commit_count.strip()

    # Create the version.py file
    with open(version_file, "w") as fp:
        fp.write('# Autogenerated by setup.py\n__version__ = "{0}"'.format(version))

if exists(version_file):
    with open(version_file, "r") as fp:
        exec(fp.read(), globals())
else:
    __version__ = "main"

with open(join(setup_dir, "README.rst"), "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="frccontrol",
    version=__version__,
    description="Wrappers around Python Control for making development of state-space models for the FIRST Robotics Competition easier",
    long_description=long_description,
    author="Tyler Veness",
    maintainer="Tyler Veness",
    maintainer_email="calcmogul@gmail.com",
    url="https://github.com/calcmogul/frccontrol",
    keywords="frc first robotics control",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=["scipy", "numpy"],
    license="BSD License",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
    ],
)
