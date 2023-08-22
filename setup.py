from distribute_setup import use_setuptools

use_setuptools()

import sys
import numpy as np
from setuptools import setup, find_packages, findall
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext

    USE_CYTHON = True
    ext = ".pyx"
except ImportError:
    USE_CYTHON = False
    ext = ".c"

if sys.platform == "win32":
    library_dirs = ["."]
    extra_link_args = []
else:
    library_dirs = []
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "cresample",
        ["vg/time_series_analysis/cresample" + ext],
        include_dirs=[np.get_include()],
        library_dirs=library_dirs,
    ),
    Extension(
        "ctimes",
        ["vg/ctimes" + ext],
        include_dirs=[np.get_include()],
        library_dirs=library_dirs,
    ),
    Extension(
        "meteox2y_cy",
        ["vg/meteo/meteox2y_cy" + ext],
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        library_dirs=library_dirs,
    ),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass = dict(build_ext=build_ext)
else:
    ext_modules = extensions
    cmdclass = {}

setup(
    name="vg",
    version="1.3",
    packages=["vg", "vg.core", "vg.meteo", "vg.time_series_analysis"],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    # ext_modules=cythonize(extensions),
    # ext_modules=extensions,
    scripts=["distribute_setup.py"],
    setup_requires=["numpy"],
    python_requires=">=3.5",
    install_requires=[
        "matplotlib",
        "scipy",
        "numpy",
        "pandas",
        "cython",
        "tqdm",
        "future",
        "timezonefinder",
        "PICOS",
        "cvxopt",
    ],
    # dependency_links=[
    #     'http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.1.1/',
    #     'http://sourceforge.net/projects/scipy/files/scipy/0.11.0b1/',
    #     'http://sourceforge.net/projects/numpy/files/NumPy/1.6.2/',
    #     ],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.dat", "*.met", "*.rst", "*.pyx", "*.c"],
        # 'doc': ['*.html', '*.rst'],
    },
    include_package_data=True,
    # metadata for upload to PyPI
    author="Dirk Schlabing",
    author_email="32363199+iskur@users.noreply.github.com",
    description="A Vector Autoregressive Weather Generator",
    license="BSD",
    keywords=("weather generator vector-autoregressive time-series-analysis"),
    long_description="""The weather generator VG is a single-site Vector-Autoregressive
 weather generator that was developed for hydrodynamic and ecologic
 modelling of lakes. It includes a number of possibilities to define
 what-if scenarios.""",
)
