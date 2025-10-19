import sys
import numpy as np
from setuptools import setup
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
    extra_compile_args = []
    extra_link_args = []
elif sys.platform == "darwin":
    # macOS clang doesn't support -fopenmp by default
    library_dirs = []
    extra_compile_args = []
    extra_link_args = []
else:
    # Linux with OpenMP support
    library_dirs = []
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = [
    Extension(
        "vg.time_series_analysis.cresample",
        ["src/vg/time_series_analysis/cresample" + ext],
        include_dirs=[np.get_include()],
        library_dirs=library_dirs,
    ),
    Extension(
        "vg.ctimes",
        ["src/vg/ctimes" + ext],
        include_dirs=[np.get_include()],
        library_dirs=library_dirs,
    ),
    Extension(
        "vg.meteo.meteox2y_cy",
        ["src/vg/meteo/meteox2y_cy" + ext],
        include_dirs=[np.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
    ),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions, language_level="3", force=True)
    cmdclass = dict(build_ext=build_ext)
else:
    ext_modules = extensions
    cmdclass = {}

setup(
    name="vg",
    ext_modules=ext_modules,
)
