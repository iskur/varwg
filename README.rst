VG: Vector Autoregressive Moving Average Weather generator
##########################################################

.. Contents::

What is VG?
***********

The weather generator VG is a single-site Vector-Autoregressive
(Moving Average) weather generator that was developed for hydrodynamic and
ecologic modelling of lakes. It includes a number of possibilities to
define climate scenarios. For example, changes in mean or in the variability of
air temperature can be set. Correlations during simulations are preserved, so
that these changes propagate from the air temperature to the other simulated
variables.


Installation
************

Prerequisites
=============

- numpy_
    ..  _numpy: http://numpy.scipy.org/
- scipy_
    ..  _scipy: http://www.scipy.org/
- matplotlib_
    ..  _matplotlib: http://matplotlib.sourceforge.net/
- pandas_
    .. _pandas: http://pandas.pydata.org/
- tqdm
    .. _tqdm: https://pypi.python.org/pypi/tqdm


Recommended additional software
===============================

- ipython_
    .. _ipython: http://ipython.org/
- numexpr_
    .. _numexpr: http://code.google.com/p/numexpr/

Instead of installing this software manually, a python software
distribution like anaconda can be used.

..  _anaconda: https://www.anaconda.com/distribution/

Download the package, uncompress it and then install via::

    python setup.py install

Documentation
*************

The documentation can be accessed online at http://iskur.bitbucket.org.

The source package also ships with the sphinx-based documentation source in the
``doc`` folder. Having sphinx_ installed, it can be build by typing::

    make html

inside the ``doc`` folder.

.. _sphinx: sphinx.pocoo.org

Release notes
*************

1.2
===

- Scenarios can be guided through changes in a variable that is not normally
  distributed.
- Disaggregation recreates seasonal changes in daily cycles.
- All scipy.stats.distributions can be used to fit variables.
- Bugfixes when disaggregating in the presence of nans. 

1.1
===

This release makes VG more tolerant to "dirty" input data.

- Non-evenly spaced time series are allowed. WARNING: linear interpolation is 
  used to regularize the data set.
- Gaps/NaNs are allowed. They are not filled in by linear interpolation, but
  actively ignored by the estimators.
- Disaggregation works on variables that have lower and/or upper bounds.

1.0
===

Initial release.

Web sites
*********

Code is hosted at: https://github.com/iskur/vg/

# Documentation: http://iskur.bitbucket.org

License information
*******************

See the file "LICENSE" for information on the history of this
software, terms & conditions for usage, and a DISCLAIMER OF ALL
WARRANTIES.

