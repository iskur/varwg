Configuration
#############

Most of VG's configuration is done at initialization of the VG object.
But to ease usage , some defaults can be defined and stored in a configuration
file.

A template for a configuration file is included in the distribution
(``config_template.py``). VG requires that at least an input file
(``met_file``) is specified. Edit the configuration file, rename it to
``config.py`` and put it somewhere in your ``PYTHONPATH``.

The settings in the configuration template file allows to test the weather
generator. The ``met_file`` variable points to a sample input file
``sample.met``.

The configuration file ``config.py``
************************************

The configuration file is a python module.

*to be extended*

Input data (``met_file``) requirements
**************************************

The format of the ``met_file`` is ascii. The first line defines the variable
names.

- Data should be evenly-spaced without gaps.
  - Hourly or daily values (hourly spacing allows Disaggregation later)
- Date/time information can be given in various ways:
    - ``time``: ISO formatted timestring ``%Y-%m-%dT%H:%M:%S.%f`` or ``%Y-%m-%dT%H:%M:%S``. Example: 1990-01-01T00:00:00
    - ``date``: ``%d.%m.%Y %H:%M:%S``. Example: 01.01.1990 00:00:00
    - ``Julian``: julian decimal day ``%y%j``. Example: 1990001.00
    - ``chron``: ``(%m/%d/%y %H:%M:%S)`` Example: (01.01.90 00:00:00)
    - and more...
  
  See Python's `time formatting specifiers <http://docs.python.org/library/datetime.html?highlight=time#strftime-and-strptime-behavior>`
- Naming of non-time variables in the input file (``met_file``):
    - air temperature: ``theta``
    - short-wave radiation: ``Qsw``
    - wind direction: ``wdir``
    - wind speed: ``U``

  The other variables can be named as preferred, but must stay consistent with
  other configurations made in the ``config.py`` file.

.. No further preprocessing is required. Removal of trend is discouraged, as 
    it is useful to estimate the dependencies between variables when doing climate
    scenario simulations (correlation due to common trend is often regarded as
    "spurious" - here however, it is exactly what we need).


.. Marginals & Seasonalities
    *************************
