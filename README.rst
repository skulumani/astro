Astrodynamics in Python
=======================

``astro`` is Python package to perform basic astrodynamics.

+-------------------------+---------------------+--------------------------+------------+-----------------------+
| Build Status            | Code Coverage       | Docs                     | Citation   | PyPi                  |
+=========================+=====================+==========================+============+=======================+
| |Travis Build Status|   | |Coverage Status|   | |Documentation Status|   | |Citation| | |PyPi|                |
+-------------------------+---------------------+--------------------------+------------+-----------------------+

.. |Travis Build Status| image:: https://travis-ci.org/skulumani/astro.svg?branch=master
   :target: https://travis-ci.org/skulumani/astro
.. |Coverage Status| image:: https://coveralls.io/repos/github/skulumani/astro/badge.svg?branch=master
   :target: https://coveralls.io/github/skulumani/astro?branch=master
.. |Documentation Status| image:: https://readthedocs.org/projects/astro-python/badge/?version=latest
    :target: http://astro-python.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. |Citation| image:: https://zenodo.org/badge/95155784.svg
    :target: https://zenodo.org/badge/latestdoi/95155784
.. |PyPi| image:: https://badge.fury.io/py/astro.svg
    :target: https://badge.fury.io/py/astro

Installation
============

Install ``astro`` by running : ``pip install astro`` to install from pypi

To install a development version (for local testing), you can clone the 
repository and run ``pip install -e .`` from the source directory.

Documentation
=============

Docs will be hosted on Read the Docs

Citing ``astro``
================

If you find this package useful it would be very helpful if you can cite it in your work.
You can use the citation link above.

Dependencies
============

There are a limited number of dependencies.
Much of ``astro`` is built on top of ``numpy``, which is usually included
in any scientific Python enviornment.
Some additional dependencies are used to offer convient wrappers for 
common operations, such as downloading two line element sets or interfacing
with SPICE.

* ``numpy`` 
* ``spiceypy``
* ``spacetrack``
* ``kinematics``
