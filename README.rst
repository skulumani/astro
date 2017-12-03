Astrodynamics in Python
=======================

``astro`` is Python package to perform basic astrodynamics.

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

Create a citation

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
