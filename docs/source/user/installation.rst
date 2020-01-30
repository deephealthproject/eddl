Installation
************

Orange3-Recommendation has a couple of prerequisites that need to be installed
first, but once met, the rest of picky requirements are automatically handle by
the installer.


Prerequisites
=============

Python3 + pip
-------------

Orange3-Recommendation currently requires **Python3** to run.
*(Note: The algorithms have been design using Numpy, Scipy and Scikit-learn.
Therefore, the algorithms could work with Python 2.7. But due to dependencies
related with Orange3 and its data.Tables, Python3 must be used)*


Numpy, Scikit-learn and Orange3
-------------------------------

The required dependencies to build the software are *Numpy >= 1.9.0*,
*Scikit-Learn >= 0.16* and *Orange3*.

*This is automatically handled by the installer.* So you don't need to install
anything else.


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux:

    python setup.py build
    sudo python setup.py install

For development mode use:

    python setup.py develop


Widget usage
============

After the installation, the widgets from this add-on are registered with Orange.
To run Orange from the terminal use:

    python3 -m Orange.canvas

new widgets are in the toolbox bar under *Recommendation* section.