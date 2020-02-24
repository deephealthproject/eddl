.. raw:: html

   <style>
   .rst-content .section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>


Installation
============

.. image:: ../_static/images/logos/conda.svg

Using the Conda package
-----------------------

A package for EDDL is available on the conda package manager.

.. code::

    conda install -c conda-forge eddl


.. image:: ../_static/images/logos/debian.svg


Using the Debian package
------------------------

A package for EDDL is available on Debian.

.. code::

    sudo apt-get install eddl

.. note::

    Not yet available

.. image:: ../_static/images/logos/homebrew.svg


Using the Homebrew package
--------------------------

A package for EDDL is available on the homebrew package manager.

.. code::

    brew tap salvacarrion/homebrew-tap
    brew install eddl


.. image:: ../_static/images/logos/cmake.svg

From source with cmake
----------------------

You can also install ``EDDL`` from source with cmake.
On Unix platforms, from the source directory:

.. code::

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
    make install

On Windows platforms, from the source directory:

.. code::

    mkdir build
    cd build
    cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=path_to_prefix ..
    nmake
    nmake install

``path_to_prefix`` is the absolute path to the folder where cmake searches for
dependencies and installs libraries. ``EDDL`` installation from cmake assumes
this folder contains ``include`` and ``lib`` subfolders.

See the :doc:`build-options` section for more details about cmake options.


Including EDDL in your project
---------------------------------

The different packages of ``EDDL`` are built with cmake, so whatever the
installation mode you choose, you can add ``EDDL`` to your project using cmake:

.. code::

    find_package(eddl REQUIRED)
    target_include_directories(your_target PUBLIC ${eddl_INCLUDE_DIRS})
    target_link_libraries(your_target PUBLIC eddl)
