Building with cmake
-------------------

A better alternative for building programs using `EDDL` is to use `cmake`, especially if you are
developing for several platforms. Assuming the following folder structure:

.. code:: bash

    first_example
       |- src
       |   |- main.cpp
       |- CMakeLists.txt

The following minimal ``CMakeLists.txt`` is enough to build the first example:

.. code:: cmake

    cmake_minimum_required(VERSION 3.15)
    project(first_example)

    find_package(eddl REQUIRED)

    add_executable(first_example src/main.cpp)
    target_link_libraries(first_example eddl)

`cmake` has to know where to find the headers, this is done through the ``CMAKE_INSTALL_PREFIX``
variable. Note that ``CMAKE_INSTALL_PREFIX`` is usually the path to a folder containing the following
subfolders: ``include``, ``lib`` and ``bin``, so you don't have to pass any additional option for linking.
Examples of valid values for ``CMAKE_INSTALL_PREFIX`` on Unix platforms are ``/usr/local``, ``/opt``.

The following commands create a directory for building (avoid building in the source folder), builds
the first example with cmake and then runs the program:

.. code:: bash

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=your_prefix ..
    make
    ./first_example

See :ref:`build-configuration` for more details about the build options.

