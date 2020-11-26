Building with cmake
-------------------

A better alternative for building programs using `EDDL` is to use `cmake`, especially if you are
developing for several platforms. Assuming the following folder structure:

.. code:: bash

    first_example/
       |- main.cpp
       |- CMakeLists.txt

A folder named ``first_example`` with two files inside, a ``*.cpp`` and a ``CMakeLists.txt``.
Now, you can copy the following lines to the ``CMakeLists.txt`` file so that we can build the first example:

.. code:: cmake

    cmake_minimum_required(VERSION 3.9.2)
    project(first_example)

    add_executable(first_example main.cpp)

    find_package(EDDL REQUIRED)
    target_link_libraries(first_example PUBLIC EDDL::eddl)


`cmake` has to know where to find the headers, this is done through the ``CMAKE_INSTALL_PREFIX``
variable. Note that ``CMAKE_INSTALL_PREFIX`` is usually the path to a folder containing the following
subfolders: ``include``, ``lib`` and ``bin``, so you don't have to pass any additional option for linking.
Examples of valid values for ``CMAKE_INSTALL_PREFIX`` on Unix platforms are ``/usr/local``, ``/opt``.

The following commands create a directory for building (avoid building in the source folder), builds
the first example with cmake and then runs the program:

.. code:: bash

    mkdir build
    cd build
    cmake ..
    make
    ./first_example

See :ref:`build-configuration` for more details about the build options.

