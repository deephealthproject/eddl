A computing service is an object that provides hardware transparency so you can easily change the hardware on which
the code will be executed.

CPU
====

.. doxygenfunction:: eddl::CS_CPU(int th=-1, const string& mem="full_mem")

Example:

.. code-block:: c++

    build(net,
          sgd(0.01),            // Optimizer
          {"soft_cross_entropy"},   // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU(4),                // CPU with 4 threads
    );


GPU
====

.. doxygenfunction:: eddl::CS_GPU(const vector<int>& g, const string& mem="full_mem")

.. doxygenfunction:: eddl::CS_GPU(const vector<int>& g, int lsb, const string& mem="full_mem")



Example:

.. code-block:: c++

    build(imported_net,
          sgd(0.01),            // Optimizer
          {"soft_cross_entropy"},   // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}),              // one GPU
          false
    );


FPGA
====

.. doxygenfunction:: eddl::CS_FPGA(const vector<int> &f, int lsb=1)

.. code-block:: c++

    build(imported_net,
          sgd(0.01),            // Optimizer
          {"soft_cross_entropy"},   // Losses
          {"categorical_accuracy"}, // Metrics
          CS_FPGA({1}),              // FPGA
    );



COMPSS
======

.. doxygenfunction:: CS_COMPSS(string filename)

.. code-block:: c++

    build(imported_net,
          sgd(0.01f),                   // Optimizer
          {"soft_cross_entropy"},          // Losses
          {"categorical_accuracy"},        // Metrics
          CS_COMPSS("filename.cfg"),       // COMPSS config file
    );


Serialization
==============
A computing service configuration can be stored and loaded to create a
new equivalent computing service. To do it we serialize the configuration
using protocol buffers and the ONNX standard definition.

Export to file
------------------

.. doxygenfunction:: save_compserv_to_onnx_file

Example:

.. code-block:: c++

    compserv cs = CS_GPU({1});
    save_compserv_to_onnx_file(cs, "my_cs.onnx");


Import from file
------------------

.. doxygenfunction:: import_compserv_from_onnx_file

Example:

.. code-block:: c++

    compserv cs = import_compserv_from_onnx_file("my_cs.onnx");
