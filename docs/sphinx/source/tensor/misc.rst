Misc
======


hardware_supported
----------------------------

.. doxygenfunction:: hardware_supported();

.. code-block:: c++

    vector<string> = Tensor::hardware_supported();
    // {"cpu", "cuda", "cuda", "fpga"}


is_hardware_supported
----------------------------

.. doxygenfunction:: is_hardware_supported(string hardware)

.. code-block:: c++

    bool supported = Tensor::is_hardware_supported("cudnn");
    // true


getDeviceID
----------------------------

.. doxygenfunction:: getDeviceID(const string& dev)

.. code-block:: c++

    Tensor* new_t = new Tensor({3, 3}, Tensor::getDeviceID("cuda:0"));
