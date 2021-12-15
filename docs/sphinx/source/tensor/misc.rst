Misc
======


hardware_supported
----------------------------

.. doxygenfunction:: vector<string> hardware_supported()

.. code-block:: c++

    vector<string> = new Tensor:hardware_supported();
    // {"cpu", "cuda", "cuda", "fpga"}


is_hardware_supported
----------------------------

.. doxygenfunction:: is_hardware_supported(string hardware)

.. code-block:: c++

    bool supported = new Tensor:is_hardware_supported("cudnn");
    // true


getDeviceID
----------------------------

.. doxygenfunction:: getDeviceID(const string& dev)

.. code-block:: c++

    Tensor* new_t = new Tensor({3, 3}, Tensor::getDeviceID("cuda:0"));
