Misc
======


getHardwareSupported
----------------------------

.. doxygenfunction:: vector<string> hardware_supported()

.. code-block:: c++

    vector<string> = new Tensor:hardware_supported();
    // {"cpu", "cuda", "cuda", "fpga"}


isHardwareSupported?
----------------------------

.. doxygenfunction:: is_hardware_supported(string hardware)

.. code-block:: c++

    bool supported = new Tensor:is_hardware_supported("cudnn");
    // true
