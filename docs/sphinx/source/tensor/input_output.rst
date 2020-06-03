Input/Output Operations
========================

.. note::

    Section in progress

    Read this: https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md


Input
-----------------------



loadfs
^^^^^^^^^^^

.. doxygenfunction:: Tensor::loadfs

.. code-block:: c++

    static Tensor* loadfs(std::ifstream &ifs, string format="");
    
load
^^^^^^^^^^^

.. doxygenfunction:: Tensor::load(const string&, string)

.. code-block:: c++

    static Tensor* load(const string& filename, string format="");
    template<typename T> static Tensor* load(const string& filename, string format="");
    


Output
-----------------------


savefs
^^^^^^^^

.. doxygenfunction:: Tensor::savefs

.. code-block:: c++

    void savefs(std::ofstream &ofs, string format="");

.. note::
    ONNX not yet implemented

save
^^^^^^^^

.. doxygenfunction:: Tensor::save

.. code-block:: c++

    void save(const string& filename, string format="");

.. note::
    ONNX not yet implemented


save2txt
^^^^^^^^

.. doxygenfunction:: Tensor::save2txt(const string&, const char, const vector<string>&)

.. code-block:: c++

    void save2txt(const string& filename, const char delimiter=',', const vector<string> &header={});

