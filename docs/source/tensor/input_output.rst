Input/Output Operations
========================

.. note::

    Section in progress

    Read this: <https://github.com/deephealthproject/eddl/blob/master/eddl_progress_tensor.md>


Input
-----------------------



loadfs
^^^^^^^^^^^

Load tensor from filestream

.. code-block:: c++

    static Tensor* loadfs(std::ifstream &ifs, string format="");
    
load
^^^^^^^^^^^

Load tensor from these filetypes:

Images: jpg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm

Numpy: npy, npz

Text: csv, tsv, txt,...


.. code-block:: c++

    static Tensor* load(const string& filename, string format="");
    template<typename T> static Tensor* load(const string& filename, string format="");
    
load_from_txt
^^^^^^^^^^^^^



.. code-block:: c++

    static Tensor* load_from_txt(const string& filename, const char delimiter=',', int headerRows=1);



Output
-----------------------


Example:

.. code-block:: c++
   :linenos:

    void savefs(std::ofstream &ofs, string format="");
    void save(const string& filename, string format="");
    void save2txt(const string& filename, const char delimiter=',', const vector<string> &header={});

