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

Load tensor from filestream.

  Parameters:

  - ``&ifs``: Filestream
  - ``format``: File format. Accepted formats are: bin, onnx, csv, tsv, txt.

.. code-block:: c++

    static Tensor* loadfs(std::ifstream &ifs, string format="");
    
load
^^^^^^^^^^^

.. doxygenfunction:: Tensor::load(const string&, string)


Load tensor from file.

  Parameters:

  - ``filename``: Name of the file to load the tensor from.
  - ``format``: Filetype. The accepted filetypes are the following:

    - Images: jpg, jpeg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm.
    - Numpy: npy, npz
    - Text: csv, tsv, txt
    - Other: bin, onnx


.. code-block:: c++

    static Tensor* load(const string& filename, string format="");
    template<typename T> static Tensor* load(const string& filename, string format="");
    
load_from_txt
^^^^^^^^^^^^^

.. doxygenfunction:: Tensor::load_from_txt(const string&, const char, int)

Load data from a text file

  Parameters:
  
  - ``filename``: Name of the file to load the tensor from.
  - ``delimiter``: Character used to separate the columns of the file.
  - ``headerRows``: Number of top rows to avoid, generally because they correspond to the header.

.. code-block:: c++

    static Tensor* load_from_txt(const string& filename, const char delimiter=',', int headerRows=1);



Output
-----------------------


savefs
^^^^^^^^

.. doxygenfunction:: Tensor::savefs

Save tensor to a filestream.

  Parameters:

  - ``&ofs``: Filestream.
  - ``format``: Format to use. The accepted formats are the following:

    - Text: csv, tsv, txt
    - Other: bin, onnx

.. code-block:: c++

    void savefs(std::ofstream &ofs, string format="");

.. note::
    ONNX not yet implemented

save
^^^^^^^^

.. doxygenfunction:: Tensor::save

Save tensor to a file.

  Parameters:

  - ``filename``: Name of the file to save the tensor to.
  - ``format``: Filetype. The accepted filetypes are the following:

    - Images: png, bmp, tga, jpg, jpeg, hdr.
    - Numpy: npy, npz
    - Text: csv, tsv, txt
    - Other: bin, onnx

.. code-block:: c++

    void save(const string& filename, string format="");

.. note::
    ONNX not yet implemented


save2txt
^^^^^^^^

.. doxygenfunction:: Tensor::save2txt(const string&, const char, const vector<string>&)

Save tensor to a text file.

  Parameters:
  
  - ``filename``: Name of the file to save the tensor to.
  - ``delimiter``: Character to use to separate the columns of the file.
  - ``header``: Header rows.

.. code-block:: c++

    void save2txt(const string& filename, const char delimiter=',', const vector<string> &header={});

