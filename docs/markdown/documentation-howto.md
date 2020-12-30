# Documentation

> **ATTENTION**: Regularly check the *Warnings*, especially after updating the conda environment, even if no modifications have been made to the documentation files.

> **NOTE**: The paths indicated are the current ones. If the project code is restructured, these paths will change.

## Configuration files

The Doxygen configuration file is **docs/doxygen/Doxyfile** and the Sphinx and Breathe configuration file is **docs/sphinx/source/conf.py**. Generally, these files will not need any modification unless the path to the project is changed.

## Doxygen

### Compilation

It should be done in the **docs/doxygen** folder using the following command:

    $ doxygen Doxyfile

### Code description

The description of the functions must be written, in this case, in the files **eddl.h** and **tensor.h** just above the corresponding function.

This description must follow a specific format, which is the one already used in the mentioned files:

    /**
      * @brief Explanation of function foo
      *
      * @param param1 Explanation of parameter param1
      * @param param2 Explanation of parameter param2
      * @return       What the function returns, or "(void)"
      *
    */
    void foo(int param1, float param2);


## Sphinx and Breathe

Documentation in Sphinx is written in .rst files following the style described in [this quick guide on reStructuredText](https://docutils.sourceforge.io/docs/user/rst/quickref.html) or in [this other link](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). Full documentation on Sphinx can be found at [this link](https://www.sphinx-doc.org/en/master/contents.html).

Breathe links the Doxygen documentation with Sphinx using specific directives that can be found at [the following link](https://breathe.readthedocs.io/en/latest/directives.html).

### Compilation

It must be done in the **docs/ sphinx/source** folder using the following command:

    $ make clean; make html

The build result is in the **_build/html** folder.

To view the generated documentation, you must open the **index.html** file (or the one corresponding to the page you want to see) in a browser, for example using the following command:

    $ firefox _build/html/index.html &

### Adding new functions

To add a new function, you must write the complete header of the function preceded by ``.. doxygenfunction :: `` so that the documentation generated using Doxygen is carried over. Example:

    .. doxygenfunction:: save_net_to_onnx_file( Net *net, string path )

> Most of the *warnings* that appear after updating the Sphinx or Breathe version have to do with how this line is written, mainly due to the use of spaces: ``*net`` vs. `` * net``, ``<vector>`` vs. ``< vector >``, ...

You can also write just the name of the function, without the arguments, in case there is only one function with that name. This is not recommended because if another function is defined later with the same name and different arguments, *warnings* will appear in the documentation that must be resolved.

### Adding examples

To add a new example, the line ``.. code-block:: c++`` must be added so that the example appears in a block formatted as c++ code. Example:

    .. code-block:: c++

        Net* net = import_net_from_onnx_file("my_model.onnx", DEV_CPU);

### Adding new sections

To add a section, its title must be underlined with one of the symbols indicated [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections). In our case, we mainly use the following:

    Sections underlined with =
    ==========================

    Subsections underlined with -
    -----------------------------

    Subsubsections underlined with ^
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

> If the underline is shorter than the title, a *warning* appears when compiling, although it is not visible when displaying the documentation in the browser.

The content can be divided into different files to avoid being too long when there are many subsections. The files with the subsections are included in the corresponding section file by the statement ``.. include:: `` followed by the path to the file. Example:

    .. include:: examples/first_example.rst
    .. include:: cmake_building.rst

### Adding new pages

To add new pages, they must be indicated in **index.rst** using the following structure:

    .. toctree::
        :maxdepth: 1
        :caption: Título a mostrar en la barra lateral

        archivo/a/incluir_1.rst
        archivo/a/incluir_2.rst
        archivo/a/incluir_n.rst

