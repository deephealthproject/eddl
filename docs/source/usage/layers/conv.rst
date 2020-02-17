
Convolutional layers
---------------------

Convolutional
^^^^^^^^^^^^^^^^


.. cpp:function:: LConv::LConv(Layer *parent, const vector<int> &ks, const vector<int> &st, \
                  const vector<int> &p, string name, int dev, int mem)

.. cpp:function:: LConv::LConv(Layer *parent, int filters, const vector<int> &kernel_size, \ 
                    const vector<int> &strides, string padding, int groups, \ 
                    const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem)

    Not implemented


.. cpp:function:: LConv::LConv(Layer *parent, ConvolDescriptor *D, string name, int dev, int mem)

resize
"""""""
.. cpp:function:: void LConv::resize(int batch)

mem_delta
"""""""""""""

.. cpp:function:: void LConv::mem_delta()

forward
"""""""""""""

.. cpp:function:: void LConv::forward()


backward
"""""""""""""

.. cpp:function:: void LConv::backward()


update_weights
"""""""""""""

.. cpp:function:: void LConv::update_weights(Tensor* w, Tensor* bias)


accumulate_accumulated_gradients
"""""""""""""

.. cpp:function:: void LConv::accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias)


reset_accumulated_gradients
"""""""""""""

.. cpp:function:: void LConv::reset_accumulated_gradients() 


apply_accumulated_gradients
"""""""""""""

.. cpp:function:: void LConv::apply_accumulated_gradients()


share
"""""""""""""

.. cpp:function:: Layer *LConv::share(int c, int bs, vector<Layer *> p)


clone
"""""""""""""

.. cpp:function:: Layer *LConv::clone(int c, int bs, vector<Layer *> p, int todev)


plot
"""""""""""""

.. cpp:function:: string LConv::plot(int c)


reset_name_counter
"""""""""""""

.. cpp:function:: void LConv::reset_name_counter() 


enable_distributed
"""""""""""""

.. cpp:function:: void LConv::enable_distributed() 


    

Transposed conv
^^^^^^^^^^^^^^^^

.. cpp:function:: LConvT::LConvT(Layer *parent, int filters, const vector<int> &kernel_size, \
                  const vector<int> &output_padding, string padding, const vector<int> &dilation_rate, \
                  const vector<int> &strides, bool use_bias, string name, int dev, int mem) 

    Not implemented

.. cpp:function:: LConvT::LConvT(Layer *parent, ConvolDescriptor *cd, string name, int dev, int mem) 



UpSampling
^^^^^^^^^^^^^^^^

.. cpp:function:: LUpSampling::LUpSampling(Layer *parent, const vector<int> &size, \ 
                    string interpolation, string name, int dev, int mem) 


forward
"""""""""""""""""
.. cpp:function:: void LUpSampling::forward()


backward
"""""""""""""""""
.. cpp:function:: void LUpSampling::backward()


share
"""""""""""""""""
.. cpp:function:: Layer *LUpSampling::share(int c, int bs, vector<Layer *> p)


clone
"""""""""""""""""
.. cpp:function:: Layer *LUpSampling::clone(int c, int bs, vector<Layer *> p, int todev) 

plot
"""""""""""""""""
.. cpp:function:: string LUpSampling::plot(int c)