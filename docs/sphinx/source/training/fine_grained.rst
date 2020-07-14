Fine-grained training
=====================

(todo)

random_indices
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::random_indices(int,int)

.. code-block:: c++
   
    vector<int> random_indices(int batch_size, int num_samples);
    
  


next_batch
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::next_batch(vector<Tensor *>, vector<Tensor *>)

.. code-block:: c++
    
    void next_batch(vector<Tensor *> in,vector<Tensor *> out);


train_batch
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::train_batch(model, vector<Tensor*>, vector<Tensor*>, vector<int>)

.. doxygenfunction:: eddl::train_batch(model, vector<Tensor*>, vector<Tensor*>)

.. code-block:: c++
    
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out);


eval_batch
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::eval_batch(model, vector<Tensor*>, vector<Tensor*>, vector<int>)

.. doxygenfunction:: eddl::eval_batch(model, vector<Tensor*>, vector<Tensor*>)

.. code-block:: c++

    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);   
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out);
        

set_mode
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: set_mode

.. code-block:: c++
    
    void set_mode(model net, int mode);
          
        
reset_loss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::reset_loss(model m)

.. code-block:: c++
    
    void reset_loss(model m);
          
forward
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::forward(model, vector<Layer*>)

.. doxygenfunction:: eddl::forward(model, vector<Tensor*>)

.. doxygenfunction:: eddl::forward(model)

.. doxygenfunction:: eddl::forward(model, int)

.. code-block:: c++
    
    vlayer forward(model m,vector<Layer *> in);
    vlayer forward(model m,vector<Tensor *> in);
    vlayer forward(model m);
    vlayer forward(model m,int b);


zeroGrads
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::zeroGrads

.. code-block:: c++
    
    void zeroGrads(model m);
          


backward
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::backward(model, vector<Tensor*>)

.. code-block:: c++
    
    void backward(model m,vector<Tensor *> target);
    void backward(model net);
    void backward(loss l);
          


update
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::update(model)

.. code-block:: c++
    
    void update(model m);
          

print_loss       
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::print_loss

.. code-block:: c++
    
    void print_loss(model m, int batch);
          


clamp
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::clamp

.. code-block:: c++
    
    void clamp(model m,float min,float max);
          
compute_loss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::compute_loss(loss)

.. code-block:: c++
    
    float compute_loss(loss L);
          

compute_metric
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::compute_metric(loss)

.. code-block:: c++
    
    float compute_metric(loss L);
          

getLoss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::getLoss

.. code-block:: c++
    
    Loss* getLoss(string type);
          

newloss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::newloss(const std::function<Layer*Layer*>&, Layer*, string)

.. code-block:: c++
    
    loss newloss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    
.. doxygenfunction:: eddl::newloss(const std::function<Layer*vector<Layer*>>&, vector<Layer*>, string)
.. code-block:: c++

    loss newloss(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
          
        

getMetric
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::getMetric

.. code-block:: c++
    
    Metric* getMetric(string type);
          


newmetric
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::newmetric(const std::function<Layer*Layer*>&, Layer*, string)

.. doxygenfunction:: eddl::newmetric(const std::function<Layer*vector<Layer*>>&, vector<Layer*>, string)

.. code-block:: c++
    
    loss newmetric(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    loss newmetric(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
          
        
detach
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::detach(layer)

.. doxygenfunction:: eddl::detach(vlayer)

.. code-block:: c++
    
    layer detach(layer l);
    vlayer detach(vlayer l);

