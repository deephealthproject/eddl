Fine-grained training
=====================

(todo)

random_indices
^^^^^^^^^^^^^^^^^

.. code-block:: c++
   
    vector<int> random_indices(int batch_size, int num_samples);
    
  


next_batch
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void next_batch(vector<Tensor *> in,vector<Tensor *> out);


train_batch
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out);


eval_batch
^^^^^^^^^^^^^^^^^

.. code-block:: c++

    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);   
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out);
        

set_mode
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void set_mode(model net, int mode);
          
        
reset_loss
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void reset_loss(model m);
          
forward
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    vlayer forward(model m,vector<Layer *> in);
    vlayer forward(model m,vector<Tensor *> in);
    vlayer forward(model m);
    vlayer forward(model m,int b);


zeroGrads
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void zeroGrads(model m);
          


backward
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void backward(model m,vector<Tensor *> target);
    void backward(model net);
    void backward(loss l);
          


update
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void update(model m);
          

print_loss       
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void print_loss(model m, int batch);
          


clamp
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    void clamp(model m,float min,float max);
          
compute_loss
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    float compute_loss(loss L);
          

compute_metric
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    float compute_metric(loss L);
          

getLoss
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    Loss* getLoss(string type);
          

newloss
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    loss newloss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    loss newloss(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
          
        

getMetric
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    Metric* getMetric(string type);
          


newmetric
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    loss newmetric(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    loss newmetric(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
          
        
detach
^^^^^^^^^^^^^^^^^

.. code-block:: c++
    
    layer detach(layer l);
    vlayer detach(vlayer l);

