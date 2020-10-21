Fine-grained training
=====================


random_indices
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::random_indices(int,int)

Example:

.. code-block:: c++
   
    tshape s = x_train->getShape();
    int num_batches = s[0]/batch_size;
 
    for(i=0; i<epochs; i++) {
        reset_loss(net);
        for(j=0; j<num_batches; j++) {
            vector<int> indices = random_indices(batch_size, s[0]);
            train_batch(net, {x_train}, {y_train}, indices); 
        }
    }


next_batch
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::next_batch(vector<Tensor*>, vector<Tensor*>)

Example:

.. code-block:: c++
    
    Tensor* xbatch = new Tensor({batch_size, 784});
    Tensor* ybatch = new Tensor({batch_size, 10})

    tshape s = x_train->getShape();
    int num_batches = s[0]/batch_size;
    
    for(i=0; i<epochs; i++) {
       reset_loss(net);
       for(j=0; j<num_batches; j++) {
           next_batch({x_train, y_train}, {xbatch, ybatch});
           train_batch(net, {xbatch}, {ybatch});
       }
    }
   


train_batch
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::train_batch(model, vector<Tensor*>, vector<Tensor*>, vector<int>)

Example:

.. code-block:: c++

    tshape s = x_train->getShape();
    int num_batches = s[0]/batch_size;
 
    for(i=0; i<epochs; i++) {
        reset_loss(net);
        for(j=0; j<num_batches; j++) {
            vector<int> indices = random_indices(batch_size, s[0]);
            train_batch(net, {x_train}, {y_train}, indices); 
        }
    }


.. doxygenfunction:: eddl::train_batch(model, vector<Tensor*>, vector<Tensor*>)

Example:

.. code-block:: c++
    
    Tensor* xbatch = new Tensor({batch_size, 784});
    Tensor* ybatch = new Tensor({batch_size, 10})

    tshape s = x_train->getShape();
    int num_batches = s[0]/batch_size;
    
    for(i=0; i<epochs; i++) {
       reset_loss(net);
       for(j=0; j<num_batches; j++) {
           next_batch({x_train, y_train}, {xbatch, ybatch});
           train_batch(net, {xbatch}, {ybatch});
       }
    }



eval_batch
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::eval_batch(model, vector<Tensor*>, vector<Tensor*>, vector<int>)

Example:

.. code-block:: c++

    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
            indices[i]=(j*batch_size)+i;
            eval_batch(net, {x_test}, {y_test}, indices);
        }
    }


.. doxygenfunction:: eddl::eval_batch(model, vector<Tensor*>, vector<Tensor*>)

        

set_mode
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: set_mode

    
          
        
reset_loss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::reset_loss(model m)

Example:

.. code-block:: c++
    
    reset_loss(net);

          
forward
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::forward(model, vector<Layer*>)

.. doxygenfunction:: eddl::forward(model, vector<Tensor*>)

Example:

.. code-block:: c++

   forward(net, {xbatch});

.. doxygenfunction:: eddl::forward(model)

.. doxygenfunction:: eddl::forward(model, int)



zeroGrads
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::zeroGrads

Example:

.. code-block:: c++
    
    zeroGrads(net);


backward
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::backward(model, vector<Tensor*>)

.. code-block:: c++
    
    backward(net, {ybatch});
          
.. doxygenfunction:: eddl::backward(model)

.. doxygenfunction:: eddl::backward(loss)


update
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::update(model)

Example:

.. code-block:: c++
    
    update(net);
          

print_loss       
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::print_loss(model, int)

Example:

.. code-block:: c++
    
    print_loss(net, j);
          


clamp
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::clamp(model, float, float)

Example:

.. code-block:: c++
    
    void clamp(model m,float min,float max);
          
compute_loss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::compute_loss(loss)

Example:

.. code-block:: c++
    
    loss mse = newloss(mse_loss, {out, target}, "mse_loss"); 
    float my_loss = 0.0;
    
    for(j=0; j<num_batches; j++) {
        next_batch({x_train},{batch});
        zeroGrads(net);
        forward(net, {batch});
        my_loss += compute_loss(mse)/batch_size;
        update(net);
    } 


compute_metric
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::compute_metric(loss)

        

getLoss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::getLoss

         

newloss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::newloss(const std::function<Layer*Layer*>&, Layer*, string)


.. doxygenfunction:: eddl::newloss(const std::function<Layer*vector<Layer*>>&, vector<Layer*>, string)

Example:

.. code-block:: c++

   layer mse_loss(vector<layer> in) {
       layer diff = Diff(in[0], in[1]);
       return Mult(diff, diff);
   }
   
   loss mse = newloss(mse_loss, {out, target}, "mse_loss");

        

getMetric
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::getMetric



newmetric
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::newmetric(const std::function<Layer*Layer*>&, Layer*, string)

.. doxygenfunction:: eddl::newmetric(const std::function<Layer*vector<Layer*>>&, vector<Layer*>, string)

   
        
detach
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::detach(layer)

.. doxygenfunction:: eddl::detach(vlayer)


