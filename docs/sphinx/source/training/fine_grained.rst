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

.. doxygenfunction:: eddl::train_batch(model net, vector<Tensor*> in, vector<Tensor*> out, vector<int> indices)

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


.. doxygenfunction:: eddl::train_batch(model net, vector<Tensor*> in, vector<Tensor*> out)

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

.. doxygenfunction:: eddl::eval_batch(model net, vector<Tensor*> in, vector<Tensor*> out, vector<int> indices)

Example:

.. code-block:: c++

    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
            indices[i]=(j*batch_size)+i;
            eval_batch(net, {x_test}, {y_test}, indices);
        }
    }


.. doxygenfunction:: eddl::eval_batch(model net, vector<Tensor*> in, vector<Tensor*> out)

        

set_mode
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: set_mode

.. code-block:: c++
    
    set_mode(net, 0); // Test model
    set_mode(net, 1); // Train model

          
        
reset_loss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::reset_loss(model m)

Example:

.. code-block:: c++
    
    reset_loss(net);

          
forward
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::forward(model m)

Example:

.. code-block:: c++

   forward(net);

   //Other ways

   forward(net, layers); // Using vector of layers ``layers``
   forward(net, tensors); // Using vector of tensors ``tensors``
   forward(net, 10); // Forward resizing the batch size to 10




zeroGrads
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::zeroGrads

Example:

.. code-block:: c++
    
    zeroGrads(net);


backward
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::backward(model net)
.. doxygenfunction:: eddl::backward(model m, vector<Tensor*> target)

.. code-block:: c++
    
    backward(net);

    // Other ways

    backward(net, target); // Using vector of tensors
    backward(loss); // Computes gradients on the model associated to the loss object passed.


.. doxygenfunction:: eddl::backward(loss l)


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

.. code-block:: c++
    
    clamp(net, 0.5, 0.7); // Clamps all the weights of the model between 0.5 and 0.7
          


          
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

.. code-block:: c++
    
    loss mse = newloss(mse_loss, {out, target}, "mse_loss"); 
    float my_loss = 0.0;
    
    for(j=0; j<num_batches; j++) {
        next_batch({x_train},{batch});
        zeroGrads(net);
        forward(net, {batch});
        my_loss += compute_metric(mse)/batch_size;
        update(net);
    }         

getLoss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::getLoss

.. code-block:: c++
    getLoss("mse_loss"); // Gets MSE Loss       

newloss
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::newloss(const std::function<Layer*(Layer*)> &f, Layer *in, string name)


.. doxygenfunction:: eddl::newloss(const std::function<Layer*(vector<Layer*>)> &f, vector<Layer*> in, string name)

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

.. doxygenfunction:: eddl::newmetric(const std::function<Layer*(Layer*)> &f, Layer *in, string name)

.. doxygenfunction:: eddl::newmetric(const std::function<Layer*(vector<Layer*>)> &f, vector<Layer*> in, string name)

   
        
detach
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: eddl::detach(layer l)

.. doxygenfunction:: eddl::detach(vlayer l)


