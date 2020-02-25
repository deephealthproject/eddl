Fine-grained training
=====================

(todo)

.. code-block:: c++
   :linenos:

    All methods:

    vector<int> random_indices(int batch_size, int num_samples);
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);
    void next_batch(vector<Tensor *> in,vector<Tensor *> out);
    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out);
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out);
    void set_mode(model net, int mode);
    void reset_loss(model m);
    vlayer forward(model m,vector<Layer *> in);
    vlayer forward(model m,vector<Tensor *> in);
    vlayer forward(model m);
    vlayer forward(model m,int b);
    void zeroGrads(model m);
    void backward(model m,vector<Tensor *> target);
    void backward(model net);
    void backward(loss l);
    void update(model m);
    void print_loss(model m, int batch);
    void clamp(model m,float min,float max);
    float compute_loss(loss L);
    float compute_metric(loss L);
    Loss* getLoss(string type);
    loss newloss(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    loss newloss(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
    Metric* getMetric(string type);
    loss newmetric(const std::function<Layer*(vector<Layer*>)>& f, vector<Layer*> in, string name);
    loss newmetric(const std::function<Layer*(Layer*)>& f, Layer *in, string name);
    layer detach(layer l);
    vlayer detach(vlayer l);

