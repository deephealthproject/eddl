target_sources(EDDLL
        PRIVATE

        # Convolutional layers
        conv/layer_conv.cpp
        conv/layer_transposed_conv.cpp
        conv/layer_upsampling.cpp

        # Merge layers
        merge/layer_add.cpp
        merge/layer_concat.cpp
        merge/layer_average.cpp
        merge/layer_matmul.cpp
        merge/layer_maximum.cpp
        merge/layer_minimum.cpp
        merge/layer_subtract.cpp

        # Noise layers
        noise/layer_gaussian.cpp

        # Pooling layers
        pool/layer_pool.cpp
        pool/layer_avgpool.cpp
        pool/layer_globalavgpool.cpp
        pool/layer_globalmaxpool.cpp
        pool/layer_maxpool.cpp

        # Recurrent layers
        recurrent/layer_rnn.cpp
        recurrent/layer_lstm.cpp

        # Core layers
        layer.cpp
        layer.h
        layer_activation.cpp
        layer_batchnorm.cpp
        layer_dense.cpp
        layer_drop.cpp
        layer_embedding.cpp
        layer_input.cpp
        layer_reshape.cpp
        layer_tensor.cpp
        layer_transpose.cpp
        )