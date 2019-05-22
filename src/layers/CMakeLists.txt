target_sources(EDDLL
        PRIVATE

        # Convolutional layers
        conv/layer_conv.cpp
        conv/layer_transposed_conv.cpp
        conv/layer_upsampling.cpp

        # Core layers
        core/layer_core.cpp
        core/layer_core.h
        core/layer_activation.cpp
        core/layer_batchnorm.cpp
        core/layer_dense.cpp
        core/layer_drop.cpp
        core/layer_embedding.cpp
        core/layer_input.cpp
        core/layer_reshape.cpp
        core/layer_tensor.cpp
        core/layer_transpose.cpp

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

        # Operators
        operators/layer_abs.cpp
        operators/layer_diff.cpp
        operators/layer_div.cpp
        operators/layer_exp.cpp
        operators/layer_log.cpp
        operators/layer_mean.cpp
        operators/layer_mult.cpp
        operators/layer_operator.cpp
        operators/layer_pow.cpp
        operators/layer_sqrt.cpp
        operators/layer_sum.cpp
        operators/layer_var.cpp

        # Pooling layers
        pool/layer_pool.cpp
        pool/layer_avgpool.cpp
        pool/layer_globalavgpool.cpp
        pool/layer_globalmaxpool.cpp
        pool/layer_maxpool.cpp

        # Recurrent layers
        recurrent/layer_rnn.cpp
        recurrent/layer_lstm.cpp
        )