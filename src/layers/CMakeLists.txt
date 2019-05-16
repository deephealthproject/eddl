target_sources(EDDLL
        PRIVATE
        layer.cpp
        layer.h

        # Merge layers
        layer_add.cpp
        layer_cat.cpp
        layer_average.cpp
        layer_matmul.cpp
        layer_maximum.cpp
        layer_minimum.cpp
        layer_subtract.cpp

        # Other layers
        layer_activation.cpp
        layer_dense.cpp
        layer_input.cpp
        layer_conv.cpp
        layer_embedding.cpp
        layer_reshape.cpp
        layer_tensor.cpp
        layer_transpose.cpp
        layer_pool.cpp

        # Normalization layers
        layer_batchnorm.cpp
        layer_drop.cpp

        # Noise layers
        layer_noise.cpp

        # Recurrent layers
        layer_rnn.cpp
        layer_lstm.cpp
        )