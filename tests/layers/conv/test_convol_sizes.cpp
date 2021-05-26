#include <gtest/gtest.h>
#include <string>

#include "eddl/descriptors/descriptors.h"


using namespace std;


TEST(Convol2DTestSuite, output_size)
{
    vector<string> padding = {"valid", "same"};
    vector<int> output_sizes;
    vector<int> input_sizes = {3, 5, 8, 10};
    vector<int> strides = {1, 2, 3};
    vector<int> kernels = {1, 2, 3};

    // Tests results ****************
    output_sizes = {
            // padding = valid
            3, 2, 1, 2, 1, 1, 1, 1, 1,  // input_size = 3
            5, 4, 3, 3, 2, 2, 2, 2, 1,  // input_size = 5
            8, 7, 6, 4, 4, 3, 3, 3, 2,  // input_size = 8
            10, 9, 8, 5, 5, 4, 4, 3, 3, // input_size = 10

            // padding = same
            3, 3, 3, 2, 2, 2, 1, 1, 1,  // input_size = 3
            5, 5, 5, 3, 3, 3, 2, 2, 2,  // input_size = 5
            8, 8, 8, 4, 4, 4, 3, 3, 3,  // input_size = 8
            10, 10, 10, 5, 5, 5, 4, 4, 4, // input_size = 10
    };

    // Test grid of values
    int i = 0;
    for(auto& p : padding){
        for(auto& is : input_sizes){
            for(auto& s : strides){
                for(auto& k : kernels){
                    // Check output size
                    int os = ConvolDescriptor::compute_output(p, is, k, s);
                    ASSERT_EQ(os, output_sizes[i++]);

//                    // Check output2intput
//                    int is2 = ConvolDescriptor::compute_input(p, os, k, s);
////                    ASSERT_EQ(is2, is);
//                    if (is2!=is){
//                        int asdasd =3;
//                        int is3 = ConvolDescriptor::compute_input(p, os, k, s);
//                        cout<< "nope"<<endl;
//                    }
                }
            }
        }
    }
}
