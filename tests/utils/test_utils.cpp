#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/utils.h"

using namespace std;


TEST(UtilsTestSuite, compute_squeeze){
    vector<int> v1 = compute_squeeze({1, 7, 1}, -1, false);
    ASSERT_TRUE(v1 == vector<int>({7}));

    vector<int> v2 = compute_squeeze({1, 7, 1}, 0, false);
    ASSERT_TRUE(v2 == vector<int>({7, 1}));

    vector<int> v3 = compute_squeeze({1, 7, 1}, 1, false);
    ASSERT_TRUE(v3 == vector<int>({1, 7, 1}));

    vector<int> v4 = compute_squeeze({1, 7, 1}, 2, false);
    ASSERT_TRUE(v4 == vector<int>({1, 7}));

    vector<int> vb1 = compute_squeeze({1, 7, 1}, -1, true);
    ASSERT_TRUE(vb1 == vector<int>({1, 7}));

    vector<int> vb2 = compute_squeeze({1, 7, 1}, 0, true);
    ASSERT_TRUE(vb2 == vector<int>({1, 7, 1}));

    vector<int> vb3 = compute_squeeze({1, 7, 1}, 1, true);
    ASSERT_TRUE(vb3 == vector<int>({1, 7}));

    // Error checking
//    vector<int> vb4 = compute_squeeze({1, 7, 2}, 5, false);
//    vector<int> vb5 = compute_squeeze({1, 7, 2}, -5, false);
//    vector<int> vb6 = compute_squeeze({1, 7, 2}, 5, true);
//    vector<int> vb7 = compute_squeeze({1, 7, 2}, -5, true);
}



TEST(UtilsTestSuite, compute_unsqueeze){
    vector<int> v1 = compute_unsqueeze({1, 7, 2}, 0, false);
    ASSERT_TRUE(v1 == vector<int>({1, 1, 7, 2}));

    vector<int> v2 = compute_unsqueeze({1, 7, 2}, 1, false);
    ASSERT_TRUE(v2 == vector<int>({1, 1, 7, 2}));

    vector<int> v3 = compute_unsqueeze({1, 7, 2}, 2, false);
    ASSERT_TRUE(v3 == vector<int>({1, 7, 1, 2}));

    vector<int> vb1 = compute_unsqueeze({1, 7, 2}, 0, true);
    ASSERT_TRUE(vb1 == vector<int>({1, 1, 7, 2}));

    vector<int> vb2 = compute_unsqueeze({1, 7, 2}, 1, true);
    ASSERT_TRUE(vb2 == vector<int>({1, 7, 1, 2}));

    vector<int> vb3 = compute_unsqueeze({1, 7, 2}, 2, true);
    ASSERT_TRUE(vb3 == vector<int>({1, 7, 2, 1}));

    // Error checking
//    vector<int> vb4 = compute_unsqueeze({1, 7, 2}, 5, false);
//    vector<int> vb5 = compute_unsqueeze({1, 7, 2}, -5, false);
//    vector<int> vb6 = compute_unsqueeze({1, 7, 2}, 5, true);
//    vector<int> vb7 = compute_unsqueeze({1, 7, 2}, -5, true);
}

TEST(UtilsTestSuite, parse_indices){
    vector<vector<int>> indices;
    vector<vector<int>> indices_ref;

    // Test cases: "x" ***************************************************
    indices = parse_indices({"0", "0", "0"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 0}, {0, 0}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"2", "27", "27"}, {3, 28, 28});
    indices_ref = {{2, 2}, {27, 27}, {27, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"1", "5", "15"}, {3, 28, 28});
    indices_ref = {{1, 1}, {5, 5}, {15, 15}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: "-x" ***************************************************
    indices = parse_indices({"-0", "-0", "-0"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 0}, {0, 0}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"-3", "-28", "-28"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 0}, {0, 0}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"-1", "-5", "-18"}, {3, 28, 28});
    indices_ref = {{2, 2}, {23, 23}, {10, 10}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: ":" ***************************************************
    indices = parse_indices({":", ":", ":"}, {3, 28, 28});
    indices_ref = {{0, 2}, {0, 27}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: "x:" ***************************************************
    indices = parse_indices({"0:", "0:", "0:"}, {3, 28, 28});
    indices_ref = {{0, 2}, {0, 27}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"1:", "5:", "27:"}, {3, 28, 28});
    indices_ref = {{1, 2}, {5, 27}, {27, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: "-x:" ***************************************************
    indices = parse_indices({"-0:", "-0:", "-0:"}, {3, 28, 28});
    indices_ref = {{0, 2}, {0, 27}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"-1:", "-5:", "-28:"}, {3, 28, 28});
    indices_ref = {{2, 2}, {23, 27}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: ":x" ***************************************************
    indices = parse_indices({":3", ":28", ":28"}, {3, 28, 28});
    indices_ref = {{0, 2}, {0, 27}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({":1", ":1", ":1"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 0}, {0, 0}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: ":-x" ***************************************************
    indices = parse_indices({":-2", ":-27", ":-27"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 0}, {0, 0}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({":-1", ":-1", ":-1"}, {3, 28, 28});
    indices_ref = {{0, 1}, {0, 26}, {0, 26}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({":-2", ":-5", ":-2"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 22}, {0, 25}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }


    // Test cases: "x:x" ***************************************************
    indices = parse_indices({"0:28", "0:28", "0:28"}, {28, 28, 28});
    indices_ref = {{0, 27}, {0, 27}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"10:28", "5:13", "0:17"}, {28, 28, 28});
    indices_ref = {{10, 27}, {5, 12}, {0, 16}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: "x:-x" ***************************************************
    indices = parse_indices({"0:-2", "0:-27", "0:-27"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 0}, {0, 0}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"1:-1", "5:-1", "3:-1"}, {3, 28, 28});
    indices_ref = {{1, 1}, {5, 26}, {3, 26}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: "-x:-x" ***************************************************
    indices = parse_indices({"-0:-2", "-0:-27", "-0:-27"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 0}, {0, 0}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"-2:-1", "-5:-3", "-10:-1"}, {3, 28, 28});
    indices_ref = {{1, 1}, {23, 24}, {18, 26}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    // Test cases: mixes ***************************************************
    indices = parse_indices({"0:1", "0:28", "0:28"}, {3, 28, 28});
    indices_ref = {{0, 0}, {0, 27}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"0:-1", "6:-5", ":"}, {3, 28, 28});
    indices_ref = {{0, 1}, {6, 22}, {0, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }

    indices = parse_indices({"-2:-1", ":-5", "16:"}, {3, 28, 28});
    indices_ref = {{1, 1}, {0, 22}, {16, 27}};
    ASSERT_TRUE(indices.size()==indices_ref.size());
    for(int i=0; i<indices.size(); i++){ ASSERT_TRUE(indices[i] == indices_ref[i]); }
}