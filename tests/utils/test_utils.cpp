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