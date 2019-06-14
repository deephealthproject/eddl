#include <catch2/catch.hpp>

#include "tensor/tensor.h"

TEST_CASE("Tensor operations")
{
    REQUIRE(Tensor({1, 1, 1}, DEV_CPU).device == DEV_CPU);
    REQUIRE(Tensor({1, 1, 1}, DEV_CPU).ndim == 3);
}