Constraints
-----------

CMaxNorm
^^^^^^^^
.. cpp:function:: CMaxNorm::CMaxNorm(float max_value, int axis) : Constraint("max_norm")  


CMinMaxNorm
^^^^^^^^^^^
.. cpp:function:: CMinMaxNorm::CMinMaxNorm(float min_value, float max_value, float rate, int axis) : Constraint("min_max_norm") {
    
    Not implemented

apply
""""""
.. cpp:function:: float CMinMaxNorm::apply(Tensor* T)


CNonNeg
^^^^^^^^^^^^
.. cpp:function:: CNonNeg::CNonNeg() : Constraint("non_neg")

    Not implemented

apply
""""""
.. cpp:function: float CNonNeg::apply(Tensor* T)

CUnitNorm
^^^^^^^^^^^^
.. cpp:function:: CUnitNorm::CUnitNorm(int axis) : Constraint("unit_norm")
    
    Not implemented

apply
"""""
.. cpp:function:: float CUnitNorm::apply(Tensor* T)