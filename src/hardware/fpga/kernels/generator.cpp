#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_RAND_UNIFORM
void k_rand_uniform(float * A, float v) {
}
#endif

#ifdef K_ENABLED_RAND_SIGNED_UNIFORM
void k_rand_signed_uniform(float * A, float v) {
}
#endif

#ifdef K_ENABLED_RAND_BINARY
void k_rand_binary(float * A, float v) {
}
#endif

#ifdef K_ENABLED_RAND_NORMAL
void k_rand_normal(float * A, float m, float s, bool fast_math) {
}
#endif

}
