#include "gemx_wrapper.h"
#include <gemx_types.h>
#include <host/gemx_gen_gemm.h>

ProgramType *current_program;
unsigned int pageA;
unsigned int pageB;
unsigned int pageX;
unsigned int pageC;

int gemx_setup(unsigned long m, unsigned long k, unsigned long n, unsigned long lda, unsigned long ldb, unsigned long ldc, unsigned long ldx) {

    unsigned int rows_a = m;
    unsigned int cols_a = k;
    unsigned int rows_b = cols_a;
    unsigned int cols_b = n;

    GenGemm gemm;

    if (!gemm.check(m, k, n, lda, ldb, ldc, ldx)) {
        std::cout << "CAN'T RUN ON GEMX" << std::endl;
        return 0;
    }

    if (current_program != NULL) delete current_program;
    current_program = new ProgramType();

#ifdef GEMX_DEBUG
        std::cout << "Adding instruction GEMM (" << m << "x" << k <<" * "<< k << "x" << n << " + " << m << "x" << n << ")\n";
#endif

    bool unused;
    pageA = current_program->allocPages("A", unused, rows_a * cols_a);
    pageB = current_program->allocPages("B", unused, rows_b * cols_b);
    pageX = current_program->allocPages("X", unused, rows_a * cols_b);
    pageC = current_program->allocPages("C", unused, rows_a * cols_b);

    GemmArgsType l_gemmArgs(
        pageA, pageB, pageC, pageX,
        m, k, n,
        lda, ldb, ldc, ldx,
        1
      );
    KargsType l_kargs;
    l_kargs.setGemmArgs(l_gemmArgs);
    l_kargs.store(current_program->addInstr(), 0);

    return 1;

}

void *gemx_instr_buffer() {
    return current_program->getMemDesc().data();
}

size_t gemx_instr_buffer_size() {
    return current_program->getMemDesc().sizeBytes();
}

unsigned long int gemx_cycle_count() {
    KargsType l_kargsRes;
    KargsOpType l_op = l_kargsRes.load(current_program->getBaseResAddr(), 0);
    assert(l_op == KargsType::OpResult || l_op == KargsType::OpControl);
    gemx::InstrResArgs l_instrRes = l_kargsRes.getInstrResArgs();

    return l_instrRes.getDuration();
}

unsigned int gemx_page_A() {
    return pageA;
}

unsigned int gemx_page_B() {
    return pageB;
}

unsigned int gemx_page_C() {
    return pageC;
}

unsigned int gemx_page_X() {
    return pageX;
}

void *gemx_buff_A() {
    return current_program->getPageAddr(pageA);
}

void *gemx_buff_B() {
    return current_program->getPageAddr(pageB);
}

void *gemx_buff_C() {
    return current_program->getPageAddr(pageC);
}

void *gemx_buff_X() {
    return current_program->getPageAddr(pageX);
}
