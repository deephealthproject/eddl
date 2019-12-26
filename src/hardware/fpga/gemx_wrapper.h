#pragma once

#include <stddef.h>

int gemx_setup(unsigned long m, unsigned long k, unsigned long n, unsigned long lda, unsigned long ldb, unsigned long ldc, unsigned long ldx);
void *gemx_instr_buffer();
size_t gemx_instr_buffer_size();
unsigned long int gemx_cycle_count();

unsigned int gemx_page_A();
unsigned int gemx_page_B();
unsigned int gemx_page_C();
unsigned int gemx_page_X();

void *gemx_buff_A();
void *gemx_buff_B();
void *gemx_buff_C();
void *gemx_buff_X();
