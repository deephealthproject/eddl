/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <new>      // included for std::bad_alloc

#include "utils.h"
#include <random>

#include "system_info.h"
#include <fstream>
#include <string.h>

#ifdef EDDL_LINUX
#include "sys/mman.h"
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#ifdef EDDL_APPLE
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#endif

#ifdef EDDL_WINDOWS
#include <windows.h>
#endif

#define PI 3.1415926
#define MAX_RTABLE 100000

float *RTable=nullptr;
int posTable=0;

// Default seed
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()


float uniform(float min, float max) {
    // rand() may not generate numbers uniformly and is therefore discouraged
    // Talk: https://channel9.msdn.com/Events/GoingNative/2013/rand-Considered-Harmful
    std::uniform_real_distribution<float> distr(min, max);
    return distr(gen);
}

float suniform() {
    return (2 * uniform()) - 1;
}

float gaussgen() {
    float x, u1, u2;
    int i;

    u1 = uniform();
    u2 = uniform();

    while (u1 == 0.0) u1 = uniform();

    x = sqrt(log(1 / u1)) * cos(2 * PI * u2);

    return x;
}


float gauss(float mean, float sd) {
    return (gaussgen() * sd) + mean;
}

void gen_rtable()
{
  if (RTable==nullptr) {
    printf("Generating Random Table\n");
    RTable=get_fmem(MAX_RTABLE, "gen_rtable");

    for(int i=0;i<MAX_RTABLE;i++)
      RTable[i]=gaussgen();
  }
}

float gauss(int s, float mean, float sd) {
    posTable=(posTable+s)%MAX_RTABLE;
    return (RTable[posTable] * sd) + mean;
}


float *get_fmem(int size, char *str){
    float* ptr = nullptr;
    bool error = false;

    // New vs Malloc *******************
    // New is the C++ way of doing it
    // New is type-safe, Malloc is not
    // New calls your type constructor, Malloc not - Same for destructor
    // New is an operator, Malloc a function (slower)
    try{
        ptr = new float[size];
    }
    catch (std::bad_alloc& badAlloc){
        error=true;
    }

    // Check for errors
    // mlock tell the system to lock to a specified memory range, and to not allow
    // that memory to be page
    // More info:
    // https://stackoverflow.com/questions/48585079/malloc-on-linux-without-overcommitting
    // https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_MRG/1.3/html/Realtime_Reference_Guide/sect-Realtime_Reference_Guide-Memory_allocation-Using_mlock_to_avoid_memory_faults.html

    // Check if free memory is bigger than requested
    unsigned long freemem = get_free_mem();
    if (size*sizeof(float) > freemem) {
        error=true;
    }

    // Not enough free memory
    if (error) {
        delete ptr;
        fprintf(stderr, "Error allocating %s in %s\n", humanSize(size*sizeof(float)), str);
        exit(EXIT_FAILURE);
    }

    return ptr;
}

char *humanSize(uint64_t bytes){
    char *suffix[] = {"B", "KB", "MB", "GB", "TB"};
    char length = sizeof(suffix) / sizeof(suffix[0]);

    int i = 0;
    double dblBytes = bytes;

    if (bytes > 1024) {
        for (i = 0; (bytes / 1024) > 0 && i<length-1; i++, bytes /= 1024)
            dblBytes = bytes / 1024.0;
    }

    static char output[200];
    sprintf(output, "%.02lf %s", dblBytes, suffix[i]);
    return output;
}


#ifdef EDDL_LINUX
    unsigned long get_free_mem() {
        std::string token;
        std::string type = "MemFree:";
        std::ifstream file("/proc/meminfo");
        while(file >> token) {
            if(token == type) {
                unsigned long mem;
                if(file >> mem) {
                    return mem * 1024; // From kB to Bytes
                } else {
                    return 0;
                }
            }
            // ignore rest of the line
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return 0; // nothing found
    }
#endif

#ifdef EDDL_APPLE
unsigned long get_free_mem() {
    // TODO: Review. This doesn't work correctly
    mach_port_t host_port;
    mach_msg_type_number_t host_size;
    vm_size_t pagesize;
    host_port = mach_host_self();
    host_size = sizeof(vm_statistics64) / sizeof(integer_t);
    host_page_size(host_port, &pagesize);
    pagesize = 0;
    int mib[2] = { CTL_HW, HW_PAGESIZE };
    size_t length = sizeof(pagesize);
    const int sysctlResult = sysctl(mib, 2, &pagesize, &length, NULL, 0);
    struct vm_statistics64 vm_stat{};
    if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
        fprintf(stderr,"Failed to fetch vm statistics");
        exit(EXIT_FAILURE);
    }
    unsigned long mem_free = vm_stat.free_count * pagesize;
    return mem_free;
}

#endif

#ifdef EDDL_WINDOWS
unsigned long get_free_mem() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return -1;
}
#endif
