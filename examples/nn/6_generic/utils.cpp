
#include <cstdio>
#include <cstdlib>
#ifdef EDDL_LINUX
#include <unistd.h>
#endif
#ifdef EDDL_APPLE
#include <unistd.h>
#endif

#ifdef EDDL_WINDOWS
#include "getopt.h"
// yes, it is very ugly, but it is a workaround for Windows
#include "getopt.c"
#endif

void process_arguments(int argc, char** argv, char* path, char* tr_images,
        char* tr_labels, char* ts_images, char* ts_labels, 
        int* epochs, int* batch_size, int* num_classes, 
        int* channels, int* width, int* height, 
        float* lr, int* initial_mpi_avg, int* chunks, int* use_bi8,
        int* use_distr_dataset) {

    int ch;
    int opterr = 0;
    
    while ((ch = getopt(argc, argv, "m:w:h:c:z:b:e:a:l:s:d8")) != -1) {
        switch (ch) {
            case 'm':
                printf("model path:'%s'\n", optarg);
                sprintf(path, "%s", optarg);
                break;
            case 'w':
                printf("width:'%s'\n", optarg);
                *width = atoi(optarg);
                break;
            case 'h':
                printf("height:'%s'\n", optarg);
                *height = atoi(optarg);
                break;
            case 'z':
                printf("channels:'%s'\n", optarg);
                *channels = atoi(optarg);
                break;
            case 'c':
                printf("classes:'%s'\n", optarg);
                *num_classes = atoi(optarg);
                break;
            case 'b':
                printf("batch size:'%s'\n", optarg);
                *batch_size = atoi(optarg);
                break;
            case 'e':
                printf("epochs:'%s'\n", optarg);
                *epochs = atoi(optarg);
                break;
            case 'a':
                printf("mpi-average:'%s'\n", optarg);
                *initial_mpi_avg = atoi(optarg);
                break;
            case 'l':
                printf("learning-rate:'%s'\n", optarg);
                *lr = std::atof(optarg);
                break;
            case 's':
                printf("use-chunks:'%s'\n", "yes");
                *chunks = atoi(optarg);
                break;
            case 'd':
                printf("use-distr_dataset:'%s'\n", "yes");
                *use_distr_dataset = 1;
                break;
            case '8':
                printf("8-bit dataset format:'%s'\n", "yes");
                *use_bi8 = 1;
                break;
            default:
                printf("other %c\n", ch);
        }
        if (*use_bi8) {
            sprintf(tr_images, "%s/%s", path, "train-images.bi8");
            sprintf(tr_labels, "%s/%s", path, "train-labels.bi8");
            sprintf(ts_images, "%s/%s", path, "test-images.bi8");
            sprintf(ts_labels, "%s/%s", path, "test-labels.bi8");
        } else {
            sprintf(tr_images, "%s/%s", path, "train-images.bin");
            sprintf(tr_labels, "%s/%s", path, "train-labels.bin");
            sprintf(ts_images, "%s/%s", path, "test-images.bin");
            sprintf(ts_labels, "%s/%s", path, "test-labels.bin");
        }
    }
}