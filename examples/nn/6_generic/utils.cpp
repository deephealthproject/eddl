

#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void error_fatal
(
        char *msg_format,
        char *msg_value
        ) {
    fprintf(stderr, msg_format, msg_value);
    exit(1);
} /* end error_fatal */

void process_arguments(int argc, char** argv, char* path, char* tr_images,
        char* tr_labels, char* ts_images, char* ts_labels,
        int* epochs, int* batch_size, int* num_classes,
        int* channels, int* width, int* height,
        float* lr, int* initial_mpi_avg, int* chunks, int* use_bi8,
        int* use_distr_dataset, int* ptmodel, char* test_file,
        bool* use_cpu, bool* use_mpi) {

    int argn;
    char *uso =
            "\nError:\n"
            "%s -p path-to-dataset -w width -h height -z channels -c nr-of-classes -b batch-size \n"
            "\t -e nr-of-epochs -a batch-average -l learning-rate -s nr-of-parts \n"
            "\t -d (distr dataset) -8 (8-bit dataset bin format)\n"
            "\t -n pre-trained-model -t model-file-to-test \n"
            "\t --cpu (use cpu instead of gpu) --mpi (use mpi instead f NCCL) \n"
            "";

    argn = 1;

    while (argn < argc) {
        if (!strncmp(argv[argn], "-p", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);
            sprintf(path, "%s", argv[argn]);
            printf("dataset path:'%s'\n", path);

            std::ifstream ifile(path);
            if (!ifile) {
                fprintf(stderr, "Error: %s. dataset not found\n", path);
                error_fatal(uso, argv[0]);
            } /* endif */
        } else if (!strncmp(argv[argn], "-w", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);

            *width = atoi(argv[argn]);
            printf("width: %d\n", *width);
        } else if (!strncmp(argv[argn], "-h", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);

            *height = atoi(argv[argn]);
            printf("height: %d\n", *height);
        } else if (!strncmp(argv[argn], "-z", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);

            *channels = atoi(argv[argn]);
            printf("channels: %d\n", *channels);
        } else if (!strncmp(argv[argn], "-c", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);

            *num_classes = atoi(argv[argn]);
            printf("classes: %d\n", *num_classes);
        } else if (!strncmp(argv[argn], "-b", 2)) {
            argn++;

            *batch_size = atoi(argv[argn]);
            printf("batch size: %d\n", *batch_size);
        } else if (!strncmp(argv[argn], "-e", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);

            *epochs = atoi(argv[argn]);
            printf("epochs:'%d'\n", *epochs);
        } else if (!strncmp(argv[argn], "-a", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);

            *initial_mpi_avg = atoi(argv[argn]);
            printf("mpi-average:'%d'\n", *initial_mpi_avg);
        } else if (!strncmp(argv[argn], "-l", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);
            *lr = atof(argv[argn]);
            printf("learning-rate:'%f'\n", *lr);
        } else if (!strncmp(argv[argn], "-n", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);
            *ptmodel = atoi(argv[argn]);
            printf("pre-trained model:'%d'\n", *ptmodel);
        } else if (!strncmp(argv[argn], "-s", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);
            *chunks = atoi(argv[argn]);
            printf("nr of chunks:'%d'\n", *chunks);
        } else if (!strncmp(argv[argn], "-d", 2)) {
            *use_distr_dataset = 1;
            printf("use distr dataset:'%s'\n", "yes");
        } else if (!strncmp(argv[argn], "-8", 2)) {
            *use_bi8 = 1;
            printf("use 8-bit bin format:'%s'\n", "yes");
        } else if (!strncmp(argv[argn], "-t", 2)) {
            argn++;
            if (argn == argc)
                error_fatal(uso, argv[0]);
            sprintf(test_file, "%s", argv[argn]);
            printf("file to test:'%s'\n", test_file);

            std::ifstream ifile(test_file);
            if (!ifile) {
                fprintf(stderr, "Error: %s. file not found\n", test_file);
                error_fatal(uso, argv[0]);
            } /* endif */
        } else if (!strncmp(argv[argn], "--cpu", 5)) {
            *use_cpu = 1;
            printf("use cpu:'%s'\n", "yes");
        } else if (!strncmp(argv[argn], "--mpi", 5)) {
            *use_mpi = 1;
            printf("use mpi:'%s'\n", "yes");
        } else {
            fprintf(stderr, "Error: %s. invalid argument\n", argv[argn]);
            error_fatal(uso, argv[0]);
        }
        if (*use_bi8) {
            sprintf(tr_images, "%s/%s", path, "train-images.bi8");
            sprintf(tr_labels, "%s/%s", path, "train-labels.bi8");
            sprintf(ts_images, "%s/%s", path, "val-images.bi8");
            sprintf(ts_labels, "%s/%s", path, "val-labels.bi8");
            //sprintf(ts_images, "%s/%s", path, "test-images.bi8");
            //sprintf(ts_labels, "%s/%s", path, "test-labels.bi8");
        } else {
            sprintf(tr_images, "%s/%s", path, "train-images.bin");
            sprintf(tr_labels, "%s/%s", path, "train-labels.bin");
            sprintf(ts_images, "%s/%s", path, "val-images.bin");
            sprintf(ts_labels, "%s/%s", path, "val-labels.bin");
            //sprintf(ts_images, "%s/%s", path, "test-images.bin");
            //sprintf(ts_labels, "%s/%s", path, "test-labels.bin");
        }
        argn++;
    }

}
