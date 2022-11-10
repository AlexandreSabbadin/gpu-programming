#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 1024
#define dpi acos(-1.)

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
long int NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
long int NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *dd, *dr, *rr;

__global__ void compute(float *ra_real, float *decl_real, float *ra_sim, float *decl_sim, long int size, unsigned int *dd, unsigned int *dr, unsigned int *rr) {
    // ID
    long int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < size * size) {
        int i = id / size;
        int j = id % size;
        // Coordinates
        float ar_1 = ra_real[i] * dpi / 10800.0f;
        float dr_1 = decl_real[i] * dpi / 10800.0f;
        float ar_2 = ra_real[j] * dpi / 10800.0f;
        float dr_2 = decl_real[j] * dpi / 10800.0f;
        float as_1 = ra_sim[i] * dpi / 10800.0f;
        float ds_1 = decl_sim[i] * dpi / 10800.0f;
        float as_2 = ra_sim[j] * dpi / 10800.0f;
        float ds_2 = decl_sim[j] * dpi / 10800.0f;

        // Histogram bin
        unsigned int bin;
        // Argument before acos
        float arg;
        // Angle between coordinates
        float theta;

        // Histogram DD
        arg = sinf(dr_1) * sinf(dr_2) + cosf(dr_1) * cosf(dr_2) * cosf(ar_1-ar_2);
        if (arg > 1.0f) arg = 1.0f; // Avoid errors
        else if (arg < -1.0f) arg = -1.0f;
        theta = acosf(arg) * (180.0f / dpi);    // To degrees
        bin = (unsigned int) (theta * binsperdegree);   // Bin
        atomicAdd(&dd[bin], 1U);    // Synchronize

        // Histogram DR
        arg = sinf(dr_1) * sinf(ds_2) + cosf(dr_1) * cosf(ds_2) * cosf(ar_1-as_2);
        if (arg > 1.0f) arg = 1.0f; // Avoid errors
        else if (arg < -1.0f) arg = -1.0f;
        theta = acosf(arg) * (180.0f / dpi);    // To degrees
        bin = (unsigned int) (theta * binsperdegree);   // Bin
        atomicAdd(&dr[bin], 1U);    // Synchronize

        // Histogram RR
        arg = sinf(ds_1) * sinf(ds_2) + cosf(ds_1) * cosf(ds_2) * cosf(as_1-as_2);
        if (arg > 1.0f) arg = 1.0f; // Avoid errors
        else if (arg < -1.0f) arg = -1.0f;
        theta = acosf(arg) * (180.0f / dpi);    // To degrees
        bin = (unsigned int) (theta * binsperdegree);   // Bin
        atomicAdd(&rr[bin], 1U);    // Synchronize
    }
}

int main(int argc, char *argv[]) {
    long int noofblocks;
    int readdata(char *argv1, char *argv2);
    int getDevice(int deviceno);
    long int histogramDRsum, histogramDDsum, histogramRRsum;
    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;

    FILE *outfil;

    // Bad usage
    if (argc != 4 ) {
        printf("Usage: %s real_data random_data output_data\n", argv[0]);
        return(-1);
    }

    if (getDevice(0) != 0) return(-1);

    if (readdata(argv[1], argv[2]) != 0 ) return(-1);

    // Define data size
    long int size;
    if (NoofReal == NoofSim) {
        size = NoofReal;
    } else {
        printf("Different data sizes: Real() != Random().\n", NoofReal, NoofSim);
        return(-1);
    }

    // Allocate memory on the GPU
    size_t Data_Size = size * sizeof(float);
    // Real data
    float *d_ra_real, *d_decl_real;
    cudaMalloc(&d_ra_real, Data_Size);
    cudaMalloc(&d_decl_real, Data_Size);
    // Random data
    float *d_ra_sim, *d_decl_sim;
    cudaMalloc(&d_ra_sim, Data_Size);
    cudaMalloc(&d_decl_sim, Data_Size);

    // Copy data from the CPU to the GPU
    // Real data
    cudaMemcpy(d_ra_real, ra_real, Data_Size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_real, decl_real, Data_Size, cudaMemcpyHostToDevice);
    // Random data
    cudaMemcpy(d_ra_sim, ra_sim, Data_Size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_decl_sim, decl_sim, Data_Size, cudaMemcpyHostToDevice);

    // Histogram size
    size_t histogramSize = binsperdegree * totaldegrees * sizeof(unsigned int);

    // Allocate memory on the CPU
    histogramDD = (unsigned int*) malloc(histogramSize);
    histogramDR = (unsigned int*) malloc(histogramSize);
    histogramRR = (unsigned int*) malloc(histogramSize);

    // Allocate memory on the GPU
    cudaMalloc(&dd, histogramSize);
    cudaMalloc(&dr, histogramSize);
    cudaMalloc(&rr, histogramSize);

    // Define number of blocks
    noofblocks = (NoofReal*NoofSim + threadsperblock - 1) / threadsperblock;

    // Start time
    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    // Copy data from the CPU to the GPU
    cudaMemcpy(dd, histogramDD, histogramSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, histogramDR, histogramSize, cudaMemcpyHostToDevice);
    cudaMemcpy(rr, histogramRR, histogramSize, cudaMemcpyHostToDevice);

    // Start kernel
    compute<<<noofblocks, threadsperblock>>>(d_ra_real, d_decl_real, d_ra_sim, d_decl_sim, size, dd, dr, rr);

    // Copy data from the GPU to the CPU
    cudaMemcpy(histogramDD, dd, histogramSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramDR, dr, histogramSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(histogramRR, rr, histogramSize, cudaMemcpyDeviceToHost);

    // End time
    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
    kerneltime += end-start;

    // Histogram sums
    histogramDDsum = 0;
    histogramDRsum = 0;
    histogramRRsum = 0;
    // Write results to file
    outfil = fopen(argv[3], "w");
    fprintf(outfil, "bin start\tomega\thist_DD\thist_DR\thist_RR\n");
    for (int i = 0; i < binsperdegree * totaldegrees; i++) {
        // Calculate difference omega
        float omega = (histogramDD[i] - 2.*histogramDR[i] + histogramRR[i]) / (1.*histogramRR[i]);
        // Add to sums
        histogramDDsum += histogramDD[i];
        histogramDRsum += histogramDR[i];
        histogramRRsum += histogramRR[i];
        // Write to file
        fprintf(outfil, "%f\t%f\t%d\t%d\t%d\n", i/4., omega, histogramDD[i], histogramDR[i], histogramRR[i]);
    }
    fclose(outfil);

    // Print sums (must be equal to size * size)
    printf("\n");
    printf("Histogram DD sum: %ld\n", histogramDDsum);
    printf("Histogram DR sum: %ld\n", histogramDRsum);
    printf("Histogram RR sum: %ld\n", histogramRRsum);

    // Print time
    printf("\nTime: %fs\n", kerneltime);

    // Free CPU arrays
    free(ra_real);
    free(decl_real);
    free(ra_sim);
    free(decl_sim);
    free(histogramDD);
    free(histogramDR);
    free(histogramRR);
    
    // Free GPU arrays
    cudaFree(d_ra_real);
    cudaFree(d_decl_real);
    cudaFree(d_ra_sim);
    cudaFree(d_decl_sim);
    cudaFree(dd);
    cudaFree(dr);
    cudaFree(rr);

    // Exit
    return(0);
}


int readdata(char *argv1, char *argv2) {
    int i, linecount;
    char inbuf[180];
    double ra, dec;
    FILE *infil;

    // Warning
    printf("   Assuming input data is given in arc minutes!\n");

    // Open file for real data
    infil = fopen(argv1, "r");

    // IO error
    if (infil == NULL) {
        printf("Cannot open input file %s\n",argv1);
        return(-1);
    }

    // Read the number of galaxies in the input file
    int announcednumber;
    // No annouced number of lines on first line
    if (fscanf(infil, "%d\n", &announcednumber) != 1) {
        printf(" No annouced number of lines on first line of file %s\n", argv1);
        return(-1);
    }

    // Line count
    linecount = 0;
    while (fgets(inbuf,180,infil) != NULL) ++linecount;
    rewind(infil);

    // Compare annouced and real line counts
    if (linecount == announcednumber) {
        printf("   %s contains %d galaxies\n", argv1, linecount);
    } else {
        printf("   %s does not contain %d galaxies but %d\n", argv1, announcednumber, linecount);
        return(-1);
    }

    // Number of lines for real data
    NoofReal = linecount;
    // Right ascension for real data
    ra_real = (float*) calloc(NoofReal, sizeof(float));
    // Declination for real data
    decl_real = (float*) calloc(NoofReal, sizeof(float));

    // Skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL) return(-1);
    i = 0;
    while (fgets(inbuf,80,infil) != NULL) {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2) {
            printf("   Cannot read line %d in %s\n", i+1, argv1);
            fclose(infil);
            return(-1);
        }
        ra_real[i] = (float) ra;
        decl_real[i] = (float) dec;
        ++i;
    }

    fclose(infil);

    if (i != NoofReal) {
        printf("   Cannot read %s correctly\n",argv1);
        return(-1);
    }

    // Open file for random data
    infil = fopen(argv2,"r");

    // IO error
    if (infil == NULL) {
        printf("Cannot open input file %s\n",argv2);
        return(-1);
    }

    // No annouced number of lines on first line
    if (fscanf(infil,"%d\n", &announcednumber) != 1) {
        printf(" No annouced number of lines on first line of file %s\n",argv2);
        return(-1);
    }

    // Line count
    linecount = 0;
    while (fgets(inbuf, 80, infil) != NULL) ++linecount;
    rewind(infil);

    // Compare annouced and real line counts
    if (linecount == announcednumber) {
        printf("   %s contains %d galaxies\n", argv2, linecount);
    } else {
        printf("   %s does not contain %d galaxies but %d\n", argv2, announcednumber, linecount);
        return(-1);
    }

    // Number of lines for random data
    NoofSim = linecount;
    // Right ascension for random data
    ra_sim   = (float *)calloc(NoofSim,sizeof(float));
    // Declination for random data
    decl_sim = (float *)calloc(NoofSim,sizeof(float));

    // Skip the number of galaxies in the input file
    if (fgets(inbuf, 180, infil) == NULL) return(-1);
    i = 0;
    while (fgets(inbuf,80,infil) != NULL) {
        if (sscanf(inbuf, "%lf %lf", &ra, &dec) != 2) {
            printf("   Cannot read line %d in %s\n", i+1, argv2);
            fclose(infil);
            return(-1);
        }
        ra_sim[i]   = (float)ra;
        decl_sim[i] = (float)dec;
        ++i;
    }

    fclose(infil);

    if (i != NoofSim) {
        printf("   Cannot read %s correctly\n",argv2);
        return(-1);
    }

    // Exit
    return(0);
}

// Get device info
int getDevice(int deviceNo) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128 ) return(-1);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("      Device %s                  device %d\n", deviceProp.name,device);
        printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
        printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
        printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
        printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
        printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
        printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
        printf("         maxGridSize                   =   %d x %d x %d\n",
                            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("         concurrentKernels             =   ");
        if (deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
        printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
        if (deviceProp.deviceOverlap == 1)
        printf("            Concurrently copy memory/execute kernel\n");
    }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead", device);
    else printf("   Using CUDA device %d\n\n", device);

    return(0);
}

