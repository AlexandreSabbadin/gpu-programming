#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
long int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
long int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;

__global__ void compute(float *ra1, float *decl1, float *ra2, float *decl2, long int N1, long int N2, unsigned int *d_histogram) {
    double dpi = acos(-1.);
    long int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N1 * N2) {
        int i = id / N1;
        int j = id % N1;
        float a1 = ra1[i]/60.0f * dpi/180.0f;
        float d1 = decl1[i]/60.0f * dpi/180.0f;
        float a2 = ra2[j]/60.0f * dpi/180.0f;
        float d2 = decl2[j]/60.0f * dpi/180.0f;
        float arg = sinf(d1)*sinf(d2)+cosf(d1)*cosf(d2)*cosf(a1-a2);
        if (arg > 1.0f) arg = 1.0f; 
        else if (arg < -1.0f) arg = -1.0f;
        float theta = acosf(arg);
        // Range
        unsigned int r = (unsigned int) floorf(theta * (180.0f / dpi) * binsperdegree);
        // Count histo
        atomicAdd(&d_histogram[r], 1U);        
    }
}

int main(int argc, char *argv[])
{
   int    i;
   int    noofblocks;
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   long int histogramDRsum, histogramDDsum, histogramRRsum;
   double w;
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;
   cudaError_t myError;

   FILE *outfil;

   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   if ( getDevice(0) != 0 ) return(-1);

   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

   // allocate mameory on the GPU
   float *d_ra_real, *d_decl_real;
   cudaMalloc(&d_ra_real, NoofReal*sizeof(float));
   cudaMalloc(&d_decl_real, NoofReal*sizeof(float));
   float *d_ra_sim, *d_decl_sim;
   cudaMalloc(&d_ra_sim, NoofSim*sizeof(float));
   cudaMalloc(&d_decl_sim, NoofSim*sizeof(float));

   // copy data to the GPU
   cudaMemcpy(d_ra_real, ra_real, NoofReal*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_decl_real, decl_real, NoofReal*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ra_sim, ra_sim, NoofSim*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_decl_sim, decl_sim, NoofSim*sizeof(float), cudaMemcpyHostToDevice);

   size_t histogramSize = binsperdegree * totaldegrees * sizeof(unsigned int);
   
   histogramDD = (unsigned int*) malloc(histogramSize);
   histogramDR = (unsigned int*) malloc(histogramSize);
   histogramRR = (unsigned int*) malloc(histogramSize);
   cudaMalloc(&d_histogram, histogramSize);

   // run the kernels on the GPU
   long int grid;

   kerneltime = 0.0;
   gettimeofday(&_ttime, &_tzone);
   start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

   cudaMemcpy(d_histogram, histogramDD, histogramSize, cudaMemcpyHostToDevice);
   grid = (NoofReal*NoofReal + threadsperblock - 1) / threadsperblock;
   printf("%li\n", grid);
   compute<<<grid, threadsperblock>>>(d_ra_real, d_decl_real, d_ra_real, d_decl_real, NoofReal, NoofReal, d_histogram);
   cudaMemcpy(histogramDD, d_histogram, histogramSize, cudaMemcpyDeviceToHost);
   
   cudaMemcpy(d_histogram, histogramDR, histogramSize, cudaMemcpyHostToDevice);
   grid = (NoofReal*NoofSim + threadsperblock - 1) / threadsperblock;
   printf("%li\n", grid);
   compute<<<grid, threadsperblock>>>(d_ra_real, d_decl_real, d_ra_sim, d_decl_sim, NoofReal, NoofSim, d_histogram);
   cudaMemcpy(histogramDR, d_histogram, histogramSize, cudaMemcpyDeviceToHost);
   
   cudaMemcpy(d_histogram, histogramRR, histogramSize, cudaMemcpyHostToDevice);
   grid = (NoofSim*NoofSim + threadsperblock - 1) / threadsperblock;
   printf("%li\n", grid);
   compute<<<grid, threadsperblock>>>(d_ra_sim, d_decl_sim, d_ra_sim, d_decl_sim, NoofSim, NoofSim, d_histogram);
   cudaMemcpy(histogramRR, d_histogram, histogramSize, cudaMemcpyDeviceToHost);
   
   // copy the results back to the CPU
   gettimeofday(&_ttime, &_tzone);
   end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
   kerneltime += end-start;

   // calculate omega values on the CPU
   histogramDDsum = 0;
   histogramDRsum = 0;
   histogramRRsum = 0;
   outfil = fopen(argv[3], "w");
   fprintf(outfil, "bin start\tomega\thist_DD\thist_DR\thist_RR\n");
   for (int i = 0; i < binsperdegree * totaldegrees; i++) {
        float omega = (histogramDD[i] - 2.*histogramDR[i] + histogramRR[i]) / (1.*histogramRR[i]);
        histogramDDsum += histogramDD[i];
        histogramDRsum += histogramDR[i];
        histogramRRsum += histogramRR[i];
        fprintf(outfil, "%f\t%f\t%d\t%d\t%d\n", i/4., omega, histogramDD[i], histogramDR[i], histogramRR[i]);
   }
   fclose(outfil);

   printf("Histogram DD sum: %ld\n", histogramDDsum);
   printf("Histogram DR sum: %ld\n", histogramDRsum);
   printf("Histogram RR sum: %ld\n", histogramRRsum);

   printf("Time: %f\n", kerneltime);

   cudaFree(d_ra_real);
   cudaFree(d_decl_real);
   cudaFree(d_ra_sim);
   cudaFree(d_decl_sim);
   cudaFree(d_histogram);
  

   return(0);
}


int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}




int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
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
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

