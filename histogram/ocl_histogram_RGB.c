#define STB_IMAGE_IMPLEMENTATION
#include <stdio.h>
#include "stb_image.h"
#include <math.h>
#ifdef __APPLE__
#  include "OpenCL/opencl.h"
#else
#  include "CL/cl.h"
#endif

#define KERNEL(x) #x

#define HIST_SIZE 256 
#define TIMING 
#define VERBOSE 

/*
typedef struct {
    unsigned int R[HIST_SIZE];
    unsigned int G[HIST_SIZE];
    unsigned int B[HIST_SIZE];
} Histogram; 
*/

typedef unsigned int* Histogram; 
typedef unsigned int* Image;


cl_context context;
cl_command_queue queue;
cl_program program;

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

/*
Counts the number of elements with value of 0 
in the array one-dimensional array M. 
Final value should be stored in count, and each
thread should process Mn elements. 
*/
const char* kernelString = KERNEL(
__kernel void histogram_v1(const int height, const int width, const __global unsigned int* img, __global unsigned int* hist) {
    int index = get_global_id(0);
    int histSize = 256;

    // Process the pixels assigned to this thread
    unsigned int tmp = img[index]; 
    atomic_inc(&hist[tmp & 0x00FF0000 >> 16]);
    atomic_inc(&hist[histSize+tmp & 0x0000FF00 >> 8]);
    atomic_inc(&hist[histSize*2+tmp & 0x000000FF]);
    
}

__kernel void histogram_v2(const int height, const int width, const __global unsigned int* img, __global unsigned int* hist) {
    // Local histogram shared by the work-group
    __local unsigned int localHist[256 * 3];

    int histSize = 256;
    int localSize = get_local_size(0);
    int globalSize = get_global_size(0);
    int index = get_global_id (0);
    int id = get_local_id(0);
    
    // Initialize the local histogram
    for (int i = id; i < histSize * 3; i += localSize) {
            localHist[i] = 0;
        }

    // Synchronize to make sure the local histogram is initialized
    barrier(CLK_LOCAL_MEM_FENCE);

    // Process the pixels assigned to this work-group
    for (int i = index; i < width*height; i += globalSize) {
        unsigned int tmp = img[i]; 
        atomic_inc(&localHist[tmp & 0x00FF0000 >> 16]);
        atomic_inc(&localHist[histSize+tmp & 0x0000FF00 >> 8]);
        atomic_inc(&localHist[histSize*2+tmp & 0x000000FF]);
    }

    // Synchronize to make sure all work-items have updated the local histogram    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (id == 0 ) {
        for (int i = 0; i < histSize * 3; i++) {
            atomic_add(&hist[i], localHist[i]);
        }
    }
    
}
);


void unprepare() {
    // Clean-up OpenCL
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
}

void prepare() {

    static int setup = 0;
    if(setup) return;
    setup = 1;

    // Opencl Setup
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
    char deviceName[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);

    // Compile the kernel
    program = clCreateProgramWithSource(context, 1, &kernelString, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

    // Check for compilation errors
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    char* messages = (char*)malloc((1+logSize)*sizeof(char));
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
    messages[logSize] = '\0';
    if (logSize > 10) { printf(">>> Compiler message: %s\n", messages); }
    free(messages);

    atexit(unprepare);
}

/*
Returns an array containing the pixel colours of the image; 
3rd byte is RED, 2nd byte is BLUE and 1st byte is GREEN. 
The count of bytes starts with the least significant byte.
*/
unsigned int* load_img_to_array(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* img = stbi_load(filename, width, height, &channels, 3); // 3 channels for RGB
    if (!img) {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        return NULL;
    }

    int imgSize = (*width) * (*height);
    int* imageArray = (int*)malloc(imgSize * sizeof(int));
    if (!imageArray) {
        perror("Error allocating memory");
        stbi_image_free(img);
        return NULL;
    }

    for (int i = 0; i < imgSize; i++) {
        int index = i * 3;
        imageArray[i] = (img[index] << 16) | (img[index + 1] << 8) | img[index + 2];
    }

    stbi_image_free(img);
    return imageArray;
}
void impl_hist_ver1_opencl( unsigned int width, unsigned int height, Image img, Histogram hist) {

    cl_int err = CL_SUCCESS;
    cl_event event = NULL;

    struct timeval before, after;
    double time = 0.0;
    
    gettimeofday(&before, NULL);
    
    // Initiliazie the histogram
    for (int i = 0; i < HIST_SIZE * 3; i++) {
        hist[i] = 0;
    }

    // Prepare program
    prepare();

    size_t global[1] = {height*width};

    // Prepare OpenCL memory objects
    cl_mem bufImg = clCreateBuffer(context, CL_MEM_READ_ONLY, height*width*sizeof(unsigned int), NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    cl_mem bufHist = clCreateBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE * 3 * sizeof(unsigned int), NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufImg, CL_TRUE, 0, width*height*sizeof(unsigned int), img, 0, NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    clEnqueueWriteBuffer(queue, bufHist, CL_TRUE, 0, HIST_SIZE * 3 * sizeof(unsigned int), hist, 0, NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    // Setup call to the opencl kernel
    cl_kernel kernel = clCreateKernel(program, "histogram_v1", &err);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 0, sizeof(unsigned int), (void*)&width);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 1, sizeof(unsigned int), (void*)&height);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufImg);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufHist);

    // Call the opencl kernel for every row of the M matrix
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                 global, NULL, 0, NULL, &event);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clWaitForEvents(1, &event);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    // Copy back the resulting C matrix to host memory
    clEnqueueReadBuffer(queue, bufHist, CL_TRUE, 0, HIST_SIZE * 3 * sizeof(unsigned int), hist, 0, NULL, &event);
    err = clWaitForEvents(1, &event);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    gettimeofday(&after, NULL);

    time = (double)(after.tv_sec - before.tv_sec) +
        (double)(after.tv_usec - before.tv_usec) / 1e6;

#ifdef TIMING    
    printf("Kernel took: %.6f seconds.\n", time);
#endif

    // Free the OpenCL memory objects
    clReleaseMemObject(bufImg);
    clReleaseMemObject(bufHist);

    clReleaseKernel(kernel);
}

void impl_hist_ver2_opencl( unsigned int width, unsigned int height, Image img, Histogram hist) {

    cl_int err = CL_SUCCESS;
    cl_event event = NULL;

    struct timeval before, after;
    double time = 0.0;
    
    gettimeofday(&before, NULL);  

    // Initiliazie the histogram
    for (int i = 0; i < HIST_SIZE * 3; i++) {
        hist[i] = 0;
    }

    // Prepare program
    prepare();

    size_t local[1] = {256};
    size_t global[1] = {height};
    
    // Prepare OpenCL memory objects
    cl_mem bufImg = clCreateBuffer(context, CL_MEM_READ_ONLY, height*width*sizeof(unsigned int), NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    cl_mem bufHist = clCreateBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE * 3 * sizeof(unsigned int), NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufImg, CL_TRUE, 0, width*height*sizeof(unsigned int), img, 0, NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    clEnqueueWriteBuffer(queue, bufHist, CL_TRUE, 0, HIST_SIZE * 3 * sizeof(unsigned int), hist, 0, NULL, NULL);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    // Setup call to the opencl kernel
    cl_kernel kernel = clCreateKernel(program, "histogram_v2", &err);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 0, sizeof(unsigned int), (void*)&width);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 1, sizeof(unsigned int), (void*)&height);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&bufImg);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufHist);

    // Call the opencl kernel for every row of the M matrix
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                 global, NULL, 0, NULL, &event);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));
    err = clWaitForEvents(1, &event);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    // Copy back the resulting C matrix to host memory
    clEnqueueReadBuffer(queue, bufHist, CL_TRUE, 0, HIST_SIZE * 3 * sizeof(unsigned int), hist, 0, NULL, &event);
    err = clWaitForEvents(1, &event);
    if(err != CL_SUCCESS)    printf("%i: %s\n", __LINE__, getErrorString(err));

    gettimeofday(&after, NULL);

    time = (double)(after.tv_sec - before.tv_sec) +
        (double)(after.tv_usec - before.tv_usec) / 1e6;

#ifdef TIMING    
    printf("Kernel took: %.6f seconds.\n", time);
#endif

    // Free the OpenCL memory objects
    clReleaseMemObject(bufImg);
    clReleaseMemObject(bufHist);

    clReleaseKernel(kernel);
}

int image_gen(int width, int height, Image img) {
  int i, img_size=width*height;
  /* Initialize image. */
  for (i = 0; i < img_size; i++) {
      unsigned int tmp = i % HIST_SIZE; 
      img[i] = tmp<<16 | tmp<<8 | tmp ;
  }
  return 0;   
} 

int impl_hist_ver0_seq(int width, int height, Image img, Histogram hist) {
    int i;
    int img_size = width * height;
    struct timeval before, after;
    double time = 0.0;
    
    gettimeofday(&before, NULL);
    
    /* Initialize bins. */
    for (i = 0; i < HIST_SIZE; i++) {
        hist[i] = 0;
        hist[HIST_SIZE+i] = 0;
        hist[HIST_SIZE*2+i] = 0;
    } 

    for (i = 0; i < img_size; i++) {
      unsigned int tmp = img[i];
      hist[tmp & 0x00FF0000 >> 16]++;
      hist[HIST_SIZE+tmp & 0x0000FF00 >> 8]++;
      hist[HIST_SIZE*2+tmp & 0x000000FF]++;  
    }

    gettimeofday(&after, NULL);

    time = (double)(after.tv_sec - before.tv_sec) +
        (double)(after.tv_usec - before.tv_usec) / 1e6;

#ifdef TIMING    
    printf("Kernel took: %.6f seconds.\n", time);
#endif

    return 0;
}

int hist_diff(Histogram h1, Histogram h2) {
    int i;
    unsigned int diffR = 0, diffG=0, diffB=0;  
    for (i=0; i<HIST_SIZE; i++) {
        diffR+=(h1[i] != h2[i]);
        diffG+=(h1[HIST_SIZE+i] != h2[HIST_SIZE+i]); 
        diffB+=(h1[HIST_SIZE*2+i] != h2[HIST_SIZE*2+i]);   
    }  
#ifdef VERBOSE
    if (diffR != 0) {printf("Error in the R channel: %u \n", diffR);}
    if (diffR != 0) {printf("Error in the G channel: %u \n", diffG);}
    if (diffR != 0) {printf("Error in the B channel: %u \n", diffB);}
#endif
    return (diffR!=0 || diffG!=0 || diffB !=0);
}

void calc_gen_img_histogram(int width, int height) {
    Image img = (unsigned int *)malloc(height*width * sizeof(unsigned int)); 
    Histogram _v0_Hist, _v1_Hist, _v2_Hist;

    printf("Input image width, height: %d, %d\n", width, height);
    
    _v0_Hist = (unsigned int*)malloc(HIST_SIZE*3*sizeof(unsigned int));
    _v1_Hist = (unsigned int*)malloc(HIST_SIZE*3*sizeof(unsigned int));  
    _v2_Hist = (unsigned int*)malloc(HIST_SIZE*3*sizeof(unsigned int));  

    image_gen(width, height, img); 
    impl_hist_ver0_seq(width, height, img, _v0_Hist);
    impl_hist_ver1_opencl(width, height, img, _v1_Hist);
    impl_hist_ver2_opencl(width, height, img, _v2_Hist);

    if (hist_diff(_v0_Hist, _v1_Hist)) printf("Error in histogram -- chech ver1! \n");
    if (hist_diff(_v0_Hist, _v2_Hist)) printf("Error in histogram -- chech ver2! \n");

   
    free(img);
    free(_v0_Hist);
    free(_v1_Hist);
    free(_v2_Hist);
    printf("\n");
}

void calc_img_histogram(const char* filename, int width, int height) {
    // Load image
    Image img = load_img_to_array(filename, &width, &height);
    
    // Define histograms
    Histogram _v0_Hist, _v1_Hist, _v2_Hist;

    printf("Input %s; Width: %d, Height: %d\n", filename, width, height);

    // Check each implementation 
    _v0_Hist = (unsigned int*)malloc(HIST_SIZE*3*sizeof(unsigned int));
    _v1_Hist = (unsigned int*)malloc(HIST_SIZE*3*sizeof(unsigned int));  
    _v2_Hist = (unsigned int*)malloc(HIST_SIZE*3*sizeof(unsigned int));  
 
    impl_hist_ver0_seq(width, height, img, _v0_Hist);
    impl_hist_ver1_opencl(width, height, img, _v1_Hist);
    impl_hist_ver2_opencl(width, height, img, _v2_Hist);
    
    if (hist_diff(_v0_Hist, _v1_Hist)) printf("Error in histogram -- chech ver1! \n");
    if (hist_diff(_v0_Hist, _v2_Hist)) printf("Error in histogram -- chech ver2! \n"); 

    free(img);
    free(_v0_Hist);
    free(_v1_Hist);
    free(_v2_Hist);
    printf("\n");
}

// gcc -o ocl_histogram_RGB ocl_histogram_RGB.c -lOpenCL -Wno-deprecated-declarations -Wno-implicit-function-declaration -lm
// ./ocl_histogram_RGB

int main(int argc, char* argv[]) {
    calc_img_histogram("img_1.jpeg",1280, 720);
    calc_img_histogram("img_2.jpeg",1920, 1080);
    calc_gen_img_histogram(10000, 10000);
    printf("All done ... \n");
}
