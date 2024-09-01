#define STB_IMAGE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "stb_image.h"

#define HIST_SIZE	256

#define VERBOSE
#define TIMING


typedef struct {
    unsigned int R[HIST_SIZE];
    unsigned int G[HIST_SIZE];
    unsigned int B[HIST_SIZE];
} Histogram; 

typedef unsigned int* Image;

/*
Image generation: change as you see fit if you want to observe the impact of the image 
on the avhieved performance.
*/
int image_gen(int width, int height, Image img) {
  int i, img_size=width*height;
  /* Initialize image. */
  for (i = 0; i < img_size; i++) {
      unsigned int tmp = i % HIST_SIZE; 
      img[i] = tmp<<16 | tmp<<8 | tmp ;
  }
  return 0;   
} 

/*
Histogram: sequential implementation, used as reference. 
*/
int impl_hist_ver0_seq(int width, int height, Image img, Histogram *hist) {
    int i;
    int img_size = width * height;
    struct timeval before, after;
    double time = 0.0;
    
    /* Initialize bins. */
    for (i = 0; i < HIST_SIZE; i++) {
        hist->R[i] = 0;
        hist->G[i] = 0;
        hist->B[i] = 0;
    } 

    gettimeofday(&before, NULL);

    for (i = 0; i < img_size; i++) {
      unsigned int tmp = img[i];
      hist->R[tmp & 0x00FF0000 >> 16]++;
      hist->G[tmp & 0x0000FF00 >> 8]++;
      hist->B[tmp & 0x000000FF]++;  
    }

    gettimeofday(&after, NULL);

    time = (double)(after.tv_sec - before.tv_sec) +
        (double)(after.tv_usec - before.tv_usec) / 1e6;

#ifdef TIMING    
    printf("Kernel took: %.6f seconds.\n", time);
#endif

    return 0;
}

/*
Histogram: first OpenMP implementation. 
Currently, the code is identical to the code in the sequential method. 
*/
int impl_hist_ver1_openmp(int width, int height, Image img, Histogram *hist) {
/*

Make sure to add the OpenMP pragmas!
Feel free to change any of the code in this function, but not in ver0
Make sure ver1 and ver2 are __different parallel algorithms__ !! 
Don't forget to compile using the right flag - e.g., -fopenmp for gcc

*/

    int i;
    int img_size = width * height;
    struct timeval before, after;
    double time = 0.0;
    
    /* Initialize bins. */
    for (i = 0; i < HIST_SIZE; i++) {
        hist->R[i] = 0;
        hist->G[i] = 0;
        hist->B[i] = 0;
    } 

    // Start timer
    gettimeofday(&before, NULL);

    // Parallelize the calculation using atomic operations
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned int tmp = img[i*width+j];
            #pragma omp atomic
            hist->R[tmp & 0x00FF0000 >> 16]++;
            #pragma omp atomic
            hist->G[tmp & 0x0000FF00 >> 8]++;
            #pragma omp atomic
            hist->B[tmp & 0x000000FF]++;
        }
    }

    // Stop timer
    gettimeofday(&after, NULL);

    time = (double)(after.tv_sec - before.tv_sec) +
        (double)(after.tv_usec - before.tv_usec) / 1e6;

#ifdef TIMING    
    printf("Kernel took: %.6f seconds.\n", time);
#endif
    return 0; 
}


/*
Histogram: second OpenMP implementation. 
Currently, the code is identical to the code in the sequential method. 
*/
int impl_hist_ver2_openmp(int width, int height, Image img, Histogram *hist) {
/*
Make sure to add the OpenMP pragmas!
Feel free to change any of the code in this function, but not in ver0
Make sure ver1 and ver2 are __different parallel algorithms__ !! 
Don't forget to compile using the right flag - e.g., -fopenmp for gcc
*/

    int i;
    int img_size = width * height;
    struct timeval before, after;
    double time = 0.0;
    
    /* Initialize bins. */
    for (i = 0; i < HIST_SIZE; i++) {
        hist->R[i] = 0;
        hist->G[i] = 0;
        hist->B[i] = 0;
    } 

    // Start timer 
    gettimeofday(&before, NULL);

    // Parallelizing the calculation using local copies and reductions
    #pragma omp parallel
    {
        // Private histogram arrays for each thread
        int local_R[HIST_SIZE] = {0};
        int local_G[HIST_SIZE] = {0};
        int local_B[HIST_SIZE] = {0};
        #pragma omp for
        for (int i = 0; i < img_size; i++) {
            unsigned int tmp = img[i];
            local_R[tmp & 0x00FF0000 >> 16]++;
            local_G[tmp & 0x0000FF00 >> 8]++;
            local_B[tmp & 0x000000FF]++;
        }

        // Combine local histograms into the global histogram
        #pragma omp critical
        {
            for (int j = 0; j < HIST_SIZE; j++) {
                hist->R[j] += local_R[j];
                hist->G[j] += local_G[j];
                hist->B[j] += local_B[j];
            }
        }
    }

    // Stop timer
    gettimeofday(&after, NULL);

    time = (double)(after.tv_sec - before.tv_sec) +
        (double)(after.tv_usec - before.tv_usec) / 1e6;

#ifdef TIMING    
    printf("Kernel took: %.6f seconds.\n", time);
#endif
    return 0; 
}

/*
Checks the difference between two histograms.
*/
int hist_diff(Histogram h1, Histogram h2) {
    int i;
    unsigned int diffR = 0, diffG=0, diffB=0;  
    for (i=0; i<HIST_SIZE; i++) {
        diffR+=(h1.R[i] != h2.R[i]);
        diffG+=(h1.G[i] != h2.G[i]); 
        diffB+=(h1.B[i] != h2.B[i]);   
    }  
#ifdef VERBOSE
    if (diffR != 0) {
        printf("Error in the R channel: %u \n", diffR);
    }
    if (diffG != 0) {
        printf("Error in the G channel: %u \n", diffG);
    } 
    if (diffB != 0) {
        printf("Error in the B channel: %u \n", diffB); 
    }
#endif
    return (diffR!=0 || diffG!=0 || diffB !=0);
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
/* 
Calculates the histogram of an image
*/
void calc_img_histogram(const char* filename, int width, int height) {
    // Load image
    Image img = load_img_to_array(filename, &width, &height);
    
    // Define histograms
    Histogram _v0_Hist, _v1_Hist, _v2_Hist;

    printf("Input %s; Width: %d, Height: %d\n", filename, width, height);

    // Check each implementation 
    impl_hist_ver0_seq(width, height, img, &_v0_Hist);
    impl_hist_ver1_openmp(width, height, img, &_v1_Hist);
    impl_hist_ver2_openmp(width, height, img, &_v2_Hist);
     
    if (hist_diff(_v0_Hist, _v1_Hist)) printf("Error in histogram -- chech ver1! \n");
    if (hist_diff(_v0_Hist, _v2_Hist)) printf("Error in histogram -- chech ver2! \n");
    
    // Free up the space allocated to the image
    free(img);

    // Newline separator for the output of the calculation
    printf("\n");
}

/* 
Calculates the histogram of a generated image
*/
void calc_gen_img_histogram(int width, int height) {
    // Generate image
    Image img = (unsigned int *)malloc(height*width * sizeof(unsigned int)); 
    image_gen(width, height, img);
    
    // Define Histograms
    Histogram _v0_Hist, _v1_Hist, _v2_Hist;

    printf("Input generated image; Width: %d, Height: %d\n", width, height);
    
    // Checks each implementation 
    impl_hist_ver0_seq(width, height, img, &_v0_Hist);
    impl_hist_ver1_openmp(width, height, img, &_v1_Hist);
    impl_hist_ver2_openmp(width, height, img, &_v2_Hist);
     
    if (hist_diff(_v0_Hist, _v1_Hist)) printf("Error in histogram -- chech ver1! \n");
    if (hist_diff(_v0_Hist, _v2_Hist)) printf("Error in histogram -- chech ver2! \n");
    
    // Free up the space allocated to the image
    free(img);

    // Newline separator for the output of the calculation
    printf("\n");
}


// To compile the file, run the command:    gcc -fopenmp -o omp_histogram_RGB omp_histogram_RGB.c
// To run the code, run the command:        ./omp_histogram_RGB

int main(int argc, char* argv[]) {
    
    // Calculating histograms for image 1
    calc_img_histogram("img_1.jpeg", 1280, 720);
    
    // Calculating histograms for image 2
    calc_img_histogram("img_2.jpeg", 1920, 1080);
    
    // Calculating the histogram for a generated image; Try experimenting with the size of the image and see how it affects the results
    calc_gen_img_histogram(10000, 10000);

    // Finalized calculating the histograms for all the images
    printf("\nAll done ... \n");
}
