#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            printf("labwork 1 CPU openmp ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU(FALSE);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            int nDevices = 0;
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

#include <omp.h>

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    // do something here
     omp_set_num_threads(16);
    
        for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!        
        #pragma omp schedule dynamic
            for (int i = 0; i < pixelCount; i++) {
                outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                            (int) inputImage->buffer[i * 3 + 2]) / 3);
                outputImage[i * 3 + 1] = outputImage[i * 3];
                outputImage[i * 3 + 2] = outputImage[i * 3];
            }
        }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

// mplement Labwork::labwork2_GPU() to extract
// information about your GPU(s)
// • Device name
// • Core info: clock rate, core counts, multiprocessor count,
// wrap size
// • Memory info: clock rate, bus width and [optional] bandwidth
// • Hint
// • [Optional] Use cudaGetDeviceCount() to get number of
// NVIDIA GPUs, if you have more than one ©
// • Use cudaGetDeviceProperties(*cudaDeviceProp
// propOut, int deviceId)
// • Examine cudaDeviceProp struc

void Labwork::labwork2_GPU() {
    // get all devices
}

__global__ void grayscale(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
    printf("1\n");
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    uchar3* dev_in = nullptr;
    uchar3* dev_out = nullptr;
    printf("2\n");

    // Allocate CUDA memory    
    cudaMalloc(&dev_in, pixelCount*3);
    cudaMalloc(&dev_out, pixelCount*3);

    printf("4\n");
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(dev_in, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
    printf("5\n");
    // Processing
    int blockSize = 64;
    int numBlock = pixelCount / blockSize;
    grayscale<<<numBlock, blockSize>>>(dev_in, dev_out);
    printf("6\n");
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy( outputImage, dev_out, pixelCount*3, cudaMemcpyDeviceToHost);
    printf("7\n");
    // Cleaning
    cudaFree(dev_in);
    cudaFree(dev_out);
    //free(outputImage);
    printf("8\n");
}

__global__ void grayscale4(uchar3 *input, uchar3 *output, int width, int height ) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if(tidx < width and tidy < height ){
        int tid = tidy*width + tidx;
        output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
        output[tid].z = output[tid].y = output[tid].x;
    }    
}

void Labwork::labwork4_GPU() {
    printf("1\n");
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    uchar3* dev_in = nullptr;
    uchar3* dev_out = nullptr;
    printf("2\n");

    // Allocate CUDA memory    
    cudaMalloc(&dev_in, pixelCount*3);
    cudaMalloc(&dev_out, pixelCount*3);

    printf("3\n");
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(dev_in, inputImage->buffer, pixelCount*3, cudaMemcpyHostToDevice);
    printf("4\n");
    // Processing
    dim3 blockSize = dim3(32,32);
    int gridx, gridy;
    printf("size of image: %d x %d\n", inputImage->width, inputImage->height);
    printf("size of block: %d x %d\n", blockSize.x, blockSize.y);
    if((inputImage->width)%(blockSize.x) == 0 ){
        gridx = (inputImage->width)/(blockSize.x);
    }
    else{
        gridx = (inputImage->width)/(blockSize.x) + 1;
    }

    if((inputImage->height)%(blockSize.y) == 0 ){
        gridy = (inputImage->height)/(blockSize.y);
    }
    else{
        gridy = (inputImage->height)/(blockSize.y) + 1;
    }
    
    dim3 gridSize = dim3(gridx, gridy);
    /* 
    //allocate and copy width and height to device
    int* width ;
    int* height; 
    cudaMalloc(&width, sizeof(int));
    cudaMemcpy(width, &inputImage->width, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&height, sizeof(int));
    cudaMemcpy(height, &inputImage->height, sizeof(int), cudaMemcpyHostToDevice);
 */


    grayscale4<<<gridSize, blockSize>>>(dev_in, dev_out, inputImage->width, inputImage->height );
    printf("5\n");
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy( outputImage, dev_out, pixelCount*3, cudaMemcpyDeviceToHost);
    printf("6\n");
    // Cleaning
    cudaFree(dev_in);
    cudaFree(dev_out);
    //free(outputImage);
    printf("7\n");
}

void Labwork::labwork5_CPU() {
}

void Labwork::labwork5_GPU(bool shared) {
}

void Labwork::labwork6_GPU() {
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























