#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <sys/time.h>
#include <cblas.h>
#include <f77blas.h>
#include <pthread.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <arrayfire.h>
#include <af/cuda.h>
#include <fitsio.h>
#include <cublasXt.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_filter.h>
#include <gsl/gsl_min.h>


// rarmijo@158.170.35.147

//nvcc otroconfloat.cu -lcudart -lcublas -lcuda -lblasx -I/opt/arrayfire/include/ -L/opt/arrayfire/lib64/ -lafcuda -lcfitsio -o otroconfloat
/* sudo scp /home/yoyisaurio/Desktop/juguetes\ de\ CUDA/otroconfloat.cu rarmijo@158.170.35.139:/home/rarmijo/Desktop/ */


// nvcc calCompreInfo.cu -lcudart -lcublas -lcuda -lblasx -I/opt/arrayfire/include/ -L/opt/arrayfire/lib64/ -lafcuda -lcfitsio -lgsl -lgslcblas -lm -o calCompreInfo
// sudo scp /home/yoyisaurio/Desktop/proyecto/calCompreInfo.cu rarmijo@beam.diinf.usach.cl:/home/rarmijo
// nvcc calCompreInfo.cu -lcudart -lcublas -lcuda -lblasx -I/opt/arrayfire/include/ -L/opt/arrayfire/lib64/ -lafcuda -lcfitsio -lgsl -lgslcblas -lm -o calCompreInfo
// ./calCompreInfo

// sudo scp /home/yoyisaurio/Desktop/proyecto/nuevo.cu rarmijo@158.170.35.139:/home/rarmijo/Desktop/proyecto

// sudo scp rarmijo@beam.diinf.usach.cl:/home/rarmijo/float_calCompresion_baseNormal_cota99/ite0/reconsImg.fit /home/yoyisaurio/Desktop/ds9/reconsImg.fit

// sudo scp rarmijo@158.170.35.139:/home/rarmijo/Desktop/proyecto/float_calCompresion_baseNormal_cota99/ite0/reconsImg.fit /home/yoyisaurio/Desktop/ds9/nuevito.fit

// nvcc nuevo.cu -lcudart -lcublas -lcuda -lblasx -I/opt/arrayfire/include/ -L/opt/arrayfire/lib64/ -lafcuda -lcfitsio -o nuevo

struct parametros_BaseRect
{
  float* u;
  float* v;
  float* w;
  float delta_u;
  float delta_v;
  float* matrizDeUnos;
  long cantVisi;
  long N;
  float estrechezDeBorde;
};

struct parametros_BaseNormal
{
  float* u;
  float* v;
  float* w;
  float delta_u;
  float delta_v;
  long cantVisi;
  long N;
};

static int Stopping_Rule(float x0, float x1, float tolerance);

#define sqrt5 2.236067977499789696

char* numAString(int* numero)
{
  int cantCarac = (*numero)/10 + 1;
  char* numComoString = (char*) malloc(sizeof(char)*cantCarac);
  return numComoString;
}

float calPendiente(float* x, int largoDeX, float* y)
{
  float sumadeYs = 0.0;
  float sumadeXs = 0.0;
  float sumaDeLosCuadradosdeXs = 0.0;
  float sumaDeMultdeXsconYs = 0.0;
  for(int i=0; i<largoDeX; i++)
  {
    float xActual = x[i];
    float yActual = y[i];
    sumadeYs += yActual;
    sumadeXs += xActual;
    sumaDeMultdeXsconYs += xActual * yActual;
    sumaDeLosCuadradosdeXs += xActual * xActual;
  }
  float cuadradoDeLaSumadeXs = sumadeXs * sumadeXs;
  float numerador = largoDeX * sumaDeMultdeXsconYs - sumadeXs * sumadeYs;
  float denominador = largoDeX * sumaDeLosCuadradosdeXs - cuadradoDeLaSumadeXs;
  return numerador/denominador;
}

float* linspace(float a, float b, long n)
{
    float c;
    int i;
    float* u;
    cudaMallocManaged(&u, n*sizeof(float));
    c = (b - a)/(n - 1);
    for(i = 0; i < n - 1; ++i)
        u[i] = a + i*c;
    u[n - 1] = b;
    return u;
}

void imprimirVector(float* lista, int tamanoLista)
{
  int i;
  for(i=0;i<tamanoLista;i++)
  {
    printf("%f\n",lista[i]);
  }
  printf("\n");
}

void imprimirMatrizColumna(float* vector, long cantFilas, long cantColumnas)
{
  long i,j;
  for(i=0;i<cantFilas;i++)
  {
    for(j=0;j<cantColumnas;j++)
    {
      printf("%.12e ", vector[(((j)*(cantFilas))+(i))]);
    }
    printf("\n");
  }
  printf("\n");
}

void imprimirMatrizPura(float* matriz, int cantFilas, int cantColumnas)
{
  for(int i=0; i<cantFilas; i++)
  {
    for(int j=0; j<cantColumnas; j++)
    {
      printf("%f ", matriz[i*cantColumnas+j]);
    }
    printf("\n");
  }
  printf("\n");
}

void escribirCoefs(float* coefs, char* nombreArchivo, long cantFilas, long cantColumnas)
{
  FILE* archivo = fopen(nombreArchivo, "w");
  for(long i=0;i<cantFilas;i++)
  {
    for(long j=0;j<cantColumnas;j++)
    {
      fprintf(archivo, "%.12e ", coefs[(((j)*(cantFilas))+(i))]);
    }
    fprintf(archivo, "\n");
  }
  fclose(archivo);
}

float** crearMatrizDouble(int cantFilas, int cantColumnas)
{
  float** matriz = (float**) calloc(cantFilas, sizeof(float*));
  int i;
  for(i=0;i<cantFilas;i++)
  {
    matriz[i] = (float*) calloc(cantColumnas, sizeof(float));
  }
  return matriz;
}

void inicializarMatriz(float** matriz, int cantFilas, int cantColumnas)
{
  int i;
  int j;
  int contador = 0;
  for(i=0;i<cantFilas;i++)
  {
      for(j=0;j<cantColumnas;j++)
      {
        matriz[i][j] = contador;
        contador++;
      }
  }
}

float* transformarMatrizAMatrizColumna(float** matriz, int cantFilas, int cantColumnas)
{
  float* nuevoVector = (float*) calloc(cantFilas*cantColumnas,sizeof(float));
  int i,j;
  for(j=0;j<cantColumnas;j++)
  {
    for(i=0;i<cantFilas;i ++)
    {
      nuevoVector[(((j)*(cantFilas))+(i))]= matriz[i][j];
    }
  }
  return nuevoVector;
}

float** transformarMatrizColumnaAMatriz(float* matrizColumna, int cantFilas, int cantColumnas)
{
  float** matriz = crearMatrizDouble(cantFilas,cantColumnas);
  int i,j;
  for(j=0;j<cantColumnas;j++)
  {
    for(i=0;i<cantFilas;i ++)
    {
      matriz[i][j] = matrizColumna[(((j)*(cantFilas))+(i))];
    }
  }
  return matriz;
}

void multMatrices(float* a, long m, long k, float* b, long n, float* c)
{
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasXtHandle_t handle;
  stat = cublasXtCreate(&handle);
  int devices[1] = { 0 };
  if(cublasXtDeviceSelect(handle, 1, devices) != CUBLAS_STATUS_SUCCESS)
  {
    printf("set devices fail\n");
  }
  float al = 1.0;
  float bet = 0.0;
  stat = cublasXtSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
  cudaDeviceSynchronize();
  for(long i=0; i<m*n;i++)
  {
    if(isnan(c[i]))
    {
      printf("Valor nan encontrado en multMatrices.\n");
      break;
    }
  }
  cublasXtDestroy(handle);
}

// void multMatrices(float* a, long m, long k, float* b, long n, float* c)
// {
//   cudaError_t cudaStat;
//   cublasStatus_t stat;
//   cublasHandle_t handle;
//   stat = cublasCreate(&handle);
//   float al = 1.0;
//   float bet = 0.0;
//   stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
//   cudaDeviceSynchronize();
//   for(long i=0; i<m*n;i++)
//   {
//     if(isnan(c[i]))
//     {
//       printf("Valor nan encontrado en multMatrices.\n");
//       break;
//     }
//   }
//   cublasDestroy(handle);
// }

// void combinacionLinealMatrices(float al, float* a, long m, long k, float bet, float* c)
// {
//   long n = k;
//   cudaError_t cudaStat;
//   cublasStatus_t stat;
//   cublasXtHandle_t handle;
//   float* b;
//   cudaMallocManaged(&b, k*n*sizeof(float));
//   cudaMemset(b, 0, k*n*sizeof(float));
//   for(int i=0; i<n; i++)
//   {
//     b[(i*n+i)] = 1.0;
//   }
//   stat = cublasXtCreate(&handle);
//   int devices[1] = { 0 };
//   if(cublasXtDeviceSelect(handle, 1, devices) != CUBLAS_STATUS_SUCCESS)
//   {
//     printf("set devices fail\n");
//   }
//   stat = cublasXtSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
//   cudaDeviceSynchronize();
//   for(long i=0; i<m*n;i++)
//   {
//     if(isnan(c[i]))
//     {
//       printf("Valor nan encontrado en combLinealMatrices.\n");
//       break;
//     }
//   }
//   cudaFree(b);
//   cublasXtDestroy(handle);
// }

__global__ void multMatrizPorConstante_kernel(float* matrizA, long cantFilas, long cantColumnas, float constante)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    matrizA[miId] = constante * matrizA[miId];
  }
}

void multMatrizPorConstante(float* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, float constante)
{
  long cantBloques = ceil((float) cantFilasMatrizA*cantColumnasMatrizA/1024);
  multMatrizPorConstante_kernel<<<cantBloques,1024>>>(matrizA, cantFilasMatrizA, cantColumnasMatrizA, constante);
  cudaDeviceSynchronize();
}

__global__ void combinacionLinealMatrices_kernel(float al, float* matrizA, long cantFilas, long cantColumnas, float bet, float* matrizB)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    matrizB[miId] = al * matrizA[miId] + bet * matrizB[miId];
  }
}

void combinacionLinealMatrices(float al, float* matrizA, long cantFilas, long cantColumnas, float bet, float* matrizB)
{
  long cantBloques = ceil((float) cantFilas*cantColumnas/1024);
  combinacionLinealMatrices_kernel<<<cantBloques,1024>>>(al, matrizA, cantFilas, cantColumnas, bet, matrizB);
  cudaDeviceSynchronize();
}

// void combinacionLinealMatrices(float al, float* a, long m, long k, float bet, float* c)
// {
//   long n = k;
//   cudaError_t cudaStat;
//   cublasStatus_t stat;
//   cublasHandle_t handle;
//   float* b;
//   cudaMallocManaged(&b, k*n*sizeof(float));
//   cudaMemset(b, 0, k*n*sizeof(float));
//   for(int i=0; i<n; i++)
//   {
//     b[(i*n+i)] = 1.0;
//   }
//   stat = cublasCreate(&handle);
//   stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
//   cudaDeviceSynchronize();
//   for(long i=0; i<m*n;i++)
//   {
//     if(isnan(c[i]))
//     {
//       printf("Valor nan encontrado en combLinealMatrices.\n");
//       break;
//     }
//   }
//   cudaFree(b);
//   cublasDestroy(handle);
// }

// void transponerMatriz(float* matriz, int cantFilas, int cantColumnas, float* matrizTranspuesta)
// {
//   for(int i=0;i<cantFilas;i++)
//   {
//     for(int j=0;j<cantColumnas;j++)
//     {
//       matrizTranspuesta[(((i)*(cantColumnas))+(j))] = matriz[(((j)*(cantFilas))+(i))];
//     }
//   }
// }

// __global__ void transponerMatriz_kernel(float* matrizA, float* matrizA_T, long cantFilas, long cantColumnas)
// {
//   long miId = threadIdx.x + blockDim.x * blockIdx.x * blockDim.x * blockDim.y + blockIdx.y * gridDim.x * blockDim.x * blockDim.y;
//   if(miId < cantFilas*cantColumnas)
//   {
//     long i = miId%cantFilas;
//     long j = miId/cantFilas;
//     matrizA_T[(i*cantColumnas+j)] = matrizA[(j*cantFilas+i)];
//   }
// }

__global__ void transponerMatriz_kernel(float* matrizA, float* matrizA_T, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    long i = miId%cantFilas;
    long j = miId/cantFilas;
    matrizA_T[(i*cantColumnas+j)] = matrizA[(j*cantFilas+i)];
  }
}

void transponerMatriz(float* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, float* resultado)
{
  long cantBloques = ceil((float) cantFilasMatrizA*cantColumnasMatrizA/1024);
  transponerMatriz_kernel<<<cantBloques,1024>>>(matrizA, resultado, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

__global__ void restaVectorColumnaConVector_kernel(float* vectorA, long largoVectorA, float* vectorB, long largoVectorB, float* resultado)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < largoVectorA*largoVectorB)
  {
    long i = miId%largoVectorA;
    long j = miId/largoVectorA;
    resultado[miId] = vectorA[i] - vectorB[j];
  }
}

float* restaVectorColumnaConVector(float* vectorA, long largoVectorA, float* vectorB, long largoVectorB)
{
  float* resultado;
  cudaMallocManaged(&resultado,largoVectorA*largoVectorB*sizeof(float));
  long cantBloques = ceil((float) largoVectorA*largoVectorB/1024);
  restaVectorColumnaConVector_kernel<<<cantBloques,1024>>>(vectorA, largoVectorA, vectorB, largoVectorB, resultado);
  cudaDeviceSynchronize();
  return resultado;
}

void vectorColumnaAMatriz(float* vectorA, long cantFilas, long cantColumnas, float* nuevaMatriz)
{
  float* vectorDeUnos;
  cudaMallocManaged(&vectorDeUnos,cantColumnas*sizeof(float));
  for(long i=0; i<cantColumnas; i++)
  {
    vectorDeUnos[i] = 1.0;
  }
  multMatrices(vectorA, cantFilas, 1, vectorDeUnos, cantColumnas, nuevaMatriz);
  cudaFree(vectorDeUnos);
}

__global__ void hadamardProduct_kernel(float* matrizA, float* matrizB, float* resultado, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    resultado[miId] = matrizA[miId]*matrizB[miId];
  }
}

void hadamardProduct(float* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, float* matrizB, float* resultado)
{
  long cantBloques = ceil((float) cantFilasMatrizA*cantColumnasMatrizA/1024);
  hadamardProduct_kernel<<<cantBloques,1024>>>(matrizA, matrizB, resultado, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

float dotProduct(float* x, long n, float* y)
{
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);
  float result;
  stat = cublasSdot(handle,n,x,1,y,1,&result);
  cublasDestroy(handle);
  return result;
}

__global__ void calcularExp_kernel(float* a, float* c, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    c[miId] = exp(a[miId]);
  }
}

void calcularExp(float* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA)
{
  long cantBloques = ceil((float) cantFilasMatrizA*cantColumnasMatrizA/1024);
  calcularExp_kernel<<<cantBloques,1024>>>(matrizA, matrizA, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

__global__ void calcularInvFrac_kernel(float* a, float* c, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    c[miId] = 1.0/a[miId];
  }
}

void calcularInvFrac(float* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA)
{
  long cantBloques = ceil((float) cantFilasMatrizA*cantColumnasMatrizA/1024);
  calcularInvFrac_kernel<<<cantBloques,1024>>>(matrizA, matrizA, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

void calVisModelo(float* MV, long cantFilasMV, long cantColumnasMV, float* MC, long cantColumnasMU, float* MU, float* matrizDeUnosTamN, float* visModelo_paso3)
{
  float* MU_T;
  cudaMallocManaged(&MU_T, cantFilasMV*cantColumnasMU*sizeof(float));
  transponerMatriz(MU, cantFilasMV, cantColumnasMU, MU_T);
  float* visModelo_paso1;
  cudaMallocManaged(&visModelo_paso1, cantColumnasMV*cantFilasMV*sizeof(float));
  cudaMemset(visModelo_paso1, 0, cantColumnasMV*cantFilasMV*sizeof(float));
  multMatrices(MC, cantColumnasMV, cantColumnasMU, MU_T, cantFilasMV, visModelo_paso1);
  cudaFree(MU_T);
  float* transpuesta;
  cudaMallocManaged(&transpuesta, cantColumnasMV*cantFilasMV*sizeof(float));
  transponerMatriz(visModelo_paso1, cantColumnasMV, cantFilasMV, transpuesta);
  cudaFree(visModelo_paso1);
  float* visModelo_paso2;
  cudaMallocManaged(&visModelo_paso2, cantFilasMV*cantColumnasMV*sizeof(float));
  hadamardProduct(MV, cantFilasMV, cantColumnasMV, transpuesta, visModelo_paso2);
  cudaFree(transpuesta);
  multMatrices(visModelo_paso2, cantFilasMV, cantColumnasMV, matrizDeUnosTamN, 1, visModelo_paso3);
  cudaFree(visModelo_paso2);
}

float* calResidual(float* visObs, float* MV, long cantFilasMV, long cantColumnasMV, float* MC, long cantColumnasMU, float* MU, float* matrizDeUnosTamN)
{
  float* visModelo;
  cudaMallocManaged(&visModelo, cantFilasMV*sizeof(float));
  cudaMemset(visModelo, 0, cantFilasMV*sizeof(float));
  calVisModelo(MV, cantFilasMV, cantColumnasMV, MC, cantColumnasMU, MU, matrizDeUnosTamN, visModelo);
  combinacionLinealMatrices(-1.0, visObs, cantFilasMV, 1, 1.0, visModelo);
  return visModelo;
}

float calCosto(float* residual, long cantVisi, float* w)
{
  float* resultado;
  cudaMallocManaged(&resultado, cantVisi*sizeof(float));
  hadamardProduct(residual, cantVisi, 1, w, resultado);
  float total = dotProduct(resultado, cantVisi, residual);
  cudaFree(resultado);
  return total;
}

__global__ void MultPorDifer_kernel(float* matrizA, float* matrizB, float* resultado, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    long posicionEnB = miId%cantFilas;
    resultado[miId] = matrizA[miId]*matrizB[posicionEnB];
  }
}

void MultPorDifer(float* matrizA, long cantFilas, long cantColumnas, float* diferencias, float* resultado)
{
  long cantBloques = ceil((float) cantFilas*cantColumnas/1024);
  MultPorDifer_kernel<<<cantBloques,1024>>>(matrizA, diferencias, resultado, cantFilas, cantColumnas);
  cudaDeviceSynchronize();
}

void calGradiente(float* residual, float* MV, long cantFilasMV, long cantColumnasMV, float* MU, long cantColumnasMU, float* w, float* total_paso2)
{
  float* diferencia;
  cudaMallocManaged(&diferencia, cantFilasMV*sizeof(float));
  hadamardProduct(residual, cantFilasMV, 1, w, diferencia);
  float* total_paso1;
  cudaMallocManaged(&total_paso1, cantColumnasMV*cantFilasMV*sizeof(float));
  MultPorDifer(MV, cantFilasMV, cantColumnasMV, diferencia, total_paso1);
  cudaFree(diferencia);
  float* total_paso1_5;
  cudaMallocManaged(&total_paso1_5, cantColumnasMV*cantFilasMV*sizeof(float));
  transponerMatriz(total_paso1, cantFilasMV, cantColumnasMV, total_paso1_5);
  cudaFree(total_paso1);
  multMatrices(total_paso1_5, cantColumnasMV, cantFilasMV, MU, cantColumnasMU, total_paso2);
  cudaFree(total_paso1_5);
}

float calAlpha(float* gradiente, long cantFilasMC, long cantColumnasMC, float* pActual, float* MV, long cantFilasMV, long cantColumnasMV, float* MU, long cantColumnasMU, float* w, float* matrizDeUnosTamN, int* flag_NOESPOSIBLEMINIMIZAR)
{
  float* gradienteNegativo;
  cudaMallocManaged(&gradienteNegativo, cantFilasMC*cantColumnasMC*sizeof(float));
  cudaMemset(gradienteNegativo, 0, cantFilasMC*cantColumnasMC*sizeof(float));
  combinacionLinealMatrices(-1.0, gradiente, cantFilasMC, cantColumnasMC, 0.0, gradienteNegativo);
  float numerador = dotProduct(gradienteNegativo, cantFilasMC*cantColumnasMC, pActual);
  cudaFree(gradienteNegativo);
  float* visModeloP;
  cudaMallocManaged(&visModeloP, cantFilasMV*sizeof(float));
  cudaMemset(visModeloP, 0, cantFilasMV*sizeof(float));
  calVisModelo(MV, cantFilasMV, cantColumnasMV, pActual, cantColumnasMU, MU, matrizDeUnosTamN, visModeloP);
  float* gradP;
  cudaMallocManaged(&gradP, cantFilasMC * cantColumnasMC*sizeof(float));
  cudaMemset(gradP, 0, cantFilasMC * cantColumnasMC*sizeof(float));
  calGradiente(visModeloP, MV, cantFilasMV, cantColumnasMV, MU, cantColumnasMU, w, gradP);
  cudaFree(visModeloP);
  float denominador = dotProduct(pActual, cantFilasMC * cantColumnasMC, gradP);
  cudaFree(gradP);
  if(denominador == 0.0)
  {
    *flag_NOESPOSIBLEMINIMIZAR = 1;
  }
  return numerador/denominador;
}

float calBeta_Fletcher_Reeves(float* gradienteActual, long tamanoGradiente, float* gradienteAnterior)
{
  float numerador = dotProduct(gradienteActual, tamanoGradiente, gradienteActual);
  float denominador = dotProduct(gradienteAnterior, tamanoGradiente, gradienteAnterior);
  float resultado = numerador/denominador;
  return resultado;
}

float* calInfoFisherDiag(float* MV, long cantFilasMV, long cantColumnasMV, float* MU, float* w)
{
  float* MV_T;
  cudaMallocManaged(&MV_T, cantFilasMV*cantColumnasMV*sizeof(float));
  transponerMatriz(MV, cantFilasMV, cantColumnasMV, MV_T);
  float* primeraMatriz_fase1;
  cudaMallocManaged(&primeraMatriz_fase1, cantColumnasMV*cantFilasMV*sizeof(float));
  hadamardProduct(MV_T, cantColumnasMV, cantFilasMV, MV_T, primeraMatriz_fase1);
  cudaFree(MV_T);
  float* wMatriz;
  cudaMallocManaged(&wMatriz, cantFilasMV*cantColumnasMV*sizeof(float));
  cudaMemset(wMatriz, 0, cantFilasMV*cantColumnasMV*sizeof(float));
  vectorColumnaAMatriz(w, cantFilasMV, cantColumnasMV, wMatriz);
  float* wmatriz_T;
  cudaMallocManaged(&wmatriz_T, cantFilasMV*cantColumnasMV*sizeof(float));
  transponerMatriz(wMatriz, cantFilasMV, cantColumnasMV, wmatriz_T);
  cudaFree(wMatriz);
  float* primeraMatriz_fase2;
  cudaMallocManaged(&primeraMatriz_fase2, cantColumnasMV*cantFilasMV*sizeof(float));
  hadamardProduct(primeraMatriz_fase1, cantColumnasMV, cantFilasMV, wmatriz_T, primeraMatriz_fase2);
  cudaFree(primeraMatriz_fase1);
  cudaFree(wmatriz_T);
  float* MU_T;
  cudaMallocManaged(&MU_T, cantFilasMV*cantColumnasMV*sizeof(float));
  transponerMatriz(MU, cantFilasMV, cantColumnasMV, MU_T);
  float* segundaMatriz;
  cudaMallocManaged(&segundaMatriz, cantFilasMV*cantColumnasMV*sizeof(float));
  hadamardProduct(MU_T, cantFilasMV, cantColumnasMV, MU_T, segundaMatriz);
  cudaFree(MU_T);
  float* resultado_fase1;
  cudaMallocManaged(&resultado_fase1, cantColumnasMV*cantFilasMV*sizeof(float));
  hadamardProduct(primeraMatriz_fase2, cantColumnasMV, cantFilasMV, segundaMatriz, resultado_fase1);
  cudaFree(primeraMatriz_fase2);
  cudaFree(segundaMatriz);
  float* vectorDeUnos;
  cudaMallocManaged(&vectorDeUnos, cantFilasMV*sizeof(float));
  float* resultado_fase2;
  cudaMallocManaged(&resultado_fase2, cantColumnasMV*sizeof(float));
  cudaMemset(resultado_fase2, 0, cantColumnasMV*sizeof(float));
  for(long i=0; i<cantFilasMV; i++)
  {
    vectorDeUnos[i] = 1;
  }
  multMatrices(resultado_fase1, cantColumnasMV, cantFilasMV, vectorDeUnos, 1, resultado_fase2);
  cudaFree(resultado_fase1);
  float medidaInfoMaximoDiagonal = 0.0;
  for (long i=0; i<cantColumnasMV; i++)
  {
      if(resultado_fase2[i] > medidaInfoMaximoDiagonal)
        medidaInfoMaximoDiagonal = resultado_fase2[i];
  }
  float medidaInfoSumaDiagonal = dotProduct(resultado_fase2, cantColumnasMV, vectorDeUnos);
  cudaFree(vectorDeUnos);
  cudaFree(resultado_fase2);
  float* medidasDeInfo = (float*) malloc(sizeof(float)*2);
  medidasDeInfo[0] = medidaInfoSumaDiagonal;
  medidasDeInfo[1] = medidaInfoMaximoDiagonal;
  return medidasDeInfo;
}

float* estimacionDePlanoDeFourier(float* MV, long cantFilasMV, long cantColumnasMV, float* MC, long cantFilasMC, long cantColumnasMC, float* MU)
{
  float* MU_T;
  cudaMallocManaged(&MU_T, cantFilasMV*cantColumnasMV*sizeof(float));
  transponerMatriz(MU, cantFilasMV, cantColumnasMV, MU_T);
  float* resultado_paso1;
  cudaMallocManaged(&resultado_paso1, cantFilasMC*cantFilasMV*sizeof(float));
  cudaMemset(resultado_paso1, 0, cantFilasMC*cantFilasMV*sizeof(float));
  multMatrices(MC, cantFilasMC, cantColumnasMC, MU_T, cantFilasMV, resultado_paso1);
  cudaFree(MU_T);
  float* resultado_paso2;
  cudaMallocManaged(&resultado_paso2, cantFilasMV*cantFilasMV*sizeof(float));
  cudaMemset(resultado_paso2, 0, cantFilasMV*cantFilasMV*sizeof(float));
  multMatrices(MV, cantFilasMV, cantColumnasMV, resultado_paso1, cantFilasMV, resultado_paso2);
  cudaFree(resultado_paso1);
  return resultado_paso2;
}

void printerror_cfitsio( int status)
{
    if (status)
    {
       fits_report_error(stderr, status);
       exit( status );
    }
    return;
}

void escribirTransformadaInversaFourier2D(float* estimacionFourier_ParteImag, float* estimacionFourier_ParteReal, long N, char* nombreArchivo)
{
  af::array estimacionFourier_ParteImag_GPU(N, N, estimacionFourier_ParteImag);
  af::array estimacionFourier_ParteReal_GPU(N, N, estimacionFourier_ParteReal);
  af::array mapaFourierRecons = af::complex(estimacionFourier_ParteReal_GPU, estimacionFourier_ParteImag_GPU);
  estimacionFourier_ParteImag_GPU.unlock();
  estimacionFourier_ParteReal_GPU.unlock();
  mapaFourierRecons = af::shift(mapaFourierRecons, (mapaFourierRecons.dims(0)+1)/2, (mapaFourierRecons.dims(1)+1)/2);
  mapaFourierRecons = af::ifft2(mapaFourierRecons, N, N);
  mapaFourierRecons = af::shift(mapaFourierRecons, (mapaFourierRecons.dims(0)+1)/2, (mapaFourierRecons.dims(1)+1)/2);
  mapaFourierRecons = af::real(mapaFourierRecons);
  mapaFourierRecons = af::flip(mapaFourierRecons, 0);
  mapaFourierRecons = af::transpose(mapaFourierRecons);
  float* auxiliar_mapaFourierRecons = mapaFourierRecons.device<float>();
  float* inver_visi = (float*) calloc(N*N, sizeof(float));
  cudaMemcpy(inver_visi, auxiliar_mapaFourierRecons, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  mapaFourierRecons.unlock();
  fitsfile *fptr;
  int status;
  long fpixel, nelements;
  int bitpix = FLOAT_IMG;
  long naxis = 2;
  long naxes[2] = {N, N};
  remove(nombreArchivo);
  status = 0;
  if (fits_create_file(&fptr, nombreArchivo, &status))
    printerror_cfitsio(status);
  if (fits_create_img(fptr, bitpix, naxis, naxes, &status))
    printerror_cfitsio(status);
  fpixel = 1;
  nelements = naxes[0] * naxes[1];
  if (fits_write_img(fptr, TFLOAT, fpixel, nelements, inver_visi, &status))
    printerror_cfitsio(status);
  if (fits_close_file(fptr, &status))
    printerror_cfitsio(status);
  free(inver_visi);
}

float* calcularMV_Rect(float* v, float delta_v, long cantVisi, long N, float estrechezDeBorde, float ancho, float* matrizDeUnos)
{
  float* desplazamientoEnV = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  float* primeraFraccionV;
  cudaMallocManaged(&primeraFraccionV, cantVisi * N * sizeof(float));
  cudaMemset(primeraFraccionV, 0, cantVisi * N * sizeof(float));
  float* segundaFraccionV;
  cudaMallocManaged(&segundaFraccionV, cantVisi * N * sizeof(float));
  for(long i=0; i<(cantVisi*N); i++)
  {
    segundaFraccionV[i] = 1.0;
  }
  float* matrizDiferenciaV = restaVectorColumnaConVector(v, cantVisi, desplazamientoEnV, N);
  cudaFree(desplazamientoEnV);
  combinacionLinealMatrices(-1.0 * estrechezDeBorde, matrizDiferenciaV, cantVisi, N, 0.0, primeraFraccionV);
  combinacionLinealMatrices(estrechezDeBorde, matrizDiferenciaV, cantVisi, N, -1 * estrechezDeBorde * ancho, segundaFraccionV);
  cudaFree(matrizDiferenciaV);
  calcularExp(primeraFraccionV, cantVisi, N);
  calcularExp(segundaFraccionV, cantVisi, N);
  combinacionLinealMatrices(1.0, matrizDeUnos, cantVisi, N, 1.0, primeraFraccionV);
  combinacionLinealMatrices(1.0, matrizDeUnos, cantVisi, N, 1.0, segundaFraccionV);
  calcularInvFrac(primeraFraccionV, cantVisi, N);
  calcularInvFrac(segundaFraccionV, cantVisi, N);
  float* MV;
  cudaMallocManaged(&MV, cantVisi * N * sizeof(float));
  for(long i=0; i<(cantVisi*N); i++)
  {
    MV[i] = 1.0/ancho;
  }
  combinacionLinealMatrices(1.0, primeraFraccionV, cantVisi, N, 1.0, segundaFraccionV);
  cudaFree(primeraFraccionV);
  combinacionLinealMatrices(1.0/ancho, segundaFraccionV, cantVisi, N, -1.0, MV);
  cudaFree(segundaFraccionV);
  return MV;
}

float* calcularMV_Rect_estFourier(float ancho, long N, float delta_v, float* matrizDeUnos, float estrechezDeBorde, float* matrizDeUnosEstFourier)
{
  float* coordenadasVCentrosCeldas = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(0.5 * delta_v, matrizDeUnosEstFourier, N, 1, 1.0, coordenadasVCentrosCeldas);
  float* MV_AF = calcularMV_Rect(coordenadasVCentrosCeldas, delta_v, N, N, estrechezDeBorde, ancho, matrizDeUnos);
  cudaFree(coordenadasVCentrosCeldas);
  return MV_AF;
}

float* calcularMV_Normal(float* v, float delta_v, long cantVisi, long N, float anchoV)
{
  float* CV;
  cudaMallocManaged(&CV, N * sizeof(float));
  for(long i=0;i<N;i++)
  {
    CV[i] = 0.5 * delta_v;
  }
  float* CV_sinescalar = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(1.0, CV_sinescalar, N, 1, 1.0, CV);
  cudaFree(CV_sinescalar);
  float* MV = restaVectorColumnaConVector(v, cantVisi, CV, N);
  cudaFree(CV);
  multMatrizPorConstante(MV, cantVisi, N, 1.0/anchoV);
  hadamardProduct(MV, cantVisi, N, MV, MV);
  multMatrizPorConstante(MV, cantVisi, N, -0.5);
  calcularExp(MV, cantVisi, N);
  multMatrizPorConstante(MV, cantVisi, N, 1.0/sqrt(2.0 * M_PI * anchoV * anchoV));
  return MV;
}

// float* calcularMV_Normal(float* v, float delta_v, int cantVisi, int N, float anchoV)
// {
//   float* CV = (float*) calloc(N, sizeof(float));
//   float* matrizDeCeros = (float*) calloc(cantVisi * N, sizeof(float));
//   for(int i=0;i<N;i++)
//   {
//     CV[i] = 0.5 * delta_v;
//   }
//   float* CV_sinescalar = linspace((-N/2.0) * delta_v, ((N/2.0) - 1) * delta_v, N);
//   combinacionLinealMatrices(1.0, CV_sinescalar, N, 1, 1.0, CV);
//   free(CV_sinescalar);
//   float* MV = restaVectorColumnaConVector(v, cantVisi, CV, N);
//   free(CV);
//   combinacionLinealMatrices(0.0, matrizDeCeros, cantVisi, N, 1.0/anchoV, MV);
//   hadamardProduct(MV, cantVisi, N, MV, MV);
//   combinacionLinealMatrices(0.0, matrizDeCeros, cantVisi, N, -0.5, MV);
//   calcularExp(MV, cantVisi, N);
//   combinacionLinealMatrices(0.0, matrizDeCeros, cantVisi, N, 1.0/sqrt(2.0 * M_PI * anchoV * anchoV), MV);
//   free(matrizDeCeros);
//   return MV;
// }

float* calcularMV_Normal_estFourier(float anchoV, long N, float delta_v, float* matrizDeUnosEstFourier)
{
  float* coordenadasVCentrosCeldas = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(0.5 * delta_v, matrizDeUnosEstFourier, N, 1, 1.0, coordenadasVCentrosCeldas);
  float* MV_AF = calcularMV_Normal(coordenadasVCentrosCeldas, delta_v, N, N, anchoV);
  cudaFree(coordenadasVCentrosCeldas);
  return MV_AF;
}

int calCompresionSegunCota(char* nombreArCoef_comp_imag, char* nombreArCoef_comp_real, float* MC_imag, float* MC_imag_comp, float* MC_real, float* MC_real_comp, long cantFilas, long cantColumnas, float cotaEnergia)
{
  long largo = cantFilas * cantColumnas;
  float* MC_img_cuadrado;
  cudaMallocManaged(&MC_img_cuadrado, cantFilas*cantColumnas*sizeof(float));
  float* MC_modulo;
  cudaMallocManaged(&MC_modulo, cantFilas*cantColumnas*sizeof(float));
  hadamardProduct(MC_imag, cantFilas, cantColumnas, MC_imag, MC_img_cuadrado);
  hadamardProduct(MC_real, cantFilas, cantColumnas, MC_real, MC_modulo);
  combinacionLinealMatrices(1.0, MC_img_cuadrado, cantFilas, cantColumnas, 1.0, MC_modulo);
  cudaFree(MC_img_cuadrado);
  af::array MC_modulo_GPU(cantFilas*cantColumnas, MC_modulo);
  af::array MC_modulo_indicesOrde_GPU(cantFilas*cantColumnas);
  af::array MC_modulo_Orde_GPU(cantFilas*cantColumnas);
  af::sort(MC_modulo_Orde_GPU, MC_modulo_indicesOrde_GPU, MC_modulo_GPU, 0, false);
  float total = af::sum<float>(MC_modulo_GPU);
  MC_modulo_Orde_GPU = MC_modulo_Orde_GPU/total;
  af::eval(MC_modulo_Orde_GPU);
  af::sync();
  float* auxiliar_MC_modulo_Orde_GPU = MC_modulo_Orde_GPU.device<float>();
  float* coefsNormalizados = (float*) calloc(largo, sizeof(float));
  cudaMemcpy(coefsNormalizados, auxiliar_MC_modulo_Orde_GPU, cantFilas*cantColumnas*sizeof(float), cudaMemcpyDeviceToHost);
  MC_modulo_Orde_GPU.unlock();
  long cantCoefsParaCota = 0;
  float sumador = 0.0;
  for(long i=0; i<largo; i++)
  {
     sumador += coefsNormalizados[i];
     cantCoefsParaCota++;
     if(sumador >= cotaEnergia)
     {
       break;
     }
  }
  cudaFree(MC_modulo);
  free(coefsNormalizados);

  MC_modulo_GPU = MC_modulo_indicesOrde_GPU(af::seq(0,(cantCoefsParaCota-1)));
  af::array indRepComp = af::constant(0, largo);
  indRepComp(MC_modulo_GPU) = 1;
  MC_modulo_GPU.unlock();
  MC_modulo_indicesOrde_GPU.unlock();

  af::array MC_imag_GPU(cantFilas*cantColumnas, MC_imag);
  af::array MC_real_GPU(cantFilas*cantColumnas, MC_real);
  MC_imag_GPU = MC_imag_GPU * indRepComp;
  MC_real_GPU = MC_real_GPU * indRepComp;
  af::eval(MC_imag_GPU);
  af::eval(MC_real_GPU);
  af::sync();
  indRepComp.unlock();
  float* auxiliar_MC_imag_GPU = MC_imag_GPU.device<float>();
  float* auxiliar_MC_real_GPU = MC_real_GPU.device<float>();
  cudaMemcpy(MC_imag_comp, auxiliar_MC_imag_GPU, cantFilas*cantColumnas*sizeof(float), cudaMemcpyDeviceToHost);
  MC_imag_GPU.unlock();
  cudaMemcpy(MC_real_comp, auxiliar_MC_real_GPU, cantFilas*cantColumnas*sizeof(float), cudaMemcpyDeviceToHost);
  MC_real_GPU.unlock();
  escribirCoefs(MC_imag_comp, nombreArCoef_comp_imag, cantFilas, cantColumnas);
  escribirCoefs(MC_real_comp, nombreArCoef_comp_real, cantFilas, cantColumnas);
  return cantCoefsParaCota;
}

float* minGradConjugado_MinCuadra_escritura(char* nombreArchivoMin, char* nombreArchivoCoefs, float* MV, float* MU, float* visibilidades, float* w, long cantVisi, long N, float* matrizDeUnosTamN, int maxIter, float tol)
{
  int flag_NOESPOSIBLEMINIMIZAR = 0;
  float* MC;
  cudaMallocManaged(&MC, N*N*sizeof(float));
  cudaMemset(MC, 0, N*N*sizeof(float));
  float* residualInit = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN);
  float* gradienteActual;
  cudaMallocManaged(&gradienteActual,N*N*sizeof(float));
  cudaMemset(gradienteActual, 0, N*N*sizeof(float));
  float* gradienteAnterior;
  cudaMallocManaged(&gradienteAnterior,N*N*sizeof(float));
  cudaMemset(gradienteAnterior, 0, N*N*sizeof(float));
  float* pActual;
  cudaMallocManaged(&pActual,N*N*sizeof(float));
  cudaMemset(pActual, 0, N*N*sizeof(float));
  float costoInicial = calCosto(residualInit, cantVisi, w);
  float costoAnterior = costoInicial;
  float costoActual = costoInicial;
  calGradiente(residualInit, MV, cantVisi, N, MU, N, w, gradienteAnterior);
  cudaFree(residualInit);
  // for(int i=0; i<N*N; i++)
  // {
  //   if(gradienteAnterior[i] != 0.0)
  //   {
  //     printf("En la linea %d es %f\n", i, gradienteAnterior[i]);
  //   }
  // }
  // exit(-1);
  combinacionLinealMatrices(-1.0, gradienteAnterior, N, N, 0.0, pActual);
  float diferenciaDeCosto = 1.0;
  int i = 0;
  float alpha = 0.0;
  float epsilon = 1e-10;
  float normalizacion = costoAnterior + costoActual + epsilon;
  FILE* archivoMin = fopen(nombreArchivoMin, "w");
  if(archivoMin == NULL)
  {
       printf("Error al crear o abrir el archivo para almacenar la minimizacion.\n");
       exit(0);
  }
  while(maxIter > i && 2.0 * diferenciaDeCosto > tol * normalizacion)
  {
    alpha = calAlpha(gradienteAnterior, N, N, pActual, MV, cantVisi, N, MU, N, w, matrizDeUnosTamN, &flag_NOESPOSIBLEMINIMIZAR);
    if(flag_NOESPOSIBLEMINIMIZAR == 1)
    {
      break;
    }
    combinacionLinealMatrices(alpha, pActual, N, N, 1.0, MC);
    float* residual = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN);
    costoActual = calCosto(residual, cantVisi, w);
    cudaMallocManaged(&gradienteActual,N*N*sizeof(float));
    cudaMemset(gradienteActual, 0, N*N*sizeof(float));
    calGradiente(residual, MV, cantVisi, N, MU, N, w, gradienteActual);
    cudaFree(residual);
    float beta = calBeta_Fletcher_Reeves(gradienteActual, N*N, gradienteAnterior);
    combinacionLinealMatrices(-1.0, gradienteActual, N, N, beta, pActual);
    diferenciaDeCosto = abs(costoAnterior - costoActual);
    normalizacion = costoAnterior + costoActual + epsilon;
    float otro = costoActual - costoAnterior;
    costoAnterior = costoActual;
    float* auxiliar = gradienteAnterior;
    gradienteAnterior = gradienteActual;
    cudaFree(auxiliar);
    i++;
    printf( "En la iteracion %d el valor de la funcion de costos es %f con un z de %.12e la diferencia con respecto al anterior costo es %.12e.\n", i, costoActual, alpha, otro);
    fprintf(archivoMin, "En la iteracion %d el valor de la funcion de costos es %f con un z de %.12e la diferencia con respecto al anterior costo es %.12e.\n", i, costoActual, alpha, otro);
  }
  fclose(archivoMin);
  cudaFree(gradienteAnterior);
  cudaFree(pActual);
  escribirCoefs(MC, nombreArchivoCoefs, N, N);
  return MC;
}

float* minGradConjugado_MinCuadra(float* MV, float* MU, float* visibilidades, float* w, long cantVisi, long N, float* matrizDeUnosTamN, int maxIter, float tol)
{
  int flag_NOESPOSIBLEMINIMIZAR = 0;
  float* MC;
  cudaMallocManaged(&MC, N*N*sizeof(float));
  cudaMemset(MC, 0, N*N*sizeof(float));
  float* residualInit = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN);
  float* gradienteActual;
  cudaMallocManaged(&gradienteActual,N*N*sizeof(float));
  cudaMemset(gradienteActual, 0, N*N*sizeof(float));
  float* gradienteAnterior;
  cudaMallocManaged(&gradienteAnterior,N*N*sizeof(float));
  cudaMemset(gradienteAnterior, 0, N*N*sizeof(float));
  float* pActual;
  cudaMallocManaged(&pActual,N*N*sizeof(float));
  cudaMemset(pActual, 0, N*N*sizeof(float));
  float costoInicial = calCosto(residualInit, cantVisi, w);
  float costoAnterior = costoInicial;
  float costoActual = costoInicial;
  calGradiente(residualInit, MV, cantVisi, N, MU, N, w, gradienteAnterior);
  cudaFree(residualInit);
  combinacionLinealMatrices(-1.0, gradienteAnterior, N, N, 0.0, pActual);
  float diferenciaDeCosto = 1.0;
  int i = 0;
  float alpha = 0.0;
  float epsilon = 1e-10;
  float normalizacion = costoAnterior + costoActual + epsilon;
  while(maxIter > i && 2.0 * diferenciaDeCosto > tol * normalizacion)
  {
    alpha = calAlpha(gradienteAnterior, N, N, pActual, MV, cantVisi, N, MU, N, w, matrizDeUnosTamN, &flag_NOESPOSIBLEMINIMIZAR);
    if(flag_NOESPOSIBLEMINIMIZAR == 1)
    {
      break;
    }
    combinacionLinealMatrices(alpha, pActual, N, N, 1.0, MC);
    float* residual = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN);
    costoActual = calCosto(residual, cantVisi, w);
    cudaMallocManaged(&gradienteActual,N*N*sizeof(float));
    cudaMemset(gradienteActual, 0, N*N*sizeof(float));
    calGradiente(residual, MV, cantVisi, N, MU, N, w, gradienteActual);
    cudaFree(residual);
    float beta = calBeta_Fletcher_Reeves(gradienteActual, N*N, gradienteAnterior);
    combinacionLinealMatrices(-1.0, gradienteActual, N, N, beta, pActual);
    diferenciaDeCosto = abs(costoAnterior - costoActual);
    normalizacion = costoAnterior + costoActual + epsilon;
    float otro = costoActual - costoAnterior;
    costoAnterior = costoActual;
    float* auxiliar = gradienteAnterior;
    gradienteAnterior = gradienteActual;
    cudaFree(auxiliar);
    i++;
    printf( "En la iteracion %d el valor de la funcion de costos es %f con un z de %.12e la diferencia con respecto al anterior costo es %.12e.\n", i, costoActual, alpha, otro);
  }
  cudaFree(gradienteAnterior);
  cudaFree(pActual);
  return MC;
}

float calculateSD(float* data, float mean, long cantElementos)
{
    float SD = 0.0;
    for (long i = 0; i < cantElementos; i++)
        SD += pow(data[i] - mean, 2);
    return sqrt(SD / 10);
}

float calculoDePSNRDeRecorte(float* estimacionFourier_ParteImag, float* estimacionFourier_ParteReal, long N, char* nombreArchivo, clock_t* tiempoTransInver_MejorCompresion)
{
  int columnaDeInicio = 150;
  int columnaDeTermino = 450;
  int filaDeInicio = 100;
  int filaDeTermino = 400;
  *tiempoTransInver_MejorCompresion = clock();
  af::array estimacionFourier_ParteImag_GPU(N, N, estimacionFourier_ParteImag);
  af::array estimacionFourier_ParteReal_GPU(N, N, estimacionFourier_ParteReal);
  af::array mapaFourierRecons = af::complex(estimacionFourier_ParteReal_GPU, estimacionFourier_ParteImag_GPU);
  estimacionFourier_ParteImag_GPU.unlock();
  estimacionFourier_ParteReal_GPU.unlock();
  mapaFourierRecons = af::shift(mapaFourierRecons, (mapaFourierRecons.dims(0)+1)/2, (mapaFourierRecons.dims(1)+1)/2);
  mapaFourierRecons = af::ifft2(mapaFourierRecons, N, N);
  mapaFourierRecons = af::shift(mapaFourierRecons, (mapaFourierRecons.dims(0)+1)/2, (mapaFourierRecons.dims(1)+1)/2);
  mapaFourierRecons = af::real(mapaFourierRecons);
  *tiempoTransInver_MejorCompresion = clock() - *tiempoTransInver_MejorCompresion;
  mapaFourierRecons = af::flip(mapaFourierRecons, 0);
  mapaFourierRecons = af::transpose(mapaFourierRecons);
  float* auxiliar_mapaFourierRecons = mapaFourierRecons.device<float>();
  float* inver_visi = (float*) calloc(N*N, sizeof(float));
  cudaMemcpy(inver_visi, auxiliar_mapaFourierRecons, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  mapaFourierRecons.unlock();

  int cantFilasARecorrer = columnaDeTermino - columnaDeInicio + 1;
  int cantColumnasARecorrer = filaDeTermino - filaDeInicio + 1;
  int contador = 0;
  int contadorEleExternos = 0;
  float sumaDeValoresExternos = 0.0;
  float maximoValorInterno = 0;
  float* nuevaImagen = (float*) calloc(cantFilasARecorrer*cantColumnasARecorrer, sizeof(float));
  float* elementosExternos = (float*) calloc(N*N, sizeof(float));
  for(int j=0; j<N; j++)
  {
    for(int i=0; i<N; i++)
    {
      if(columnaDeInicio <= i && i <= columnaDeTermino && filaDeInicio <= j && j <= filaDeTermino)
      {
          nuevaImagen[contador] = inver_visi[i+j*N];
          if(maximoValorInterno < inver_visi[i+j*N])
          {
            maximoValorInterno = inver_visi[i+j*N];
          }
          contador++;
      }
      else
      {
        elementosExternos[contadorEleExternos] = inver_visi[i+j*N];
        sumaDeValoresExternos += elementosExternos[contadorEleExternos];
        contadorEleExternos++;
      }
    }
  }
  float mediaExterna = sumaDeValoresExternos/contadorEleExternos;
  float desvEstandar = calculateSD(elementosExternos, mediaExterna, contadorEleExternos);
  free(elementosExternos);
  float PSNR = maximoValorInterno/desvEstandar;
  // printf("El contador es %d\n", contador);
  // printf("La wea total es %d\n", cantFilasARecorrer*cantColumnasARecorrer);
  // printf("La cantidad de elementos externos es %d\n", contadorEleExternos);

  fitsfile *fptr;
  int status;
  long fpixel, nelements;
  int bitpix = FLOAT_IMG;
  long naxis = 2;
  // long naxes[2] = {cantFilasARecorrer, cantColumnasARecorrer};
  long naxes[2] = {N, N};
  remove(nombreArchivo);
  status = 0;
  if (fits_create_file(&fptr, nombreArchivo, &status))
    printerror_cfitsio(status);
  if (fits_create_img(fptr, bitpix, naxis, naxes, &status))
    printerror_cfitsio(status);
  fpixel = 1;
  nelements = naxes[0] * naxes[1];
  // if (fits_write_img(fptr, TFLOAT, fpixel, nelements, nuevaImagen, &status))
  if (fits_write_img(fptr, TFLOAT, fpixel, nelements, inver_visi, &status))
    printerror_cfitsio(status);
  if (fits_close_file(fptr, &status))
    printerror_cfitsio(status);
  free(inver_visi);
  free(nuevaImagen);
  return PSNR;
}

void calCompSegunAncho_Normal_escritura(char nombreDirPrin[], char* nombreDirSec, float ancho, float cotaEnergia, int iterActual, int maxIter, float tol, float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal, float delta_u, float delta_v, long cantVisi, long N, float* matrizDeUnosTamN, float estrechezDeBorde)
{
  // ############### CONFIG. DE NOMBRES DE ARCHIVOS  ##############
  char nombreArReconsImg[] = "reconsImg.fit";
  char nombreArReconsCompreImg[] = "reconsCompreImg.fit";
  char nombreArMin_imag[] = "minCoefs_imag.txt";
  char nombreArCoef_imag[] = "coefs_imag.txt";
  char nombreArCoef_comp_imag[] = "coefs_comp_imag.txt";
  char nombreArMin_real[] = "minCoefs_real.txt";
  char nombreArCoef_real[] = "coefs_real.txt";
  char nombreArCoef_comp_real[] = "coefs_comp_real.txt";
  char nombreArInfoCompresion[] = "infoCompre.txt";
  char nombreArInfoTiemposEjecu[] = "infoTiemposEjecu.txt";


  // ############### CALCULO DE MU Y MV - CREACION DE DIRECTORIO SEGUNDARIO  ##############
  printf("...Comenzando calculo de MV...\n");
  clock_t tiempoCalculoMV;
  tiempoCalculoMV = clock();
  float* MV = calcularMV_Normal(v, delta_v, cantVisi, N, ancho);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  float tiempoTotalCalculoMV = ((float)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

  printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  float* MU = calcularMV_Normal(u, delta_u, cantVisi, N, ancho);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  float tiempoTotalCalculoMU = ((float)tiempoCalculoMU)/CLOCKS_PER_SEC;
  printf("Calculo de MU completado.\n");

  char* rutaADirecSec = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*sizeof(char)+sizeof(char)*3);
  strcpy(rutaADirecSec, nombreDirPrin);
  strcat(rutaADirecSec, "/");
  strcat(rutaADirecSec, nombreDirSec);
  if(mkdir(rutaADirecSec, 0777) == -1)
  {
      printf("ERROR: No se pudo crear subdirectorio.");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  strcat(rutaADirecSec, "/");


  // ############### MINIMIZACION DE COEFS, PARTE IMAGINARIA  ##############
  char* nombreArchivoMin_imag = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArMin_imag)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_imag, rutaADirecSec);
  strcat(nombreArchivoMin_imag, nombreArMin_imag);
  char* nombreArchivoCoefs_imag = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_imag)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_imag, rutaADirecSec);
  strcat(nombreArchivoCoefs_imag, nombreArCoef_imag);
  printf("...Comenzando minimizacion de coeficientes parte imaginaria...\n");
  clock_t tiempoMinPartImag;
  tiempoMinPartImag = clock();
  float* MC_imag = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_imag, nombreArchivoCoefs_imag, MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol);
  tiempoMinPartImag = clock() - tiempoMinPartImag;
  float tiempoTotalMinPartImag = ((float)tiempoMinPartImag)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte imaginaria terminado.\n");
  free(nombreArchivoMin_imag);
  free(nombreArchivoCoefs_imag);


  // ############### MINIMIZACION DE COEFS, PARTE REAL  ##############
  char* nombreArchivoMin_real = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArMin_real)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_real, rutaADirecSec);
  strcat(nombreArchivoMin_real, nombreArMin_real);
  char* nombreArchivoCoefs_real = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_real)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_real, rutaADirecSec);
  strcat(nombreArchivoCoefs_real, nombreArCoef_real);
  printf("...Comenzando minimizacion de coeficientes parte real...\n");
  clock_t tiempoMinPartReal;
  tiempoMinPartReal = clock();
  float* MC_real = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_real, nombreArchivoCoefs_real, MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol);
  tiempoMinPartReal = clock() - tiempoMinPartReal;
  float tiempoTotalMinPartReal = ((float)tiempoMinPartReal)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");
  free(nombreArchivoMin_real);
  free(nombreArchivoCoefs_real);


  // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  float* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w);
  tiempoInfo = clock() - tiempoInfo;
  float tiempoTotalInfo = ((float)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);


  // ############### CALCULO DE GRADO DE COMPRESION ##############
  float* MC_comp_imag;
  cudaMallocManaged(&MC_comp_imag,N*N*sizeof(float));
  cudaMemset(MC_comp_imag, 0, N*N*sizeof(float));
  float* MC_comp_real;
  cudaMallocManaged(&MC_comp_real,N*N*sizeof(float));
  cudaMemset(MC_comp_real, 0, N*N*sizeof(float));
  char* nombreArchivoCoef_comp_imag = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_comp_imag)*sizeof(char)*2);
  strcpy(nombreArchivoCoef_comp_imag, rutaADirecSec);
  strcat(nombreArchivoCoef_comp_imag, nombreArCoef_comp_imag);
  char* nombreArchivoCoef_comp_real = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_comp_real)*sizeof(char)*2);
  strcpy(nombreArchivoCoef_comp_real, rutaADirecSec);
  strcat(nombreArchivoCoef_comp_real, nombreArCoef_comp_real);
  printf("...Comenzando calculo de compresion...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  int cantCoefs = calCompresionSegunCota(nombreArchivoCoef_comp_imag, nombreArchivoCoef_comp_real, MC_imag, MC_comp_imag, MC_real, MC_comp_real, N, N, cotaEnergia);
  tiempoCompresion = clock() - tiempoCompresion;
  float tiempoTotalCompresion = ((float)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresion terminado.\n");
  free(nombreArchivoCoef_comp_imag);
  free(nombreArchivoCoef_comp_real);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  FILE* archivo = fopen(nombreArchivoInfoComp, "a");
  float nivelDeCompresion = 1.0 - 1.0 * cantCoefs / N*N;
  fprintf(archivo, "%d %.12f %12.f %.12e %.12e %.12f %.12d\n", iterActual, ancho/delta_u, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, cantCoefs);
  fclose(archivo);
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);


  // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  float* MV_AF = calcularMV_Normal_estFourier(ancho, N, delta_v, matrizDeUnosTamN);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  float tiempoTotalCalculoMV_AF = ((float)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  float* MU_AF = calcularMV_Normal_estFourier(ancho, N, delta_u, matrizDeUnosTamN);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  float tiempoTotalCalculoMU_AF = ((float)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  float* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  float tiempoTotalReconsFourierPartImag = ((float)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  cudaFree(MC_imag);
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  float* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  float tiempoTotalReconsFourierPartReal = ((float)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  cudaFree(MC_real);
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  float tiempoTotalReconsTransInver = ((float)tiempoReconsTransInver)/CLOCKS_PER_SEC;
  cudaFree(estimacionFourier_ParteImag);
  cudaFree(estimacionFourier_ParteReal);
  free(nombreArchivoReconsImg);


  // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION COMPRIMIDA DE LA IMAGEN ##############
  char* nombreArchivoReconsImgComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsCompreImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImgComp, rutaADirecSec);
  strcat(nombreArchivoReconsImgComp, nombreArReconsCompreImg);
  clock_t tiempoReconsFourierPartImagComp;
  tiempoReconsFourierPartImagComp = clock();
  float* estimacionFourier_compre_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_imag, N, N, MU_AF);
  tiempoReconsFourierPartImagComp = clock() - tiempoReconsFourierPartImagComp;
  float tiempoTotalReconsFourierPartImagComp = ((float)tiempoReconsFourierPartImagComp)/CLOCKS_PER_SEC;
  cudaFree(MC_comp_imag);
  clock_t tiempoReconsFourierPartRealComp;
  tiempoReconsFourierPartRealComp = clock();
  float* estimacionFourier_compre_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_real, N, N, MU_AF);
  tiempoReconsFourierPartRealComp = clock() - tiempoReconsFourierPartRealComp;
  float tiempoTotalReconsFourierPartRealComp = ((float)tiempoReconsFourierPartRealComp)/CLOCKS_PER_SEC;
  cudaFree(MC_comp_real);
  clock_t tiempoReconsTransInverComp;
  tiempoReconsTransInverComp = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_compre_ParteImag, estimacionFourier_compre_ParteReal, N, nombreArchivoReconsImgComp);
  tiempoReconsTransInverComp = clock() - tiempoReconsTransInverComp;
  float tiempoTotalReconsTransInverComp = ((float)tiempoReconsTransInverComp)/CLOCKS_PER_SEC;
  cudaFree(estimacionFourier_compre_ParteImag);
  cudaFree(estimacionFourier_compre_ParteReal);
  free(nombreArchivoReconsImgComp);
  cudaFree(MU_AF);
  cudaFree(MV_AF);


  // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
  fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalMinPartImag, tiempoTotalMinPartReal, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver, tiempoTotalReconsFourierPartImagComp, tiempoTotalReconsFourierPartRealComp, tiempoTotalReconsTransInverComp);
  fclose(archivoInfoTiemposEjecu);
  free(nombreArchivoInfoComp);

  free(rutaADirecSec);
}

float calPSNRDeDistintasCompresiones(float inicioIntervalo, float finIntervalo, int cantParamEvaInfo, char rutaADirecSec[], char rutaADirecTer[], char nombreArReconsCompreImg[], float* MC_imag, float* MC_real, float* MV_AF, float* MU_AF, long N, clock_t* tiempoReconsParteImag_MejorCompresion, clock_t* tiempoReconsParteReal_MejorCompresion, clock_t* tiempoTransInver_MejorCompresion)
{
  float cotaMinPSNR = 0.75;
  float cotaMinCompresion = 0.2;
  // float limiteInferior = 0.3;
  // float limiteSuperior = 0.8;
  float* datosDelMin = (float*) malloc(sizeof(float)*4);
  long cantCoefsMejorCompre = 0;
  char nombreArchivoTXTCompre[] = "compresiones.txt";
  char nombreArchivoDatosMinPSNR[] = "mejorTradeOffPSNRCompre.txt";
  char nombreArchivoCompreImg[] = "compreImg";


  float* paramEvaInfo = linspace(inicioIntervalo/100.0, finIntervalo/100.0, cantParamEvaInfo);
  float* MC_comp_imag;
  cudaMallocManaged(&MC_comp_imag,N*N*sizeof(float));
  cudaMemset(MC_comp_imag, 0, N*N*sizeof(float));
  float* MC_comp_real;
  cudaMallocManaged(&MC_comp_real,N*N*sizeof(float));
  cudaMemset(MC_comp_real, 0, N*N*sizeof(float));
  long largo = N * N;
  float* MC_img_cuadrado;
  cudaMallocManaged(&MC_img_cuadrado, N*N*sizeof(float));
  float* MC_modulo;
  cudaMallocManaged(&MC_modulo, N*N*sizeof(float));
  hadamardProduct(MC_imag, N, N, MC_imag, MC_img_cuadrado);
  hadamardProduct(MC_real, N, N, MC_real, MC_modulo);
  combinacionLinealMatrices(1.0, MC_img_cuadrado, N, N, 1.0, MC_modulo);
  cudaFree(MC_img_cuadrado);
  af::array MC_modulo_GPU(N*N, MC_modulo);
  cudaFree(MC_modulo);
  af::array MC_modulo_indicesOrde_GPU(N*N);
  af::array MC_modulo_Orde_GPU(N*N);
  af::sort(MC_modulo_Orde_GPU, MC_modulo_indicesOrde_GPU, MC_modulo_GPU, 0, false);
  float total = af::sum<float>(MC_modulo_GPU);
  MC_modulo_Orde_GPU = MC_modulo_Orde_GPU/total;
  af::eval(MC_modulo_Orde_GPU);
  af::eval(MC_modulo_indicesOrde_GPU);
  af::sync();
  float* auxiliar_MC_modulo_Orde_GPU = MC_modulo_Orde_GPU.device<float>();
  float* auxiliar_MC_modulo_indicesOrde_GPU = MC_modulo_indicesOrde_GPU.device<float>();
  float* coefsNormalizados = (float*) malloc(largo*sizeof(float));
  cudaMemcpy(coefsNormalizados, auxiliar_MC_modulo_Orde_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  int* MC_modulo_indicesOrde_CPU = (int*) malloc(largo*sizeof(int));
  cudaMemcpy(MC_modulo_indicesOrde_CPU, auxiliar_MC_modulo_indicesOrde_GPU, N*N*sizeof(int), cudaMemcpyDeviceToHost);
  MC_modulo_Orde_GPU.unlock();
  MC_modulo_GPU.unlock();
  MC_modulo_indicesOrde_GPU.unlock();
  long cantCoefsParaCota = 0;
  float sumador = 0.0;
  long iExterno = 0;
  float* cantidadPorcentualDeCoefs = linspace(0.0, largo, largo+1);
  combinacionLinealMatrices(0.0, cantidadPorcentualDeCoefs, largo+1, 1, 1.0/largo, cantidadPorcentualDeCoefs);
  char* nombreArchivoCompresiones = (char*) malloc(sizeof(char)*strlen(rutaADirecSec)*strlen(nombreArchivoTXTCompre)+sizeof(char)*4);
  strcpy(nombreArchivoCompresiones, rutaADirecSec);
  strcat(nombreArchivoCompresiones, "/");
  strcat(nombreArchivoCompresiones, nombreArchivoTXTCompre);
  FILE* archivoPSNR = fopen(nombreArchivoCompresiones, "a");
  float* vectorDePSNR = (float*) calloc(cantParamEvaInfo, sizeof(float));
  float* porcenReal = (float*) calloc(cantParamEvaInfo, sizeof(float));
  float* porcenIdeal = (float*) calloc(cantParamEvaInfo, sizeof(float));
  long* cantCoefsUsadas = (long*) calloc(cantParamEvaInfo, sizeof(long));
  float* vectorDePorcenEnergia = (float*) calloc(cantParamEvaInfo, sizeof(float));
  float* vectorDeDifePSNREntrePtosAdya = (float*) calloc(cantParamEvaInfo, sizeof(float));
  int flag_inicioDeVentana = 1;
  int cantPtsVentana = 0;
  int inicioDeVentana = 0;
  clock_t tiempoCualquiera;
  for(long j=0; j<cantParamEvaInfo; j++)
  {
    sumador = 0.0;
    cantCoefsParaCota = 0;
    iExterno = 0;
    for(long i=0; i<largo+1; i++)
    {
      if(cantidadPorcentualDeCoefs[i] < paramEvaInfo[cantParamEvaInfo-1-j])
      {
        sumador += coefsNormalizados[i];
        cantCoefsParaCota++;
      }
      else
      {
        iExterno = i;
        printf("Del %f%% solicitado, el mas cercano correspondiente al %f%% de coefs, lo que corresponde a %ld coeficientes los cuales poseen el %f%% de la energia.\n", paramEvaInfo[cantParamEvaInfo-1-j] * 100, cantidadPorcentualDeCoefs[i] * 100, cantCoefsParaCota, sumador * 100);
        break;
      }
    }
    if(cantCoefsParaCota != 0)
    {
      int* indicesATomar_CPU = (int*) calloc(cantCoefsParaCota, sizeof(int));
      for(int k=0; k<cantCoefsParaCota; k++)
      {
        indicesATomar_CPU[k] = MC_modulo_indicesOrde_CPU[k];
      }
      af::array indicesATomar_GPU(cantCoefsParaCota, indicesATomar_CPU);
      free(indicesATomar_CPU);
      af::array indRepComp = af::constant(0, largo);
      indRepComp(indicesATomar_GPU) = 1;
      indicesATomar_GPU.unlock();
      af::array MC_imag_GPU(N*N, MC_imag);
      af::array MC_real_GPU(N*N, MC_real);
      MC_imag_GPU = MC_imag_GPU * indRepComp;
      MC_real_GPU = MC_real_GPU * indRepComp;
      af::eval(MC_imag_GPU);
      af::eval(MC_real_GPU);
      af::sync();
      indRepComp.unlock();
      float* auxiliar_MC_imag_GPU = MC_imag_GPU.device<float>();
      float* auxiliar_MC_real_GPU = MC_real_GPU.device<float>();
      cudaMemcpy(MC_comp_imag, auxiliar_MC_imag_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
      MC_imag_GPU.unlock();
      cudaMemcpy(MC_comp_real, auxiliar_MC_real_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
      MC_real_GPU.unlock();
      float* estimacionFourier_compre_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_imag, N, N, MU_AF);
      float* estimacionFourier_compre_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_real, N, N, MU_AF);
      int numero = j+1;
      char* numComoString = numAString(&numero);
      sprintf(numComoString, "%d", numero);
      char* nombreArchivoReconsImgComp = (char*) malloc(sizeof(char)*strlen(rutaADirecTer)*strlen(numComoString)*strlen(nombreArchivoCompreImg)+sizeof(char)*7);
      strcpy(nombreArchivoReconsImgComp, rutaADirecTer);
      strcat(nombreArchivoReconsImgComp, "/");
      strcat(nombreArchivoReconsImgComp, nombreArchivoCompreImg);
      strcat(nombreArchivoReconsImgComp, "_");
      strcat(nombreArchivoReconsImgComp, numComoString);
      strcat(nombreArchivoReconsImgComp, ".fit");
      float PSNRActual = calculoDePSNRDeRecorte(estimacionFourier_compre_ParteImag, estimacionFourier_compre_ParteReal, N, nombreArchivoReconsImgComp, &tiempoCualquiera);
      porcenIdeal[j] = 1-paramEvaInfo[cantParamEvaInfo-1-j];
      vectorDePSNR[j] = PSNRActual;
      porcenReal[j] = 1-cantidadPorcentualDeCoefs[iExterno];
      cantCoefsUsadas[j] = cantCoefsParaCota;
      vectorDePorcenEnergia[j] = sumador;
      fprintf(archivoPSNR, "%f %f %f\n", 1-cantidadPorcentualDeCoefs[iExterno], 1-paramEvaInfo[cantParamEvaInfo-1-j], PSNRActual);
      cudaFree(estimacionFourier_compre_ParteImag);
      cudaFree(estimacionFourier_compre_ParteReal);
      free(numComoString);
      free(nombreArchivoReconsImgComp);
    }
  }
  fclose(archivoPSNR);

  float* vectorDePSNRFiltrado = (float*) calloc(cantParamEvaInfo, sizeof(float));
  gsl_vector* vectorDePSNREnGSL = gsl_vector_alloc(cantParamEvaInfo);
  gsl_vector* vectorDePSNREnGSLFiltrado = gsl_vector_alloc(cantParamEvaInfo);
  for(int i=0; i<cantParamEvaInfo; i++)
  {
    gsl_vector_set(vectorDePSNREnGSL, i, vectorDePSNR[i]);
  }
  gsl_filter_gaussian_workspace* gauss_p = gsl_filter_gaussian_alloc(5);
  gsl_filter_gaussian(GSL_FILTER_END_PADVALUE, 1.0, 0, vectorDePSNREnGSL, vectorDePSNREnGSLFiltrado, gauss_p);
  for(int i=0; i<cantParamEvaInfo; i++)
  {
    vectorDePSNRFiltrado[i] = gsl_vector_get(vectorDePSNREnGSLFiltrado, i);
  }
  gsl_vector_free(vectorDePSNREnGSL);
  gsl_vector_free(vectorDePSNREnGSLFiltrado);
  gsl_filter_gaussian_free(gauss_p);


  // float* listaDeMetricas = (float*) malloc(sizeof(float)*cantParamEvaInfo);
  // float* primeraRecta_subListaDeX = (float*) calloc(cantParamEvaInfo, sizeof(float));
  // float* primeraRecta_subListaDeY = (float*) calloc(cantParamEvaInfo, sizeof(float));
  // float* segundaRecta_subListaDeX = (float*) calloc(cantParamEvaInfo, sizeof(float));
  // float* segundaRecta_subListaDeY = (float*) calloc(cantParamEvaInfo, sizeof(float));
  // memcpy(segundaRecta_subListaDeX, porcenReal, sizeof(float)*cantParamEvaInfo);
  // memcpy(segundaRecta_subListaDeY, vectorDePSNRFiltrado, sizeof(float)*cantParamEvaInfo);
  // primeraRecta_subListaDeX[0] = porcenReal[0];
  // primeraRecta_subListaDeY[0] = vectorDePSNRFiltrado[0];
  // float metricaMin;
  // float metricaActual;
  // int flagPrimerValorParaMetricaMin = 0;
  // printf("7\n");
  // for(int i=1; i<cantParamEvaInfo-1; i++)
  // {
  //     primeraRecta_subListaDeX[i] = porcenReal[i];
  //     primeraRecta_subListaDeY[i] = vectorDePSNRFiltrado[i];
  //     float pendienteDePrimeraRecta = calPendiente(primeraRecta_subListaDeX, i+1, primeraRecta_subListaDeY);
  //     segundaRecta_subListaDeX[i-1] = 0.0;
  //     segundaRecta_subListaDeY[i-1] = 0.0;
  //     float pendienteDeSegundaRecta = calPendiente(&(segundaRecta_subListaDeX[i]), cantParamEvaInfo-i, &(segundaRecta_subListaDeY[i]));
  //     metricaActual = -1.0 * pendienteDeSegundaRecta/pendienteDePrimeraRecta;
  //     listaDeMetricas[i] = metricaActual;
  //     if(limiteInferior <= porcenReal[i] && porcenReal[i] <= limiteSuperior)
  //     {
  //       if(flagPrimerValorParaMetricaMin == 0)
  //       {
  //         metricaMin = metricaActual;
  //         datosDelMin[0] = porcenIdeal[i];
  //         datosDelMin[1] = porcenReal[i];
  //         cantCoefsMejorCompre = cantCoefsUsadas[i];
  //         datosDelMin[2] = vectorDePorcenEnergia[i];
  //         datosDelMin[3] = vectorDePSNR[i];
  //         flagPrimerValorParaMetricaMin = 1;
  //       }
  //       if(metricaActual < metricaMin)
  //       {
  //         metricaMin = metricaActual;
  //         datosDelMin[0] = porcenIdeal[i];
  //         datosDelMin[1] = porcenReal[i];
  //         cantCoefsMejorCompre = cantCoefsUsadas[i];
  //         datosDelMin[2] = vectorDePorcenEnergia[i];
  //         datosDelMin[3] = vectorDePSNR[i];
  //       }
  //     }
  // }

  FILE* archivoRandom = fopen("wea.txt", "w");
  for(int i=0; i<cantParamEvaInfo; i++)
  {
      fprintf(archivoRandom, "%f\n", vectorDePSNRFiltrado[i]);
  }
  fclose(archivoRandom);

  // free(vectorDePSNRFiltrado);
  // free(primeraRecta_subListaDeX);
  // free(primeraRecta_subListaDeY);
  // free(segundaRecta_subListaDeX);
  // free(segundaRecta_subListaDeY);
  // free(porcenIdeal);
  // free(porcenReal);
  // free(cantCoefsUsadas);
  // free(vectorDePorcenEnergia);
  // free(vectorDePSNR);


  for(int j=0; j<cantParamEvaInfo; j++)
  {
    float porcenActual = porcenReal[j];
    float porcenDifActual = vectorDePSNRFiltrado[j]/vectorDePSNRFiltrado[0];
    if(j >= 1)
    {
      if(porcenActual >= cotaMinCompresion && porcenDifActual >= cotaMinPSNR)
      {
        if(flag_inicioDeVentana)
        {
          inicioDeVentana = j;
          flag_inicioDeVentana = 0;
        }
        vectorDeDifePSNREntrePtosAdya[cantPtsVentana] = vectorDePSNRFiltrado[j] - vectorDePSNRFiltrado[j-1];
        // printf("%.12e\n", vectorDeDifePSNREntrePtosAdya[cantPtsVentana]);
        cantPtsVentana++;
      }
    }
  }

  af::array vectorDeDifePSNREntrePtosAdya_GPU(cantPtsVentana, vectorDeDifePSNREntrePtosAdya);
  free(vectorDeDifePSNREntrePtosAdya);
  af::array vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU(cantPtsVentana);
  af::array vectorDeDifePSNREntrePtosAdya_Orde_GPU(cantPtsVentana);
  af::sort(vectorDeDifePSNREntrePtosAdya_Orde_GPU, vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU, vectorDeDifePSNREntrePtosAdya_GPU, 0, true);
  vectorDeDifePSNREntrePtosAdya_GPU.unlock();
  vectorDeDifePSNREntrePtosAdya_Orde_GPU.unlock();
  int* auxiliar_vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU = vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU.device<int>();
  int* vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU = (int*) malloc(sizeof(int)*cantPtsVentana);
  cudaMemcpy(vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU, auxiliar_vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU, cantPtsVentana*sizeof(int), cudaMemcpyDeviceToHost);
  vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU.unlock();
  // int indiceElegido = vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU[0] + inicioDeVentana - 1;
  printf("El indice elegido es %d\n", indiceElegido);
  free(vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU);
  datosDelMin[0] = porcenIdeal[indiceElegido];
  datosDelMin[1] = porcenReal[indiceElegido];
  cantCoefsMejorCompre = cantCoefsUsadas[indiceElegido];
  datosDelMin[2] = vectorDePorcenEnergia[indiceElegido];
  datosDelMin[3] = vectorDePSNR[indiceElegido];
  free(vectorDePSNRFiltrado);
  free(porcenIdeal);
  free(porcenReal);
  free(cantCoefsUsadas);
  free(vectorDePorcenEnergia);
  free(vectorDePSNR);


  char* nombreArchivoMejorCompre = (char*) malloc(sizeof(char)*strlen(rutaADirecSec)*strlen(nombreArchivoDatosMinPSNR)+sizeof(char)*4);
  strcpy(nombreArchivoMejorCompre, rutaADirecSec);
  strcat(nombreArchivoMejorCompre, "/");
  strcat(nombreArchivoMejorCompre, nombreArchivoDatosMinPSNR);
  FILE* archivoMejorCompre = fopen(nombreArchivoMejorCompre, "w");
  fprintf(archivoMejorCompre, "El tradeoff seleccionado corresponde al %f%% de coefs, el mas cercano correspondiente al %f%% de coefs, lo que corresponde a %ld coeficientes los cuales poseen el %f%% de la energia y un PSNR de %f%%.\n", datosDelMin[0]  * 100, datosDelMin[1]  * 100, cantCoefsMejorCompre, datosDelMin[2]  * 100, datosDelMin[3]);
  free(nombreArchivoMejorCompre);
  free(datosDelMin);
  fclose(archivoMejorCompre);
  float* indicesATomar_CPU = (float*) malloc(cantCoefsMejorCompre*sizeof(float));
  for(int k=0; k<cantCoefsMejorCompre; k++)
  {
    indicesATomar_CPU[k] = MC_modulo_indicesOrde_CPU[k];
  }
  af::array indicesATomar_GPU(cantCoefsMejorCompre, indicesATomar_CPU);
  free(indicesATomar_CPU);
  af::array indRepComp = af::constant(0, largo);
  indRepComp(indicesATomar_GPU) = 1;
  indicesATomar_GPU.unlock();
  af::array MC_imag_GPU(N*N, MC_imag);
  af::array MC_real_GPU(N*N, MC_real);
  MC_imag_GPU = MC_imag_GPU * indRepComp;
  MC_real_GPU = MC_real_GPU * indRepComp;
  af::eval(MC_imag_GPU);
  af::eval(MC_real_GPU);
  af::sync();
  indRepComp.unlock();
  float* auxiliar_MC_imag_GPU = MC_imag_GPU.device<float>();
  float* auxiliar_MC_real_GPU = MC_real_GPU.device<float>();
  cudaMemcpy(MC_comp_imag, auxiliar_MC_imag_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  MC_imag_GPU.unlock();
  cudaMemcpy(MC_comp_real, auxiliar_MC_real_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  MC_real_GPU.unlock();
  *tiempoReconsParteImag_MejorCompresion = clock();
  float* estimacionFourier_compre_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_imag, N, N, MU_AF);
  *tiempoReconsParteImag_MejorCompresion = clock() - *tiempoReconsParteImag_MejorCompresion;
  *tiempoReconsParteReal_MejorCompresion  = clock();
  float* estimacionFourier_compre_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_real, N, N, MU_AF);
  *tiempoReconsParteReal_MejorCompresion = clock() - *tiempoReconsParteReal_MejorCompresion;
  char* nombreArchivoReconsImgComp = (char*) malloc(sizeof(char)*strlen(rutaADirecSec)*strlen(nombreArReconsCompreImg)+sizeof(char)*4);
  strcpy(nombreArchivoReconsImgComp, rutaADirecSec);
  strcat(nombreArchivoReconsImgComp, "/");
  strcat(nombreArchivoReconsImgComp, nombreArReconsCompreImg);
  float PSNRActual = calculoDePSNRDeRecorte(estimacionFourier_compre_ParteImag, estimacionFourier_compre_ParteReal, N, nombreArchivoReconsImgComp, tiempoTransInver_MejorCompresion);
  cudaFree(estimacionFourier_compre_ParteImag);
  cudaFree(estimacionFourier_compre_ParteReal);

  cudaFree(MC_comp_imag);
  cudaFree(MC_comp_real);
  cudaFree(cantidadPorcentualDeCoefs);
  cudaFree(paramEvaInfo);
  cudaFree(MU_AF);
  cudaFree(MV_AF);
  free(coefsNormalizados);
  free(MC_modulo_indicesOrde_CPU);
  free(nombreArchivoCompresiones);
  return cantCoefsMejorCompre;
}

void calCompSegunAncho_Rect_escritura(char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], float ancho, float cotaEnergia, int iterActual, int maxIter, float tol, float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal, float delta_u, float delta_v, float* matrizDeUnos, long cantVisi, long N, float* matrizDeUnosTamN, float estrechezDeBorde)
{
  float inicioPorcenCompre = 0.0;
  float terminoPorcenCompre = 0.2;
  int cantPorcen = 101;
  // int cantPorcen = 2;


  // ############### CONFIG. DE NOMBRES DE ARCHIVOS  ##############
  char nombreArReconsImg[] = "reconsImg.fit";
  char nombreArReconsCompreImg[] = "reconsCompreImg.fit";
  char nombreArMin_imag[] = "minCoefs_imag.txt";
  char nombreArCoef_imag[] = "coefs_imag.txt";
  char nombreArCoef_comp_imag[] = "coefs_comp_imag.txt";
  char nombreArMin_real[] = "minCoefs_real.txt";
  char nombreArCoef_real[] = "coefs_real.txt";
  char nombreArCoef_comp_real[] = "coefs_comp_real.txt";
  char nombreArInfoCompresion[] = "infoCompre.txt";
  char nombreArInfoTiemposEjecu[] = "infoTiemposEjecu.txt";


  // ############### CALCULO DE MU Y MV - CREACION DE DIRECTORIO SEGUNDARIO  ##############
  printf("...Comenzando calculo de MV...\n");
  clock_t tiempoCalculoMV;
  tiempoCalculoMV = clock();
  float* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  float tiempoTotalCalculoMV = ((float)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

  printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  float* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  float tiempoTotalCalculoMU = ((float)tiempoCalculoMU)/CLOCKS_PER_SEC;
  printf("Calculo de MU completado.\n");

  char* rutaADirecSec = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*sizeof(char)+sizeof(char)*3);
  strcpy(rutaADirecSec, nombreDirPrin);
  strcat(rutaADirecSec, "/");
  strcat(rutaADirecSec, nombreDirSec);
  if(mkdir(rutaADirecSec, 0777) == -1)
  {
      printf("ERROR: No se pudo crear subdirectorio.");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  strcat(rutaADirecSec, "/");


  // ############### MINIMIZACION DE COEFS, PARTE IMAGINARIA  ##############
  char* nombreArchivoMin_imag = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArMin_imag)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_imag, rutaADirecSec);
  strcat(nombreArchivoMin_imag, nombreArMin_imag);
  char* nombreArchivoCoefs_imag = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_imag)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_imag, rutaADirecSec);
  strcat(nombreArchivoCoefs_imag, nombreArCoef_imag);
  printf("...Comenzando minimizacion de coeficientes parte imaginaria...\n");
  clock_t tiempoMinPartImag;
  tiempoMinPartImag = clock();
  float* MC_imag = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_imag, nombreArchivoCoefs_imag, MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol);
  tiempoMinPartImag = clock() - tiempoMinPartImag;
  float tiempoTotalMinPartImag = ((float)tiempoMinPartImag)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte imaginaria terminado.\n");
  free(nombreArchivoMin_imag);
  free(nombreArchivoCoefs_imag);


  // ############### MINIMIZACION DE COEFS, PARTE REAL  ##############
  char* nombreArchivoMin_real = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArMin_real)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_real, rutaADirecSec);
  strcat(nombreArchivoMin_real, nombreArMin_real);
  char* nombreArchivoCoefs_real = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_real)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_real, rutaADirecSec);
  strcat(nombreArchivoCoefs_real, nombreArCoef_real);
  printf("...Comenzando minimizacion de coeficientes parte real...\n");
  clock_t tiempoMinPartReal;
  tiempoMinPartReal = clock();
  float* MC_real = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_real, nombreArchivoCoefs_real, MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol);
  tiempoMinPartReal = clock() - tiempoMinPartReal;
  float tiempoTotalMinPartReal = ((float)tiempoMinPartReal)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");
  free(nombreArchivoMin_real);
  free(nombreArchivoCoefs_real);


  // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  float* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w);
  tiempoInfo = clock() - tiempoInfo;
  float tiempoTotalInfo = ((float)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);


  // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  float* MV_AF = calcularMV_Rect_estFourier(ancho, N, delta_v, matrizDeUnos, estrechezDeBorde, matrizDeUnosTamN);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  float tiempoTotalCalculoMV_AF = ((float)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  float* MU_AF = calcularMV_Rect_estFourier(ancho, N, delta_u, matrizDeUnos, estrechezDeBorde, matrizDeUnosTamN);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  float tiempoTotalCalculoMU_AF = ((float)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  float* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  float tiempoTotalReconsFourierPartImag = ((float)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  float* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  float tiempoTotalReconsFourierPartReal = ((float)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  float tiempoTotalReconsTransInver = ((float)tiempoReconsTransInver)/CLOCKS_PER_SEC;
  cudaFree(estimacionFourier_ParteImag);
  cudaFree(estimacionFourier_ParteReal);
  free(nombreArchivoReconsImg);


  // ############### CALCULO DE GRADO DE COMPRESION ##############
  char* rutaADirecTer = (char*) malloc(strlen(rutaADirecSec)*strlen(nombreDirTer)*sizeof(char)+sizeof(char)*3);
  strcpy(rutaADirecTer, rutaADirecSec);
  strcat(rutaADirecTer, "/");
  strcat(rutaADirecTer, nombreDirTer);
  if(mkdir(rutaADirecTer, 0777) == -1)
  {
    printf("ERROR: No se pudo crear subdirectorio.\n");
    printf("PROGRAMA ABORTADO.\n");
    exit(0);
  }
  strcat(rutaADirecTer, "/");
  clock_t tiempoReconsFourierPartImagComp;
  clock_t tiempoReconsFourierPartRealComp;
  clock_t tiempoReconsTransInverComp;
  printf("...Comenzando calculo de compresiones...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  int cantCoefs = calPSNRDeDistintasCompresiones(inicioPorcenCompre, terminoPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, &tiempoReconsFourierPartImagComp, &tiempoReconsFourierPartRealComp, &tiempoReconsTransInverComp);
  tiempoCompresion = clock() - tiempoCompresion;
  float tiempoTotalCompresion = ((float)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresiones terminado.\n");
  free(rutaADirecTer);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  FILE* archivo = fopen(nombreArchivoInfoComp, "a");
  float nivelDeCompresion = 1.0 - cantCoefs * 1.0 / N*N;
  fprintf(archivo, "%d %.12f %.12e %.12e %.12f %.12d\n", iterActual, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, cantCoefs);
  fclose(archivo);
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);

  cudaFree(MC_real);
  cudaFree(MC_imag);
  cudaFree(MU_AF);
  cudaFree(MV_AF);
  float tiempoTotalReconsFourierPartImagComp = ((float)tiempoReconsFourierPartImagComp)/CLOCKS_PER_SEC;
  float tiempoTotalReconsFourierPartRealComp = ((float)tiempoReconsFourierPartRealComp)/CLOCKS_PER_SEC;
  float tiempoTotalReconsTransInverComp = ((float)tiempoReconsTransInverComp)/CLOCKS_PER_SEC;

  // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
  fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalMinPartImag, tiempoTotalMinPartReal, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver, tiempoTotalReconsFourierPartImagComp, tiempoTotalReconsFourierPartRealComp, tiempoTotalReconsTransInverComp);
  fclose(archivoInfoTiemposEjecu);

  free(rutaADirecSec);
}

double funcOptiInfo_Traza_Rect(double ancho, void* params)
{
  struct parametros_BaseRect* ps = (struct parametros_BaseRect*) params;
  float* MV = calcularMV_Rect(ps->v, ps->delta_v, ps->cantVisi, ps->N, ps->estrechezDeBorde, ancho, ps->matrizDeUnos);
  float* MU = calcularMV_Rect(ps->u, ps->delta_u, ps->cantVisi, ps->N, ps->estrechezDeBorde, ancho, ps->matrizDeUnos);
  float* medidasDeInfo = calInfoFisherDiag(MV, ps->cantVisi, ps->N, MU, ps->w);
  float medidaSumaDeLaDiagonal = medidasDeInfo[0];
  free(medidasDeInfo);
  cudaFree(MV);
  cudaFree(MU);
  return -1 * medidaSumaDeLaDiagonal;
}

double funcOptiInfo_Traza_Normal(double ancho, void* params)
{
  struct parametros_BaseNormal* ps = (struct parametros_BaseNormal*) params;
  float* MV = calcularMV_Normal(ps->v, ps->delta_v, ps->cantVisi, ps->N, ancho);
  float* MU = calcularMV_Normal(ps->u, ps->delta_u, ps->cantVisi, ps->N, ancho);
  float* medidasDeInfo = calInfoFisherDiag(MV, ps->cantVisi, ps->N, MU, ps->w);
  float medidaSumaDeLaDiagonal = medidasDeInfo[0];
  free(medidasDeInfo);
  cudaFree(MV);
  cudaFree(MU);
  return -1 * medidaSumaDeLaDiagonal;
}

double goldenMin_BaseRect(float* u, float* v, float* w, float delta_u, float delta_v, float* matrizDeUnos, long cantVisi, long N, float estrechezDeBorde)
{
  int status;
  int iter = 0, max_iter = 100;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  gsl_function F;
  parametros_BaseRect actual;
  actual.u = u;
  actual.v = v;
  actual.w = w;
  actual.delta_u = delta_u;
  actual.delta_v = delta_v;
  actual.matrizDeUnos = matrizDeUnos;
  actual.cantVisi = cantVisi;
  actual.N = N;
  actual.estrechezDeBorde = estrechezDeBorde;
  double m;
  double a = 1.0 * actual.delta_u, b = 5.0 * actual.delta_u;
  F.function = &funcOptiInfo_Traza_Rect;
  void* punteroVoidAActual = &actual;
  F.params = punteroVoidAActual;

  T = gsl_min_fminimizer_quad_golden;
  s = gsl_min_fminimizer_alloc (T);
  gsl_set_error_handler_off();

  m = 1.0 * actual.delta_u;
  int status_interval = gsl_min_fminimizer_set (s, &F, m, a, b);


  while(status_interval)
  {
    m += 0.001 * actual.delta_u;
    printf("m ahora es %f\n", m/actual.delta_u);
    status_interval = gsl_min_fminimizer_set (s, &F, m, a, b);
  }

  printf ("using %s method\n",
          gsl_min_fminimizer_name (s));

  printf ("%5s [%9s, %9s] %9s\n",
          "iter", "lower", "upper", "min");

  printf ("%5d [%.7f, %.7f] %.7f\n",
          iter, a, b, m);

  do
    {
      iter++;
      status = gsl_min_fminimizer_iterate (s);

      m = gsl_min_fminimizer_x_minimum (s);
      a = gsl_min_fminimizer_x_lower (s);
      b = gsl_min_fminimizer_x_upper (s);

      status = gsl_min_test_interval (a, b, 0.01, 0.01);

      if (status == GSL_SUCCESS)
        printf ("Converged:\n");

      printf ("%5d [%.7f, %.7f] "
              "%.7f\n",
              iter, a/delta_u, b/delta_u, m/delta_u);
    }
  while (status == GSL_CONTINUE && iter < max_iter);

  gsl_min_fminimizer_free (s);
  return m;
}

double goldenMin_BaseNormal(float* u, float* v, float* w, float delta_u, float delta_v, long cantVisi, long N)
{
  int status;
  int iter = 0, max_iter = 100;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  gsl_function F;
  parametros_BaseNormal actual;
  actual.u = u;
  actual.v = v;
  actual.w = w;
  actual.delta_u = delta_u;
  actual.delta_v = delta_v;
  actual.cantVisi = cantVisi;
  actual.N = N;
  double m = 1.5 * actual.delta_u, m_expected = M_PI;
  double a = 1.0 * actual.delta_u, b = 5.0 * actual.delta_u;
  F.function = &funcOptiInfo_Traza_Normal;
  void* punteroVoidAActual = &actual;
  F.params = punteroVoidAActual;

  T = gsl_min_fminimizer_quad_golden;
  s = gsl_min_fminimizer_alloc (T);
  gsl_set_error_handler_off();

  m = 1.0 * actual.delta_u;
  int status_interval = gsl_min_fminimizer_set (s, &F, m, a, b);


  while(status_interval)
  {
    m += 0.001 * actual.delta_u;
    printf("m ahora es %f\n", m/actual.delta_u);
    status_interval = gsl_min_fminimizer_set (s, &F, m, a, b);
  }

  printf ("using %s method\n",
          gsl_min_fminimizer_name (s));

  printf ("%5s [%9s, %9s] %9s\n",
          "iter", "lower", "upper", "min");

  printf ("%5d [%.7f, %.7f] %.7f\n",
          iter, a, b, m);

  do
    {
      iter++;
      status = gsl_min_fminimizer_iterate (s);

      m = gsl_min_fminimizer_x_minimum (s);
      a = gsl_min_fminimizer_x_lower (s);
      b = gsl_min_fminimizer_x_upper (s);

      status
        = gsl_min_test_interval (a, b, 0.001, 0.0);

      if (status == GSL_SUCCESS)
        printf ("Converged:\n");

      printf ("%5d [%.7f, %.7f] "
              "%.7f\n",
              iter, a/delta_u, b/delta_u,m/delta_u);
    }
  while (status == GSL_CONTINUE && iter < max_iter);

  gsl_min_fminimizer_free (s);
  return m;
}

void lecturaDeTXT(char nombreArchivo[], float* frecuencia, float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal, long cantVisi)
{
  long contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  float c_constant = 2.99792458E8;
  fp = fopen(nombreArchivo, "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  while ((read = getline(&line, &len, fp)) != -1)
  {
    *frecuencia = atof(strtok(line, " "));
    visi_parteReal[contador] = atof(strtok(NULL, " "));
    visi_parteImaginaria[contador] = atof(strtok(NULL, " "));
    u[contador] = atof(strtok(NULL, " ")) * (*frecuencia)/c_constant;
    v[contador] = atof(strtok(NULL, " ")) * (*frecuencia)/c_constant;
    w[contador] = atof(strtok(NULL, " "));
    contador++;
    if(contador == cantVisi)
      break;
	}
  free(line);
  fclose(fp);
}

void lectCantVisi(char nombreArchivo[], long* cantVisi)
{
  long contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  char* nombreNuevoTXT = (char*) malloc(strlen(nombreArchivo)*sizeof(char)+sizeof(char)*20);
  strcpy(nombreNuevoTXT, nombreArchivo);
  strcat(nombreNuevoTXT, "cantvisi.txt");
  fp = fopen(nombreNuevoTXT, "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  read = getline(&line, &len, fp);
  printf("Se han leido %s visibilidades.\n", line);
  *cantVisi = atoi(line);
  free(line);
  free(nombreNuevoTXT);
  fclose(fp);
}

void lectDeTXTcreadoDesdeMS(char nombreArchivo[], float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal)
{
  long contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  char* nombreNuevoTXT = (char*) malloc(strlen(nombreArchivo)*sizeof(char)+sizeof(char)*5);
  strcpy(nombreNuevoTXT, nombreArchivo);
  strcat(nombreNuevoTXT, ".txt");
  fp = fopen(nombreNuevoTXT, "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  while ((read = getline(&line, &len, fp)) != -1)
  {
    visi_parteReal[contador] = atof(strtok(line, " "));
    visi_parteImaginaria[contador] = atof(strtok(NULL, " "));
    u[contador] = atof(strtok(NULL, " "));
    v[contador] = atof(strtok(NULL, " "));
    w[contador] = atof(strtok(NULL, " "));
    contador++;
	}
  printf("El contador es %ld\n", contador);
  free(line);
  free(nombreNuevoTXT);
  fclose(fp);
}

void lectDeTXTcreadoDesdeMSConLimite(char nombreArchivo[], float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal, long inicio, long fin, long cantVisi)
{
  long contador = 0;
  long contadorIte = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  char* nombreNuevoTXT = (char*) malloc(strlen(nombreArchivo)*sizeof(char)+sizeof(char)*5);
  strcpy(nombreNuevoTXT, nombreArchivo);
  strcat(nombreNuevoTXT, ".txt");
  fp = fopen(nombreNuevoTXT, "r");
  printf("Nombre nuevo es %s\n", nombreNuevoTXT);
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  while ((read = getline(&line, &len, fp)) != -1)
  {
    if (contadorIte >= inicio)
    {
      visi_parteReal[contador] = atof(strtok(line, " "));
      visi_parteImaginaria[contador] = atof(strtok(NULL, " "));
      u[contador] = atof(strtok(NULL, " "));
      v[contador] = atof(strtok(NULL, " "));
      w[contador] = atof(strtok(NULL, " "));
      contador++;
    }
    contadorIte++;
    if(contadorIte >= fin)
      break;
	}
  printf("El contador es %ld\n", contador);
  free(line);
  free(nombreNuevoTXT);
  fclose(fp);
}

void escrituraDeArchivoConParametros_Normal(char nombreArchivoPara[], char nombreArchivo[], char nombreDirPrin[], int cantVisi, int N, int maxIter, float tolGrad)
{
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  FILE* archivoDePara = fopen(nombreArchivoPara, "w");
  fprintf(archivoDePara, "Programa inicio su ejecucion con fecha: %d-%d-%d %d:%d:%d\n", tm.tm_year + 1900, tm.tm_mon + 1,tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  fprintf(archivoDePara, "Compresion con base normal utilizando informacion del archivo %s cuyos parametros de ejecucion fueron:\n", nombreArchivo);
  fprintf(archivoDePara, "Cantidad de visibilidades(cantVisi): %d\n", cantVisi);
  fprintf(archivoDePara, "Cantidad de Coefs(N x N): %d x %d = %d\n", N, N, N*N);
  fprintf(archivoDePara, "Maximo de iteraciones impuesto para la minimizacion de coeficientes(maxIter): %d\n", maxIter);
  fprintf(archivoDePara, "Grado de tolerancia a la minimizacion de los coefs(tolGrad): %.12e\n", tolGrad);
  fclose(archivoDePara);
}

void escrituraDeArchivoConParametros_Rect(char nombreArchivoPara[], char nombreArchivo[], char nombreDirPrin[], long cantVisi, long N, int maxIter, float tolGrad, float estrechezDeBorde)
{
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  FILE* archivoDePara = fopen(nombreArchivoPara, "w");
  fprintf(archivoDePara, "Programa inicio su ejecucion con fecha: %d-%d-%d %d:%d:%d\n", tm.tm_year + 1900, tm.tm_mon + 1,tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  fprintf(archivoDePara, "Compresion con base rectangular utilizando informacion del archivo %s cuyos parametros de ejecucion fueron:\n", nombreArchivo);
  fprintf(archivoDePara, "Estrechez de borde: %f\n", estrechezDeBorde);
  fprintf(archivoDePara, "Cantidad de visibilidades(cantVisi): %ld\n", cantVisi);
  fprintf(archivoDePara, "Cantidad de Coefs(N x N): %ld x %ld = %ld\n", N, N, N*N);
  fprintf(archivoDePara, "Maximo de iteraciones impuesto para la minimizacion de coeficientes(maxIter): %d\n", maxIter);
  fprintf(archivoDePara, "Grado de tolerancia a la minimizacion de los coefs(tolGrad): %.12e\n", tolGrad);
  fclose(archivoDePara);
}

void calculoDeInfoCompre_BaseNormal(char nombreArchivo[], int maxIter, float tolGrad, float tolGolden, float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal, float delta_u, float delta_v, long cantVisi, long N, float cotaEnergia, char nombreDirPrin[], char nombreDirSec[], int cantParamEvaInfo, float inicioIntervalo, float finIntervalo, float* matrizDeUnosEstFourier, float estrechezDeBorde)
{
  float inicioIntervaloEscalado = inicioIntervalo * delta_u;
  float finIntervaloEscalado = finIntervalo * delta_u;
  char nombreArPara[] = "parametrosEjecucion.txt";
  if(cotaEnergia > 1.0)
  {
      printf("ERROR: La cota de energia debe estar expresado en decimales, no en porcentajes.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  int cotaEnergiaInt = cotaEnergia * 100;
  char* cotaEnergiaString = numAString(&cotaEnergiaInt);
  sprintf(cotaEnergiaString, "%d", cotaEnergiaInt);
  strcat(nombreDirPrin, cotaEnergiaString);
  if(mkdir(nombreDirPrin, 0777) == -1)
  {
      printf("ERROR: El directorio EXISTE, PELIGRO DE SOBREESCRITURA, por favor eliga otro nombre de directorio.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  else
      printf("Directorio creado.\n");
  char* nombreArchivoPara = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArPara)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoPara, nombreDirPrin);
  strcat(nombreArchivoPara, "/");
  strcat(nombreArchivoPara, nombreArPara);
  escrituraDeArchivoConParametros_Rect(nombreArchivoPara, nombreArchivo, nombreDirPrin, cantVisi, N, maxIter, tolGrad, estrechezDeBorde);
  free(nombreArchivoPara);

  // goldenMin_BaseNormal(u, v, w, delta_u, delta_v, cantVisi, N);
  // printf("El optimo esta en %.12f\n", optimo);

  // printf("inicio del intervalo es %.12f y el fin del intervalo es %.12f\n", inicioIntervalo, finIntervalo);
  // float* paramEvaInfo = linspace(inicioIntervaloEscalado, finIntervaloEscalado, cantParamEvaInfo);
  // int i = 0;
  // // for(int i=0; i<cantParamEvaInfo; i++)
  // // {
  //   char* numComoString = numAString(&i);
  //   sprintf(numComoString, "%d", i);
  //   char* nombreDirSecCopia = (char*) malloc(sizeof(char)*strlen(nombreDirSec)*strlen(numComoString));
  //   strcpy(nombreDirSecCopia, nombreDirSec);
  //   strcat(nombreDirSecCopia, numComoString);
  //   calCompSegunAncho_Normal_escritura(nombreDirPrin, nombreDirSecCopia, paramEvaInfo[i], cotaEnergia, i, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, matrizDeUnosEstFourier, estrechezDeBorde);
  //   free(numComoString);
  //   free(nombreDirSecCopia);
  // }
}

void calculoDeInfoCompre_BaseRect(char nombreArchivo[], int maxIter, float tolGrad, float tolGolden, float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal, float delta_u, float delta_v, float* matrizDeUnos, long cantVisi, long N, float cotaEnergia, char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], int cantParamEvaInfo, float inicioIntervalo, float finIntervalo, float* matrizDeUnosEstFourier, float estrechezDeBorde)
{
  float inicioIntervaloEscalado = inicioIntervalo * delta_u;
  float finIntervaloEscalado = finIntervalo * delta_u;
  char nombreArPara[] = "parametrosEjecucion.txt";
  if(cotaEnergia > 1.0)
  {
      printf("ERROR: La cota de energia debe estar expresado en decimales, no en porcentajes.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  int cotaEnergiaInt = cotaEnergia * 100;
  char* cotaEnergiaString = numAString(&cotaEnergiaInt);
  sprintf(cotaEnergiaString, "%d", cotaEnergiaInt);
  // strcat(nombreDirPrin, cotaEnergiaString);
  if(mkdir(nombreDirPrin, 0777) == -1)
  {
      printf("ERROR: El directorio EXISTE, PELIGRO DE SOBREESCRITURA, por favor eliga otro nombre de directorio.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  else
      printf("Directorio creado.\n");
  char* nombreArchivoPara = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArPara)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoPara, nombreDirPrin);
  strcat(nombreArchivoPara, "/");
  strcat(nombreArchivoPara, nombreArPara);
  escrituraDeArchivoConParametros_Rect(nombreArchivoPara, nombreArchivo, nombreDirPrin, cantVisi, N, maxIter, tolGrad, estrechezDeBorde);
  free(nombreArchivoPara);

  // float optimo = goldenMin_BaseRect(u, v, w, delta_u, delta_v, matrizDeUnos, cantVisi, N, estrechezDeBorde);
  // printf("El optimo esta en %.12f\n", optimo);

  float* paramEvaInfo = linspace(inicioIntervaloEscalado, finIntervaloEscalado, cantParamEvaInfo);
  int i = 0;
  // for(int i=0; i<cantParamEvaInfo; i++)
  // {
    char* numComoString = numAString(&i);
    sprintf(numComoString, "%d", i);
    char* nombreDirSecCopia = (char*) malloc(sizeof(char)*strlen(nombreDirSec)*strlen(numComoString));
    strcpy(nombreDirSecCopia, nombreDirSec);
    strcat(nombreDirSecCopia, numComoString);
    calCompSegunAncho_Rect_escritura(nombreDirPrin, nombreDirSecCopia, nombreDirTer, paramEvaInfo[i], cotaEnergia, i, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosEstFourier, estrechezDeBorde);
    free(numComoString);
    free(nombreDirSecCopia);
  // }
}

void calImagenesADistintasCompresiones_Rect(float inicioIntervalo, float finIntervalo, int cantParamEvaInfo, char nombreDirPrin[], float ancho, int maxIter, float tol, float* u, float* v, float* w, float* visi_parteImaginaria, float* visi_parteReal, float delta_u, float delta_v, float* matrizDeUnos, long cantVisi, long N, float* matrizDeUnosTamN, float estrechezDeBorde)
{

  if(mkdir(nombreDirPrin, 0777) == -1)
  {
      printf("ERROR: El directorio EXISTE, PELIGRO DE SOBREESCRITURA, por favor eliga otro nombre de directorio.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  else
      printf("Directorio creado.\n");
  char nombreArReconsCompreImg[] = "reconsCompreImg";
  float* paramEvaInfo = linspace(inicioIntervalo/100.0, finIntervalo/100.0, cantParamEvaInfo);


  // ############### CALCULO DE MU Y MV - CREACION DE DIRECTORIO SEGUNDARIO  ##############
  printf("...Comenzando calculo de MV...\n");
  float* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos);
  printf("Calculo de MV completado.\n");

  printf("...Comenzando calculo de MU...\n");
  float* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos);
  printf("Calculo de MU completado.\n");


  // ############### MINIMIZACION DE COEFS, PARTE IMAGINARIA  ##############
  printf("...Comenzando minimizacion de coeficientes parte imaginaria...\n");
  float* MC_imag = minGradConjugado_MinCuadra(MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol);
  printf("Proceso de minimizacion de coeficientes parte imaginaria terminado.\n");


  // ############### MINIMIZACION DE COEFS, PARTE REAL  ##############
  printf("...Comenzando minimizacion de coeficientes parte real...\n");
  float* MC_real = minGradConjugado_MinCuadra(MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol);
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");


  float* MV_AF = calcularMV_Rect_estFourier(ancho, N, delta_v, matrizDeUnos, estrechezDeBorde, matrizDeUnosTamN);
  float* MU_AF = calcularMV_Rect_estFourier(ancho, N, delta_u, matrizDeUnos, estrechezDeBorde, matrizDeUnosTamN);


  float* MC_comp_imag;
  cudaMallocManaged(&MC_comp_imag,N*N*sizeof(float));
  cudaMemset(MC_comp_imag, 0, N*N*sizeof(float));
  float* MC_comp_real;
  cudaMallocManaged(&MC_comp_real,N*N*sizeof(float));
  cudaMemset(MC_comp_real, 0, N*N*sizeof(float));

  long largo = N * N;
  float* MC_img_cuadrado;
  cudaMallocManaged(&MC_img_cuadrado, N*N*sizeof(float));
  float* MC_modulo;
  cudaMallocManaged(&MC_modulo, N*N*sizeof(float));
  hadamardProduct(MC_imag, N, N, MC_imag, MC_img_cuadrado);
  hadamardProduct(MC_real, N, N, MC_real, MC_modulo);
  combinacionLinealMatrices(1.0, MC_img_cuadrado, N, N, 1.0, MC_modulo);
  cudaFree(MC_img_cuadrado);
  af::array MC_modulo_GPU(N*N, MC_modulo);
  cudaFree(MC_modulo);
  af::array MC_modulo_indicesOrde_GPU(N*N);
  af::array MC_modulo_Orde_GPU(N*N);
  af::sort(MC_modulo_Orde_GPU, MC_modulo_indicesOrde_GPU, MC_modulo_GPU, 0, false);
  float total = af::sum<float>(MC_modulo_GPU);
  MC_modulo_Orde_GPU = MC_modulo_Orde_GPU/total;
  af::eval(MC_modulo_Orde_GPU);
  af::eval(MC_modulo_indicesOrde_GPU);
  af::sync();
  float* auxiliar_MC_modulo_Orde_GPU = MC_modulo_Orde_GPU.device<float>();
  float* auxiliar_MC_modulo_indicesOrde_GPU = MC_modulo_indicesOrde_GPU.device<float>();
  float* coefsNormalizados = (float*) malloc(largo*sizeof(float));
  cudaMemcpy(coefsNormalizados, auxiliar_MC_modulo_Orde_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  int* MC_modulo_indicesOrde_CPU = (int*) malloc(largo*sizeof(int));
  cudaMemcpy(MC_modulo_indicesOrde_CPU, auxiliar_MC_modulo_indicesOrde_GPU, N*N*sizeof(int), cudaMemcpyDeviceToHost);
  MC_modulo_Orde_GPU.unlock();
  MC_modulo_GPU.unlock();
  MC_modulo_indicesOrde_GPU.unlock();

  long cantCoefsParaCota = 0;
  float sumador = 0.0;
  float* cantCoefsPorParametro = (float*) malloc(sizeof(float)*cantParamEvaInfo);
  float* cantidadPorcentualDeCoefs = linspace(1.0, largo, largo);
  combinacionLinealMatrices(0.0, cantidadPorcentualDeCoefs, largo, 1, 1.0/N, cantidadPorcentualDeCoefs);
  for(long j=0; j<cantParamEvaInfo; j++)
  {
    sumador = 0.0;
    cantCoefsParaCota = 0;
    for(long i=0; i<largo; i++)
    {
       sumador += coefsNormalizados[i];
       cantCoefsParaCota++;
       if(cantidadPorcentualDeCoefs[i] >= paramEvaInfo[j])
       {
         printf("Del %f%% solicitado, se ha tomado el mas cercano correspondiente al %f%% de coefs, lo que corresponde a un total de %ld coeficientes los cuales poseen el %f%% de la energia.\n", paramEvaInfo[j], cantidadPorcentualDeCoefs[i], cantCoefsParaCota, sumador);
         break;
       }
    }
    float* indicesATomar_CPU = (float*) malloc(cantCoefsParaCota*sizeof(float));
    for(int k=0; k<cantCoefsParaCota; k++)
    {
      indicesATomar_CPU[k] = MC_modulo_indicesOrde_CPU[k];
    }
    af::array indicesATomar_GPU(cantCoefsParaCota, indicesATomar_CPU);
    free(indicesATomar_CPU);
    af::array indRepComp = af::constant(0, largo);
    indRepComp(indicesATomar_GPU) = 1;
    indicesATomar_GPU.unlock();

    af::array MC_imag_GPU(N*N, MC_imag);
    af::array MC_real_GPU(N*N, MC_real);
    MC_imag_GPU = MC_imag_GPU * indRepComp;
    MC_real_GPU = MC_real_GPU * indRepComp;
    af::eval(MC_imag_GPU);
    af::eval(MC_real_GPU);
    af::sync();
    indRepComp.unlock();
    float* auxiliar_MC_imag_GPU = MC_imag_GPU.device<float>();
    float* auxiliar_MC_real_GPU = MC_real_GPU.device<float>();
    cudaMemcpy(MC_comp_imag, auxiliar_MC_imag_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    MC_imag_GPU.unlock();
    cudaMemcpy(MC_comp_real, auxiliar_MC_real_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    MC_real_GPU.unlock();
    float* estimacionFourier_compre_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_imag, N, N, MU_AF);
    float* estimacionFourier_compre_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_real, N, N, MU_AF);
    int numero = j+1;
    char* numComoString = numAString(&numero);
    sprintf(numComoString, "%d", numero);
    char* nombreArchivoReconsImgComp = (char*) malloc(sizeof(char)*strlen(nombreDirPrin)*strlen(numComoString)*strlen(nombreArReconsCompreImg)+sizeof(char)*7);
    strcpy(nombreArchivoReconsImgComp, nombreDirPrin);
    strcat(nombreArchivoReconsImgComp, "/");
    strcat(nombreArchivoReconsImgComp, nombreArReconsCompreImg);
    strcat(nombreArchivoReconsImgComp, "_");
    strcat(nombreArchivoReconsImgComp, numComoString);
    strcat(nombreArchivoReconsImgComp, ".fit");

    printf("%s\n", nombreArchivoReconsImgComp);

    escribirTransformadaInversaFourier2D(estimacionFourier_compre_ParteImag, estimacionFourier_compre_ParteReal, N, nombreArchivoReconsImgComp);
    cudaFree(estimacionFourier_compre_ParteImag);
    cudaFree(estimacionFourier_compre_ParteReal);
    free(numComoString);
    free(nombreArchivoReconsImgComp);
  }
  cudaFree(MU_AF);
  cudaFree(MV_AF);
  free(coefsNormalizados);
  free(MC_modulo_indicesOrde_CPU);
}

void filtroGaussiano()
{
  int largoVector = 100;
  float* porcenReal = (float*) malloc(sizeof(float)*largoVector);
  float* vector = (float*) malloc(sizeof(float)*largoVector);
  long contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  fp = fopen("/home/rarmijo/psnr_hd142_rect.txt", "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo");
      exit(0);
  }
  while ((read = getline(&line, &len, fp)) != -1)
  {
    porcenReal[largoVector-1-contador] = atof(strtok(line, " "));
    strtok(NULL, " ");
    vector[contador] = atof(strtok(NULL, " "));
    contador++;
	}
  printf("El contador es %ld\n", contador);
  free(line);
  fclose(fp);

  // for(int i=0; i<largoVector; i++)
  // {
  //   printf("%f\n", porcenReal[i]);
  // }
  // exit(-1);

  float* vectorFiltrado = (float*) calloc(largoVector, sizeof(float));
  gsl_vector* copiaVectorEnGSL = gsl_vector_alloc(largoVector);
  gsl_vector* vectorEnGSLFiltrado = gsl_vector_alloc(largoVector);
  for(int i=0; i<largoVector; i++)
  {
    gsl_vector_set(copiaVectorEnGSL, i, vector[largoVector-1-i]);
  }
  gsl_filter_gaussian_workspace* gauss_p = gsl_filter_gaussian_alloc(largoVector);
  gsl_filter_gaussian(GSL_FILTER_END_TRUNCATE, 1.0, 0, copiaVectorEnGSL, vectorEnGSLFiltrado, gauss_p);
  for(int i=0; i<largoVector; i++)
  {
    vectorFiltrado[i] = gsl_vector_get(copiaVectorEnGSL, i);
  }
  gsl_vector_free(copiaVectorEnGSL);
  gsl_vector_free(vectorEnGSLFiltrado);
  gsl_filter_gaussian_free(gauss_p);

  float* listaDeMetricas = (float*) malloc(sizeof(float)*largoVector);
  float* primeraRecta_subListaDeX = (float*) calloc(largoVector, sizeof(float));
  float* primeraRecta_subListaDeY = (float*) calloc(largoVector, sizeof(float));
  float* segundaRecta_subListaDeX = (float*) calloc(largoVector, sizeof(float));
  float* segundaRecta_subListaDeY = (float*) calloc(largoVector, sizeof(float));
  memcpy(segundaRecta_subListaDeX, porcenReal, sizeof(float)*largoVector);
  memcpy(segundaRecta_subListaDeY, vectorFiltrado, sizeof(float)*largoVector);
  primeraRecta_subListaDeX[0] = porcenReal[0];
  primeraRecta_subListaDeY[0] = vectorFiltrado[0];
  for(int i=1; i<largoVector-1; i++)
  {
      primeraRecta_subListaDeX[i] = porcenReal[i];
      primeraRecta_subListaDeY[i] = vectorFiltrado[i];
      float pendienteDePrimeraRecta = calPendiente(primeraRecta_subListaDeX, i+1, primeraRecta_subListaDeY);
      // printf("En la iteracion %d la pendienteDePrimeraRecta es %f\n", i, pendienteDePrimeraRecta);
      segundaRecta_subListaDeX[i-1] = 0.0;
      segundaRecta_subListaDeY[i-1] = 0.0;
      float pendienteDeSegundaRecta = calPendiente(&(segundaRecta_subListaDeX[i]), largoVector-i, &(segundaRecta_subListaDeY[i]));
      // printf("En la iteracion %d la pendienteDeSegundaRecta es %f\n", i, pendienteDeSegundaRecta);
      listaDeMetricas[i] = -1 * pendienteDeSegundaRecta/pendienteDePrimeraRecta;
      printf("%f\n", listaDeMetricas[i]);
  }
  free(primeraRecta_subListaDeX);
  free(primeraRecta_subListaDeY);
  free(segundaRecta_subListaDeX);
  free(segundaRecta_subListaDeY);
}

int main()
{
  // PARAMETROS GENERALES
  long cantVisi = 15034;
  long inicio = 0;
  long fin = 15034;

  // long cantVisi = 30000;
  // long inicio = 0;
  // long fin = 30000;

  int N = 512;
  // long N = 1600; //HLTau_B6cont.calavg.tav300s
  int maxIter = 100;

  float tolGrad = 1E-12;

  float delta_x = 0.02;
  // float delta_x = 0.005; //HLTau_B6cont.calavg.tav300s
  // float delta_x = 0.03; //co65
  float delta_x_rad = (delta_x * M_PI)/648000.0;
  float delta_u = 1.0/(N*delta_x_rad);
  float delta_v = 1.0/(N*delta_x_rad);

  //PARAMETROS PARTICULARES DE BASE RECT
  float estrechezDeBorde = 1000.0;

  // float frecuencia;
  // float *u, *v, *w, *visi_parteImaginaria, *visi_parteReal;
  // cudaMallocManaged(&u, cantVisi*sizeof(float));
  // cudaMallocManaged(&v, cantVisi*sizeof(float));
  // cudaMallocManaged(&w, cantVisi*sizeof(float));
  // cudaMallocManaged(&visi_parteImaginaria, cantVisi*sizeof(float));
  // cudaMallocManaged(&visi_parteReal, cantVisi*sizeof(float));
  // char nombreArchivo[] = "hd142_b9cont_self_tav.0.0.txt";
  // lecturaDeTXT(nombreArchivo, &frecuencia, u, v, w, visi_parteImaginaria, visi_parteReal, cantVisi);

  // // ########### NOTEBOOK ##############
  // char nombreArchivo[] = "/home/yoyisaurio/Desktop/HLTau_B6cont.calavg.tav300s";
  // char comandoCasaconScript[] = "/home/yoyisaurio/casa-pipeline-release-5.6.2-2.el7/bin/casa -c /home/yoyisaurio/Desktop/proyecto/deMSaTXT.py";

  // // ########### PC-LAB ##############
  // char nombreArchivo[] = "/home/rarmijo/Desktop/proyecto/HLTau_B6cont.calavg.tav300s";
  // char comandoCasaconScript[] = "/home/rarmijo/casa-pipeline-release-5.6.2-2.el7/bin/casa -c ./deMSaTXT.py";

  // // ########### PC-LAB ##############
  // char nombreArchivo[] = "./co65.ms";
  // char comandoCasaconScript[] = "/home/rarmijo/casa-pipeline-release-5.6.2-2.el7/bin/casa -c ./deMSaTXT.py";

  // // ########### BEAM ##############
  // char nombreArchivo[] = "./HLTau_B6cont.calavg.tav300s";
  // char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // // ########### BEAM ##############
  // char nombreArchivo[] = "./FREQ78.ms";
  // char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // // // ########### BEAM ##############
  // char nombreArchivo[] = "./co65.ms";
  // char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // ########### BEAM ##############
  char nombreArchivo[] = "./hd142_b9cont_self_tav.ms";
  char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // // ########### BEAM ##############
  // char nombreArchivo[] = "/home/rarmijo/HLTau_Band6_CalibratedData/HLTau_B6cont.calavg";
  // char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // char* comandoScriptMSaTXT = (char*) malloc(strlen(comandoCasaconScript)*strlen(nombreArchivo)*sizeof(char)+sizeof(char)*3);
  // strcpy(comandoScriptMSaTXT, comandoCasaconScript);
  // strcat(comandoScriptMSaTXT, " ");
  // strcat(comandoScriptMSaTXT, nombreArchivo);
  // system(comandoScriptMSaTXT);
  // free(comandoScriptMSaTXT);


  lectCantVisi(nombreArchivo, &cantVisi);
  float *u, *v, *w, *visi_parteImaginaria, *visi_parteReal;
  cudaMallocManaged(&u, cantVisi*sizeof(float));
  cudaMallocManaged(&v, cantVisi*sizeof(float));
  cudaMallocManaged(&w, cantVisi*sizeof(float));
  cudaMallocManaged(&visi_parteImaginaria, cantVisi*sizeof(float));
  cudaMallocManaged(&visi_parteReal, cantVisi*sizeof(float));
  lectDeTXTcreadoDesdeMS(nombreArchivo, u, v, w, visi_parteImaginaria, visi_parteReal);
  // lectDeTXTcreadoDesdeMSConLimite(nombreArchivo, u, v, w, visi_parteImaginaria, visi_parteReal, inicio, fin, cantVisi);

  float* matrizDeUnos, *matrizDeUnosEstFourier;
  cudaMallocManaged(&matrizDeUnos, cantVisi*N*sizeof(float));
  for(long i=0; i<(cantVisi*N); i++)
  {
    matrizDeUnos[i] = 1.0;
  }
  cudaMallocManaged(&matrizDeUnosEstFourier, N*sizeof(float));
  for(long i=0; i<N; i++)
  {
    matrizDeUnosEstFourier[i] = 1.0;
  }

  // float* rango = linspace(1.0 * delta_u, 5.0 * delta_u, 100);
  // for(int i=0; i<100; i++)
  // {
  //   float* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, rango[i], matrizDeUnos);
  //   float* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, rango[i], matrizDeUnos);
  //   float* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w);
  //   cudaFree(MU);
  //   cudaFree(MV);
  //   float medidaSumaDeLaDiagonal = medidasDeInfo[0];
  //   free(medidasDeInfo);
  //   float info = -1 * medidaSumaDeLaDiagonal;
  //   printf("%.12e\n", info);
  // }


  // goldenMin(u, v, w, delta_u, delta_v, matrizDeUnos, cantVisi, N, estrechezDeBorde);
  //
  // // filtroGaussiano();
  // exit(-1);


  // // double ancho = delta_u;
  // //
  // // // float* MV = calcularMV_Normal(v, delta_v, cantVisi, N, ancho);
  // // float* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos);
  // // // float* MU = calcularMV_Normal(u, delta_u, cantVisi, N, ancho);
  // // float* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos);
  // // float* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w);
  // // float medidaSumaDeLaDiagonal = medidasDeInfo[0];
  // // printf("%.12e\n", medidaSumaDeLaDiagonal);


  float cotaEnergia = 0.99;
  // char nombreDirPrin[] = "float_calCompresion_baseNormal_cota";
  char nombreDirPrin[] = "experi_hd142_solo80";
  char nombreDirSec[] = "ite";
  char nombreDirTer[] = "compresiones";
  char nombreArchivoTiempo[] = "tiempo.txt";
  int cantParamEvaInfo = 80;
  // float inicioIntervalo = 0.8;
  float inicioIntervalo = 1.0;
  float finIntervalo = 3.0;
  float tolGolden = 1E-12;
  int iterActual = 0;
  clock_t t;
  t = clock();
  // calculoDeInfoCompre_BaseNormal(nombreArchivo, maxIter, tolGrad, tolGolden, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, cotaEnergia, nombreDirPrin, nombreDirSec, nombreDirTer, cantParamEvaInfo, inicioIntervalo, finIntervalo, matrizDeUnosEstFourier, estrechezDeBorde);
  calculoDeInfoCompre_BaseRect(nombreArchivo, maxIter, tolGrad, tolGolden, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, cotaEnergia, nombreDirPrin, nombreDirSec, nombreDirTer, cantParamEvaInfo, inicioIntervalo, finIntervalo, matrizDeUnosEstFourier, estrechezDeBorde);
  t = clock() - t;
  float time_taken = ((float)t)/CLOCKS_PER_SEC;
  char* nombreCompletoArchivoTiempo = (char*) malloc(sizeof(char)*strlen(nombreArchivoTiempo)*strlen(nombreDirPrin)+sizeof(char)*3);
  strcpy(nombreCompletoArchivoTiempo, nombreDirPrin);
  strcat(nombreCompletoArchivoTiempo, "/");
  strcat(nombreCompletoArchivoTiempo, nombreArchivoTiempo);
  FILE* archivoTiempo = fopen(nombreCompletoArchivoTiempo, "w");
  float minutitos = time_taken/60;
  float horas = minutitos/60;
  printf("El tiempo de ejecucion fue %.12f segundos o %.12f minutos o %.12f horas.\n", time_taken, minutitos, horas);
  fprintf(archivoTiempo, "El tiempo de ejecucion fue %.12f segundos o %.12f minutos o %.12f horas.\n", time_taken, minutitos, horas);
  fclose(archivoTiempo);

  // // char nombreDirPrin[] = "calCompresiones_Normal";
  // // char nombreArchivoTiempo[] = "tiempo.txt";
  // // int cantParamEvaInfo = 100;
  // // float inicioIntervalo = 1.0;
  // // float finIntervalo = 100.0;
  // // float tolGolden = 1E-12;
  // // float nuevoAncho = 1.0 * delta_u;
  // // clock_t t;
  // // t = clock();
  // // calPSNRDeDistintasCompresiones_Normal(inicioIntervalo, finIntervalo, cantParamEvaInfo, nombreDirPrin, nuevoAncho, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, matrizDeUnosEstFourier, estrechezDeBorde);
  // // // calPSNRDeDistintasCompresiones_Rect(inicioIntervalo, finIntervalo, cantParamEvaInfo, nombreDirPrin, nuevoAncho, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosEstFourier, estrechezDeBorde);
  // // // calImagenesADistintasCompresiones_Rect(inicioIntervalo, finIntervalo, cantParamEvaInfo, nombreDirPrin, nuevoAncho, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosEstFourier, estrechezDeBorde);
  // // t = clock() - t;
  // // float time_taken = ((float)t)/CLOCKS_PER_SEC;
  // // char* nombreCompletoArchivoTiempo = (char*) malloc(sizeof(char)*strlen(nombreArchivoTiempo)*strlen(nombreDirPrin)+sizeof(char)*3);
  // // strcpy(nombreCompletoArchivoTiempo, nombreDirPrin);
  // // strcat(nombreCompletoArchivoTiempo, "/");
  // // strcat(nombreCompletoArchivoTiempo, nombreArchivoTiempo);
  // // FILE* archivoTiempo = fopen(nombreCompletoArchivoTiempo, "w");
  // // float minutitos = time_taken/60;
  // // float horas = minutitos/60;
  // // printf("El tiempo de ejecucion fue %.12f segundos o %.12f minutos o %.12f horas.\n", time_taken, minutitos, horas);
  // // fprintf(archivoTiempo, "El tiempo de ejecucion fue %.12f segundos o %.12f minutos o %.12f horas.\n", time_taken, minutitos, horas);
  // // fclose(archivoTiempo);
  //
  //
  // // printf("...Comenzando calculo de MV...\n");
  // // clock_t tiempoCalculoMV;
  // // tiempoCalculoMV = clock();
  // // float* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, delta_v, matrizDeUnos);
  // // tiempoCalculoMV = clock() - tiempoCalculoMV;
  // // float tiempoTotalCalculoMV = ((float)tiempoCalculoMV)/CLOCKS_PER_SEC;
  // // printf("Calculo de MV completado.\n");
  // //
  // // printf("...Comenzando calculo de MU...\n");
  // // clock_t tiempoCalculoMU;
  // // tiempoCalculoMU = clock();
  // // float* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, delta_u, matrizDeUnos);
  // // tiempoCalculoMU = clock() - tiempoCalculoMU;
  // // float tiempoTotalCalculoMU = ((float)tiempoCalculoMU)/CLOCKS_PER_SEC;
  // // printf("Calculo de MU completado.\n");
  // //
  // // int blockSize;   // The launch configurator returned block size
  // // int minGridSize; // The minimum grid size needed to achieve the
  // //                  // maximum occupancy for a full device launch
  // // int gridSize;    // The actual grid size needed, based on input size
  // //
  // // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, transponerMatriz_kernel, 0, 0);
  // // // Round up according to array size
  // // gridSize = (cantVisi*N + blockSize - 1) / blockSize;
  // //
  // // // long cantBloques = ceil((float) cantFilas*N/1024);
  // // // hadamardProduct_kernel<<<gridSize,blockSize>>>(MU, MV, matrizDeUnos, cantVisi, N);
  // // // combinacionLinealMatrices_kernel<<<gridSize,blockSize>>>(5.0, MU, cantVisi, N, 5.0, MV);
  // // transponerMatriz_kernel<<<gridSize,blockSize>>>(MU, matrizDeUnos, cantVisi, N);
  // // cudaDeviceSynchronize();
  // //
  // //   // calculate theoretical occupancy
  // // int maxActiveBlocks;
  // // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, transponerMatriz_kernel, blockSize, 0);
  // //
  // // int device;
  // // cudaDeviceProp props;
  // // cudaGetDevice(&device);
  // // cudaGetDeviceProperties(&props, device);
  // //
  // // float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
  // //                   (float)(props.maxThreadsPerMultiProcessor /
  // //                           props.warpSize);
  // //
  // // printf("Launched blocks of size %d. Theoretical occupancy: %f\n",
  // //        blockSize, occupancy);
  //
  // cudaFree(u);
  // cudaFree(v);
  // cudaFree(w);
  // cudaFree(visi_parteImaginaria);
  // cudaFree(visi_parteReal);
  // cudaFree(matrizDeUnos);
  // cudaFree(matrizDeUnosEstFourier);
}
