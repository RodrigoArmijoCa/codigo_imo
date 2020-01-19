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
#include <cusparse.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define BLOCK_DIM 16
#define TILE_WIDTH 16

// BEAM: rarmijo@158.170.35.147
// PC-LAB: rarmijo@158.170.35.139
//nvcc calCompreInfo.cu -lcudart -lcublas -lcuda -lblasx -I/opt/arrayfire/include/ -L/opt/arrayfire/lib64/ -lafcuda -lcusparse -Xcompiler -fopenmp -lcfitsio -I/usr/include/gsl -lgsl -lgslcblas -lm -I/usr/lib/cuda-10.0/samples/common/inc -o calCompreInfo

struct parametros_BaseRect
{
  double* u;
  double* v;
  double* w;
  double delta_u;
  double delta_v;
  double* matrizDeUnos;
  long cantVisi;
  long N;
  double estrechezDeBorde;
};

struct parametros_BaseNormal
{
  double* u;
  double* v;
  double* w;
  double delta_u;
  double delta_v;
  long cantVisi;
  long N;
};

struct parametros_Minl1
{
  long cantVisi;
  long N;
  double* MU;
  double* MC;
  double* MV;
  double* residual;
  double* w;
  double* pActual;
  double* matrizDeUnosTamN;
  double param_lambda;
  double tamBloque;
  double numGPU;
};

#define sqrt5 2.236067977499789696

char* numAString(int* numero)
{
  int cantCarac = (*numero)/10 + 1;
  char* numComoString = (char*) malloc(sizeof(char)*cantCarac);
  return numComoString;
}

double calPendiente(double* x, int largoDeX, double* y)
{
  double sumadeYs = 0.0;
  double sumadeXs = 0.0;
  double sumaDeLosCuadradosdeXs = 0.0;
  double sumaDeMultdeXsconYs = 0.0;
  for(int i=0; i<largoDeX; i++)
  {
    double xActual = x[i];
    double yActual = y[i];
    sumadeYs += yActual;
    sumadeXs += xActual;
    sumaDeMultdeXsconYs += xActual * yActual;
    sumaDeLosCuadradosdeXs += xActual * xActual;
  }
  double cuadradoDeLaSumadeXs = sumadeXs * sumadeXs;
  double numerador = largoDeX * sumaDeMultdeXsconYs - sumadeXs * sumadeYs;
  double denominador = largoDeX * sumaDeLosCuadradosdeXs - cuadradoDeLaSumadeXs;
  return numerador/denominador;
}

double* linspace(double a, double b, long n)
{
    double c;
    int i;
    double* u;
    cudaMallocManaged(&u, n*sizeof(double));
    c = (b - a)/(n - 1);
    for(i = 0; i < n - 1; ++i)
        u[i] = a + i*c;
    u[n - 1] = b;
    return u;
}

void linspaceSinBordes(double a, double b, long n, double* u)
{
    double c;
    int i;
    c = (b - a)/(n - 1);
    for(i = 0; i < n - 1; ++i)
        u[i] = a + i*c;
    u[n - 1] = b;
}

double* linspaceNoEquiespaciado(double* limitesDeZonas, double* cantPuntosPorZona, int cantParesDePuntos)
{
  // double c1, double b1, double a1, double a2, double b2, double c2, int nc, int nb, int na
  int cantZonas = cantParesDePuntos*2-1;
  int cantPuntosTotales = cantParesDePuntos*2;
  for(int i=0; i<cantZonas; i++)
  {
    cantPuntosTotales += cantPuntosPorZona[i%cantParesDePuntos];
  }
  double* puntosTotales;
  cudaMallocManaged(&puntosTotales, cantPuntosTotales*sizeof(double));
  int posicionActual = 0;
  int posicionCantPuntosPorZona = 0;
  for(int i=0; i<cantZonas; i++)
  {
    int cantPuntosIntermediosAInsertar;
    if(posicionCantPuntosPorZona < cantParesDePuntos)
    {
      cantPuntosIntermediosAInsertar = cantPuntosPorZona[posicionCantPuntosPorZona]+2;
      posicionCantPuntosPorZona++;
    }
    else
    {
      int nuevaPosicion = cantZonas - posicionCantPuntosPorZona - 1;
      cantPuntosIntermediosAInsertar = cantPuntosPorZona[nuevaPosicion]+2;
      posicionCantPuntosPorZona++;
    }
    linspaceSinBordes(limitesDeZonas[i], limitesDeZonas[i+1], cantPuntosIntermediosAInsertar, &(puntosTotales[posicionActual]));
    posicionActual += cantPuntosIntermediosAInsertar-1;
  }
  for(int i=0; i<cantPuntosTotales; i++)
  {
    printf("%f ", puntosTotales[i]);
  }
  printf("\n");
  return puntosTotales;
}

double* linspaceNoEquiespaciadoMitad(double* limitesDeZonas, double* cantPuntosPorZona, int cantPtosLimites)
{
  // double c1, double b1, double a1, double a2, double b2, double c2, int nc, int nb, int na
  int cantPuntosTotales = cantPtosLimites;
  for(int i=0; i<cantPtosLimites-1; i++)
  {
    cantPuntosTotales += cantPuntosPorZona[i];
  }
  printf("La cantidad de puntos totales es %d\n", cantPuntosTotales);
  double* puntosTotales;
  cudaMallocManaged(&puntosTotales, cantPuntosTotales*sizeof(double));
  int posicionActual = 0;
  for(int i=0; i<cantPtosLimites-1; i++)
  {
    int cantPuntosAInsertar = cantPuntosPorZona[i]+2;
    linspaceSinBordes(limitesDeZonas[i], limitesDeZonas[i+1], cantPuntosAInsertar, &(puntosTotales[posicionActual]));
    posicionActual += cantPuntosAInsertar-1;
  }
  // for(int i=0; i<cantPuntosTotales; i++)
  // {
  //   printf("%f ", puntosTotales[i]);
  // }
  // printf("\n");
  return puntosTotales;
}

void imprimirVector(double* lista, int tamanoLista)
{
  int i;
  for(i=0;i<tamanoLista;i++)
  {
    printf("%f\n",lista[i]);
  }
  printf("\n");
}

void imprimirMatrizColumna(double* vector, long cantFilas, long cantColumnas)
{
  long i,j;
  for(i=0;i<cantFilas;i++)
  {
    for(j=0;j<cantColumnas;j++)
    {
      // printf("%.12e ", vector[(((j)*(cantFilas))+(i))]);
      printf("%.12f ", vector[(((j)*(cantFilas))+(i))]);
    }
    printf("\n");
  }
  printf("\n");
}

void imprimirMatrizPura(double* matriz, int cantFilas, int cantColumnas)
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

void escribirCoefs(double* coefs, char* nombreArchivo, long cantFilas, long cantColumnas)
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

double** crearMatrizDouble(int cantFilas, int cantColumnas)
{
  double** matriz = (double**) calloc(cantFilas, sizeof(double*));
  int i;
  for(i=0;i<cantFilas;i++)
  {
    matriz[i] = (double*) calloc(cantColumnas, sizeof(double));
  }
  return matriz;
}

void inicializarMatriz(double** matriz, int cantFilas, int cantColumnas)
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

double* transformarMatrizAMatrizColumna(double** matriz, int cantFilas, int cantColumnas)
{
  double* nuevoVector = (double*) calloc(cantFilas*cantColumnas,sizeof(double));
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

double** transformarMatrizColumnaAMatriz(double* matrizColumna, int cantFilas, int cantColumnas)
{
  double** matriz = crearMatrizDouble(cantFilas,cantColumnas);
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

void multMatrices(double* a, long m, long k, double* b, long n, double* c, int numGPU)
{
  cublasXtHandle_t handle;
  cublasXtCreate(&handle);
  int devices[1] = {numGPU};
  if(cublasXtDeviceSelect(handle, 1, devices) != CUBLAS_STATUS_SUCCESS)
  {
    printf("set devices fail\n");
  }
  double al = 1.0;
  double bet = 0.0;
  cublasXtDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
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

// void multMatrices(double* A, long M, long K, double* B, long N, double* C, int numGPU)
// {
//   printf("MultMatrices1\n");
//   cusparseHandle_t handle;	cusparseCreate(&handle);
//   double *d_C_dense;
//   cudaMallocManaged(&d_C_dense, M*N*sizeof(double));
//   printf("MultMatrices2\n");
//
//   double *D;
//   cudaMallocManaged(&D, M*N*sizeof(double));
//   cudaMemset(D, 0, M*N*sizeof(double));
//   printf("MultMatrices3\n");
//
// 	// --- Descriptor for sparse matrix A
// 	cusparseMatDescr_t descrA;
//   cusparseCreateMatDescr(&descrA);
// 	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
//
// 	// --- Descriptor for sparse matrix B
// 	cusparseMatDescr_t descrB;
//   cusparseCreateMatDescr(&descrB);
// 	cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE);
//
// 	// --- Descriptor for sparse matrix C
// 	cusparseMatDescr_t descrC;
//   cusparseCreateMatDescr(&descrC);
// 	cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE);
//
//   // --- Descriptor for sparse matrix D
//   cusparseMatDescr_t descrD;
//   cusparseCreateMatDescr(&descrD);
//   cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL);
//   cusparseSetMatIndexBase(descrD, CUSPARSE_INDEX_BASE_ONE);
//
//   int nnzA = 0;							// --- Number of nonzero elements in dense matrix A
//   int nnzB = 0;							// --- Number of nonzero elements in dense matrix B
//   int nnzD = 0;							// --- Number of nonzero elements in dense matrix B
//
//   const int lda = M;						// --- Leading dimension of dense matrix
//   const int ldb = K;						// --- Leading dimension of dense matrix
//   const int ldd = M;						// --- Leading dimension of dense matrix
//
//   printf("MultMatrices4\n");
//
//   cusparseStatus_t estado;
//   // --- Device side number of nonzero elements per row of matrix A
//   int *nnzPerVectorA;
//   printf("MultMatrices4.1\n");
//   cudaMallocManaged(&nnzPerVectorA, M * sizeof(*nnzPerVectorA));
//   printf("MultMatrices4.2\n");
//   estado = cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, K, descrA, A, lda, nnzPerVectorA, &nnzA);
//   if(estado != CUSPARSE_STATUS_SUCCESS)
//   {
//     if(CUSPARSE_STATUS_NOT_INITIALIZED == estado)
//       printf("CUSPARSE_STATUS_NOT_INITIALIZED\n");
//     else if(CUSPARSE_STATUS_ALLOC_FAILED == estado)
//       printf("CUSPARSE_STATUS_ALLOC_FAILED\n");
//     else if(CUSPARSE_STATUS_INVALID_VALUE == estado)
//       printf("CUSPARSE_STATUS_INVALID_VALUE\n");
//     else if(CUSPARSE_STATUS_ARCH_MISMATCH == estado)
//       printf("CUSPARSE_STATUS_ARCH_MISMATCH\n");
//     else if(CUSPARSE_STATUS_EXECUTION_FAILED == estado)
//       printf("CUSPARSE_STATUS_EXECUTION_FAILED\n");
//     else if(CUSPARSE_STATUS_INTERNAL_ERROR == estado)
//       printf("CUSPARSE_STATUS_INTERNAL_ERROR\n");
//     else if(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED == estado)
//       printf("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n");
//   }
//   printf("MultMatrices4.3\n");
//
//   // --- Device side number of nonzero elements per row of matrix B
//   int *nnzPerVectorB;
//   printf("MultMatrices4.4\n");
//   cudaMallocManaged(&nnzPerVectorB, K * sizeof(*nnzPerVectorB));
//   printf("MultMatrices4.5\n");
//   cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, K, N, descrB, B, ldb, nnzPerVectorB, &nnzB);
//   printf("MultMatrices4.6\n");
//
//   // --- Device side number of nonzero elements per row of matrix B
//   int *nnzPerVectorD;
//   printf("MultMatrices4.7\n");
//   cudaMallocManaged(&nnzPerVectorD, M * sizeof(*nnzPerVectorD));
//   printf("MultMatrices4.8\n");
//   cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descrD, D, ldd, nnzPerVectorD, &nnzD);
//
//   printf("MultMatrices5\n");
//   // --- Device side sparse matrix
// 	double *csrValA; cudaMallocManaged(&csrValA, nnzA * sizeof(*csrValA));
//   double *csrValB; cudaMallocManaged(&csrValB, nnzB * sizeof(*csrValB));
//   double *csrValD; cudaMallocManaged(&csrValD, nnzD * sizeof(*csrValD));
//
//   printf("MultMatrices6\n");
//
//   int *csrRowPtrA; cudaMallocManaged(&csrRowPtrA, (M + 1) * sizeof(*csrRowPtrA));
// 	int *csrRowPtrB; cudaMallocManaged(&csrRowPtrB, (K + 1) * sizeof(*csrRowPtrB));
//   int *csrRowPtrD; cudaMallocManaged(&csrRowPtrD, (M + 1) * sizeof(*csrRowPtrD));
// 	int *csrColIndA; cudaMallocManaged(&csrColIndA, nnzA * sizeof(*csrColIndA));
//   int *csrColIndB; cudaMallocManaged(&csrColIndB, nnzB * sizeof(*csrColIndB));
//   int *csrColIndD; cudaMallocManaged(&csrColIndD, nnzD * sizeof(*csrColIndD));
//
//   printf("MultMatrices7\n");
//
//   cusparseSdense2csr(handle, M, K, descrA, A, lda, nnzPerVectorA, csrValA, csrRowPtrA, csrColIndA);
// 	cusparseSdense2csr(handle, K, N, descrB, B, ldb, nnzPerVectorB, csrValB, csrRowPtrB, csrColIndB);
//   cusparseSdense2csr(handle, M, N, descrD, D, ldd, nnzPerVectorD, csrValD, csrRowPtrD, csrColIndD);
//
//   printf("MultMatrices8\n");
//
//   // assume matrices A, B and D are ready.
//   int baseC, nnzC;
//   csrgemm2Info_t info = NULL;
//   size_t bufferSize;
//   void *buffer = NULL;
//   // nnzTotalDevHostPtr points to host memory
//   int *nnzTotalDevHostPtr = &nnzC;
//   double alpha = 1.0;
//   double beta  = 1.0;
//   cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
//
//   printf("MultMatrices9\n");
//
//   // step 1: create an opaque structure
//   cusparseCreateCsrgemm2Info(&info);
//
//   printf("MultMatrices10\n");
//
//   // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
//   cusparseScsrgemm2_bufferSizeExt(handle, M, N, K, &alpha,
//       descrA, nnzA, csrRowPtrA, csrColIndA,
//       descrB, nnzB, csrRowPtrB, csrColIndB,
//       &beta,
//       descrD, nnzD, csrRowPtrD, csrColIndD,
//       info,
//       &bufferSize);
//
//   printf("MultMatrices11\n");
//
//   cudaMallocManaged(&buffer, bufferSize);
//
//   printf("MultMatrices12\n");
//
//   // step 3: compute csrRowPtrC
//   int *csrRowPtrC;
//   cudaMallocManaged((void**)&csrRowPtrC, sizeof(int)*(M+1));
//
//   printf("MultMatrices13\n");
//
//   cusparseXcsrgemm2Nnz(handle, M, N, K,
//           descrA, nnzA, csrRowPtrA, csrColIndA,
//           descrB, nnzB, csrRowPtrB, csrColIndB,
//           descrD, nnzD, csrRowPtrD, csrColIndD,
//           descrC, csrRowPtrC, nnzTotalDevHostPtr,
//           info, buffer);
//
//   printf("MultMatrices14\n");
//
//   if (NULL != nnzTotalDevHostPtr)
//   {
//       nnzC = *nnzTotalDevHostPtr;
//   }
//   else
//   {
//       cudaMemcpy(&nnzC, csrRowPtrC+M, sizeof(int), cudaMemcpyDeviceToHost);
//       cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
//       nnzC -= baseC;
//   }
//
//   printf("MultMatrices15\n");
//
//   int *csrColIndC;
//   cudaMallocManaged((void**)&csrColIndC, sizeof(int)*nnzC);
//   double *csrValC;
//   cudaMallocManaged((void**)&csrValC, sizeof(double)*nnzC);
//
//   printf("MultMatrices16\n");
//
//   cusparseScsrgemm2(handle, M, N, K, &alpha,
//           descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
//           descrB, nnzB, csrValB, csrRowPtrB, csrColIndB,
//           &beta,
//           descrD, nnzD, csrValD, csrRowPtrD, csrColIndD,
//           descrC, csrValC, csrRowPtrC, csrColIndC,
//           info, buffer);
//
//   printf("MultMatrices17\n");
//
//   cusparseScsr2dense(handle, M, N, descrC, csrValC, csrRowPtrC, csrColIndC, d_C_dense, M);
//   cudaFree(D);
//   cudaFree(buffer);
//   cudaFree(nnzPerVectorA);
//   cudaFree(nnzPerVectorB);
//   cudaFree(nnzPerVectorD);
//   cudaFree(csrValA);
//   cudaFree(csrValB);
//   cudaFree(csrValC);
//   cudaFree(csrValD);
//   cudaFree(csrRowPtrA);
//   cudaFree(csrRowPtrB);
//   cudaFree(csrRowPtrC);
//   cudaFree(csrRowPtrD);
//   cudaFree(csrColIndA);
//   cudaFree(csrColIndB);
//   cudaFree(csrColIndC);
//   cudaFree(csrColIndD);
//
//   printf("MultMatrices18\n");
//
//   cudaMemcpy(C, d_C_dense, M * N * sizeof(double), cudaMemcpyDeviceToHost);
//
//   printf("MultMatrices19\n");
//
//   cudaFree(d_C_dense);
//   cusparseDestroyCsrgemm2Info(info);
//   cusparseDestroyMatDescr(descrA);
//   cusparseDestroyMatDescr(descrB);
//   cusparseDestroyMatDescr(descrC);
//   cusparseDestroyMatDescr(descrD);
//   cusparseDestroy(handle);
//
//   printf("MultMatrices20\n");
// }

// void multMatrices(double* a, long m, long k, double* b, long n, double* c, int numGPU)
// {
//   cublasXtHandle_t handle;
//   cublasXtCreate(&handle);
//   int devices[] = {0};
//   if(cublasXtDeviceSelect(handle, 1, devices) != CUBLAS_STATUS_SUCCESS)
//   {
//     printf("set devices fail\n");
//     exit(-1);
//   }
//   // int* devices = (int*) malloc(sizeof(int)*numGPU);
//   // for(int i=0; i<numGPU; i++)
//   //   devices[i] = i;
//   // int   devices[]
//   // int contadorDeIntentos = 0;
//   // int numLocalGPUs = numGPU;
//   // while(cublasXtDeviceSelect(handle, numLocalGPUs, devices) != CUBLAS_STATUS_SUCCESS)
//   // {
//   //   printf("set devices fail\n");
//   //   contadorDeIntentos++;
//   //   if(contadorDeIntentos == 20)
//   //   {
//   //     printf("No se ha logrado ejecutar mult con %d GPUs.\n", numLocalGPUs);
//   //     contadorDeIntentos = 0;
//   //     numLocalGPUs--;
//   //     if(numLocalGPUs == 0)
//   //     {
//   //       printf("No se ha conseguido usar ni una sola GPU.\n");
//   //       exit(-1);
//   //     }
//   //     devices = (int*) realloc(devices, sizeof(int)*numLocalGPUs);
//   //     printf("Reintentando mult con %d GPUs.\n", numLocalGPUs);
//   //   }
//   // }
//   double al = 1.0;
//   double bet = 0.0;
//   cublasXtSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
//   cudaDeviceSynchronize();
//   for(long i=0; i<m*n;i++)
//   {
//     if(isnan(c[i]))
//     {
//       printf("Valor nan encontrado en multMatrices.\n");
//       break;
//     }
//   }
//   cublasXtDestroy(handle);
//   // free(devices);
// }

// void multMatrices(double* a, long m, long k, double* b, long n, double* c, int numGPU)
// {
//   cublasHandle_t handle;
//   cudaSetDevice(1);
//   cublasCreate(&handle);
//   double al = 1.0;
//   double bet = 1.0;
//   cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
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

// void combinacionLinealMatrices(double al, double* a, long m, long k, double bet, double* c)
// {
//   long n = k;
//   cudaError_t cudaStat;
//   cublasStatus_t stat;
//   cublasXtHandle_t handle;
//   double* b;
//   cudaMallocManaged(&b, k*n*sizeof(double));
//   cudaMemset(b, 0, k*n*sizeof(double));
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

void vectorColumnaAMatriz(double* vectorA, long cantFilas, long cantColumnas, double* nuevaMatriz, int numGPU)
{
  double* vectorDeUnos;
  cudaMallocManaged(&vectorDeUnos,cantColumnas*sizeof(double));
  for(long i=0; i<cantColumnas; i++)
  {
    vectorDeUnos[i] = 1.0;
  }
  multMatrices(vectorA, cantFilas, 1, vectorDeUnos, cantColumnas, nuevaMatriz, numGPU);
  cudaFree(vectorDeUnos);
}

__global__ void multMatrizPorConstante_kernel(double* matrizA, long cantFilas, long cantColumnas, double constante)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    matrizA[miId] = constante * matrizA[miId];
  }
}

void multMatrizPorConstante(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, double constante, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/tamBloque);
  // cudaSetDevice(numGPU);
  multMatrizPorConstante_kernel<<<cantBloques,tamBloque>>>(matrizA, cantFilasMatrizA, cantColumnasMatrizA, constante);
  cudaDeviceSynchronize();
}

// __global__ void multMatrizPorConstante_kernel_multiGPU(double* matrizA, long cantFilas, long cantColumnas, double constante, int gpuId)
// {
//   long miId = threadIdx.x + blockDim.x * (blockIdx.x + gpuId * gridDim.x);
//   if(miId < cantFilas*cantColumnas)
//   {
//     matrizA[miId] = constante * matrizA[miId];
//   }
// }
//
// void multMatrizPorConstante(double* matrizA, long cantFilas, long cantColumnas, double constante, int tamBloque, int numGPU)
// {
//   long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque*numGPU);
//   #pragma omp parallel num_threads(numGPU)
//   {
//     int thread_id = omp_get_thread_num();
//     cudaSetDevice(thread_id);
//     multMatrizPorConstante_kernel_multiGPU<<<cantBloques,tamBloque>>>(matrizA, cantFilas, cantColumnas, constante, thread_id);
//   }
//   cudaDeviceSynchronize();
// }

// __global__ void combinacionLinealMatrices_kernel_multiGPU(double al, double* matrizA, long cantFilas, long cantColumnas, double bet, double* matrizB, int gpuId)
// {
//   long miId = threadIdx.x + blockDim.x * (blockIdx.x + gpuId * gridDim.x);
//   if(miId < cantFilas*cantColumnas)
//   {
//     matrizB[miId] = al * matrizA[miId] + bet * matrizB[miId];
//   }
// }
//
// void combinacionLinealMatrices(double al, double* matrizA, long cantFilas, long cantColumnas, double bet, double* matrizB, int tamBloque, int numGPU)
// {
//   long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque*numGPU);
//   #pragma omp parallel num_threads(numGPU)
//   {
//     int thread_id = omp_get_thread_num();
//     cudaSetDevice(thread_id);
//     combinacionLinealMatrices_kernel_multiGPU<<<cantBloques,tamBloque>>>(al, matrizA, cantFilas, cantColumnas, bet, matrizB, thread_id);
//   }
//   cudaDeviceSynchronize();
// }

__global__ void combinacionLinealMatrices_kernel(double al, double* matrizA, long cantFilas, long cantColumnas, double bet, double* matrizB)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    matrizB[miId] = al * matrizA[miId] + bet * matrizB[miId];
  }
}

void combinacionLinealMatrices(double al, double* matrizA, long cantFilas, long cantColumnas, double bet, double* matrizB, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque);
  // cudaSetDevice(numGPU);
  combinacionLinealMatrices_kernel<<<cantBloques,tamBloque>>>(al, matrizA, cantFilas, cantColumnas, bet, matrizB);
  cudaDeviceSynchronize();
}

__global__ void sumarMatrizConstante_kernel(double al, double* matrizA, long cantFilas, long cantColumnas, double bet)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    matrizA[miId] = al * matrizA[miId] + bet;
  }
}

void sumarMatrizConstante(double al, double* matrizA, long cantFilas, long cantColumnas, double bet, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque);
  sumarMatrizConstante_kernel<<<cantBloques,tamBloque>>>(al, matrizA, cantFilas, cantColumnas, bet);
  cudaDeviceSynchronize();
}

// __global__ void transponerMatriz_kernel(double* matrizA, double* matrizA_T, long cantFilas, long cantColumnas)
// {
//   long miId = threadIdx.x + blockDim.x * blockIdx.x;
//   if(miId < cantFilas*cantColumnas)
//   {
//     long i = miId%cantFilas;
//     long j = miId/cantFilas;
//     matrizA_T[(i*cantColumnas+j)] = matrizA[(j*cantFilas+i)];
//   }
// }
//
// void transponerMatriz(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, double* resultado, int numGPU)
// {
//   long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/1024);
//   transponerMatriz_kernel<<<cantBloques,1024>>>(matrizA, resultado, cantFilasMatrizA, cantColumnasMatrizA);
//   cudaDeviceSynchronize();
// }


__global__ void transponerMatriz_kernel(double* idata, double* odata, long width, long height)
{
  __shared__ double block[BLOCK_DIM][BLOCK_DIM+1];
	long xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	long yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		long index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}
	__syncthreads();
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		long index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

void transponerMatriz(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, double* resultado, int numGPU)
{
  dim3 grid(cantFilasMatrizA/BLOCK_DIM,  cantColumnasMatrizA/BLOCK_DIM, 1);
  dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
  // cudaSetDevice(numGPU);
  transponerMatriz_kernel<<<grid,threads>>>(matrizA, resultado, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

__global__ void restaVectorColumnaConVector_kernel(double* vectorA, long largoVectorA, double* vectorB, long largoVectorB, double* resultado)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < largoVectorA*largoVectorB)
  {
    long i = miId%largoVectorA;
    long j = miId/largoVectorA;
    resultado[miId] = vectorA[i] - vectorB[j];
  }
}

double* restaVectorColumnaConVector(double* vectorA, long largoVectorA, double* vectorB, long largoVectorB, int tamBloque, int numGPU)
{
  double* resultado;
  cudaMallocManaged(&resultado,largoVectorA*largoVectorB*sizeof(double));
  long cantBloques = ceil((double) largoVectorA*largoVectorB/tamBloque);
  // cudaSetDevice(numGPU);
  restaVectorColumnaConVector_kernel<<<cantBloques,tamBloque>>>(vectorA, largoVectorA, vectorB, largoVectorB, resultado);
  cudaDeviceSynchronize();
  return resultado;
}

// __global__ void restaVectorColumnaConVector_kernel_multiGPU(double* vectorA, long largoVectorA, double* vectorB, long largoVectorB, double* resultado, int gpuId)
// {
//   long miId = threadIdx.x + blockDim.x * (blockIdx.x + gpuId * gridDim.x);
//   if(miId < largoVectorA*largoVectorB)
//   {
//     long i = miId%largoVectorA;
//     long j = miId/largoVectorA;
//     resultado[miId] = vectorA[i] - vectorB[j];
//   }
// }
//
// double* restaVectorColumnaConVector(double* vectorA, long largoVectorA, double* vectorB, long largoVectorB, int tamBloque, int numGPU)
// {
//   double* resultado;
//   cudaMallocManaged(&resultado,largoVectorA*largoVectorB*sizeof(double));
//   long cantBloques = ceil((double) largoVectorA*largoVectorB/tamBloque*numGPU);
//   #pragma omp parallel num_threads(numGPU)
//   {
//     int thread_id = omp_get_thread_num();
//     cudaSetDevice(thread_id);
//     restaVectorColumnaConVector_kernel_multiGPU<<<cantBloques,tamBloque>>>(vectorA, largoVectorA, vectorB, largoVectorB, resultado, thread_id);
//   }
//   cudaDeviceSynchronize();
//   return resultado;
// }

// __global__ void hadamardProduct_kernel_multiGPU(double* matrizA, double* matrizB, double* resultado, long cantFilas, long cantColumnas, int gpuId)
// {
//   long miId = threadIdx.x + blockDim.x * (blockIdx.x + gpuId * gridDim.x);
//   if(miId < cantFilas*cantColumnas)
//   {
//     resultado[miId] = matrizA[miId]*matrizB[miId];
//   }
// }
//
// void hadamardProduct(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, double* matrizB, double* resultado, int tamBloque, int numGPU)
// {
//   long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/tamBloque*numGPU);
//   #pragma omp parallel num_threads(numGPU)
//   {
//     int thread_id = omp_get_thread_num();
//     cudaSetDevice(thread_id);
//     hadamardProduct_kernel_multiGPU<<<cantBloques,tamBloque>>>(matrizA, matrizB, resultado, cantFilasMatrizA, cantColumnasMatrizA, thread_id);
//   }
//   cudaDeviceSynchronize();
// }

__global__ void hadamardProduct_kernel(double* matrizA, double* matrizB, double* resultado, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    resultado[miId] = matrizA[miId]*matrizB[miId];
  }
}

void hadamardProduct(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, double* matrizB, double* resultado, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/tamBloque);
  // cudaSetDevice(numGPU);
  hadamardProduct_kernel<<<cantBloques,tamBloque>>>(matrizA, matrizB, resultado, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

__global__ void MultPorDifer_kernel(double* matrizA, double* matrizB, double* resultado, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    long posicionEnB = miId%cantFilas;
    resultado[miId] = matrizA[miId]*matrizB[posicionEnB];
  }
}

void MultPorDifer(double* matrizA, long cantFilas, long cantColumnas, double* diferencias, double* resultado, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque);
  // cudaSetDevice(numGPU);
  MultPorDifer_kernel<<<cantBloques,tamBloque>>>(matrizA, diferencias, resultado, cantFilas, cantColumnas);
  cudaDeviceSynchronize();
}

// __global__ void MultPorDifer_kernel_multiGPU(double* matrizA, double* matrizB, double* resultado, long cantFilas, long cantColumnas, int gpuId)
// {
//   long miId = threadIdx.x + blockDim.x * (blockIdx.x + gpuId * gridDim.x);
//   if(miId < cantFilas*cantColumnas)
//   {
//     long posicionEnB = miId%cantFilas;
//     resultado[miId] = matrizA[miId]*matrizB[posicionEnB];
//   }
// }
//
// void MultPorDifer(double* matrizA, long cantFilas, long cantColumnas, double* diferencias, double* resultado, int tamBloque, int numGPU)
// {
//   long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque*numGPU);
//   #pragma omp parallel num_threads(numGPU)
//   {
//     int thread_id = omp_get_thread_num();
//     cudaSetDevice(thread_id);
//     MultPorDifer_kernel_multiGPU<<<cantBloques,tamBloque>>>(matrizA, diferencias, resultado, cantFilas, cantColumnas, thread_id);
//   }
//   cudaDeviceSynchronize();
// }

double dotProduct(double* x, long n, double* y, int numGPU)
{
  // cudaSetDevice(numGPU);
  cublasHandle_t handle;
  cublasCreate(&handle);
  double result;
  cublasDdot(handle,n,x,1,y,1,&result);
  cublasDestroy(handle);
  return result;
}

__global__ void calcularExp_kernel(double* a, double* c, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    c[miId] = exp(a[miId]);
  }
}

void calcularExp(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA)
{
  long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/1024);
  calcularExp_kernel<<<cantBloques,1024>>>(matrizA, matrizA, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

__global__ void calcularExp_kernel_multiGPU(double* a, double* c, long cantFilas, long cantColumnas, int gpuId)
{
  long miId = threadIdx.x + blockDim.x * (blockIdx.x + gpuId * gridDim.x);
  if(miId < cantFilas*cantColumnas)
  {
    c[miId] = exp(a[miId]);
  }
}

void calcularExp2(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/tamBloque*numGPU);
  #pragma omp parallel num_threads(numGPU)
  {
    int thread_id = omp_get_thread_num();
    // cudaSetDevice(thread_id);
    calcularExp_kernel_multiGPU<<<cantBloques,tamBloque>>>(matrizA, matrizA, cantFilasMatrizA, cantColumnasMatrizA, thread_id);
  }
  cudaDeviceSynchronize();
}

__global__ void calcularInvFrac_kernel(double* a, double* c, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    c[miId] = 1.0/a[miId];
  }
}

void calcularInvFrac(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA)
{
  long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/1024);
  calcularInvFrac_kernel<<<cantBloques,1024>>>(matrizA, matrizA, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
}

__global__ void calcularInvFrac_kernel_multiGPU(double* a, double* c, long cantFilas, long cantColumnas, int gpuId)
{
  long miId = threadIdx.x + blockDim.x * (blockIdx.x + gpuId * gridDim.x);
  if(miId < cantFilas*cantColumnas)
  {
    c[miId] = 1.0/a[miId];
  }
}

void calcularInvFrac2(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/tamBloque*numGPU);
  #pragma omp parallel num_threads(numGPU)
  {
    int thread_id = omp_get_thread_num();
    // cudaSetDevice(thread_id);
    calcularInvFrac_kernel_multiGPU<<<cantBloques,tamBloque>>>(matrizA, matrizA, cantFilasMatrizA, cantColumnasMatrizA, thread_id);
  }
  cudaDeviceSynchronize();
}

void calVisModelo(double* MV, long cantFilasMV, long cantColumnasMV, double* MC, long cantColumnasMU, double* MU, double* matrizDeUnosTamN, double* visModelo_paso3, int tamBloque, int numGPU)
{
  double* MU_T;
  cudaMallocManaged(&MU_T, cantFilasMV*cantColumnasMU*sizeof(double));
  transponerMatriz(MU, cantFilasMV, cantColumnasMU, MU_T, numGPU);
  double* visModelo_paso1;
  cudaMallocManaged(&visModelo_paso1, cantColumnasMV*cantFilasMV*sizeof(double));
  cudaMemset(visModelo_paso1, 0, cantColumnasMV*cantFilasMV*sizeof(double));
  multMatrices(MC, cantColumnasMV, cantColumnasMU, MU_T, cantFilasMV, visModelo_paso1, numGPU);
  cudaFree(MU_T);
  double* transpuesta;
  cudaMallocManaged(&transpuesta, cantColumnasMV*cantFilasMV*sizeof(double));
  transponerMatriz(visModelo_paso1, cantColumnasMV, cantFilasMV, transpuesta, numGPU);
  cudaFree(visModelo_paso1);
  double* visModelo_paso2;
  cudaMallocManaged(&visModelo_paso2, cantFilasMV*cantColumnasMV*sizeof(double));
  hadamardProduct(MV, cantFilasMV, cantColumnasMV, transpuesta, visModelo_paso2, tamBloque, numGPU);
  cudaFree(transpuesta);
  multMatrices(visModelo_paso2, cantFilasMV, cantColumnasMV, matrizDeUnosTamN, 1, visModelo_paso3, numGPU);
  cudaFree(visModelo_paso2);
}

double* calResidual(double* visObs, double* MV, long cantFilasMV, long cantColumnasMV, double* MC, long cantColumnasMU, double* MU, double* matrizDeUnosTamN, int tamBloque, int numGPU)
{
  double* visModelo;
  cudaMallocManaged(&visModelo, cantFilasMV*sizeof(double));
  cudaMemset(visModelo, 0, cantFilasMV*sizeof(double));
  calVisModelo(MV, cantFilasMV, cantColumnasMV, MC, cantColumnasMU, MU, matrizDeUnosTamN, visModelo, tamBloque, numGPU);
  combinacionLinealMatrices(-1.0, visObs, cantFilasMV, 1, 1.0, visModelo, tamBloque, numGPU);
  return visModelo;
}

double calCosto(double* residual, long cantVisi, double* w, int tamBloque, int numGPU)
{
  double* resultado;
  cudaMallocManaged(&resultado, cantVisi*sizeof(double));
  hadamardProduct(residual, cantVisi, 1, w, resultado, tamBloque, numGPU);
  double total = dotProduct(resultado, cantVisi, residual, numGPU);
  cudaFree(resultado);
  return total;
}

void calGradiente(double* residual, double* MV, long cantFilasMV, long cantColumnasMV, double* MU, long cantColumnasMU, double* w, double* total_paso2, int tamBloque, int numGPU)
{
  double* diferencia;
  cudaMallocManaged(&diferencia, cantFilasMV*sizeof(double));
  hadamardProduct(residual, cantFilasMV, 1, w, diferencia, tamBloque, numGPU);
  double* total_paso1;
  cudaMallocManaged(&total_paso1, cantColumnasMV*cantFilasMV*sizeof(double));
  MultPorDifer(MV, cantFilasMV, cantColumnasMV, diferencia, total_paso1, tamBloque, numGPU);
  cudaFree(diferencia);
  double* total_paso1_5;
  cudaMallocManaged(&total_paso1_5, cantColumnasMV*cantFilasMV*sizeof(double));
  transponerMatriz(total_paso1, cantFilasMV, cantColumnasMV, total_paso1_5, numGPU);
  cudaFree(total_paso1);
  multMatrices(total_paso1_5, cantColumnasMV, cantFilasMV, MU, cantColumnasMU, total_paso2, numGPU);
  cudaFree(total_paso1_5);
}

double calAlpha(double* gradiente, long cantFilasMC, long cantColumnasMC, double* pActual, double* MV, long cantFilasMV, long cantColumnasMV, double* MU, long cantColumnasMU, double* w, double* matrizDeUnosTamN, int* flag_NOESPOSIBLEMINIMIZAR, int tamBloque, int numGPU)
{
  double* gradienteNegativo;
  cudaMallocManaged(&gradienteNegativo, cantFilasMC*cantColumnasMC*sizeof(double));
  cudaMemset(gradienteNegativo, 0, cantFilasMC*cantColumnasMC*sizeof(double));
  combinacionLinealMatrices(-1.0, gradiente, cantFilasMC, cantColumnasMC, 0.0, gradienteNegativo, tamBloque, numGPU);
  double numerador = dotProduct(gradienteNegativo, cantFilasMC*cantColumnasMC, pActual, numGPU);
  cudaFree(gradienteNegativo);
  double* visModeloP;
  cudaMallocManaged(&visModeloP, cantFilasMV*sizeof(double));
  cudaMemset(visModeloP, 0, cantFilasMV*sizeof(double));
  calVisModelo(MV, cantFilasMV, cantColumnasMV, pActual, cantColumnasMU, MU, matrizDeUnosTamN, visModeloP, tamBloque, numGPU);
  double* gradP;
  cudaMallocManaged(&gradP, cantFilasMC * cantColumnasMC*sizeof(double));
  cudaMemset(gradP, 0, cantFilasMC * cantColumnasMC*sizeof(double));
  calGradiente(visModeloP, MV, cantFilasMV, cantColumnasMV, MU, cantColumnasMU, w, gradP, tamBloque, numGPU);
  cudaFree(visModeloP);
  double denominador = dotProduct(pActual, cantFilasMC * cantColumnasMC, gradP, numGPU);
  cudaFree(gradP);
  if(denominador == 0.0)
  {
    *flag_NOESPOSIBLEMINIMIZAR = 1;
  }
  return numerador/denominador;
}

__global__ void matrizDeSigno_kernel(double* a, double* c, double lambda, long cantFilas, long cantColumnas)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    double valor = a[miId];
    if (valor != 0.0)
    {
      c[miId] = lambda * valor/abs(valor);
    }
  }
}

double* matrizDeSigno(double* matrizA, long cantFilas, long cantColumnas, double lambda, int tamBloque)
{
  double* matrizDeSigno;
  cudaMallocManaged(&matrizDeSigno, cantFilas*cantColumnas*sizeof(double));
  cudaMemset(matrizDeSigno, 0, cantFilas*cantColumnas*sizeof(double));
  long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque);
  matrizDeSigno_kernel<<<cantBloques,tamBloque>>>(matrizA, matrizDeSigno, lambda, cantFilas, cantColumnas);
  cudaDeviceSynchronize();
  return matrizDeSigno;
}
//
// __global__ void matrizSoloDeSigno_kernel(double* a, double* c, long cantFilas, long cantColumnas)
// {
//   long miId = threadIdx.x + blockDim.x * blockIdx.x;
//   if(miId < cantFilas*cantColumnas)
//   {
//     double valor = a[miId];
//     if (valor != 0.0)
//     {
//       c[miId] = valor/abs(valor);
//     }
//   }
// }
//
// double* matrizSoloDeSigno(double* matrizA, long cantFilas, long cantColumnas, int tamBloque)
// {
//   double* matrizDeSigno;
//   cudaMallocManaged(&matrizDeSigno, cantFilas*cantColumnas*sizeof(double));
//   long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque);
//   matrizSoloDeSigno_kernel<<<cantBloques,tamBloque>>>(matrizA, matrizDeSigno, cantFilas, cantColumnas);
//   cudaDeviceSynchronize();
//   return matrizDeSigno;
// }

double calCosto_l1(double lambda, double* residual, long cantVisi, double* w, double* MC, int N, int tamBloque, int numGPU)
{
  af::array matrizDeCoefs_GPU(N*N, MC);
  double total_sumcoefs = af::sum<double>(abs(matrizDeCoefs_GPU));
  af::sync();
  matrizDeCoefs_GPU.unlock();
  double* resultado;
  cudaMallocManaged(&resultado, cantVisi*sizeof(double));
  hadamardProduct(residual, cantVisi, 1, w, resultado, tamBloque, numGPU);
  double total = dotProduct(resultado, cantVisi, residual, numGPU);
  cudaFree(resultado);
  return total + total_sumcoefs * lambda;
}

void calGradiente_l1(double lambda, double* residual, double* MV, long cantFilasMV, long cantColumnasMV, double* MU, long cantColumnasMU, double* w, double* MC, double* total_paso2, int N, int tamBloque, int numGPU)
{
  double* diferencia;
  cudaMallocManaged(&diferencia, cantFilasMV*sizeof(double));
  hadamardProduct(residual, cantFilasMV, 1, w, diferencia, tamBloque, numGPU);
  double* total_paso1;
  cudaMallocManaged(&total_paso1, cantColumnasMV*cantFilasMV*sizeof(double));
  MultPorDifer(MV, cantFilasMV, cantColumnasMV, diferencia, total_paso1, tamBloque, numGPU);
  cudaFree(diferencia);
  double* total_paso1_5;
  cudaMallocManaged(&total_paso1_5, cantColumnasMV*cantFilasMV*sizeof(double));
  transponerMatriz(total_paso1, cantFilasMV, cantColumnasMV, total_paso1_5, numGPU);
  cudaFree(total_paso1);
  multMatrices(total_paso1_5, cantColumnasMV, cantFilasMV, MU, cantColumnasMU, total_paso2, numGPU);
  cudaFree(total_paso1_5);
  double* matrizDeSignos_Coefs = matrizDeSigno(MC, N, N, lambda, tamBloque);
  combinacionLinealMatrices(1.0, matrizDeSignos_Coefs, cantColumnasMV, cantColumnasMU, 1.0, total_paso2, tamBloque, numGPU);
  cudaFree(matrizDeSignos_Coefs);
}

double calBeta_Fletcher_Reeves(double* gradienteActual, long tamanoGradiente, double* gradienteAnterior, int numGPU)
{
  double numerador = dotProduct(gradienteActual, tamanoGradiente, gradienteActual, numGPU);
  double denominador = dotProduct(gradienteAnterior, tamanoGradiente, gradienteAnterior, numGPU);
  double resultado = numerador/denominador;
  return resultado;
}

double* calInfoFisherDiag(double* MV, long cantFilasMV, long cantColumnasMV, double* MU, double* w, int tamBloque, int numGPU)
{
  double* MV_T;
  cudaMallocManaged(&MV_T, cantFilasMV*cantColumnasMV*sizeof(double));
  transponerMatriz(MV, cantFilasMV, cantColumnasMV, MV_T, numGPU);
  double* primeraMatriz_fase1;
  cudaMallocManaged(&primeraMatriz_fase1, cantColumnasMV*cantFilasMV*sizeof(double));
  hadamardProduct(MV_T, cantColumnasMV, cantFilasMV, MV_T, primeraMatriz_fase1, tamBloque, numGPU);
  cudaFree(MV_T);
  double* wMatriz;
  cudaMallocManaged(&wMatriz, cantFilasMV*cantColumnasMV*sizeof(double));
  cudaMemset(wMatriz, 0, cantFilasMV*cantColumnasMV*sizeof(double));
  vectorColumnaAMatriz(w, cantFilasMV, cantColumnasMV, wMatriz, numGPU);
  double* wmatriz_T;
  cudaMallocManaged(&wmatriz_T, cantFilasMV*cantColumnasMV*sizeof(double));
  transponerMatriz(wMatriz, cantFilasMV, cantColumnasMV, wmatriz_T, numGPU);
  cudaFree(wMatriz);
  double* primeraMatriz_fase2;
  cudaMallocManaged(&primeraMatriz_fase2, cantColumnasMV*cantFilasMV*sizeof(double));
  hadamardProduct(primeraMatriz_fase1, cantColumnasMV, cantFilasMV, wmatriz_T, primeraMatriz_fase2, tamBloque, numGPU);
  cudaFree(primeraMatriz_fase1);
  cudaFree(wmatriz_T);
  double* MU_T;
  cudaMallocManaged(&MU_T, cantFilasMV*cantColumnasMV*sizeof(double));
  transponerMatriz(MU, cantFilasMV, cantColumnasMV, MU_T, numGPU);
  double* segundaMatriz;
  cudaMallocManaged(&segundaMatriz, cantFilasMV*cantColumnasMV*sizeof(double));
  hadamardProduct(MU_T, cantFilasMV, cantColumnasMV, MU_T, segundaMatriz, tamBloque, numGPU);
  cudaFree(MU_T);
  double* resultado_fase1;
  cudaMallocManaged(&resultado_fase1, cantColumnasMV*cantFilasMV*sizeof(double));
  hadamardProduct(primeraMatriz_fase2, cantColumnasMV, cantFilasMV, segundaMatriz, resultado_fase1, tamBloque, numGPU);
  cudaFree(primeraMatriz_fase2);
  cudaFree(segundaMatriz);
  double* vectorDeUnos;
  cudaMallocManaged(&vectorDeUnos, cantFilasMV*sizeof(double));
  double* resultado_fase2;
  cudaMallocManaged(&resultado_fase2, cantColumnasMV*sizeof(double));
  cudaMemset(resultado_fase2, 0, cantColumnasMV*sizeof(double));
  for(long i=0; i<cantFilasMV; i++)
  {
    vectorDeUnos[i] = 1;
  }
  multMatrices(resultado_fase1, cantColumnasMV, cantFilasMV, vectorDeUnos, 1, resultado_fase2, numGPU);
  cudaFree(resultado_fase1);
  double medidaInfoMaximoDiagonal = 0.0;
  for (long i=0; i<cantColumnasMV; i++)
  {
      if(resultado_fase2[i] > medidaInfoMaximoDiagonal)
        medidaInfoMaximoDiagonal = resultado_fase2[i];
  }
  double medidaInfoSumaDiagonal = dotProduct(resultado_fase2, cantColumnasMV, vectorDeUnos, numGPU);
  cudaFree(vectorDeUnos);
  cudaFree(resultado_fase2);
  double* medidasDeInfo = (double*) malloc(sizeof(double)*2);
  medidasDeInfo[0] = medidaInfoSumaDiagonal;
  medidasDeInfo[1] = medidaInfoMaximoDiagonal;
  return medidasDeInfo;
}

double* estimacionDePlanoDeFourier(double* MV, long cantFilasMV, long cantColumnasMV, double* MC, long cantFilasMC, long cantColumnasMC, double* MU, int numGPU)
{
  double* MU_T;
  cudaMallocManaged(&MU_T, cantFilasMV*cantColumnasMV*sizeof(double));
  transponerMatriz(MU, cantFilasMV, cantColumnasMV, MU_T, numGPU);
  double* resultado_paso1;
  cudaMallocManaged(&resultado_paso1, cantFilasMC*cantFilasMV*sizeof(double));
  cudaMemset(resultado_paso1, 0, cantFilasMC*cantFilasMV*sizeof(double));
  multMatrices(MC, cantFilasMC, cantColumnasMC, MU_T, cantFilasMV, resultado_paso1, numGPU);
  cudaFree(MU_T);
  double* resultado_paso2;
  cudaMallocManaged(&resultado_paso2, cantFilasMV*cantFilasMV*sizeof(double));
  cudaMemset(resultado_paso2, 0, cantFilasMV*cantFilasMV*sizeof(double));
  multMatrices(MV, cantFilasMV, cantColumnasMV, resultado_paso1, cantFilasMV, resultado_paso2, numGPU);
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

void escribirTransformadaInversaFourier2D(double* estimacionFourier_ParteImag, double* estimacionFourier_ParteReal, long N, char* nombreArchivo)
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
  double* auxiliar_mapaFourierRecons = mapaFourierRecons.device<double>();
  double* inver_visi = (double*) calloc(N*N, sizeof(double));
  cudaMemcpy(inver_visi, auxiliar_mapaFourierRecons, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  mapaFourierRecons.unlock();
  fitsfile *fptr;
  int status;
  long fpixel, nelements;
  int bitpix = DOUBLE_IMG;
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
  if (fits_write_img(fptr, TDOUBLE, fpixel, nelements, inver_visi, &status))
    printerror_cfitsio(status);
  if (fits_close_file(fptr, &status))
    printerror_cfitsio(status);
  free(inver_visi);
}

double* calcularMV_Rect(double* v, double delta_v, long cantVisi, long N, double estrechezDeBorde, double ancho, double* matrizDeUnos, int tamBloque, int numGPU)
{
  double* desplazamientoEnV = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  double* primeraFraccionV;
  cudaMallocManaged(&primeraFraccionV, cantVisi * N * sizeof(double));
  cudaMemset(primeraFraccionV, 0, cantVisi * N * sizeof(double));
  double* segundaFraccionV;
  cudaMallocManaged(&segundaFraccionV, cantVisi * N * sizeof(double));
  for(long i=0; i<(cantVisi*N); i++)
  {
    segundaFraccionV[i] = 1.0;
  }
  double* matrizDiferenciaV = restaVectorColumnaConVector(v, cantVisi, desplazamientoEnV, N, tamBloque, numGPU);
  cudaFree(desplazamientoEnV);
  combinacionLinealMatrices(-1.0 * estrechezDeBorde, matrizDiferenciaV, cantVisi, N, 0.0, primeraFraccionV, tamBloque, numGPU);
  combinacionLinealMatrices(estrechezDeBorde, matrizDiferenciaV, cantVisi, N, -1 * estrechezDeBorde * ancho, segundaFraccionV, tamBloque, numGPU);
  cudaFree(matrizDiferenciaV);
  calcularExp(primeraFraccionV, cantVisi, N);
  calcularExp(segundaFraccionV, cantVisi, N);
  combinacionLinealMatrices(1.0, matrizDeUnos, cantVisi, N, 1.0, primeraFraccionV, tamBloque, numGPU);
  combinacionLinealMatrices(1.0, matrizDeUnos, cantVisi, N, 1.0, segundaFraccionV, tamBloque, numGPU);
  calcularInvFrac(primeraFraccionV, cantVisi, N);
  calcularInvFrac(segundaFraccionV, cantVisi, N);
  double* MV;
  cudaMallocManaged(&MV, cantVisi * N * sizeof(double));
  for(long i=0; i<(cantVisi*N); i++)
  {
    MV[i] = 1.0/ancho;
  }
  combinacionLinealMatrices(1.0, primeraFraccionV, cantVisi, N, 1.0, segundaFraccionV, tamBloque, numGPU);
  cudaFree(primeraFraccionV);
  combinacionLinealMatrices(1.0/ancho, segundaFraccionV, cantVisi, N, -1.0, MV, tamBloque, numGPU);
  cudaFree(segundaFraccionV);
  return MV;
}

double* calcularMV_Rect_estFourier(double ancho, long N, double delta_v, double* matrizDeUnos, double estrechezDeBorde, double* matrizDeUnosTamN, int tamBloque, int numGPU)
{
  double* coordenadasVCentrosCeldas = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(0.5 * delta_v, matrizDeUnosTamN, N, 1, 1.0, coordenadasVCentrosCeldas, tamBloque, numGPU);
  double* MV_AF = calcularMV_Rect(coordenadasVCentrosCeldas, delta_v, N, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  cudaFree(coordenadasVCentrosCeldas);
  return MV_AF;
}

double* calcularMV_Normal(double* v, double delta_v, long cantVisi, long N, double anchoV, int tamBloque, int numGPU)
{
  double* CV;
  cudaMallocManaged(&CV, N * sizeof(double));
  for(long i=0;i<N;i++)
  {
    CV[i] = 0.5 * delta_v;
  }
  double* CV_sinescalar = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(1.0, CV_sinescalar, N, 1, 1.0, CV, tamBloque, numGPU);
  cudaFree(CV_sinescalar);
  double* MV = restaVectorColumnaConVector(v, cantVisi, CV, N, tamBloque, numGPU);
  cudaFree(CV);
  multMatrizPorConstante(MV, cantVisi, N, 1.0/anchoV, tamBloque, numGPU);
  hadamardProduct(MV, cantVisi, N, MV, MV, tamBloque, numGPU);
  multMatrizPorConstante(MV, cantVisi, N, -0.5, tamBloque, numGPU);
  calcularExp(MV, cantVisi, N);
  multMatrizPorConstante(MV, cantVisi, N, 1.0/sqrt(2.0 * M_PI * anchoV * anchoV), tamBloque, numGPU);
  return MV;
}

// double* calcularMV_Normal(double* v, double delta_v, int cantVisi, int N, double anchoV)
// {
//   double* CV = (double*) calloc(N, sizeof(double));
//   double* matrizDeCeros = (double*) calloc(cantVisi * N, sizeof(double));
//   for(int i=0;i<N;i++)
//   {
//     CV[i] = 0.5 * delta_v;
//   }
//   double* CV_sinescalar = linspace((-N/2.0) * delta_v, ((N/2.0) - 1) * delta_v, N);
//   combinacionLinealMatrices(1.0, CV_sinescalar, N, 1, 1.0, CV);
//   free(CV_sinescalar);
//   double* MV = restaVectorColumnaConVector(v, cantVisi, CV, N);
//   free(CV);
//   combinacionLinealMatrices(0.0, matrizDeCeros, cantVisi, N, 1.0/anchoV, MV);
//   hadamardProduct(MV, cantVisi, N, MV, MV);
//   combinacionLinealMatrices(0.0, matrizDeCeros, cantVisi, N, -0.5, MV);
//   calcularExp(MV, cantVisi, N);
//   combinacionLinealMatrices(0.0, matrizDeCeros, cantVisi, N, 1.0/sqrt(2.0 * M_PI * anchoV * anchoV), MV);
//   free(matrizDeCeros);
//   return MV;
// }

double* calcularMV_Normal_estFourier(double anchoV, long N, double delta_v, double* matrizDeUnosTamN, int tamBloque, int numGPU)
{
  double* coordenadasVCentrosCeldas = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(0.5 * delta_v, matrizDeUnosTamN, N, 1, 1.0, coordenadasVCentrosCeldas, tamBloque, numGPU);
  double* MV_AF = calcularMV_Normal(coordenadasVCentrosCeldas, delta_v, N, N, anchoV, tamBloque, numGPU);
  cudaFree(coordenadasVCentrosCeldas);
  return MV_AF;
}

double* calcularMV_InvCuadra(double* v, double delta_v, long cantVisi, long N, double anchoV, int tamBloque, int numGPU)
{
  double* CV;
  cudaMallocManaged(&CV, N * sizeof(double));
  for(long i=0;i<N;i++)
  {
    CV[i] = 0.5 * delta_v;
  }
  double* CV_sinescalar = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(1.0, CV_sinescalar, N, 1, 1.0, CV, tamBloque, numGPU);
  cudaFree(CV_sinescalar);
  double* MV = restaVectorColumnaConVector(v, cantVisi, CV, N, tamBloque, numGPU);
  cudaFree(CV);
  multMatrizPorConstante(MV, cantVisi, N, 1.0/anchoV, tamBloque, numGPU);
  hadamardProduct(MV, cantVisi, N, MV, MV, tamBloque, numGPU);
  sumarMatrizConstante(1.0, MV, cantVisi, N, 1.0, tamBloque, numGPU);
  calcularInvFrac(MV, cantVisi, N);
  multMatrizPorConstante(MV, cantVisi, N, 1.0/(M_PI*anchoV), tamBloque, numGPU);
  return MV;
}

double* calcularMV_InvCuadra_estFourier(double anchoV, long N, double delta_v, double* matrizDeUnosTamN, int tamBloque, int numGPU)
{
  double* coordenadasVCentrosCeldas = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(0.5 * delta_v, matrizDeUnosTamN, N, 1, 1.0, coordenadasVCentrosCeldas, tamBloque, numGPU);
  double* MV_AF = calcularMV_InvCuadra(coordenadasVCentrosCeldas, delta_v, N, N, anchoV, tamBloque, numGPU);
  cudaFree(coordenadasVCentrosCeldas);
  return MV_AF;
}

__global__ void combinacionLinealMatrices_kernel_conretorno(double al, double* matrizA, long cantFilas, long cantColumnas, double bet, double* matrizB, double* resultado)
{
  long miId = threadIdx.x + blockDim.x * blockIdx.x;
  if(miId < cantFilas*cantColumnas)
  {
    resultado[miId] = al * matrizA[miId] + bet * matrizB[miId];
  }
}

double* combinacionLinealMatrices_conretorno(double al, double* matrizA, long cantFilas, long cantColumnas, double bet, double* matrizB, int tamBloque, int numGPU)
{
  double* resultado;
  cudaMallocManaged(&resultado, cantFilas*cantColumnas*sizeof(double));
  long cantBloques = ceil((double) cantFilas*cantColumnas/tamBloque);
  combinacionLinealMatrices_kernel_conretorno<<<cantBloques,tamBloque>>>(al, matrizA, cantFilas, cantColumnas, bet, matrizB, resultado);
  cudaDeviceSynchronize();
  return resultado;
}

double* hadamardProduct_conretorno(double* matrizA, long cantFilasMatrizA, long cantColumnasMatrizA, double* matrizB, int tamBloque, int numGPU)
{
  long cantBloques = ceil((double) cantFilasMatrizA*cantColumnasMatrizA/tamBloque);
  double* resultado;
  cudaMallocManaged(&resultado, cantFilasMatrizA*cantColumnasMatrizA*sizeof(double));
  hadamardProduct_kernel<<<cantBloques,tamBloque>>>(matrizA, matrizB, resultado, cantFilasMatrizA, cantColumnasMatrizA);
  cudaDeviceSynchronize();
  return resultado;
}

double* generarSiguienteColumna(int k, double* primero, long largo, double* segundo, double* xinf, int tamBloque, int numGPU)
{
  double* primerVectorDif_paso1 = combinacionLinealMatrices_conretorno(sqrt(2.0/k), xinf, largo, 1, 0.0, xinf, tamBloque, numGPU);
  double* primerVectorDif_paso2 = hadamardProduct_conretorno(primerVectorDif_paso1, largo, 1, segundo, tamBloque, numGPU);
  cudaFree(primerVectorDif_paso1);
  double* segundVectorDif = combinacionLinealMatrices_conretorno(sqrt(1.0 - 1.0/k), primero, largo, 1, 0.0, xinf, tamBloque, numGPU);
  double* nuevo = combinacionLinealMatrices_conretorno(1.0, primerVectorDif_paso2, largo, 1, -1.0, segundVectorDif, tamBloque, numGPU);
  cudaFree(primerVectorDif_paso2);
  cudaFree(segundVectorDif);
  return nuevo;
}

void reemplazarColumna(double* matriz, int numFilasARecorrer, long cantFilas, int* iinf, long indiceColumna, double* nuevaColumna)
{
  for(int i=0;i<numFilasARecorrer;i++)
  {
      matriz[(((indiceColumna)*(cantFilas))+(iinf[i]))] = nuevaColumna[i];
  }
}

double* hermite(double* x, long largoDeX, long deg, int tamBloque, int numGPU)
{
  double limitGauss = 5;
  double rpi = sqrt(M_PI);
  double* xsup;
  cudaMallocManaged(&xsup, largoDeX*sizeof(double));
  cudaMemset(xsup, 0, largoDeX*sizeof(double));
  int* isup;
  cudaMallocManaged(&isup, largoDeX*sizeof(int));
  cudaMemset(isup, 0, largoDeX*sizeof(int));
  int indiceisup = 0;
  double* xinf;
  cudaMallocManaged(&xinf, largoDeX*sizeof(double));
  cudaMemset(xinf, 0, largoDeX*sizeof(double));
  int* iinf;
  cudaMallocManaged(&iinf, largoDeX*sizeof(int));
  cudaMemset(iinf, 0, largoDeX*sizeof(int));
  int indiceiif = 0;
  for(int i=0; i<largoDeX; i++)
  {
    if (abs(x[i]) > limitGauss)
    {
      isup[indiceisup] = i;
      xsup[indiceisup] = x[i];
      indiceisup++;
    }
    else
    {
      iinf[indiceiif] = i;
      xinf[indiceiif] = x[i];
      indiceiif++;
    }
  }
  // isup = (int*) realloc(sizeof(int)*(indiceisup+1));
  // iinf = (int*) realloc(sizeof(int)*(indiceiif+1));
  double* v;
  cudaMallocManaged(&v, (largoDeX)*(deg+1)*sizeof(double));
  cudaMemset(v, 0, (largoDeX)*(deg+1)*sizeof(double));
  if(indiceiif > 0)
  {
    double* x22inf_paso1 = hadamardProduct_conretorno(xinf, indiceiif, 1, xinf, tamBloque, numGPU);
    double* x22inf_paso2 = combinacionLinealMatrices_conretorno(0.5, x22inf_paso1, indiceiif, 1, 0.0, x22inf_paso1, tamBloque, numGPU);
    cudaFree(x22inf_paso1);
    double* x22infnegativo = combinacionLinealMatrices_conretorno(-1.0, x22inf_paso2, indiceiif, 1, 0.0, x22inf_paso2, tamBloque, numGPU);
    calcularExp(x22infnegativo, indiceiif, 1);
    double* primeraColumna = combinacionLinealMatrices_conretorno(1.0/sqrt(rpi), x22infnegativo, indiceiif, 1, 0.0, x22infnegativo, tamBloque, numGPU);
    reemplazarColumna(v, indiceiif, largoDeX, iinf, 0, primeraColumna);
    if (deg > 0)
    {
      double* x2inf = combinacionLinealMatrices_conretorno(2, xinf, indiceiif, 1, 0.0, x22infnegativo, tamBloque, numGPU);
      double* segundaColumna_paso1 = combinacionLinealMatrices_conretorno(1.0/sqrt(2.0*rpi), x22infnegativo, indiceiif, 1, 0.0, x22inf_paso2, tamBloque, numGPU);
      double* segundaColumna = hadamardProduct_conretorno(segundaColumna_paso1, indiceiif, 1, x2inf, tamBloque, numGPU);
      cudaFree(x2inf);
      cudaFree(segundaColumna_paso1);
      reemplazarColumna(v, indiceiif, largoDeX, iinf, 1, segundaColumna);
      for(int i=2; i<(deg+1); i++)
      {
        double* auxiliar = primeraColumna;
        primeraColumna = segundaColumna;
        segundaColumna = generarSiguienteColumna(i, auxiliar, indiceiif, segundaColumna, xinf, tamBloque, numGPU);
        reemplazarColumna(v, indiceiif, largoDeX, iinf, i, segundaColumna);
        cudaFree(auxiliar);
      }
      cudaFree(segundaColumna);
    }
    cudaFree(xinf);
    cudaFree(iinf);
    cudaFree(x22inf_paso2);
    cudaFree(x22infnegativo);
  }
  if (indiceisup > 0)
  {
    double* x22sup_paso1 = hadamardProduct_conretorno(xsup, indiceisup, 1, xsup, tamBloque, numGPU);
    double* x22sup_paso2 = combinacionLinealMatrices_conretorno(0.5, x22sup_paso1, indiceisup, 1, 0.0, x22sup_paso1, tamBloque, numGPU);
    cudaFree(x22sup_paso1);
    double* x22supnegativo = combinacionLinealMatrices_conretorno(-1.0, x22sup_paso2, indiceisup, 1, 0.0, x22sup_paso2, tamBloque, numGPU);
    calcularExp(x22supnegativo, indiceisup, 1);
    double* primeraColumna_sup = combinacionLinealMatrices_conretorno(1.0/sqrt(rpi), x22supnegativo, indiceisup, 1, 0.0, x22supnegativo, tamBloque, numGPU);
    reemplazarColumna(v, indiceisup, largoDeX, isup, 0, primeraColumna_sup);
    if (deg > 0)
    {
      double* x2sup = combinacionLinealMatrices_conretorno(2, xsup, indiceisup, 1, 0.0, x22supnegativo, tamBloque, numGPU);
      double* segundaColumna_paso1_sup = combinacionLinealMatrices_conretorno(1.0/sqrt(2.0*rpi), x22supnegativo, indiceisup, 1, 0.0, x22sup_paso2, tamBloque, numGPU);
      double* segundaColumna_sup = hadamardProduct_conretorno(segundaColumna_paso1_sup, indiceisup, 1, x2sup, tamBloque, numGPU);
      cudaFree(x2sup);
      cudaFree(segundaColumna_paso1_sup);
      reemplazarColumna(v, indiceisup, largoDeX, isup, 1, segundaColumna_sup);
      for(int i=2; i<(deg+1); i++)
      {
        double* auxiliar = primeraColumna_sup;
        primeraColumna_sup = segundaColumna_sup;
        segundaColumna_sup = generarSiguienteColumna(i, auxiliar, indiceisup, segundaColumna_sup, xsup, tamBloque, numGPU);
        reemplazarColumna(v, indiceisup, largoDeX, isup, i, segundaColumna_sup);
        cudaFree(auxiliar);
      }
      cudaFree(segundaColumna_sup);
    }
    cudaFree(xsup);
    cudaFree(isup);
    cudaFree(x22sup_paso2);
    cudaFree(x22supnegativo);
  }
  return v;
}

double buscarMaximo(double* lista, int largoLista)
{
  int maximoActual = lista[0];
  for(int i=0; i<largoLista; i++)
  {
    if(lista[i] > maximoActual)
    {
      maximoActual = lista[i];
    }
  }
  return maximoActual;
}

int calCompresionSegunCota(char* nombreArCoef_comp_imag, char* nombreArCoef_comp_real, double* MC_imag, double* MC_imag_comp, double* MC_real, double* MC_real_comp, long cantFilas, long cantColumnas, double cotaEnergia, int tamBloque, int numGPU)
{
  long largo = cantFilas * cantColumnas;
  double* MC_img_cuadrado;
  cudaMallocManaged(&MC_img_cuadrado, cantFilas*cantColumnas*sizeof(double));
  double* MC_modulo;
  cudaMallocManaged(&MC_modulo, cantFilas*cantColumnas*sizeof(double));
  hadamardProduct(MC_imag, cantFilas, cantColumnas, MC_imag, MC_img_cuadrado, tamBloque, numGPU);
  hadamardProduct(MC_real, cantFilas, cantColumnas, MC_real, MC_modulo, tamBloque, numGPU);
  combinacionLinealMatrices(1.0, MC_img_cuadrado, cantFilas, cantColumnas, 1.0, MC_modulo, tamBloque, numGPU);
  cudaFree(MC_img_cuadrado);
  af::array MC_modulo_GPU(cantFilas*cantColumnas, MC_modulo);
  af::array MC_modulo_indicesOrde_GPU(cantFilas*cantColumnas);
  af::array MC_modulo_Orde_GPU(cantFilas*cantColumnas);
  af::sort(MC_modulo_Orde_GPU, MC_modulo_indicesOrde_GPU, MC_modulo_GPU, 0, false);
  double total = af::sum<double>(MC_modulo_GPU);
  MC_modulo_Orde_GPU = MC_modulo_Orde_GPU/total;
  af::eval(MC_modulo_Orde_GPU);
  af::sync();
  double* auxiliar_MC_modulo_Orde_GPU = MC_modulo_Orde_GPU.device<double>();
  double* coefsNormalizados = (double*) calloc(largo, sizeof(double));
  cudaMemcpy(coefsNormalizados, auxiliar_MC_modulo_Orde_GPU, cantFilas*cantColumnas*sizeof(double), cudaMemcpyDeviceToHost);
  MC_modulo_Orde_GPU.unlock();
  long cantCoefsParaCota = 0;
  double sumador = 0.0;
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
  double* auxiliar_MC_imag_GPU = MC_imag_GPU.device<double>();
  double* auxiliar_MC_real_GPU = MC_real_GPU.device<double>();
  cudaMemcpy(MC_imag_comp, auxiliar_MC_imag_GPU, cantFilas*cantColumnas*sizeof(double), cudaMemcpyDeviceToHost);
  MC_imag_GPU.unlock();
  cudaMemcpy(MC_real_comp, auxiliar_MC_real_GPU, cantFilas*cantColumnas*sizeof(double), cudaMemcpyDeviceToHost);
  MC_real_GPU.unlock();
  escribirCoefs(MC_imag_comp, nombreArCoef_comp_imag, cantFilas, cantColumnas);
  escribirCoefs(MC_real_comp, nombreArCoef_comp_real, cantFilas, cantColumnas);
  return cantCoefsParaCota;
}

double* minGradConjugado_MinCuadra_escritura(char* nombreArchivoMin, char* nombreArchivoCoefs, double* MV, double* MU, double* visibilidades, double* w, long cantVisi, long N, double* matrizDeUnosTamN, int maxIter, double tol, int tamBloque, int numGPU)
{
  int flag_NOESPOSIBLEMINIMIZAR = 0;
  double* MC;
  cudaMallocManaged(&MC, N*N*sizeof(double));
  cudaMemset(MC, 0, N*N*sizeof(double));
  double* residualInit = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN, tamBloque, numGPU);
  double* gradienteActual;
  cudaMallocManaged(&gradienteActual,N*N*sizeof(double));
  cudaMemset(gradienteActual, 0, N*N*sizeof(double));
  double* gradienteAnterior;
  cudaMallocManaged(&gradienteAnterior,N*N*sizeof(double));
  cudaMemset(gradienteAnterior, 0, N*N*sizeof(double));
  double* pActual;
  cudaMallocManaged(&pActual,N*N*sizeof(double));
  cudaMemset(pActual, 0, N*N*sizeof(double));
  double costoInicial = calCosto(residualInit, cantVisi, w, tamBloque, numGPU);
  double costoAnterior = costoInicial;
  double costoActual = costoInicial;
  calGradiente(residualInit, MV, cantVisi, N, MU, N, w, gradienteAnterior, tamBloque, numGPU);
  cudaFree(residualInit);
  // for(int i=0; i<N*N; i++)
  // {
  //   if(gradienteAnterior[i] != 0.0)
  //   {
  //     printf("En la linea %d es %f\n", i, gradienteAnterior[i]);
  //   }
  // }
  // exit(-1);
  combinacionLinealMatrices(-1.0, gradienteAnterior, N, N, 0.0, pActual, tamBloque, numGPU);
  double diferenciaDeCosto = 1.0;
  int i = 0;
  double alpha = 0.0;
  double epsilon = 1e-10;
  double normalizacion = costoAnterior + costoActual + epsilon;
  FILE* archivoMin = fopen(nombreArchivoMin, "w");
  if(archivoMin == NULL)
  {
       printf("Error al crear o abrir el archivo para almacenar la minimizacion.\n");
       exit(0);
  }
  while(maxIter > i && 2.0 * diferenciaDeCosto > tol * normalizacion)
  {
    alpha = calAlpha(gradienteAnterior, N, N, pActual, MV, cantVisi, N, MU, N, w, matrizDeUnosTamN, &flag_NOESPOSIBLEMINIMIZAR, tamBloque, numGPU);
    if(flag_NOESPOSIBLEMINIMIZAR == 1)
    {
      break;
    }
    combinacionLinealMatrices(alpha, pActual, N, N, 1.0, MC, tamBloque, numGPU);
    double* residual = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN, tamBloque, numGPU);
    costoActual = calCosto(residual, cantVisi, w, tamBloque, numGPU);
    cudaMallocManaged(&gradienteActual,N*N*sizeof(double));
    cudaMemset(gradienteActual, 0, N*N*sizeof(double));
    calGradiente(residual, MV, cantVisi, N, MU, N, w, gradienteActual, tamBloque, numGPU);
    cudaFree(residual);
    double beta = calBeta_Fletcher_Reeves(gradienteActual, N*N, gradienteAnterior, numGPU);
    combinacionLinealMatrices(-1.0, gradienteActual, N, N, beta, pActual, tamBloque, numGPU);
    diferenciaDeCosto = abs(costoAnterior - costoActual);
    normalizacion = costoAnterior + costoActual + epsilon;
    double otro = costoActual - costoAnterior;
    costoAnterior = costoActual;
    double* auxiliar = gradienteAnterior;
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

double* minGradConjugado_MinCuadra(double* MV, double* MU, double* visibilidades, double* w, long cantVisi, long N, double* matrizDeUnosTamN, int maxIter, double tol, int tamBloque, int numGPU)
{
  int flag_NOESPOSIBLEMINIMIZAR = 0;
  double* MC;
  cudaMallocManaged(&MC, N*N*sizeof(double));
  cudaMemset(MC, 0, N*N*sizeof(double));
  double* residualInit = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN, tamBloque, numGPU);
  double* gradienteActual;
  cudaMallocManaged(&gradienteActual,N*N*sizeof(double));
  cudaMemset(gradienteActual, 0, N*N*sizeof(double));
  double* gradienteAnterior;
  cudaMallocManaged(&gradienteAnterior,N*N*sizeof(double));
  cudaMemset(gradienteAnterior, 0, N*N*sizeof(double));
  double* pActual;
  cudaMallocManaged(&pActual,N*N*sizeof(double));
  cudaMemset(pActual, 0, N*N*sizeof(double));
  double costoInicial = calCosto(residualInit, cantVisi, w, tamBloque, numGPU);
  double costoAnterior = costoInicial;
  double costoActual = costoInicial;
  calGradiente(residualInit, MV, cantVisi, N, MU, N, w, gradienteAnterior, tamBloque, numGPU);
  cudaFree(residualInit);
  combinacionLinealMatrices(-1.0, gradienteAnterior, N, N, 0.0, pActual, tamBloque, numGPU);
  double diferenciaDeCosto = 1.0;
  int i = 0;
  double alpha = 0.0;
  double epsilon = 1e-10;
  double normalizacion = costoAnterior + costoActual + epsilon;
  while(maxIter > i && 2.0 * diferenciaDeCosto > tol * normalizacion)
  {
    alpha = calAlpha(gradienteAnterior, N, N, pActual, MV, cantVisi, N, MU, N, w, matrizDeUnosTamN, &flag_NOESPOSIBLEMINIMIZAR, tamBloque, numGPU);
    if(flag_NOESPOSIBLEMINIMIZAR == 1)
    {
      break;
    }
    combinacionLinealMatrices(alpha, pActual, N, N, 1.0, MC, tamBloque, numGPU);
    double* residual = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN, tamBloque, numGPU);
    costoActual = calCosto(residual, cantVisi, w, tamBloque, numGPU);
    cudaMallocManaged(&gradienteActual,N*N*sizeof(double));
    cudaMemset(gradienteActual, 0, N*N*sizeof(double));
    calGradiente(residual, MV, cantVisi, N, MU, N, w, gradienteActual, tamBloque, numGPU);
    cudaFree(residual);
    double beta = calBeta_Fletcher_Reeves(gradienteActual, N*N, gradienteAnterior, numGPU);
    combinacionLinealMatrices(-1.0, gradienteActual, N, N, beta, pActual, tamBloque, numGPU);
    diferenciaDeCosto = abs(costoAnterior - costoActual);
    normalizacion = costoAnterior + costoActual + epsilon;
    double otro = costoActual - costoAnterior;
    costoAnterior = costoActual;
    double* auxiliar = gradienteAnterior;
    gradienteAnterior = gradienteActual;
    cudaFree(auxiliar);
    i++;
    printf( "En la iteracion %d el valor de la funcion de costos es %f con un z de %.12e la diferencia con respecto al anterior costo es %.12e.\n", i, costoActual, alpha, otro);
  }
  cudaFree(gradienteAnterior);
  cudaFree(pActual);
  return MC;
}

double calculateSD(double* data, double mean, long cantElementos)
{
    double SD = 0.0;
    for (long i = 0; i < cantElementos; i++)
        SD += pow(data[i] - mean, 2);
    return sqrt(SD / 10);
}

double calculoDePSNRDeRecorte(double* estimacionFourier_ParteImag, double* estimacionFourier_ParteReal, long N, char* nombreArchivo, clock_t* tiempoTransInver_MejorCompresion)
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
  double* auxiliar_mapaFourierRecons = mapaFourierRecons.device<double>();
  double* inver_visi = (double*) calloc(N*N, sizeof(double));
  cudaMemcpy(inver_visi, auxiliar_mapaFourierRecons, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  mapaFourierRecons.unlock();

  int cantFilasARecorrer = columnaDeTermino - columnaDeInicio + 1;
  int cantColumnasARecorrer = filaDeTermino - filaDeInicio + 1;
  int contador = 0;
  int contadorEleExternos = 0;
  double sumaDeValoresExternos = 0.0;
  double maximoValorInterno = 0;
  double promedioValorInterno = 0;
  double* nuevaImagen = (double*) calloc(cantFilasARecorrer*cantColumnasARecorrer, sizeof(double));
  double* elementosExternos = (double*) calloc(N*N, sizeof(double));
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
          promedioValorInterno += inver_visi[i+j*N];
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
  double mediaExterna = sumaDeValoresExternos/contadorEleExternos;
  double desvEstandar = calculateSD(elementosExternos, mediaExterna, contadorEleExternos);
  free(elementosExternos);
  promedioValorInterno = promedioValorInterno/contador;
  double PSNR = maximoValorInterno/desvEstandar;
  // double PSNR = promedioValorInterno/desvEstandar;

  fitsfile *fptr;
  int status;
  long fpixel, nelements;
  int bitpix = DOUBLE_IMG;
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
  // if (fits_write_img(fptr, Tdouble, fpixel, nelements, nuevaImagen, &status))
  if (fits_write_img(fptr, TDOUBLE, fpixel, nelements, inver_visi, &status))
    printerror_cfitsio(status);
  if (fits_close_file(fptr, &status))
    printerror_cfitsio(status);
  free(inver_visi);
  free(nuevaImagen);
  return PSNR;
}

double* calPSNRDeDistintasCompresiones(double inicioIntervalo, int cantParamEvaInfo, char rutaADirecSec[], char rutaADirecTer[], char nombreArReconsCompreImg[], double* MC_imag, double* MC_real, double* MV_AF, double* MU_AF, long N, int tamBloque, int numGPU)
{
  double cotaMinPSNR = 0.75;
  double cotaMinCompresion = 0.2;
  double* datosDelMin = (double*) malloc(sizeof(double)*8);
  char nombreArchivoDatosMinPSNR[] = "mejorTradeOffPSNRCompre.txt";
  char nombreArchivoCompreImg[] = "compreImg";
  char nombreDatosDeIte[] = "datosDeIte.txt";
  char nombreDatosDeIteLegible[] = "datosDeIteLegible.txt";
  char nombreCurvaPSNRSuavizada[] = "curvaPSNRSuavizada.txt";
  char nombreRelativoCoefsCeroAporte[] = "idsCoefsCeroAporte.txt";
  double* MC_comp_imag;
  cudaMallocManaged(&MC_comp_imag,N*N*sizeof(double));
  cudaMemset(MC_comp_imag, 0, N*N*sizeof(double));
  double* MC_comp_real;
  cudaMallocManaged(&MC_comp_real,N*N*sizeof(double));
  cudaMemset(MC_comp_real, 0, N*N*sizeof(double));
  long largo = N * N;
  double* MC_img_cuadrado;
  cudaMallocManaged(&MC_img_cuadrado, N*N*sizeof(double));
  double* MC_modulo;
  cudaMallocManaged(&MC_modulo, N*N*sizeof(double));
  hadamardProduct(MC_imag, N, N, MC_imag, MC_img_cuadrado, tamBloque, numGPU);
  hadamardProduct(MC_real, N, N, MC_real, MC_modulo, tamBloque, numGPU);
  combinacionLinealMatrices(1.0, MC_img_cuadrado, N, N, 1.0, MC_modulo, tamBloque, numGPU);
  cudaFree(MC_img_cuadrado);
  af::array MC_modulo_GPU(N*N, MC_modulo);
  cudaFree(MC_modulo);
  af::array MC_modulo_indicesOrde_GPU(N*N);
  af::array MC_modulo_Orde_GPU(N*N);
  af::sort(MC_modulo_Orde_GPU, MC_modulo_indicesOrde_GPU, MC_modulo_GPU, 0, false);
  double total = af::sum<double>(MC_modulo_GPU);
  MC_modulo_Orde_GPU = MC_modulo_Orde_GPU/total;
  double porcenTotal = af::sum<double>(MC_modulo_Orde_GPU);
  af::eval(MC_modulo_Orde_GPU);
  af::eval(MC_modulo_indicesOrde_GPU);
  af::sync();
  double* auxiliar_MC_modulo_Orde_GPU = MC_modulo_Orde_GPU.device<double>();
  double* auxiliar_MC_modulo_indicesOrde_GPU = MC_modulo_indicesOrde_GPU.device<double>();
  double* coefsNormalizados = (double*) malloc(largo*sizeof(double));
  cudaMemcpy(coefsNormalizados, auxiliar_MC_modulo_Orde_GPU, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  int* MC_modulo_indicesOrde_CPU = (int*) malloc(largo*sizeof(int));
  cudaMemcpy(MC_modulo_indicesOrde_CPU, auxiliar_MC_modulo_indicesOrde_GPU, N*N*sizeof(int), cudaMemcpyDeviceToHost);
  MC_modulo_Orde_GPU.unlock();
  MC_modulo_GPU.unlock();
  MC_modulo_indicesOrde_GPU.unlock();
  char* nombreAbsolutoCoefsCeroAporte = (char*) malloc(sizeof(char)*(strlen(rutaADirecSec)+strlen(nombreRelativoCoefsCeroAporte))+sizeof(char)*4);
  strcpy(nombreAbsolutoCoefsCeroAporte, rutaADirecSec);
  strcat(nombreAbsolutoCoefsCeroAporte, "/");
  strcat(nombreAbsolutoCoefsCeroAporte, nombreRelativoCoefsCeroAporte);
  // printf("El porcenTotal es %f\n", porcenTotal);
  int contadorCoefsCeroAporte = 0;
  double valorActual = porcenTotal;
  FILE* archivoIdsCoefsCeroAporte = fopen(nombreAbsolutoCoefsCeroAporte, "a");
  for(long i=0; i<largo; i++)
  {
    valorActual = valorActual - coefsNormalizados[largo-1-i];
    if(porcenTotal > valorActual)
    {
      break;
    }
    else
    {
      fprintf(archivoIdsCoefsCeroAporte, "%ld\n", largo-1-i);
      contadorCoefsCeroAporte++;
    }
  }
  fclose(archivoIdsCoefsCeroAporte);
  free(nombreAbsolutoCoefsCeroAporte);
  // printf("La cantidad de coefs cero aporte es %d\n", contadorCoefsCeroAporte);
  double finIntervalo = ((double) (largo-contadorCoefsCeroAporte))/largo;
  // printf("La porcentaje de coefs utiles es %f\n", finIntervalo);
  double* paramEvaInfo = linspace(inicioIntervalo/100.0, finIntervalo, cantParamEvaInfo);
  long cantCoefsParaCota = 0;
  double sumador = 0.0;
  long iExterno = 0;
  double* cantidadPorcentualDeCoefs = linspace(0.0, largo, largo+1);
  combinacionLinealMatrices(0.0, cantidadPorcentualDeCoefs, largo+1, 1, 1.0/largo, cantidadPorcentualDeCoefs, tamBloque, numGPU);
  double* vectorDePSNR = (double*) calloc(cantParamEvaInfo, sizeof(double));
  double* porcenReal = (double*) calloc(cantParamEvaInfo, sizeof(double));
  double* porcenIdeal = (double*) calloc(cantParamEvaInfo, sizeof(double));
  long* cantCoefsUsadas = (long*) calloc(cantParamEvaInfo, sizeof(long));
  double* vectorDePorcenEnergia = (double*) calloc(cantParamEvaInfo, sizeof(double));
  double* vectorDeDifePSNREntrePtosAdya = (double*) calloc(cantParamEvaInfo, sizeof(double));
  double* porcenPSNRConRespectoTotal = (double*) calloc(cantParamEvaInfo, sizeof(double));
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
        break;
      }
    }
    if(cantCoefsParaCota > 0)
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
      af::array MC_imag_GPU(largo, MC_imag);
      af::array MC_real_GPU(largo, MC_real);
      MC_imag_GPU = MC_imag_GPU * indRepComp;
      MC_real_GPU = MC_real_GPU * indRepComp;
      af::eval(MC_imag_GPU);
      af::eval(MC_real_GPU);
      af::sync();
      indRepComp.unlock();
      double* auxiliar_MC_imag_GPU = MC_imag_GPU.device<double>();
      double* auxiliar_MC_real_GPU = MC_real_GPU.device<double>();
      cudaMemcpy(MC_comp_imag, auxiliar_MC_imag_GPU, largo*sizeof(double), cudaMemcpyDeviceToHost);
      MC_imag_GPU.unlock();
      cudaMemcpy(MC_comp_real, auxiliar_MC_real_GPU, largo*sizeof(double), cudaMemcpyDeviceToHost);
      MC_real_GPU.unlock();
      double* estimacionFourier_compre_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_imag, N, N, MU_AF, numGPU);
      double* estimacionFourier_compre_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_real, N, N, MU_AF, numGPU);
      int numero = j+1;
      char* numComoString = numAString(&numero);
      sprintf(numComoString, "%d", numero);
      char* nombreArchivoReconsImgComp = (char*) malloc(sizeof(char)*(strlen(rutaADirecTer)+strlen(numComoString)+strlen(nombreArchivoCompreImg))+sizeof(char)*7);
      strcpy(nombreArchivoReconsImgComp, rutaADirecTer);
      strcat(nombreArchivoReconsImgComp, "/");
      strcat(nombreArchivoReconsImgComp, nombreArchivoCompreImg);
      strcat(nombreArchivoReconsImgComp, "_");
      strcat(nombreArchivoReconsImgComp, numComoString);
      strcat(nombreArchivoReconsImgComp, ".fit");
      double PSNRActual = calculoDePSNRDeRecorte(estimacionFourier_compre_ParteImag, estimacionFourier_compre_ParteReal, N, nombreArchivoReconsImgComp, &tiempoCualquiera);
      porcenIdeal[j] = 1-paramEvaInfo[cantParamEvaInfo-1-j];
      vectorDePSNR[j] = PSNRActual;
      porcenReal[j] = 1-cantidadPorcentualDeCoefs[iExterno];
      cantCoefsUsadas[j] = cantCoefsParaCota;
      vectorDePorcenEnergia[j] = sumador;
      cudaFree(estimacionFourier_compre_ParteImag);
      cudaFree(estimacionFourier_compre_ParteReal);
      free(numComoString);
      free(nombreArchivoReconsImgComp);
    }
  }
  double maximoPSNR = 0;
  for(long j=0; j<cantParamEvaInfo; j++)
  {
    if(maximoPSNR < vectorDePSNR[j])
      maximoPSNR = vectorDePSNR[j];
  }
  char* nombreArchivoDatosDeIte = (char*) malloc(sizeof(char)*(strlen(rutaADirecSec)+strlen(nombreDatosDeIte))+sizeof(char)*4);
  strcpy(nombreArchivoDatosDeIte, rutaADirecSec);
  strcat(nombreArchivoDatosDeIte, "/");
  strcat(nombreArchivoDatosDeIte, nombreDatosDeIte);
  char* nombreArchivoDatosDeIteLegible = (char*) malloc(sizeof(char)*(strlen(rutaADirecSec)+strlen(nombreDatosDeIteLegible))+sizeof(char)*4);
  strcpy(nombreArchivoDatosDeIteLegible, rutaADirecSec);
  strcat(nombreArchivoDatosDeIteLegible, "/");
  strcat(nombreArchivoDatosDeIteLegible, nombreDatosDeIteLegible);
  for(int j=0; j<cantParamEvaInfo; j++)
  {
    porcenPSNRConRespectoTotal[j] = vectorDePSNR[j]/maximoPSNR;
    FILE* archivoDatosDeIte = fopen(nombreArchivoDatosDeIte, "a");
    fprintf(archivoDatosDeIte, "%.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", abs(porcenIdeal[j]-1), porcenIdeal[j], abs(porcenReal[j]-1), porcenReal[j], cantCoefsUsadas[j], vectorDePorcenEnergia[j], vectorDePSNR[j], porcenPSNRConRespectoTotal[j]);
    fclose(archivoDatosDeIte);
    FILE* archivoDatosDeIteLegible = fopen(nombreArchivoDatosDeIteLegible, "a");
    fprintf(archivoDatosDeIteLegible, "Porcen ideal de coefs: %f, Porcen ideal de compresion: %f, Porcen real de coefs: %f, Porcen real de compresion: %f, Cant de coefs: %ld, Porcen de energia: %f, PSNR: %f, Porcen de PSNR con respecto al total: %f\n", abs(porcenIdeal[j]-1) * 100, porcenIdeal[j] * 100, abs(porcenReal[j]-1) * 100, porcenReal[j] * 100, cantCoefsUsadas[j], vectorDePorcenEnergia[j], vectorDePSNR[j], porcenPSNRConRespectoTotal[j] * 100);
    fclose(archivoDatosDeIteLegible);
  }
  free(nombreArchivoDatosDeIte);
  free(nombreArchivoDatosDeIteLegible);
  double* vectorDePSNRFiltrado = (double*) calloc(cantParamEvaInfo, sizeof(double));
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
  char* nombreArchivoCurvaPSNRSuavizada = (char*) malloc(sizeof(char)*(strlen(rutaADirecSec)+strlen(nombreCurvaPSNRSuavizada))+sizeof(char)*4);
  strcpy(nombreArchivoCurvaPSNRSuavizada, rutaADirecSec);
  strcat(nombreArchivoCurvaPSNRSuavizada, "/");
  strcat(nombreArchivoCurvaPSNRSuavizada, nombreCurvaPSNRSuavizada);
  FILE* archivoCurvaPSNRSuavizada = fopen(nombreArchivoCurvaPSNRSuavizada, "a");
  for(int i=0; i<cantParamEvaInfo; i++)
  {
      fprintf(archivoCurvaPSNRSuavizada, "%f\n", vectorDePSNRFiltrado[i]);
  }
  fclose(archivoCurvaPSNRSuavizada);
  free(nombreArchivoCurvaPSNRSuavizada);

  for(int j=0; j<cantParamEvaInfo; j++)
  {
    double porcenActual = porcenReal[j];
    double porcenDifActual = vectorDePSNRFiltrado[j]/vectorDePSNRFiltrado[0];
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
  free(vectorDePSNRFiltrado);
  if(cantPtsVentana > 0)
  {
    af::array vectorDeDifePSNREntrePtosAdya_GPU(cantPtsVentana, vectorDeDifePSNREntrePtosAdya);
    af::array vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU(cantPtsVentana);
    af::array vectorDeDifePSNREntrePtosAdya_Orde_GPU(cantPtsVentana);
    af::sort(vectorDeDifePSNREntrePtosAdya_Orde_GPU, vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU, vectorDeDifePSNREntrePtosAdya_GPU, 0, true);
    vectorDeDifePSNREntrePtosAdya_GPU.unlock();
    vectorDeDifePSNREntrePtosAdya_Orde_GPU.unlock();
    int* auxiliar_vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU = vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU.device<int>();
    int* vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU = (int*) malloc(sizeof(int)*cantPtsVentana);
    cudaMemcpy(vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU, auxiliar_vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU, cantPtsVentana*sizeof(int), cudaMemcpyDeviceToHost);
    vectorDeDifePSNREntrePtosAdya_indicesOrde_GPU.unlock();
    int indiceElegido = vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU[0] + inicioDeVentana - 1;
    free(vectorDeDifePSNREntrePtosAdya_indicesOrde_CPU);
    datosDelMin[0] = abs(porcenIdeal[indiceElegido]-1);
    datosDelMin[1] = porcenIdeal[indiceElegido];
    datosDelMin[2] = abs(porcenReal[indiceElegido]-1);
    datosDelMin[3] = porcenReal[indiceElegido];
    datosDelMin[4] = cantCoefsUsadas[indiceElegido];
    datosDelMin[5] = vectorDePorcenEnergia[indiceElegido];
    datosDelMin[6] = vectorDePSNR[indiceElegido];
    datosDelMin[7] = porcenPSNRConRespectoTotal[indiceElegido];
  }
  else
  {
    datosDelMin[0] = 0;
    datosDelMin[1] = 0;
    datosDelMin[2] = 0;
    datosDelMin[3] = 0;
    datosDelMin[4] = 0;
    datosDelMin[5] = 0;
    datosDelMin[6] = 0;
    datosDelMin[7] = 0;
  }
  free(vectorDeDifePSNREntrePtosAdya);
  free(porcenIdeal);
  free(porcenReal);
  free(cantCoefsUsadas);
  free(vectorDePorcenEnergia);
  free(vectorDePSNR);
  free(porcenPSNRConRespectoTotal);
  char* nombreArchivoMejorCompre = (char*) malloc(sizeof(char)*(strlen(rutaADirecSec)+strlen(nombreArchivoDatosMinPSNR))+sizeof(char)*4);
  strcpy(nombreArchivoMejorCompre, rutaADirecSec);
  strcat(nombreArchivoMejorCompre, "/");
  strcat(nombreArchivoMejorCompre, nombreArchivoDatosMinPSNR);
  FILE* archivoMejorCompre = fopen(nombreArchivoMejorCompre, "w");
  fprintf(archivoMejorCompre, "%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
  fclose(archivoMejorCompre);
  free(nombreArchivoMejorCompre);
  cudaFree(MC_comp_imag);
  cudaFree(MC_comp_real);
  cudaFree(cantidadPorcentualDeCoefs);
  cudaFree(paramEvaInfo);
  free(coefsNormalizados);
  free(MC_modulo_indicesOrde_CPU);
  return datosDelMin;
}

double maximoEntre2Numeros(double primerNumero, double segundoNumero)
{
  if(primerNumero > segundoNumero)
  {
    return primerNumero;
  }
  else
  {
    return segundoNumero;
  }
}

void calCompSegunAncho_Hermite_escritura(char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], double ancho, double cotaEnergia, int iterActual, int maxIter, double tol, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, long cantVisi, long N, double* matrizDeUnosTamN, double max_radius, int tamBloque, int numGPU)
{
  double inicioPorcenCompre = 0.0;
  double terminoPorcenCompre = 0.2;
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
  long n = N-1;
  double beta_factor = ancho;


  // ############### CALCULO DE MU Y MV - CREACION DE DIRECTORIO SEGUNDARIO  ##############
  double beta_u = beta_factor/max_radius;
  double K = beta_u * (sqrt(2*n+1)+1);
  double* x_samp = combinacionLinealMatrices_conretorno(K, u, cantVisi, 1, 0.0, u, tamBloque, numGPU);
  double* y_samp = combinacionLinealMatrices_conretorno(K, v, cantVisi, 1, 0.0, v, tamBloque, numGPU);
  printf("...Comenzando calculo de MV...\n");
  clock_t tiempoCalculoMV;
  tiempoCalculoMV = clock();
  double* MV = hermite(y_samp, cantVisi, n, tamBloque, numGPU);
  // int contadorNoCeros = 0;
  // for(long i=0; i<cantVisi*N; i++)
  // {
  //   // printf("%ld\n",i);
  //   if(MV[i] != 0.0)
  //   {
  //       contadorNoCeros++;
  //       // printf("En posi %ld es %f\n", i, MV[i]);
  //   }
  // }
  // printf("%d\n", contadorNoCeros);
  // exit(0);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  double tiempoTotalCalculoMV = ((double)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");
  printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  double* MU = hermite(x_samp, cantVisi, n, tamBloque, numGPU);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  double tiempoTotalCalculoMU = ((double)tiempoCalculoMU)/CLOCKS_PER_SEC;
  printf("Calculo de MU completado.\n");
  cudaFree(y_samp);
  cudaFree(x_samp);

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
  double* MC_imag = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_imag, nombreArchivoCoefs_imag, MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartImag = clock() - tiempoMinPartImag;
  double tiempoTotalMinPartImag = ((double)tiempoMinPartImag)/CLOCKS_PER_SEC;
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
  double* MC_real = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_real, nombreArchivoCoefs_real, MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartReal = clock() - tiempoMinPartReal;
  double tiempoTotalMinPartReal = ((double)tiempoMinPartReal)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");
  free(nombreArchivoMin_real);
  free(nombreArchivoCoefs_real);

   // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, numGPU);
  tiempoInfo = clock() - tiempoInfo;
  double tiempoTotalInfo = ((double)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);

   // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);

  double* coordenadasVCentrosCeldas = linspace((-N/2.0) * delta_v, ((N/2.0) - 1.0) * delta_v, N);
  combinacionLinealMatrices(0.5 * delta_v, matrizDeUnosTamN, N, 1, 1.0, coordenadasVCentrosCeldas, tamBloque, numGPU);
  double* coordenadasUCentrosCeldas = linspace((-N/2.0) * delta_u, ((N/2.0) - 1.0) * delta_u, N);
  combinacionLinealMatrices(0.5 * delta_u, matrizDeUnosTamN, N, 1, 1.0, coordenadasUCentrosCeldas, tamBloque, numGPU);
  combinacionLinealMatrices(0.0, coordenadasUCentrosCeldas, N, 1, K, coordenadasUCentrosCeldas, tamBloque, numGPU);
  combinacionLinealMatrices(0.0, coordenadasVCentrosCeldas, N, 1, K, coordenadasVCentrosCeldas, tamBloque, numGPU);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  double* MV_AF = hermite(coordenadasVCentrosCeldas, N, n, tamBloque, numGPU);
  for(long i=0; i<N*N; i++)
  {
    if(MV_AF[i] != 0.0)
    {
        printf("En posi %ld es %f\n", i, MV_AF[i]);
    }
  }
  // exit(0);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  double tiempoTotalCalculoMV_AF = ((double)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  double* MU_AF = hermite(coordenadasUCentrosCeldas, N, n, tamBloque, numGPU);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  double tiempoTotalCalculoMU_AF = ((double)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  cudaFree(coordenadasVCentrosCeldas);
  cudaFree(coordenadasUCentrosCeldas);
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  double* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  double tiempoTotalReconsFourierPartImag = ((double)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  double* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  double tiempoTotalReconsFourierPartReal = ((double)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  double tiempoTotalReconsTransInver = ((double)tiempoReconsTransInver)/CLOCKS_PER_SEC;
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
  printf("...Comenzando calculo de compresiones...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  double* datosDelMin = calPSNRDeDistintasCompresiones(inicioPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, tamBloque, numGPU);
  tiempoCompresion = clock() - tiempoCompresion;
  double tiempoTotalCompresion = ((double)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresiones terminado.\n");
  free(rutaADirecTer);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  double nivelDeCompresion = 1.0 - datosDelMin[4] * 1.0 / N*N;
  #pragma omp critical
  {
    FILE* archivo = fopen(nombreArchivoInfoComp, "a");
    fprintf(archivo, "%d %f %.12f %.12e %.12e %.12f %.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", iterActual, ancho/delta_u, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], (long) datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
    fclose(archivo);
  }
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);
  free(datosDelMin);

  cudaFree(MC_real);
  cudaFree(MC_imag);
  cudaFree(MU_AF);
  cudaFree(MV_AF);

   // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  #pragma omp critical
  {
    FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
    fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalMinPartImag, tiempoTotalMinPartReal, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver);
    fclose(archivoInfoTiemposEjecu);
  }
  free(nombreArchivoInfoTiemposEjecu);
  free(rutaADirecSec);
}

void calCompSegunAncho_InvCuadra_escritura(char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], double ancho, double cotaEnergia, int iterActual, int maxIter, double tol, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, long cantVisi, long N, double* matrizDeUnosTamN, int tamBloque, int numGPU)
{
  double inicioPorcenCompre = 0.0;
  // double terminoPorcenCompre = 0.2;
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
  double* MV = calcularMV_InvCuadra(v, delta_v, cantVisi, N, ancho, tamBloque, numGPU);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  double tiempoTotalCalculoMV = ((double)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

   printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  double* MU = calcularMV_InvCuadra(u, delta_u, cantVisi, N, ancho, tamBloque, numGPU);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  double tiempoTotalCalculoMU = ((double)tiempoCalculoMU)/CLOCKS_PER_SEC;
  printf("Calculo de MU completado.\n");

   char* rutaADirecSec = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirSec))*sizeof(char)+sizeof(char)*3);
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
  char* nombreArchivoMin_imag = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirSec)+strlen(nombreArMin_imag))*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_imag, rutaADirecSec);
  strcat(nombreArchivoMin_imag, nombreArMin_imag);
  char* nombreArchivoCoefs_imag = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirSec)+strlen(nombreArCoef_imag))*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_imag, rutaADirecSec);
  strcat(nombreArchivoCoefs_imag, nombreArCoef_imag);
  printf("...Comenzando minimizacion de coeficientes parte imaginaria...\n");
  clock_t tiempoMinPartImag;
  tiempoMinPartImag = clock();
  double* MC_imag = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_imag, nombreArchivoCoefs_imag, MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartImag = clock() - tiempoMinPartImag;
  double tiempoTotalMinPartImag = ((double)tiempoMinPartImag)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte imaginaria terminado.\n");
  free(nombreArchivoMin_imag);
  free(nombreArchivoCoefs_imag);


   // ############### MINIMIZACION DE COEFS, PARTE REAL  ##############
  char* nombreArchivoMin_real = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirSec)+strlen(nombreArMin_real))*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_real, rutaADirecSec);
  strcat(nombreArchivoMin_real, nombreArMin_real);
  char* nombreArchivoCoefs_real = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirSec)+strlen(nombreArCoef_real))*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_real, rutaADirecSec);
  strcat(nombreArchivoCoefs_real, nombreArCoef_real);
  printf("...Comenzando minimizacion de coeficientes parte real...\n");
  clock_t tiempoMinPartReal;
  tiempoMinPartReal = clock();
  double* MC_real = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_real, nombreArchivoCoefs_real, MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartReal = clock() - tiempoMinPartReal;
  double tiempoTotalMinPartReal = ((double)tiempoMinPartReal)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");
  free(nombreArchivoMin_real);
  free(nombreArchivoCoefs_real);

   // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, numGPU);
  tiempoInfo = clock() - tiempoInfo;
  double tiempoTotalInfo = ((double)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);

   // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirSec)+strlen(nombreArReconsImg))*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  double* MV_AF = calcularMV_InvCuadra_estFourier(ancho, N, delta_v, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  double tiempoTotalCalculoMV_AF = ((double)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  double* MU_AF = calcularMV_InvCuadra_estFourier(ancho, N, delta_u, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  double tiempoTotalCalculoMU_AF = ((double)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  double* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  double tiempoTotalReconsFourierPartImag = ((double)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  double* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  double tiempoTotalReconsFourierPartReal = ((double)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  double tiempoTotalReconsTransInver = ((double)tiempoReconsTransInver)/CLOCKS_PER_SEC;
  cudaFree(estimacionFourier_ParteImag);
  cudaFree(estimacionFourier_ParteReal);
  free(nombreArchivoReconsImg);


  // ############### CALCULO DE GRADO DE COMPRESION ##############
 char* rutaADirecTer = (char*) malloc((strlen(rutaADirecSec)+strlen(nombreDirTer))*sizeof(char)+sizeof(char)*3);
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
 printf("...Comenzando calculo de compresiones...\n");
 clock_t tiempoCompresion;
 tiempoCompresion = clock();
 double* datosDelMin = calPSNRDeDistintasCompresiones(inicioPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, tamBloque, numGPU);
 tiempoCompresion = clock() - tiempoCompresion;
 double tiempoTotalCompresion = ((double)tiempoCompresion)/CLOCKS_PER_SEC;
 printf("Proceso de calculo de compresiones terminado.\n");
 free(rutaADirecTer);
 char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
 strcpy(nombreArchivoInfoComp, nombreDirPrin);
 strcat(nombreArchivoInfoComp, "/");
 strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
 // double* datosDelMin = (double*) malloc(sizeof(double)*8);
 double nivelDeCompresion = 1.0 - datosDelMin[4] * 1.0 / N*N;
 // datosDelMin[0] = 0.0;
 // datosDelMin[1] = 0.0;
 // datosDelMin[2] = 0.0;
 // datosDelMin[3] = 0.0;
 // datosDelMin[4] = 0.0;
 // datosDelMin[5] = 0.0;
 // datosDelMin[6] = 0.0;
 // datosDelMin[7] = 0.0;
 #pragma omp critical
 {
   FILE* archivo = fopen(nombreArchivoInfoComp, "a");
   fprintf(archivo, "%d %f %.12f %.12e %.12e %.12f %.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", iterActual, ancho/delta_u, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], (long) datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
   fclose(archivo);
 }
 free(nombreArchivoInfoComp);
 free(medidasDeInfo);
 free(datosDelMin);

 cudaFree(MC_real);
 cudaFree(MC_imag);
 cudaFree(MU_AF);
 cudaFree(MV_AF);

  // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
 char* nombreArchivoInfoTiemposEjecu = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreArInfoTiemposEjecu))*sizeof(char)+sizeof(char)*2);
 strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
 strcat(nombreArchivoInfoTiemposEjecu, "/");
 strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
 #pragma omp critical
 {
   FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
   fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalMinPartImag, tiempoTotalMinPartReal, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver);
   fclose(archivoInfoTiemposEjecu);
 }
 free(rutaADirecSec);
 free(nombreArchivoInfoTiemposEjecu);
}

void calCompSegunAncho_Normal_escritura(char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], double ancho, double cotaEnergia, int iterActual, int maxIter, double tol, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, long cantVisi, long N, double* matrizDeUnosTamN, int tamBloque, int numGPU)
{
  double inicioPorcenCompre = 0.0;
  double terminoPorcenCompre = 0.2;
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
  double* MV = calcularMV_Normal(v, delta_v, cantVisi, N, ancho, tamBloque, numGPU);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  double tiempoTotalCalculoMV = ((double)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

   printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  double* MU = calcularMV_Normal(u, delta_u, cantVisi, N, ancho, tamBloque, numGPU);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  double tiempoTotalCalculoMU = ((double)tiempoCalculoMU)/CLOCKS_PER_SEC;
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
  double* MC_imag = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_imag, nombreArchivoCoefs_imag, MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartImag = clock() - tiempoMinPartImag;
  double tiempoTotalMinPartImag = ((double)tiempoMinPartImag)/CLOCKS_PER_SEC;
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
  double* MC_real = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_real, nombreArchivoCoefs_real, MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartReal = clock() - tiempoMinPartReal;
  double tiempoTotalMinPartReal = ((double)tiempoMinPartReal)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");
  free(nombreArchivoMin_real);
  free(nombreArchivoCoefs_real);


   // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, numGPU);
  tiempoInfo = clock() - tiempoInfo;
  double tiempoTotalInfo = ((double)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);


   // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  double* MV_AF = calcularMV_Normal_estFourier(ancho, N, delta_v, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  double tiempoTotalCalculoMV_AF = ((double)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  double* MU_AF = calcularMV_Normal_estFourier(ancho, N, delta_u, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  double tiempoTotalCalculoMU_AF = ((double)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  double* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  double tiempoTotalReconsFourierPartImag = ((double)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  double* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  double tiempoTotalReconsFourierPartReal = ((double)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  double tiempoTotalReconsTransInver = ((double)tiempoReconsTransInver)/CLOCKS_PER_SEC;
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
  printf("...Comenzando calculo de compresiones...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  double* datosDelMin = calPSNRDeDistintasCompresiones(inicioPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, tamBloque, numGPU);
  tiempoCompresion = clock() - tiempoCompresion;
  double tiempoTotalCompresion = ((double)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresiones terminado.\n");
  free(rutaADirecTer);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  double nivelDeCompresion = 1.0 - datosDelMin[4] * 1.0 / N*N;
  #pragma omp critical
  {
    FILE* archivo = fopen(nombreArchivoInfoComp, "a");
    fprintf(archivo, "%d %f %.12f %.12e %.12e %.12f %.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", iterActual, ancho/delta_u, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], (long) datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
    fclose(archivo);
  }
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);
  free(datosDelMin);

  cudaFree(MC_real);
  cudaFree(MC_imag);
  cudaFree(MU_AF);
  cudaFree(MV_AF);

   // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  #pragma omp critical
  {
    FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
    fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalMinPartImag, tiempoTotalMinPartReal, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver);
    fclose(archivoInfoTiemposEjecu);
  }
  free(rutaADirecSec);
  free(nombreArchivoInfoTiemposEjecu);
}

void calCompSegunAncho_Rect_escritura(char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], double ancho, double cotaEnergia, int iterActual, int maxIter, double tol, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque, int numGPU, double* matrizDeUnosNxN)
{
  // hd_142
  double inicioPorcenCompre = 0.0;
  // double terminoPorcenCompre = 0.2;
  int cantPorcen = 101;


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
  double* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  double tiempoTotalCalculoMV = ((double)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

  printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  double* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  double tiempoTotalCalculoMU = ((double)tiempoCalculoMU)/CLOCKS_PER_SEC;
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
  double* MC_imag = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_imag, nombreArchivoCoefs_imag, MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartImag = clock() - tiempoMinPartImag;
  double tiempoTotalMinPartImag = ((double)tiempoMinPartImag)/CLOCKS_PER_SEC;
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
  double* MC_real = minGradConjugado_MinCuadra_escritura(nombreArchivoMin_real, nombreArchivoCoefs_real, MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartReal = clock() - tiempoMinPartReal;
  double tiempoTotalMinPartReal = ((double)tiempoMinPartReal)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");
  free(nombreArchivoMin_real);
  free(nombreArchivoCoefs_real);


  // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, numGPU);
  tiempoInfo = clock() - tiempoInfo;
  double tiempoTotalInfo = ((double)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);


  // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  double* MV_AF = calcularMV_Rect_estFourier(ancho, N, delta_v, matrizDeUnosNxN, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  double tiempoTotalCalculoMV_AF = ((double)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  double* MU_AF = calcularMV_Rect_estFourier(ancho, N, delta_u, matrizDeUnosNxN, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  double tiempoTotalCalculoMU_AF = ((double)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  double* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  double tiempoTotalReconsFourierPartImag = ((double)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  double* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  double tiempoTotalReconsFourierPartReal = ((double)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  double tiempoTotalReconsTransInver = ((double)tiempoReconsTransInver)/CLOCKS_PER_SEC;
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
  printf("...Comenzando calculo de compresiones...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  double* datosDelMin = calPSNRDeDistintasCompresiones(inicioPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, tamBloque, numGPU);
  tiempoCompresion = clock() - tiempoCompresion;
  double tiempoTotalCompresion = ((double)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresiones terminado.\n");
  free(rutaADirecTer);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  double nivelDeCompresion = 1.0 - datosDelMin[4] * 1.0 / N*N;
  #pragma omp critical
  {
    FILE* archivo = fopen(nombreArchivoInfoComp, "a");
    fprintf(archivo, "%d %f %.12f %.12e %.12e %.12f %.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", iterActual, ancho/delta_u, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], (long) datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
    fclose(archivo);
  }
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);
  free(datosDelMin);

  cudaFree(MC_real);
  cudaFree(MC_imag);
  cudaFree(MU_AF);
  cudaFree(MV_AF);

  // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  #pragma omp critical
  {
    FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
    fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalMinPartImag, tiempoTotalMinPartReal, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver);
    fclose(archivoInfoTiemposEjecu);
  }
  free(nombreArchivoInfoTiemposEjecu);
  free(rutaADirecSec);
}

double funcOptiInfo_Traza_Rect(double ancho, void* params)
{
  struct parametros_BaseRect* ps = (struct parametros_BaseRect*) params;
  double* MV = calcularMV_Rect(ps->v, ps->delta_v, ps->cantVisi, ps->N, ps->estrechezDeBorde, ancho, ps->matrizDeUnos, 1024, 1);
  double* MU = calcularMV_Rect(ps->u, ps->delta_u, ps->cantVisi, ps->N, ps->estrechezDeBorde, ancho, ps->matrizDeUnos, 1024, 1);
  double* medidasDeInfo = calInfoFisherDiag(MV, ps->cantVisi, ps->N, MU, ps->w, 1024, 1);
  double medidaSumaDeLaDiagonal = medidasDeInfo[0];
  free(medidasDeInfo);
  cudaFree(MV);
  cudaFree(MU);
  return -1 * medidaSumaDeLaDiagonal;
}

double funcOptiInfo_Traza_Normal(double ancho, void* params)
{
  struct parametros_BaseNormal* ps = (struct parametros_BaseNormal*) params;
  double* MV = calcularMV_Normal(ps->v, ps->delta_v, ps->cantVisi, ps->N, ancho, 1024, 1);
  double* MU = calcularMV_Normal(ps->u, ps->delta_u, ps->cantVisi, ps->N, ancho, 1024, 1);
  double* medidasDeInfo = calInfoFisherDiag(MV, ps->cantVisi, ps->N, MU, ps->w, 1024, 1);
  double medidaSumaDeLaDiagonal = medidasDeInfo[0];
  free(medidasDeInfo);
  cudaFree(MV);
  cudaFree(MU);
  return -1 * medidaSumaDeLaDiagonal;
}

double funcOptiValorDeZ(double z, void* params)
{
  struct parametros_Minl1* ps = (struct parametros_Minl1*) params;
  double* visModelo_pActual;
  cudaMallocManaged(&visModelo_pActual, ps->cantVisi*sizeof(double));
  cudaMemset(visModelo_pActual, 0, ps->cantVisi*sizeof(double));
  calVisModelo(ps->MV, ps->cantVisi, ps->N, ps->pActual, ps->N, ps->MU, ps->matrizDeUnosTamN, visModelo_pActual, ps->tamBloque, ps->numGPU);
  combinacionLinealMatrices(1.0, ps->residual, ps->cantVisi, 1, z, visModelo_pActual, ps->tamBloque, ps->numGPU);
  hadamardProduct(visModelo_pActual, ps->cantVisi, 1, visModelo_pActual, visModelo_pActual, ps->tamBloque, ps->numGPU);
  double total_minCuadra = dotProduct(visModelo_pActual, ps->cantVisi, ps->w, ps->numGPU);
  cudaFree(visModelo_pActual);
  af::array pActual_GPU(ps->N*ps->N, ps->pActual);
  af::array MC_GPU(ps->N*ps->N, ps->MC);
  af::array totalCoefs_GPU(ps->N*ps->N);
  totalCoefs_GPU = MC_GPU + z * pActual_GPU;
  double sumaTotal_Coefs_pActual = af::sum<double>(af::abs(totalCoefs_GPU)) * ps->param_lambda;
  af::eval(totalCoefs_GPU);
  af::sync();
  pActual_GPU.unlock();
  MC_GPU.unlock();
  totalCoefs_GPU.unlock();
  return total_minCuadra + sumaTotal_Coefs_pActual;
}

double funcValorZ(double z, double cantVisi, long N, double* MV, double* MC, double* MU, double* matrizDeUnosTamN, int tamBloque, int numGPU, double* w, double* pActual, double param_lambda, double* residual)
{
  double* visModelo_pActual;
  cudaMallocManaged(&visModelo_pActual, cantVisi*sizeof(double));
  cudaMemset(visModelo_pActual, 0, cantVisi*sizeof(double));
  // printf("%f\n", z);
  calVisModelo(MV, cantVisi, N, pActual, N, MU, matrizDeUnosTamN, visModelo_pActual, tamBloque, numGPU);
  combinacionLinealMatrices(1.0, residual, cantVisi, 1, z, visModelo_pActual, tamBloque, numGPU);
  hadamardProduct(visModelo_pActual, cantVisi, 1, visModelo_pActual, visModelo_pActual, tamBloque, numGPU);
  double total_minCuadra = dotProduct(visModelo_pActual, cantVisi, w, numGPU);
  // printf("La suma de total min es %f\n", total_minCuadra);
  cudaFree(visModelo_pActual);
  af::array pActual_GPU(N*N, pActual);
  af::array MC_GPU(N*N, MC);
  MC_GPU = MC_GPU + z * pActual_GPU;
  double sumaTotal_Coefs_pActual = af::sum<double>(af::abs(MC_GPU)) * param_lambda;
  af::eval(MC_GPU);
  af::sync();
  pActual_GPU.unlock();
  MC_GPU.unlock();
  // printf("La suma de coefs es %.12e\n", sumaTotal_Coefs_pActual);
  return  total_minCuadra + sumaTotal_Coefs_pActual;
}

double goldenMin_Minl1(int* flag_NOESPOSIBLEMINIMIZAR, double a, double b, long cantVisi, long N, double* MU, double* MC, double* MV, double* residual, double* w, double* pActual, double param_lambda, int tamBloque, int numGPU, double* matrizDeUnosTamN, double delta_u)
{
  int status;
  int iter = 0, max_iter = 100;
  const gsl_min_fminimizer_type *T;
  gsl_min_fminimizer *s;
  gsl_function F;
  parametros_Minl1 actual;
  actual.cantVisi = cantVisi;
  actual.N = N;
  actual.MU = MU;
  actual.MC = MC;
  actual.MV = MV;
  actual.residual = residual;
  actual.w = w;
  actual.pActual = pActual;
  actual.param_lambda = param_lambda;
  actual.tamBloque = tamBloque;
  actual.numGPU = numGPU;
  actual.matrizDeUnosTamN = matrizDeUnosTamN;
  double m = (b+a*0.4);
  F.function = &funcOptiValorDeZ;
  void* punteroVoidAActual = &actual;
  F.params = punteroVoidAActual;
  T = gsl_min_fminimizer_quad_golden;
  s = gsl_min_fminimizer_alloc(T);
  gsl_set_error_handler_off();
  int status_interval = gsl_min_fminimizer_set(s, &F, m, a, b);
  if(status_interval)
  {
    *flag_NOESPOSIBLEMINIMIZAR = 1;
    return -1;
  }

  // printf ("using %s method\n",
  //         gsl_min_fminimizer_name (s));
  //
  // printf ("%5s [%9s, %9s] %9s\n",
  //         "iter", "lower", "upper", "min");
  //
  // printf ("%5d [%.7f, %.7f] %.7f\n",
  //         iter, a, b, m);

  do
    {
      iter++;
      status = gsl_min_fminimizer_iterate(s);

      m = gsl_min_fminimizer_x_minimum(s);
      a = gsl_min_fminimizer_x_lower(s);
      b = gsl_min_fminimizer_x_upper(s);

      status
        = gsl_min_test_interval (a, b, 0.001, 0.0);
        // = gsl_min_test_interval(a, b, 1e-30, 1e-30);

      // if (status == GSL_SUCCESS)
        // printf ("Converged:\n");

      // printf ("%5d [%.7f, %.7f] "
      //         "%.7f\n",
      //         iter, a/delta_u, b/delta_u,m/delta_u);
    }
  while (status == GSL_CONTINUE && iter < max_iter);
  gsl_min_fminimizer_free(s);
  return m;
}

static int Stopping_Rule(double x0, double x1, double tolerance)
{
   double xm = 0.5 * fabs( x1 + x0 );

   if ( xm <= 1.0 ) return ( fabs( x1 - x0 ) < tolerance ) ? 1 : 0;
   return ( fabs( x1 - x0 ) < tolerance * xm ) ? 1 : 0;
}

void Max_Search_Golden_Section(double (*f)(double, double, long, double*, double*, double*, double*, int, int, double*, double*, double, double*), double* a, double *fa, double* b, double* fb, double tolerance, double cantVisi, long N, double* MV, double* MC, double* MU, double* matrizDeUnosTamN, int tamBloque, int numGPU, double* w, double* pActual, double param_lambda, double* residual)
{
   static const double lambda = 0.5 * (sqrt5 - 1.0);
   static const double mu = 0.5 * (3.0 - sqrt5);         // = 1 - lambda
   double x1;
   double x2;
   double fx1;
   double fx2;


                // Find first two internal points and evaluate
                // the function at the two internal points.

   x1 = *b - lambda * (*b - *a);
   x2 = *a + lambda * (*b - *a);
   fx1 = f(x1, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
   fx2 = f(x2, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);

             // Verify that the tolerance is an acceptable number

   if (tolerance <= 0.0) tolerance = sqrt(DBL_EPSILON) * (*b - *a);

           // Loop by exluding segments from current endpoints a, b
           // to current internal points x1, x2 and then calculating
           // a new internal point until the length of the interval
           // is less than or equal to the tolerance.

   while ( ! Stopping_Rule( *a, *b, tolerance) ) {
      if (fx1 < fx2) {
         *a = x1;
         *fa = fx1;
         if ( Stopping_Rule( *a, *b, tolerance) ) break;
         x1 = x2;
         fx1 = fx2;
         x2 = *b - mu * (*b - *a);
         fx2 = f(x2, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
      } else {
         *b = x2;
         *fb = fx2;
         if ( Stopping_Rule( *a, *b, tolerance) ) break;
         x2 = x1;
         fx2 = fx1;
         x1 = *a + mu * (*b - *a);
         fx1 = f(x1, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
      }
   }
   return;
}


void Min_Search_Golden_Section(double (*f)(double, double, long, double*, double*, double*, double*, int, int, double*, double*, double, double*), double* a, double *fa, double* b, double* fb, double tolerance, double cantVisi, long N, double* MV, double* MC, double* MU, double* matrizDeUnosTamN, int tamBloque, int numGPU, double* w, double* pActual, double param_lambda, double* residual)
{
   static const double lambda = 0.5 * (sqrt5 - 1.0);
   static const double mu = 0.5 * (3.0 - sqrt5);         // = 1 - lambda
   double x1;
   double x2;
   double fx1;
   double fx2;


                // Find first two internal points and evaluate
                // the function at the two internal points.

   x1 = *b - lambda * (*b - *a);
   x2 = *a + lambda * (*b - *a);
   fx1 = f(x1, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
   fx2 = f(x2, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);

             // Verify that the tolerance is an acceptable number

   if (tolerance <= 0.0) tolerance = sqrt(DBL_EPSILON) * (*b - *a);

           // Loop by exluding segments from current endpoints a, b
           // to current internal points x1, x2 and then calculating
           // a new internal point until the length of the interval
           // is less than or equal to the tolerance.

   while ( ! Stopping_Rule( *a, *b, tolerance) ) {
      if (fx1 > fx2) {
         *a = x1;
         *fa = fx1;
         if ( Stopping_Rule( *a, *b, tolerance) ) break;
         x1 = x2;
         fx1 = fx2;
         x2 = *b - mu * (*b - *a);
         fx2 = f(x2, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
      } else {
         *b = x2;
         *fb = fx2;
         if ( Stopping_Rule( *a, *b, tolerance) ) break;
         x2 = x1;
         fx2 = fx1;
         x1 = *a + mu * (*b - *a);
         fx1 = f(x1, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
      }
   }
   return;
}


double goldenMin_BaseRect(double* u, double* v, double* w, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double estrechezDeBorde)
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

double goldenMin_BaseNormal(double* u, double* v, double* w, double delta_u, double delta_v, long cantVisi, long N)
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

  T = gsl_min_fminimizer_brent;
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

void lecturaDeTXT(char nombreArchivo[], double* frecuencia, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, long cantVisi)
{
  long contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  double c_constant = 2.99792458E8;
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
  printf("%ld visibilidades leidas.\n", contador);
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

void lectDeTXTcreadoDesdeMS(char nombreArchivo[], double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal)
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

void lectDeTXTcreadoDesdeMSConLimite(char nombreArchivo[], double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, long inicio, long fin, long cantVisi)
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

void lecturaDeTXTDeCoefs(char nombreArchivo[], double* MC, long cantFilas, long cantColumnas)
{
  long contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  fp = fopen(nombreArchivo, "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  for(int i=0; i<cantFilas; i++)
  {
    getline(&line, &len, fp);
    MC[0*cantFilas+i] = atof(strtok(line, " "));
    for(int j=1; j<cantColumnas; j++)
    {
      MC[j*cantFilas+i] = atof(strtok(NULL, " "));
    }
  }
  free(line);
  fclose(fp);
}

void lectArchivoLambdaYCosto(char nombreArchivo[], int* listaDeNumIte, double* listaDeLambdas, double* listaDeCostos)
{
  long contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  fp = fopen(nombreArchivo, "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  while ((read = getline(&line, &len, fp)) != -1)
  {
    listaDeNumIte[contador] = atoi(strtok(line, " "));
    listaDeLambdas[contador] = atof(strtok(NULL, " "));
    listaDeCostos[contador] = atof(strtok(NULL, " "));
    contador++;
	}
  printf("Se han leido %ld lambdas.\n", contador);
  free(line);
  fclose(fp);
}

void seleccionarMejoresLambdas(char nombreDirSegundaEtapaDesdeRaiz[], char nombreArchivoCostoYLambda[], int cantidadDeLambdasTotales, int cantMejoresLambdasASeleccionar, int* listaMejores_NumIte, double* listaMejores_Lambda)
{
  char nombreArchivoMejoresLambdas[] = "lambdas_seleccionados.txt";
  int* listaDeNumIte;
  cudaMallocManaged(&listaDeNumIte, cantidadDeLambdasTotales*sizeof(int));
  double* listaDeLambdas;
  cudaMallocManaged(&listaDeLambdas, cantidadDeLambdasTotales*sizeof(double));
  double* listaDeCostos;
  cudaMallocManaged(&listaDeCostos, cantidadDeLambdasTotales*sizeof(double));
  lectArchivoLambdaYCosto(nombreArchivoCostoYLambda, listaDeNumIte, listaDeLambdas, listaDeCostos);
  af::array listaDeLambdas_GPU(cantidadDeLambdasTotales, listaDeCostos);
  af::array listaDeLambdas_indicesOrde_GPU(cantidadDeLambdasTotales);
  af::array listaDeLambdas_Orde_GPU(cantidadDeLambdasTotales);
  af::sort(listaDeLambdas_Orde_GPU, listaDeLambdas_indicesOrde_GPU, listaDeLambdas_GPU, 0, true);
  af::eval(listaDeLambdas_indicesOrde_GPU);
  af::sync();
  int* auxiliar_listaDeLambdas_indicesOrde_GPU = listaDeLambdas_indicesOrde_GPU.device<int>();
  listaDeLambdas_GPU.unlock();
  listaDeLambdas_indicesOrde_GPU.unlock();
  listaDeLambdas_Orde_GPU.unlock();
  int* listaDeLambdas_indicesOrde_CPU = (int*) malloc(cantidadDeLambdasTotales*sizeof(int));
  cudaMemcpy(listaDeLambdas_indicesOrde_CPU, auxiliar_listaDeLambdas_indicesOrde_GPU, cantidadDeLambdasTotales*sizeof(int), cudaMemcpyDeviceToHost);

  char* nombreArchivoMejoresCostoYLambda = (char*) malloc(sizeof(char)*(strlen(nombreDirSegundaEtapaDesdeRaiz)*strlen(nombreArchivoMejoresLambdas)+3));
  strcpy(nombreArchivoMejoresCostoYLambda, nombreDirSegundaEtapaDesdeRaiz);
  strcat(nombreArchivoMejoresCostoYLambda, "/");
  strcat(nombreArchivoMejoresCostoYLambda, nombreArchivoMejoresLambdas);
  FILE* archivoMejoresCostoYLambda = fopen(nombreArchivoMejoresCostoYLambda, "w");
  for(int i=0; i<cantMejoresLambdasASeleccionar; i++)
  {
    int idActual = listaDeLambdas_indicesOrde_CPU[i];
    listaMejores_NumIte[i] = listaDeNumIte[idActual];
    listaMejores_Lambda[i] = listaDeLambdas[idActual];
    fprintf(archivoMejoresCostoYLambda, "%d %.12e %.12e\n", listaDeNumIte[idActual], listaDeLambdas[idActual], listaDeCostos[idActual]);
  }
  fclose(archivoMejoresCostoYLambda);
  free(nombreArchivoMejoresCostoYLambda);
  free(listaDeLambdas_indicesOrde_CPU);
  cudaFree(listaDeNumIte);
  cudaFree(listaDeLambdas);
  cudaFree(listaDeCostos);
}

void escrituraDeArchivoConParametros_Hermite(char nombreArchivoPara[], char nombreArchivo[], char nombreDirPrin[], int cantVisi, int N, int maxIter, double tolGrad)
{
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  FILE* archivoDePara = fopen(nombreArchivoPara, "w");
  fprintf(archivoDePara, "Programa inicio su ejecucion con fecha: %d-%d-%d %d:%d:%d\n", tm.tm_year + 1900, tm.tm_mon + 1,tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  fprintf(archivoDePara, "Compresion con base hermite utilizando informacion del archivo %s cuyos parametros de ejecucion fueron:\n", nombreArchivo);
  fprintf(archivoDePara, "Cantidad de visibilidades(cantVisi): %d\n", cantVisi);
  fprintf(archivoDePara, "Cantidad de Coefs(N x N): %d x %d = %d\n", N, N, N*N);
  fprintf(archivoDePara, "Maximo de iteraciones impuesto para la minimizacion de coeficientes(maxIter): %d\n", maxIter);
  fprintf(archivoDePara, "Grado de tolerancia a la minimizacion de los coefs(tolGrad): %.12e\n", tolGrad);
  fclose(archivoDePara);
}

void escrituraDeArchivoConParametros_Normal(char nombreArchivoPara[], char nombreArchivo[], char nombreDirPrin[], int cantVisi, int N, int maxIter, double tolGrad)
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

void escrituraDeArchivoConParametros_Rect(char nombreArchivoPara[], char nombreArchivo[], char nombreDirPrin[], long cantVisi, long N, int maxIter, double tolGrad, double estrechezDeBorde)
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

void calculoDeInfoCompre_BaseHermite(char nombreArchivo[], int maxIter, double tolGrad, double tolGolden, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, long cantVisi, long N, double cotaEnergia, char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], int cantParamEvaInfo, double inicioIntervalo, double finIntervalo, double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque)
{
  char nombreArDetLinspace[] = "detalleslinspace.txt";
  double inicioIntervaloEscalado = inicioIntervalo * delta_u;
  double finIntervaloEscalado = finIntervalo * delta_u;
  char nombreArPara[] = "parametrosEjecucion.txt";
  if(cotaEnergia > 1.0)
  {
      printf("ERROR: La cota de energia debe estar expresado en decimales, no en porcentajes.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  // int cotaEnergiaInt = cotaEnergia * 100;
  // char* cotaEnergiaString = numAString(&cotaEnergiaInt);
  // sprintf(cotaEnergiaString, "%d", cotaEnergiaInt);
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
  escrituraDeArchivoConParametros_Hermite(nombreArchivoPara, nombreArchivo, nombreDirPrin, cantVisi, N, maxIter, tolGrad);
  free(nombreArchivoPara);

  // double optimo = goldenMin_BaseNormal(u, v, w, delta_u, delta_v, cantVisi, N);
  // printf("El optimo esta en %.12f\n", optimo);

  double limitesDeZonas[] = {0.205, 0.5, 1.0};
  double cantPuntosPorZona[] = {100, 50};
  int cantPtosLimites = 3;
  double* paramEvaInfo = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);

  char* nombreArchivoDetallesLinspace = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArDetLinspace)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoDetallesLinspace, nombreDirPrin);
  strcat(nombreArchivoDetallesLinspace, "/");
  strcat(nombreArchivoDetallesLinspace, nombreArDetLinspace);
  FILE* archivoDetLin = fopen(nombreArchivoDetallesLinspace, "a");
  for(int i=0; i<cantPtosLimites-1; i++)
  {
    fprintf(archivoDetLin, "Inicio: %f, Fin: %f, Cant. Ele: %f\n", limitesDeZonas[i], limitesDeZonas[i+1], cantPuntosPorZona[i]);
  }
  fclose(archivoDetLin);
  free(nombreArchivoDetallesLinspace);

  double maxu = buscarMaximo(u, cantVisi);
  double maxv = buscarMaximo(v, cantVisi);
  double max_radius = maximoEntre2Numeros(maxu,maxv);

  int i = 0;
  // #pragma omp parallel num_threads(4)
  // {
  //   #pragma omp for schedule(dynamic, 1)
  //   for(int i=0; i<cantParamEvaInfo; i++)
  //   {
      char* numComoString = numAString(&i);
      sprintf(numComoString, "%d", i);
      char* nombreDirSecCopia = (char*) malloc(sizeof(char)*strlen(nombreDirSec)*strlen(numComoString));
      strcpy(nombreDirSecCopia, nombreDirSec);
      strcat(nombreDirSecCopia, numComoString);
      // int thread_id = omp_get_thread_num();
      int thread_id = i;
      cudaSetDevice(thread_id);
      af::setDevice(thread_id);
      calCompSegunAncho_Hermite_escritura(nombreDirPrin, nombreDirSecCopia, nombreDirTer, paramEvaInfo[i], cotaEnergia, i, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, matrizDeUnosTamN, max_radius, tamBloque, thread_id);
      free(numComoString);
      free(nombreDirSecCopia);
  //   }
  // }
}

void calculoDeInfoCompre_BaseInvCuadra(char nombreArchivo[], int maxIter, double tolGrad, double tolGolden, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, long cantVisi, long N, double cotaEnergia, char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], int cantParamEvaInfo, double inicioIntervalo, double finIntervalo, double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque)
{
  char nombreArDetLinspace[] = "detalleslinspace.txt";
  double inicioIntervaloEscalado = inicioIntervalo * delta_u;
  double finIntervaloEscalado = finIntervalo * delta_u;
  char nombreArPara[] = "parametrosEjecucion.txt";
  if(cotaEnergia > 1.0)
  {
      printf("ERROR: La cota de energia debe estar expresado en decimales, no en porcentajes.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  if(mkdir(nombreDirPrin, 0777) == -1)
  {
      printf("ERROR: El directorio EXISTE, PELIGRO DE SOBREESCRITURA, por favor eliga otro nombre de directorio.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  else
      printf("Directorio creado.\n");

  // double optimo = goldenMin_BaseNormal(u, v, w, delta_u, delta_v, cantVisi, N);
  // printf("El optimo esta en %.12f\n", optimo);

  double limitesDeZonas[] = {0.001, 1.0, 3.0};
  double cantPuntosPorZona[] = {400, 800};
  int cantPtosLimites = 3;
  double* paramEvaInfo_pre = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  double* paramEvaInfo;
  cudaMallocManaged(&paramEvaInfo, cantParamEvaInfo*sizeof(double));
  combinacionLinealMatrices(delta_u, paramEvaInfo_pre, cantParamEvaInfo, 1, 0.0, paramEvaInfo, tamBloque, 0);
  cudaFree(paramEvaInfo_pre);

  char* nombreArchivoDetallesLinspace = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArDetLinspace)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoDetallesLinspace, nombreDirPrin);
  strcat(nombreArchivoDetallesLinspace, "/");
  strcat(nombreArchivoDetallesLinspace, nombreArDetLinspace);
  FILE* archivoDetLin = fopen(nombreArchivoDetallesLinspace, "a");
  for(int i=0; i<cantPtosLimites-1; i++)
  {
    fprintf(archivoDetLin, "Inicio: %f, Fin: %f, Cant. Ele: %f\n", limitesDeZonas[i], limitesDeZonas[i+1], cantPuntosPorZona[i]);
  }
  fclose(archivoDetLin);
  free(nombreArchivoDetallesLinspace);

  // int i = 0;
  #pragma omp parallel num_threads(16)
  {
      #pragma omp for schedule(dynamic, 1)
      for(int i=0; i<cantParamEvaInfo; i++)
      {
        char* numComoString = numAString(&i);
        sprintf(numComoString, "%d", i);
        char* nombreDirSecCopia = (char*) malloc(sizeof(char)*(strlen(nombreDirSec)+strlen(numComoString)));
        strcpy(nombreDirSecCopia, nombreDirSec);
        strcat(nombreDirSecCopia, numComoString);
        int thread_id = omp_get_thread_num();
        int deviceId = thread_id%4;
        cudaSetDevice(deviceId);
        af::setDevice(deviceId);
        calCompSegunAncho_InvCuadra_escritura(nombreDirPrin, nombreDirSecCopia, nombreDirTer, paramEvaInfo[i], cotaEnergia, i, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, matrizDeUnosTamN, tamBloque, deviceId);
        free(numComoString);
        free(nombreDirSecCopia);
      }
  }
}

void calculoDeInfoCompre_BaseNormal(char nombreArchivo[], int maxIter, double tolGrad, double tolGolden, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, long cantVisi, long N, double cotaEnergia, char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], int cantParamEvaInfo, double inicioIntervalo, double finIntervalo, double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque)
{
  char nombreArDetLinspace[] = "detalleslinspace.txt";
  double inicioIntervaloEscalado = inicioIntervalo * delta_u;
  double finIntervaloEscalado = finIntervalo * delta_u;
  char nombreArPara[] = "parametrosEjecucion.txt";
  if(cotaEnergia > 1.0)
  {
      printf("ERROR: La cota de energia debe estar expresado en decimales, no en porcentajes.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
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
  escrituraDeArchivoConParametros_Normal(nombreArchivoPara, nombreArchivo, nombreDirPrin, cantVisi, N, maxIter, tolGrad);
  free(nombreArchivoPara);

  // double optimo = goldenMin_BaseNormal(u, v, w, delta_u, delta_v, cantVisi, N);
  // printf("El optimo esta en %.12f\n", optimo);

  double limitesDeZonas[] = {0.001, 2.0, 3.0};
  double cantPuntosPorZona[] = {400, 800};
  int cantPtosLimites = 3;
  double* paramEvaInfo_pre = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  double* paramEvaInfo;
  cudaMallocManaged(&paramEvaInfo, cantParamEvaInfo*sizeof(double));
  combinacionLinealMatrices(delta_u, paramEvaInfo_pre, cantParamEvaInfo, 1, 0.0, paramEvaInfo, tamBloque, 0);
  cudaFree(paramEvaInfo_pre);

  char* nombreArchivoDetallesLinspace = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArDetLinspace)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoDetallesLinspace, nombreDirPrin);
  strcat(nombreArchivoDetallesLinspace, "/");
  strcat(nombreArchivoDetallesLinspace, nombreArDetLinspace);
  FILE* archivoDetLin = fopen(nombreArchivoDetallesLinspace, "a");
  for(int i=0; i<cantPtosLimites-1; i++)
  {
    fprintf(archivoDetLin, "Inicio: %f, Fin: %f, Cant. Ele: %f\n", limitesDeZonas[i], limitesDeZonas[i+1], cantPuntosPorZona[i]);
  }
  fclose(archivoDetLin);
  free(nombreArchivoDetallesLinspace);

  // int i = 0;
  #pragma omp parallel num_threads(70)
  {
      #pragma omp for schedule(dynamic, 1)
      for(int i=0; i<cantParamEvaInfo; i++)
      {
        char* numComoString = numAString(&i);
        sprintf(numComoString, "%d", i);
        char* nombreDirSecCopia = (char*) malloc(sizeof(char)*strlen(nombreDirSec)*strlen(numComoString));
        strcpy(nombreDirSecCopia, nombreDirSec);
        strcat(nombreDirSecCopia, numComoString);
        int thread_id = omp_get_thread_num();
        int deviceId = thread_id%4;
        cudaSetDevice(deviceId);
        af::setDevice(deviceId);
        calCompSegunAncho_Normal_escritura(nombreDirPrin, nombreDirSecCopia, nombreDirTer, paramEvaInfo[i], cotaEnergia, i, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, matrizDeUnosTamN, tamBloque, deviceId);
        free(numComoString);
        free(nombreDirSecCopia);
      }
  }
}

void calculoDeInfoCompre_BaseRect(char nombreArchivo[], int maxIter, double tolGrad, double tolGolden, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double cotaEnergia, char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], int cantParamEvaInfo, double inicioIntervalo, double finIntervalo, double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque, double* matrizDeUnosNxN)
{
  char nombreArDetLinspace[] = "detalleslinspace.txt";
  double inicioIntervaloEscalado = inicioIntervalo * delta_u;
  double finIntervaloEscalado = finIntervalo * delta_u;
  char nombreArPara[] = "parametrosEjecucion.txt";
  if(cotaEnergia > 1.0)
  {
      printf("ERROR: La cota de energia debe estar expresado en decimales, no en porcentajes.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
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

  // double optimo = goldenMin_BaseRect(u, v, w, delta_u, delta_v, matrizDeUnos, cantVisi, N, estrechezDeBorde);
  // printf("El optimo esta en %.12f\n", optimo);

  // double* paramEvaInfo = linspace(inicioIntervaloEscalado, finIntervaloEscalado, cantParamEvaInfo);

  // double limitesDeZonas[] = {0.1, 2.0, 3.0};
  // double cantPuntosPorZona[] = {100, 50};
  // int cantPtosLimites = 3;
  // double* paramEvaInfo_pre = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  // double* paramEvaInfo;
  // cudaMallocManaged(&paramEvaInfo, cantParamEvaInfo*sizeof(double));
  // combinacionLinealMatrices(delta_u, paramEvaInfo_pre, cantParamEvaInfo, 1, 0.0, paramEvaInfo, tamBloque, 0);
  // cudaFree(paramEvaInfo_pre);

  double limitesDeZonas[] = {0.001, 1.0, 3.0};
  double cantPuntosPorZona[] = {400, 100};
  int cantPtosLimites = 3;
  double* paramEvaInfo_pre = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  double* paramEvaInfo;
  cudaMallocManaged(&paramEvaInfo, cantParamEvaInfo*sizeof(double));
  combinacionLinealMatrices(delta_u, paramEvaInfo_pre, cantParamEvaInfo, 1, 0.0, paramEvaInfo, tamBloque, 0);
  cudaFree(paramEvaInfo_pre);

  char* nombreArchivoDetallesLinspace = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArDetLinspace)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoDetallesLinspace, nombreDirPrin);
  strcat(nombreArchivoDetallesLinspace, "/");
  strcat(nombreArchivoDetallesLinspace, nombreArDetLinspace);
  FILE* archivoDetLin = fopen(nombreArchivoDetallesLinspace, "a");
  for(int i=0; i<cantPtosLimites-1; i++)
  {
    fprintf(archivoDetLin, "Inicio: %f, Fin: %f, Cant. Ele: %f\n", limitesDeZonas[i], limitesDeZonas[i+1], cantPuntosPorZona[i]);
  }
  fclose(archivoDetLin);
  free(nombreArchivoDetallesLinspace);

  // int i = 0;
  #pragma omp parallel num_threads(4)
  {
    #pragma omp for schedule(dynamic, 1)
    for(int i=0; i<cantParamEvaInfo; i++)
    {
      char* numComoString = numAString(&i);
      sprintf(numComoString, "%d", i);
      char* nombreDirSecCopia = (char*) malloc(sizeof(char)*strlen(nombreDirSec)*strlen(numComoString));
      strcpy(nombreDirSecCopia, nombreDirSec);
      strcat(nombreDirSecCopia, numComoString);
      int thread_id = omp_get_thread_num();
      int deviceId = thread_id%4;
      cudaSetDevice(deviceId);
      af::setDevice(deviceId);
      calCompSegunAncho_Rect_escritura(nombreDirPrin, nombreDirSecCopia, nombreDirTer, paramEvaInfo[i], cotaEnergia, i, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosTamN, estrechezDeBorde, tamBloque, deviceId, matrizDeUnosNxN);
      free(numComoString);
      free(nombreDirSecCopia);
    }
  }
}

double* minGradConjugado_MinCuadra_escritura_l1(double param_lambda, double* costoFinal, char* nombreArchivoMin, char* nombreArchivoCoefs, double* MC, double* MV, double* MU, double* visibilidades, double* w, long cantVisi, long N, double* matrizDeUnosTamN, double delta_u, int maxIter, double tol, int tamBloque, int numGPU)
{
  double inicioIntervaloZ = -1e30;
  double finIntervaloZ = 1e30;
  int flag_NOESPOSIBLEMINIMIZAR = 0;
  double* residual = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN, tamBloque, numGPU);
  double* gradienteActual;
  cudaMallocManaged(&gradienteActual,N*N*sizeof(double));
  cudaMemset(gradienteActual, 0, N*N*sizeof(double));
  double* gradienteAnterior;
  cudaMallocManaged(&gradienteAnterior,N*N*sizeof(double));
  cudaMemset(gradienteAnterior, 0, N*N*sizeof(double));
  double* pActual;
  cudaMallocManaged(&pActual,N*N*sizeof(double));
  cudaMemset(pActual, 0, N*N*sizeof(double));
  double costoInicial = calCosto_l1(param_lambda, residual, cantVisi, w, MC, N, tamBloque, numGPU);
  double costoAnterior = costoInicial;
  double costoActual = costoInicial;
  calGradiente_l1(param_lambda, residual, MV, cantVisi, N, MU, N, w, MC, gradienteAnterior, N, tamBloque, numGPU);

  // for(int i=0; i<N*N; i++)
  // {
  //   if(gradienteAnterior[i] != 0.0)
  //   {
  //     printf("En la linea %d es %.12e\n", i, gradienteAnterior[i]);
  //   }
  // }
  // exit(-1);

  combinacionLinealMatrices(-1.0, gradienteAnterior, N, N, 0.0, pActual, tamBloque, numGPU);
  double diferenciaDeCosto = 1.0;
  int i = 0;
  double alpha = 0.0;
  double epsilon = 1e-10;
  double normalizacion = costoAnterior + costoActual + epsilon;


  // double valorcito = funcValorZ(7782347894233.96984961, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
  // printf("%.12e\n", valorcito);
  // // exit(-1);
  //
  // double valorcito1 = funcValorZ(1, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
  // printf("%.12e\n", valorcito1);
  // exit(-1);

  // double* ahora = linspace(-1e30, 1e30, 200);
  // FILE* archivorandom = fopen("/home/rarmijo/zetas.txt", "w");
  // for(int j=0; j<200; j++)
  // {
  //   // printf("%f\n",ahora[j]);
  //   double valorcito = funcValorZ(ahora[j], cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
  //   printf("%.32e\n", valorcito);
  //   fprintf(archivorandom, "%.32e\n", valorcito);
  // }
  // fclose(archivorandom);
  // exit(-1);

  FILE* archivoMin = fopen(nombreArchivoMin, "w");
  if(archivoMin == NULL)
  {
       printf("Error al crear o abrir el archivo para almacenar la minimizacion.\n");
       exit(0);
  }
  // double* ahora = linspace(-1e30, 1e30, 200);
  while(maxIter > i && 2.0 * diferenciaDeCosto > tol * normalizacion)
  {
    // alpha = calAlpha(gradienteAnterior, N, N, pActual, MV, cantVisi, N, MU, N, w, matrizDeUnosTamN, &flag_NOESPOSIBLEMINIMIZAR, tamBloque, numGPU);

    alpha = goldenMin_Minl1(&flag_NOESPOSIBLEMINIMIZAR, inicioIntervaloZ, finIntervaloZ, cantVisi, N, MU, MC, MV, residual, w, pActual, param_lambda, tamBloque, numGPU, matrizDeUnosTamN, delta_u);

    // double inicioIntervaloZ = -1e30;
    // double finIntervaloZ = 1e30;
    // double fInicioIntervaloZ = funcValorZ(inicioIntervaloZ, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
    // double fFinIntervaloZ = funcValorZ(finIntervaloZ, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
    // Min_Search_Golden_Section(funcValorZ, &inicioIntervaloZ, &fInicioIntervaloZ, &finIntervaloZ, &fFinIntervaloZ, 1e-12, cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
    // // printf("El inicio es %.30f y el fin es %.30f\n", finIntervaloZ, inicioIntervaloZ);
    // alpha = (finIntervaloZ+inicioIntervaloZ)/2.0;

    // char nombreBase[] = "/srv/nas01/rarmijo/resultados_temporales/zetas";
    // char* numComoString = numAString(&i);
    // sprintf(numComoString, "%d", i);
    // char* nombreConIteracion = (char*) malloc(sizeof(char)*(strlen(nombreBase)+strlen(numComoString)+5));
    // strcpy(nombreConIteracion, nombreBase);
    // strcat(nombreConIteracion, numComoString);
    // strcat(nombreConIteracion, ".txt");
    // FILE* archivorandom = fopen(nombreConIteracion, "w");
    // for(int j=0; j<200; j++)
    // {
    //   double valorcito = funcValorZ(ahora[j], cantVisi, N, MV, MC, MU, matrizDeUnosTamN, tamBloque, numGPU, w, pActual, param_lambda, residual);
    //   // printf("%.32e\n", valorcito);
    //   fprintf(archivorandom, "%.32e\n", valorcito);
    // }
    // fclose(archivorandom);
    // free(numComoString);
    // free(nombreConIteracion);

    if(flag_NOESPOSIBLEMINIMIZAR == 1)
    {
      printf("No fue posible minimizar\n");
      break;
    }
    combinacionLinealMatrices(alpha, pActual, N, N, 1.0, MC, tamBloque, numGPU);
    // for(int j=0; j<N*N; j++)
    // {
    //   if(MC[j] < 1e-5)
    //   {
    //     MC[j] = 0.0;
    //   }
    // }

    // for(int j=0; j<N*N; j++)
    // {
    //   if(MC[j] < 1e-12)
    //   {
    //     MC[j] = 0.0;
    //   }
    // }

    double* puntero_residualAnterior = residual;
    residual = calResidual(visibilidades, MV, cantVisi, N, MC, N, MU, matrizDeUnosTamN, tamBloque, numGPU);
    cudaFree(puntero_residualAnterior);
    costoActual = calCosto_l1(param_lambda, residual, cantVisi, w, MC, N, tamBloque, numGPU);
    cudaMallocManaged(&gradienteActual,N*N*sizeof(double));
    cudaMemset(gradienteActual, 0, N*N*sizeof(double));
    calGradiente_l1(param_lambda, residual, MV, cantVisi, N, MU, N, w, MC, gradienteActual, N, tamBloque, numGPU);
    double beta = calBeta_Fletcher_Reeves(gradienteActual, N*N, gradienteAnterior, numGPU);
    combinacionLinealMatrices(-1.0, gradienteActual, N, N, beta, pActual, tamBloque, numGPU);
    diferenciaDeCosto = abs(costoAnterior - costoActual);
    normalizacion = costoAnterior + costoActual + epsilon;
    double otro = costoActual - costoAnterior;
    costoAnterior = costoActual;
    double* puntero_GradienteAnterior = gradienteAnterior;
    gradienteAnterior = gradienteActual;
    cudaFree(puntero_GradienteAnterior);
    i++;
    printf( "En la iteracion %d el valor de la funcion de costos es %f con un z de %.12e la diferencia con respecto al anterior costo es %.12e.\n", i, costoActual, alpha, otro);
    fprintf(archivoMin, "En la iteracion %d el valor de la funcion de costos es %f con un z de %.12e la diferencia con respecto al anterior costo es %.12e.\n", i, costoActual, alpha, otro);
  }
  fclose(archivoMin);
  cudaFree(gradienteAnterior);
  cudaFree(pActual);
  escribirCoefs(MC, nombreArchivoCoefs, N, N);
  *costoFinal = costoActual;
  return MC;
}

void calCompSegunAncho_Rect_escritura_l1(double param_lambda, double* MC_imag, double* MC_real, char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], char nombreArchivoLamda[], double ancho, int iterActual, int maxIter, double tol, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque, int numGPU, double* matrizDeUnosNxN)
{
  // hd_142
  double inicioPorcenCompre = 0.0;
  // double terminoPorcenCompre = 0.2;
  int cantPorcen = 101;


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
  double* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  double tiempoTotalCalculoMV = ((double)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

  printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  double* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  double tiempoTotalCalculoMU = ((double)tiempoCalculoMU)/CLOCKS_PER_SEC;
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
  double costoParteImag;
  char* nombreArchivoMin_imag = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArMin_imag)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_imag, rutaADirecSec);
  strcat(nombreArchivoMin_imag, nombreArMin_imag);
  char* nombreArchivoCoefs_imag = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_imag)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_imag, rutaADirecSec);
  strcat(nombreArchivoCoefs_imag, nombreArCoef_imag);
  printf("...Comenzando minimizacion de coeficientes parte imaginaria...\n");
  clock_t tiempoMinPartImag;
  tiempoMinPartImag = clock();
  minGradConjugado_MinCuadra_escritura_l1(param_lambda, &costoParteImag, nombreArchivoMin_imag, nombreArchivoCoefs_imag, MC_imag, MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, delta_u, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartImag = clock() - tiempoMinPartImag;
  double tiempoTotalMinPartImag = ((double)tiempoMinPartImag)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte imaginaria terminado.\n");
  free(nombreArchivoMin_imag);
  free(nombreArchivoCoefs_imag);


  // ############### MINIMIZACION DE COEFS, PARTE REAL  ##############
  double costoParteReal;
  char* nombreArchivoMin_real = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArMin_real)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoMin_real, rutaADirecSec);
  strcat(nombreArchivoMin_real, nombreArMin_real);
  char* nombreArchivoCoefs_real = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArCoef_real)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoCoefs_real, rutaADirecSec);
  strcat(nombreArchivoCoefs_real, nombreArCoef_real);
  printf("...Comenzando minimizacion de coeficientes parte real...\n");
  clock_t tiempoMinPartReal;
  tiempoMinPartReal = clock();
  minGradConjugado_MinCuadra_escritura_l1(param_lambda, &costoParteReal, nombreArchivoMin_real, nombreArchivoCoefs_real, MC_real, MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, delta_u, maxIter, tol, tamBloque, numGPU);
  tiempoMinPartReal = clock() - tiempoMinPartReal;
  double tiempoTotalMinPartReal = ((double)tiempoMinPartReal)/CLOCKS_PER_SEC;
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");
  free(nombreArchivoMin_real);
  free(nombreArchivoCoefs_real);


  // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, numGPU);
  tiempoInfo = clock() - tiempoInfo;
  double tiempoTotalInfo = ((double)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);


  // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  double* MV_AF = calcularMV_Rect_estFourier(ancho, N, delta_v, matrizDeUnosNxN, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  double tiempoTotalCalculoMV_AF = ((double)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  double* MU_AF = calcularMV_Rect_estFourier(ancho, N, delta_u, matrizDeUnosNxN, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  double tiempoTotalCalculoMU_AF = ((double)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  double* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  double tiempoTotalReconsFourierPartImag = ((double)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  double* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  double tiempoTotalReconsFourierPartReal = ((double)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  double tiempoTotalReconsTransInver = ((double)tiempoReconsTransInver)/CLOCKS_PER_SEC;
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
  printf("...Comenzando calculo de compresiones...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  double* datosDelMin = calPSNRDeDistintasCompresiones(inicioPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, tamBloque, numGPU);
  tiempoCompresion = clock() - tiempoCompresion;
  double tiempoTotalCompresion = ((double)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresiones terminado.\n");
  free(rutaADirecTer);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  double nivelDeCompresion = 1.0 - datosDelMin[4] * 1.0 / N*N;
  #pragma omp critical
  {
    FILE* archivo = fopen(nombreArchivoInfoComp, "a");
    fprintf(archivo, "%d %f %.12f %.12e %.12e %.12f %.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", iterActual, ancho/delta_u, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], (long) datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
    fclose(archivo);
  }
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);
  free(datosDelMin);

  cudaFree(MU_AF);
  cudaFree(MV_AF);

  // ############### ESCRITURA VALOR LAMBDA ##############
  char* nombreArchivoCostoYLambda = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArchivoLamda)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoCostoYLambda, nombreDirPrin);
  strcat(nombreArchivoCostoYLambda, "/");
  strcat(nombreArchivoCostoYLambda, nombreArchivoLamda);
  #pragma omp critical
  {
    FILE* archivoCostoYLambda = fopen(nombreArchivoCostoYLambda, "a");
    fprintf(archivoCostoYLambda, "%d %.12e %.12e\n", iterActual, param_lambda, costoParteImag+costoParteReal);
    fclose(archivoCostoYLambda);
  }
  free(nombreArchivoCostoYLambda);


  // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  #pragma omp critical
  {
    FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
    fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalMinPartImag, tiempoTotalMinPartReal, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver);
    fclose(archivoInfoTiemposEjecu);
  }
  free(nombreArchivoInfoTiemposEjecu);
  free(rutaADirecSec);
}

void calculoDeInfoCompre_l1_BaseRect(char nombreArchivo[], int maxIter, double tolGrad, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double cotaEnergia, char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque, double* matrizDeUnosNxN)
{

  int cantMejoresLambdasASeleccionar = 5;
  int maxIterMejoresLambda = 1;
  char nombreArchivoCostoYLambda[] = "costoylambda.txt";
  char nombreDirPrimeraEtapa[] = "etapa1";
  char nombreDirSegundaEtapa[] = "etapa2";
  char nombreDirTerceraEtapa[] = "etapa3";
  char nombreDirCoefs[] = "/srv/nas01/rarmijo/resultados/experi_hd142_b9_model_Rect/ite401";
  char nombreArchivoCoefsImag[] = "coefs_imag.txt";
  char nombreArchivoCoefsReal[] = "coefs_real.txt";
  double ancho = delta_u * 1.0;

  char nombreArDetLinspace[] = "detalleslinspace.txt";
  char nombreArPara[] = "parametrosEjecucion.txt";
  if(cotaEnergia > 1.0)
  {
      printf("ERROR: La cota de energia debe estar expresado en decimales, no en porcentajes.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
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


  double limitesDeZonas[] = {10e-10, 10e-5, 10e-1};
  // double cantPuntosPorZona[] = {100, 100};
  double cantPuntosPorZona[] = {5, 5};
  int cantPtosLimites = 3;
  double* paramEvaInfo = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  int cantidadDeLambdasTotales = cantPtosLimites;
  for(int contaLambdas=0; contaLambdas<cantPtosLimites-1; contaLambdas++)
  {
    cantidadDeLambdasTotales += cantPuntosPorZona[contaLambdas];
  }


  // ############### PRIMERA ETAPA: CALCULO DE COSTO PARA DISTINTOS LAMBDAS ##############
  printf("Comenzando ETAPA 1\n");
  char* nombreDirPrimeraEtapaDesdeRaiz = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirPrimeraEtapa)+3)*sizeof(char));
  strcpy(nombreDirPrimeraEtapaDesdeRaiz, nombreDirPrin);
  strcat(nombreDirPrimeraEtapaDesdeRaiz, "/");
  strcat(nombreDirPrimeraEtapaDesdeRaiz, nombreDirPrimeraEtapa);
  if(mkdir(nombreDirPrimeraEtapaDesdeRaiz, 0777) == -1)
  {
      printf("ERROR: No se pudo crear subdirectorio para la PRIMERA ETAPA.");
      printf("PROGRAMA ABORTADO ANTES DE LA PRIMERA ETAPA.\n");
      exit(0);
  }

  char* nombreArchivoActual_Coefs_imag_Principal = (char*) malloc(sizeof(char)*(strlen(nombreDirCoefs)+strlen(nombreArchivoCoefsImag)+3));
  strcpy(nombreArchivoActual_Coefs_imag_Principal, nombreDirCoefs);
  strcat(nombreArchivoActual_Coefs_imag_Principal, "/");
  strcat(nombreArchivoActual_Coefs_imag_Principal, nombreArchivoCoefsImag);
  char* nombreArchivoActual_Coefs_real_Principal = (char*) malloc(sizeof(char)*(strlen(nombreDirCoefs)+strlen(nombreArchivoCoefsReal)+3));
  strcpy(nombreArchivoActual_Coefs_real_Principal, nombreDirCoefs);
  strcat(nombreArchivoActual_Coefs_real_Principal, "/");
  strcat(nombreArchivoActual_Coefs_real_Principal, nombreArchivoCoefsReal);
  double* MC_imag_principal, *MC_real_principal;
  cudaMallocManaged(&MC_imag_principal, N*N*sizeof(double));
  cudaMallocManaged(&MC_real_principal, N*N*sizeof(double));
  #pragma omp critical
  {
    lecturaDeTXTDeCoefs(nombreArchivoActual_Coefs_imag_Principal, MC_imag_principal, N, N);
    lecturaDeTXTDeCoefs(nombreArchivoActual_Coefs_real_Principal, MC_real_principal, N, N);
  }
  free(nombreArchivoActual_Coefs_imag_Principal);
  free(nombreArchivoActual_Coefs_real_Principal);
  // int i = 0;
  #pragma omp parallel num_threads(4)
  {
    #pragma omp for schedule(dynamic, 1)
    for(int i=0; i<cantidadDeLambdasTotales; i++)
    {
      char* numComoString = numAString(&i);
      sprintf(numComoString, "%d", i);
      char* nombreDirSecCopia = (char*) malloc(sizeof(char)*strlen(nombreDirSec)*strlen(numComoString));
      strcpy(nombreDirSecCopia, nombreDirSec);
      strcat(nombreDirSecCopia, numComoString);
      int thread_id = omp_get_thread_num();
      int deviceId = thread_id%4;
      cudaSetDevice(deviceId);
      af::setDevice(deviceId);
      calCompSegunAncho_Rect_escritura_l1(paramEvaInfo[i], MC_imag_principal, MC_real_principal, nombreDirPrimeraEtapaDesdeRaiz, nombreDirSecCopia, nombreDirTer, nombreArchivoCostoYLambda, ancho, i, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosTamN, estrechezDeBorde, tamBloque, deviceId, matrizDeUnosNxN);
      free(numComoString);
      free(nombreDirSecCopia);
    }
  }
  cudaFree(paramEvaInfo);
  cudaFree(MC_imag_principal);
  cudaFree(MC_real_principal);
  printf("ETAPA 1 CONCLUIDA.\n");


  // ############### SEGUNDA ETAPA: SELECCION DE MEJORES LAMBDAS ##############
  printf("Comenzando ETAPA 2\n");
  char* nombreDirSegundaEtapaDesdeRaiz = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirSegundaEtapa)+3)*sizeof(char));
  strcpy(nombreDirSegundaEtapaDesdeRaiz, nombreDirPrin);
  strcat(nombreDirSegundaEtapaDesdeRaiz, "/");
  strcat(nombreDirSegundaEtapaDesdeRaiz, nombreDirSegundaEtapa);
  if(mkdir(nombreDirSegundaEtapaDesdeRaiz, 0777) == -1)
  {
      printf("ERROR: No se pudo crear subdirectorio para la SEGUNDA ETAPA.");
      printf("PROGRAMA ABORTADO ANTES DE LA SEGUNDA ETAPA.\n");
      exit(0);
  }
  char* nombreArchivoPrimeraEtapaCostoYLambda = (char*) malloc((strlen(nombreDirPrimeraEtapaDesdeRaiz)+strlen(nombreArchivoCostoYLambda)+3)*sizeof(char));
  strcpy(nombreArchivoPrimeraEtapaCostoYLambda, nombreDirPrimeraEtapaDesdeRaiz);
  strcat(nombreArchivoPrimeraEtapaCostoYLambda, "/");
  strcat(nombreArchivoPrimeraEtapaCostoYLambda, nombreArchivoCostoYLambda);
  int* listaMejores_NumIte;
  cudaMallocManaged(&listaMejores_NumIte, cantMejoresLambdasASeleccionar*sizeof(int));
  double* listaMejores_Lambda;
  cudaMallocManaged(&listaMejores_Lambda, cantMejoresLambdasASeleccionar*sizeof(int));
  seleccionarMejoresLambdas(nombreDirSegundaEtapaDesdeRaiz, nombreArchivoPrimeraEtapaCostoYLambda, cantidadDeLambdasTotales, cantMejoresLambdasASeleccionar, listaMejores_NumIte, listaMejores_Lambda);
  free(nombreDirSegundaEtapaDesdeRaiz);
  free(nombreArchivoPrimeraEtapaCostoYLambda);
  printf("ETAPA 2 CONCLUIDA\n");


  // ############### TERCERA ETAPA: CALCULO DE COSTO PARA LOS MEJORES LAMBDAS ##############
  printf("Comenzando ETAPA 3\n");
  char* nombreDirTerceraEtapaDesdeRaiz = (char*) malloc((strlen(nombreDirPrin)+strlen(nombreDirTerceraEtapa)+3)*sizeof(char));
  strcpy(nombreDirTerceraEtapaDesdeRaiz, nombreDirPrin);
  strcat(nombreDirTerceraEtapaDesdeRaiz, "/");
  strcat(nombreDirTerceraEtapaDesdeRaiz, nombreDirTerceraEtapa);
  if(mkdir(nombreDirTerceraEtapaDesdeRaiz, 0777) == -1)
  {
      printf("ERROR: No se pudo crear subdirectorio para la TERCERA ETAPA.");
      printf("PROGRAMA ABORTADO ANTES DE LA TERCERA ETAPA.\n");
      exit(0);
  }
  char* nombreDirBaseCoefsPrimeraEtapa = (char*) malloc(sizeof(char)*(strlen(nombreDirPrimeraEtapaDesdeRaiz)+strlen(nombreDirSec)+2));
  strcpy(nombreDirBaseCoefsPrimeraEtapa, nombreDirPrimeraEtapaDesdeRaiz);
  strcat(nombreDirBaseCoefsPrimeraEtapa, "/");
  strcat(nombreDirBaseCoefsPrimeraEtapa, nombreDirSec);
  #pragma omp parallel num_threads(4)
  {
    #pragma omp for schedule(dynamic, 1)
    for(int i=0; i<cantMejoresLambdasASeleccionar; i++)
    {

      char* numComoStringCarpetaCoefs = numAString(&(listaMejores_NumIte[i]));
      sprintf(numComoStringCarpetaCoefs, "%d", listaMejores_NumIte[i]);
      char* nombreArchivoActualCoefs_imag = (char*) malloc(sizeof(char)*(strlen(nombreDirBaseCoefsPrimeraEtapa)+strlen(numComoStringCarpetaCoefs)+strlen(nombreArchivoCoefsImag)+3));
      strcpy(nombreArchivoActualCoefs_imag, nombreDirBaseCoefsPrimeraEtapa);
      strcat(nombreArchivoActualCoefs_imag, numComoStringCarpetaCoefs);
      strcat(nombreArchivoActualCoefs_imag, "/");
      strcat(nombreArchivoActualCoefs_imag, nombreArchivoCoefsImag);
      char* nombreArchivoActualCoefs_real = (char*) malloc(sizeof(char)*(strlen(nombreDirBaseCoefsPrimeraEtapa)+strlen(numComoStringCarpetaCoefs)+strlen(nombreArchivoCoefsReal)+3));
      strcpy(nombreArchivoActualCoefs_real, nombreDirBaseCoefsPrimeraEtapa);
      strcat(nombreArchivoActualCoefs_real, numComoStringCarpetaCoefs);
      strcat(nombreArchivoActualCoefs_real, "/");
      strcat(nombreArchivoActualCoefs_real, nombreArchivoCoefsReal);
      double* MC_imag, *MC_real;
      cudaMallocManaged(&MC_imag, N*N*sizeof(double));
      cudaMallocManaged(&MC_real, N*N*sizeof(double));
      #pragma omp critical
      {
        lecturaDeTXTDeCoefs(nombreArchivoActualCoefs_imag, MC_imag, N, N);
        lecturaDeTXTDeCoefs(nombreArchivoActualCoefs_real, MC_real, N, N);
      }
      free(nombreArchivoActualCoefs_imag);
      free(nombreArchivoActualCoefs_real);

      char* numComoString = numAString(&i);
      sprintf(numComoString, "%d", i);
      char* nombreDirSecCopia = (char*) malloc(sizeof(char)*strlen(nombreDirSec)*strlen(numComoString));
      strcpy(nombreDirSecCopia, nombreDirSec);
      strcat(nombreDirSecCopia, numComoString);

      int thread_id = omp_get_thread_num();
      int deviceId = thread_id%4;
      cudaSetDevice(deviceId);
      af::setDevice(deviceId);
      calCompSegunAncho_Rect_escritura_l1(listaMejores_Lambda[i], MC_imag, MC_real, nombreDirTerceraEtapaDesdeRaiz, nombreDirSecCopia, nombreDirTer, nombreArchivoCostoYLambda, ancho, i, maxIterMejoresLambda, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosTamN, estrechezDeBorde, tamBloque, deviceId, matrizDeUnosNxN);
      free(numComoStringCarpetaCoefs);
      free(numComoString);
      free(nombreDirSecCopia);
      cudaFree(MC_imag);
      cudaFree(MC_real);
    }
  }
  free(nombreDirPrimeraEtapaDesdeRaiz);
  free(nombreDirTerceraEtapaDesdeRaiz);
  free(nombreDirBaseCoefsPrimeraEtapa);
  printf("ETAPA 3 CONCLUIDA\n");
}

void lecturaDeArchivo_infoCompre(char nombreArchivo[], int* vectorDeNumItera, double* vectorDeAnchos_EnDeltaU, double* vectorDeAnchos_Real, int largoVector)
{
  // int contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  fp = fopen(nombreArchivo, "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  for(int i=0; i<largoVector; i++)
  {
    if((read = getline(&line, &len, fp)) == -1)
    {
      break;
    }
    vectorDeNumItera[i] = atoi(strtok(line, " "));
    vectorDeAnchos_EnDeltaU[i] = atof(strtok(NULL, " "));
    vectorDeAnchos_Real[i] = atof(strtok(NULL, " "));
    // printf("Num Ite: %d, Ancho en delta_u: %f, Ancho real: %f\n", vectorDeNumItera[i], vectorDeAnchos_EnDeltaU[i], vectorDeAnchos_Real[i]);
    // contador++;
	}
  // printf("%d lineas leidas del archivo infoCompre.\n", contador);
  free(line);
  fclose(fp);
}

char** lecturaDeNombresDeArchivo(char nombreArchivo[], int* cantLineasPorArchivo, int cantArchivos, int* largoDeNombres)
{
  char** nombreDeArchivosInfoCompre = (char**) malloc(sizeof(char)*cantArchivos);
  int contador = 0;
  FILE *fp;
  size_t len = 0;
  char *line = NULL;
  ssize_t read;
  fp = fopen(nombreArchivo, "r");
  if (fp == NULL)
  {
      printf("No se pudo abrir el archivo %s",nombreArchivo);
      exit(0);
  }
  for(int i=0; i<cantArchivos; i++)
  {
    if((read = getline(&line, &len, fp)) == -1)
    {
      printf("Error en lectura de archivo con nombres de archivo.\n");
      exit(-1);
    }
    nombreDeArchivosInfoCompre[i] = (char*) malloc(sizeof(char)*len);
    largoDeNombres[i] = len;
    strcpy(nombreDeArchivosInfoCompre[i], strtok(line, " "));
    cantLineasPorArchivo[i] = atoi(strtok(NULL, " "));
    printf("%s\n", nombreDeArchivosInfoCompre[i]);
    printf("Las lineas a leer para este archivo son %d\n", cantLineasPorArchivo[i]);
    contador++;
	}
  printf("%d lineas con nombres de archivo leidos.\n", contador);
  free(line);
  fclose(fp);
  return nombreDeArchivosInfoCompre;
}

void reciclador_calCompSegunAncho_Rect_escritura(double* MC_imag, double* MC_real, char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], double ancho, double ancho_enDeltaU, int iterActual, double* u, double* v, double* w, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double* matrizDeUnosTamN, int tamBloque, int numGPU, double* matrizDeUnosNxN, double estrechezDeBorde)
{
  // hd_142
  double inicioPorcenCompre = 0.0;
  // double terminoPorcenCompre = 0.2;
  int cantPorcen = 101;


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
  double* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  double tiempoTotalCalculoMV = ((double)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

  printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  double* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  double tiempoTotalCalculoMU = ((double)tiempoCalculoMU)/CLOCKS_PER_SEC;
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


  // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, numGPU);
  tiempoInfo = clock() - tiempoInfo;
  double tiempoTotalInfo = ((double)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);

  // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  double* MV_AF = calcularMV_Rect_estFourier(ancho, N, delta_v, matrizDeUnosNxN, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  double tiempoTotalCalculoMV_AF = ((double)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  double* MU_AF = calcularMV_Rect_estFourier(ancho, N, delta_u, matrizDeUnosNxN, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  double tiempoTotalCalculoMU_AF = ((double)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  double* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  double tiempoTotalReconsFourierPartImag = ((double)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  double* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  double tiempoTotalReconsFourierPartReal = ((double)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  double tiempoTotalReconsTransInver = ((double)tiempoReconsTransInver)/CLOCKS_PER_SEC;
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
  printf("...Comenzando calculo de compresiones...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  double* datosDelMin = calPSNRDeDistintasCompresiones(inicioPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, tamBloque, numGPU);
  tiempoCompresion = clock() - tiempoCompresion;
  double tiempoTotalCompresion = ((double)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresiones terminado.\n");
  free(rutaADirecTer);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  double nivelDeCompresion = 1.0 - datosDelMin[4] * 1.0 / N*N;
  #pragma omp critical
  {
    FILE* archivo = fopen(nombreArchivoInfoComp, "a");
    fprintf(archivo, "%d %f %.12f %.12e %.12e %.12f %.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", iterActual, ancho_enDeltaU, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], (long) datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
    fclose(archivo);
  }
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);
  free(datosDelMin);
  cudaFree(MU_AF);
  cudaFree(MV_AF);

  // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  #pragma omp critical
  {
    FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
    fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver);
    fclose(archivoInfoTiemposEjecu);
  }
  free(nombreArchivoInfoTiemposEjecu);
  free(rutaADirecSec);
}

void reciclador_calculoDeInfoCompre_BaseRect(char nombreArchivoConNombres[], char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], char nombreArchivoCoefs_imag[], char nombreArchivoCoefs_real[], int cantArchivos, int flag_multiThread, char nombreArchivoInfoCompre[], int maxIter, double* u, double* v, double* w, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double* matrizDeUnosTamN, int tamBloque, double* matrizDeUnosNxN, double estrechezDeBorde)
{
  if(mkdir(nombreDirPrin, 0777) == -1)
  {
      printf("ERROR: El directorio EXISTE, PELIGRO DE SOBREESCRITURA, por favor eliga otro nombre de directorio.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  else
      printf("Directorio creado.\n");

      int* cantLineasPorArchivo;
      cudaMallocManaged(&cantLineasPorArchivo, cantArchivos*sizeof(int));
      int* largoDeNombres;
      cudaMallocManaged(&largoDeNombres, cantArchivos*sizeof(int));
      char** nombres = lecturaDeNombresDeArchivo(nombreArchivoConNombres, cantLineasPorArchivo, cantArchivos, largoDeNombres);
      int contadorDeIteraciones = -1;
      for(int numArchi=0; numArchi<cantArchivos; numArchi++)
      {
          int cantLineasActual = cantLineasPorArchivo[numArchi];
          char* nombreActual = (char*) malloc(sizeof(char)*(largoDeNombres[numArchi])+sizeof(char)*strlen(nombreArchivoInfoCompre)+sizeof(char)*5);
          strcpy(nombreActual, &(nombres[numArchi][0]));
          strcat(nombreActual, "/");
          strcat(nombreActual, nombreArchivoInfoCompre);

          // printf("%s\n", nombreActual);
          int* vectorDeNumItera;
          cudaMallocManaged(&vectorDeNumItera, cantLineasActual*sizeof(int));
          double* vectorDeAnchos_EnDeltaU;
          cudaMallocManaged(&vectorDeAnchos_EnDeltaU, cantLineasActual*sizeof(double));
          double* vectorDeAnchos_Real;
          cudaMallocManaged(&vectorDeAnchos_Real, cantLineasActual*sizeof(double));
          lecturaDeArchivo_infoCompre(nombreActual, vectorDeNumItera, vectorDeAnchos_EnDeltaU, vectorDeAnchos_Real, cantLineasActual);
          free(nombreActual);
          #pragma omp parallel num_threads(20)
          {
            #pragma omp for schedule(dynamic, 1)
            for(int numLinea=0; numLinea<cantLineasActual; numLinea++)
            {
              int numIteracion;
              if(flag_multiThread)
              {
                #pragma omp critical
                {
                  contadorDeIteraciones++;
                  numIteracion = contadorDeIteraciones;
                }
              }
              else
              {
                contadorDeIteraciones++;
              }
              int numIteAUsar = vectorDeNumItera[numLinea];
              char* numComoString_iterALeer = numAString(&numIteAUsar);
              sprintf(numComoString_iterALeer, "%d", numIteAUsar);
              char* nombreDirSecLeerCoefs = (char*) malloc(sizeof(char)*strlen(nombreDirSec)+sizeof(char)*strlen(numComoString_iterALeer)+sizeof(char)*5);
              strcpy(nombreDirSecLeerCoefs, nombreDirSec);
              strcat(nombreDirSecLeerCoefs, numComoString_iterALeer);
              free(numComoString_iterALeer);

              char* nombreArchivoActualCoefs_imag = (char*) malloc(sizeof(char)*(largoDeNombres[numArchi])+sizeof(char)*strlen(nombreArchivoCoefs_imag)+sizeof(char)*10);
              strcpy(nombreArchivoActualCoefs_imag, &(nombres[numArchi][0]));
              strcat(nombreArchivoActualCoefs_imag, "/");
              strcat(nombreArchivoActualCoefs_imag, nombreDirSecLeerCoefs);
              strcat(nombreArchivoActualCoefs_imag, "/");
              strcat(nombreArchivoActualCoefs_imag, nombreArchivoCoefs_imag);
              char* nombreArchivoActualCoefs_real = (char*) malloc(sizeof(char)*(largoDeNombres[numArchi])+sizeof(char)*strlen(nombreArchivoCoefs_real)+sizeof(char)*10);
              strcpy(nombreArchivoActualCoefs_real, &(nombres[numArchi][0]));
              strcat(nombreArchivoActualCoefs_real, "/");
              strcat(nombreArchivoActualCoefs_real, nombreDirSecLeerCoefs);
              strcat(nombreArchivoActualCoefs_real, "/");
              strcat(nombreArchivoActualCoefs_real, nombreArchivoCoefs_real);
              free(nombreDirSecLeerCoefs);
              double* MC_imag_actual, *MC_real_actual ;
              cudaMallocManaged(&MC_imag_actual, N*N*sizeof(double));
              cudaMallocManaged(&MC_real_actual, N*N*sizeof(double));
              #pragma omp critical
              {
                lecturaDeTXTDeCoefs(nombreArchivoActualCoefs_imag, MC_imag_actual, N, N);
                lecturaDeTXTDeCoefs(nombreArchivoActualCoefs_real, MC_real_actual, N, N);
              }
              free(nombreArchivoActualCoefs_imag);
              free(nombreArchivoActualCoefs_real);
              char* numComoString_iterAEscribir = numAString(&numIteracion);
              sprintf(numComoString_iterAEscribir, "%d", numIteracion);
              char* nombreNuevoDirSec = (char*) malloc(sizeof(char)*strlen(nombreDirSec)+sizeof(char)*strlen(numComoString_iterAEscribir)+sizeof(char)*5);
              strcpy(nombreNuevoDirSec, nombreDirSec);
              strcat(nombreNuevoDirSec, numComoString_iterAEscribir);
              free(numComoString_iterAEscribir);
              int thread_id = omp_get_thread_num();
              int deviceId = thread_id%4;
              cudaSetDevice(deviceId);
              af::setDevice(deviceId);
              reciclador_calCompSegunAncho_Rect_escritura(MC_imag_actual, MC_real_actual, nombreDirPrin, nombreNuevoDirSec, nombreDirTer, vectorDeAnchos_Real[numLinea], vectorDeAnchos_EnDeltaU[numLinea], numIteracion, u, v, w, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosTamN, tamBloque, deviceId, matrizDeUnosNxN, estrechezDeBorde);
              free(nombreNuevoDirSec);
              cudaFree(MC_imag_actual);
              cudaFree(MC_real_actual);
            }
          }
          cudaFree(vectorDeNumItera);
          cudaFree(vectorDeAnchos_EnDeltaU);
          cudaFree(vectorDeAnchos_Real);
      }
      cudaFree(cantLineasPorArchivo);
      cudaFree(largoDeNombres);
      for(int i=0; i<cantArchivos; i++)
        free(&(nombres[i][0]));
      free(nombres);
}

void calImagenesADistintasCompresiones_Rect(double inicioIntervalo, double finIntervalo, int cantParamEvaInfo, char nombreDirPrin[], double ancho, int maxIter, double tol, double* u, double* v, double* w, double* visi_parteImaginaria, double* visi_parteReal, double delta_u, double delta_v, double* matrizDeUnos, long cantVisi, long N, double* matrizDeUnosTamN, double estrechezDeBorde, int tamBloque, int numGPU)
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
  double* paramEvaInfo = linspace(inicioIntervalo/100.0, finIntervalo/100.0, cantParamEvaInfo);


  // ############### CALCULO DE MU Y MV - CREACION DE DIRECTORIO SEGUNDARIO  ##############
  printf("...Comenzando calculo de MV...\n");
  double* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  printf("Calculo de MV completado.\n");

  printf("...Comenzando calculo de MU...\n");
  double* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, ancho, matrizDeUnos, tamBloque, numGPU);
  printf("Calculo de MU completado.\n");


  // ############### MINIMIZACION DE COEFS, PARTE IMAGINARIA  ##############
  printf("...Comenzando minimizacion de coeficientes parte imaginaria...\n");
  double* MC_imag = minGradConjugado_MinCuadra(MV, MU, visi_parteImaginaria, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  printf("Proceso de minimizacion de coeficientes parte imaginaria terminado.\n");


  // ############### MINIMIZACION DE COEFS, PARTE REAL  ##############
  printf("...Comenzando minimizacion de coeficientes parte real...\n");
  double* MC_real = minGradConjugado_MinCuadra(MV, MU, visi_parteReal, w, cantVisi, N, matrizDeUnosTamN, maxIter, tol, tamBloque, numGPU);
  printf("Proceso de minimizacion de coeficientes parte real terminado.\n");


  double* MV_AF = calcularMV_Rect_estFourier(ancho, N, delta_v, matrizDeUnos, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);
  double* MU_AF = calcularMV_Rect_estFourier(ancho, N, delta_u, matrizDeUnos, estrechezDeBorde, matrizDeUnosTamN, tamBloque, numGPU);


  double* MC_comp_imag;
  cudaMallocManaged(&MC_comp_imag,N*N*sizeof(double));
  cudaMemset(MC_comp_imag, 0, N*N*sizeof(double));
  double* MC_comp_real;
  cudaMallocManaged(&MC_comp_real,N*N*sizeof(double));
  cudaMemset(MC_comp_real, 0, N*N*sizeof(double));

  long largo = N * N;
  double* MC_img_cuadrado;
  cudaMallocManaged(&MC_img_cuadrado, N*N*sizeof(double));
  double* MC_modulo;
  cudaMallocManaged(&MC_modulo, N*N*sizeof(double));
  hadamardProduct(MC_imag, N, N, MC_imag, MC_img_cuadrado, tamBloque, numGPU);
  hadamardProduct(MC_real, N, N, MC_real, MC_modulo, tamBloque, numGPU);
  combinacionLinealMatrices(1.0, MC_img_cuadrado, N, N, 1.0, MC_modulo, tamBloque, numGPU);
  cudaFree(MC_img_cuadrado);
  af::array MC_modulo_GPU(N*N, MC_modulo);
  cudaFree(MC_modulo);
  af::array MC_modulo_indicesOrde_GPU(N*N);
  af::array MC_modulo_Orde_GPU(N*N);
  af::sort(MC_modulo_Orde_GPU, MC_modulo_indicesOrde_GPU, MC_modulo_GPU, 0, false);
  double total = af::sum<double>(MC_modulo_GPU);
  MC_modulo_Orde_GPU = MC_modulo_Orde_GPU/total;
  af::eval(MC_modulo_Orde_GPU);
  af::eval(MC_modulo_indicesOrde_GPU);
  af::sync();
  double* auxiliar_MC_modulo_Orde_GPU = MC_modulo_Orde_GPU.device<double>();
  double* auxiliar_MC_modulo_indicesOrde_GPU = MC_modulo_indicesOrde_GPU.device<double>();
  double* coefsNormalizados = (double*) malloc(largo*sizeof(double));
  cudaMemcpy(coefsNormalizados, auxiliar_MC_modulo_Orde_GPU, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  int* MC_modulo_indicesOrde_CPU = (int*) malloc(largo*sizeof(int));
  cudaMemcpy(MC_modulo_indicesOrde_CPU, auxiliar_MC_modulo_indicesOrde_GPU, N*N*sizeof(int), cudaMemcpyDeviceToHost);
  MC_modulo_Orde_GPU.unlock();
  MC_modulo_GPU.unlock();
  MC_modulo_indicesOrde_GPU.unlock();

  long cantCoefsParaCota = 0;
  double sumador = 0.0;
  double* cantCoefsPorParametro = (double*) malloc(sizeof(double)*cantParamEvaInfo);
  double* cantidadPorcentualDeCoefs = linspace(1.0, largo, largo);
  combinacionLinealMatrices(0.0, cantidadPorcentualDeCoefs, largo, 1, 1.0/N, cantidadPorcentualDeCoefs, tamBloque, numGPU);
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
    double* indicesATomar_CPU = (double*) malloc(cantCoefsParaCota*sizeof(double));
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
    double* auxiliar_MC_imag_GPU = MC_imag_GPU.device<double>();
    double* auxiliar_MC_real_GPU = MC_real_GPU.device<double>();
    cudaMemcpy(MC_comp_imag, auxiliar_MC_imag_GPU, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    MC_imag_GPU.unlock();
    cudaMemcpy(MC_comp_real, auxiliar_MC_real_GPU, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    MC_real_GPU.unlock();
    double* estimacionFourier_compre_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_imag, N, N, MU_AF, numGPU);
    double* estimacionFourier_compre_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_comp_real, N, N, MU_AF, numGPU);
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
  double* porcenReal = (double*) malloc(sizeof(double)*largoVector);
  double* vector = (double*) malloc(sizeof(double)*largoVector);
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

  double* vectorFiltrado = (double*) calloc(largoVector, sizeof(double));
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

  double* listaDeMetricas = (double*) malloc(sizeof(double)*largoVector);
  double* primeraRecta_subListaDeX = (double*) calloc(largoVector, sizeof(double));
  double* primeraRecta_subListaDeY = (double*) calloc(largoVector, sizeof(double));
  double* segundaRecta_subListaDeX = (double*) calloc(largoVector, sizeof(double));
  double* segundaRecta_subListaDeY = (double*) calloc(largoVector, sizeof(double));
  memcpy(segundaRecta_subListaDeX, porcenReal, sizeof(double)*largoVector);
  memcpy(segundaRecta_subListaDeY, vectorFiltrado, sizeof(double)*largoVector);
  primeraRecta_subListaDeX[0] = porcenReal[0];
  primeraRecta_subListaDeY[0] = vectorFiltrado[0];
  for(int i=1; i<largoVector-1; i++)
  {
      primeraRecta_subListaDeX[i] = porcenReal[i];
      primeraRecta_subListaDeY[i] = vectorFiltrado[i];
      double pendienteDePrimeraRecta = calPendiente(primeraRecta_subListaDeX, i+1, primeraRecta_subListaDeY);
      // printf("En la iteracion %d la pendienteDePrimeraRecta es %f\n", i, pendienteDePrimeraRecta);
      segundaRecta_subListaDeX[i-1] = 0.0;
      segundaRecta_subListaDeY[i-1] = 0.0;
      double pendienteDeSegundaRecta = calPendiente(&(segundaRecta_subListaDeX[i]), largoVector-i, &(segundaRecta_subListaDeY[i]));
      // printf("En la iteracion %d la pendienteDeSegundaRecta es %f\n", i, pendienteDeSegundaRecta);
      listaDeMetricas[i] = -1 * pendienteDeSegundaRecta/pendienteDePrimeraRecta;
      printf("%f\n", listaDeMetricas[i]);
  }
  free(primeraRecta_subListaDeX);
  free(primeraRecta_subListaDeY);
  free(segundaRecta_subListaDeX);
  free(segundaRecta_subListaDeY);
}

// void multMatrices3(double* matrizA, long M, long K, double* matrizB, long N, double* matrizD, double* matrizC)
// {
//   cusparseHandle_t handle;	cusparseCreate(&handle);
// 	double* A;	cudaMalloc(&A, M * K * sizeof(double));
// 	double* B;	cudaMalloc(&B, K * N * sizeof(double));
//   double* C;	cudaMalloc(&C, M * N * sizeof(double));
//   double* D;	cudaMalloc(&D, M * N * sizeof(double));
// 	cudaMemcpy(A, matrizA, M * K * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(B, matrizB, K * N * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(D, matrizD, M * N * sizeof(double), cudaMemcpyHostToDevice);
//
// 	// --- Descriptor for sparse matrix A
// 	cusparseMatDescr_t descrA;
//   cusparseCreateMatDescr(&descrA);
// 	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
//
// 	// --- Descriptor for sparse matrix B
// 	cusparseMatDescr_t descrB;
//   cusparseCreateMatDescr(&descrB);
// 	cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE);
//
// 	// --- Descriptor for sparse matrix C
// 	cusparseMatDescr_t descrC;
//   cusparseCreateMatDescr(&descrC);
// 	cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
// 	cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE);
//
//   // --- Descriptor for sparse matrix D
//   cusparseMatDescr_t descrD;
//   cusparseCreateMatDescr(&descrD);
//   cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL);
//   cusparseSetMatIndexBase(descrD, CUSPARSE_INDEX_BASE_ONE);
//
//   int nnzA = 0;							// --- Number of nonzero elements in dense matrix A
//   int nnzB = 0;							// --- Number of nonzero elements in dense matrix B
//   int nnzD = 0;							// --- Number of nonzero elements in dense matrix B
//
//   const int lda = M;						// --- Leading dimension of dense matrix
//   const int ldb = K;						// --- Leading dimension of dense matrix
//   const int ldd = M;						// --- Leading dimension of dense matrix
//
//   // --- Device side number of nonzero elements per row of matrix A
//   int *nnzPerVectorA; 	cudaMalloc(&nnzPerVectorA, M * sizeof(*nnzPerVectorA));
//   cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, K, descrA, A, lda, nnzPerVectorA, &nnzA);
//
//   // --- Device side number of nonzero elements per row of matrix B
//   int *nnzPerVectorB; 	cudaMalloc(&nnzPerVectorB, K * sizeof(*nnzPerVectorB));
//   cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, K, N, descrB, B, ldb, nnzPerVectorB, &nnzB);
//
//   // --- Device side number of nonzero elements per row of matrix B
//   int *nnzPerVectorD; 	cudaMalloc(&nnzPerVectorD, M * sizeof(*nnzPerVectorD));
//   cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descrD, D, ldd, nnzPerVectorD, &nnzD);
//
//   // --- Device side sparse matrix
// 	double *csrValA; cudaMalloc(&csrValA, nnzA * sizeof(*csrValA));
//   double *csrValB; cudaMalloc(&csrValB, nnzB * sizeof(*csrValB));
//   double *csrValD; cudaMalloc(&csrValD, nnzD * sizeof(*csrValD));
//
//   int *csrRowPtrA; cudaMalloc(&csrRowPtrA, (M + 1) * sizeof(*csrRowPtrA));
// 	int *csrRowPtrB; cudaMalloc(&csrRowPtrB, (K + 1) * sizeof(*csrRowPtrB));
//   int *csrRowPtrD; cudaMalloc(&csrRowPtrD, (M + 1) * sizeof(*csrRowPtrD));
// 	int *csrColIndA; cudaMalloc(&csrColIndA, nnzA * sizeof(*csrColIndA));
//   int *csrColIndB; cudaMalloc(&csrColIndB, nnzB * sizeof(*csrColIndB));
//   int *csrColIndD; cudaMalloc(&csrColIndD, nnzD * sizeof(*csrColIndD));
//
//   cusparseSdense2csr(handle, M, K, descrA, A, lda, nnzPerVectorA, csrValA, csrRowPtrA, csrColIndA);
// 	cusparseSdense2csr(handle, K, N, descrB, B, ldb, nnzPerVectorB, csrValB, csrRowPtrB, csrColIndB);
//   cusparseSdense2csr(handle, M, N, descrD, D, ldd, nnzPerVectorD, csrValD, csrRowPtrD, csrColIndD);
//
//
//   // assume matrices A, B and D are ready.
//   int baseC, nnzC;
//   csrgemm2Info_t info = NULL;
//   size_t bufferSize;
//   void *buffer = NULL;
//   // nnzTotalDevHostPtr points to host memory
//   int *nnzTotalDevHostPtr = &nnzC;
//   double alpha = 1.0;
//   double beta  = 1.0;
//   cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
//
//   // step 1: create an opaque structure
//   cusparseCreateCsrgemm2Info(&info);
//
//   // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
//   cusparseScsrgemm2_bufferSizeExt(handle, M, N, K, &alpha,
//       descrA, nnzA, csrRowPtrA, csrColIndA,
//       descrB, nnzB, csrRowPtrB, csrColIndB,
//       &beta,
//       descrD, nnzD, csrRowPtrD, csrColIndD,
//       info,
//       &bufferSize);
//   cudaMalloc(&buffer, bufferSize);
//
//   // step 3: compute csrRowPtrC
//   int *csrRowPtrC;
//   cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(M+1));
//   cusparseXcsrgemm2Nnz(handle, M, N, K,
//           descrA, nnzA, csrRowPtrA, csrColIndA,
//           descrB, nnzB, csrRowPtrB, csrColIndB,
//           descrD, nnzD, csrRowPtrD, csrColIndD,
//           descrC, csrRowPtrC, nnzTotalDevHostPtr,
//           info, buffer);
//   if (NULL != nnzTotalDevHostPtr)
//   {
//       nnzC = *nnzTotalDevHostPtr;
//   }
//   else
//   {
//       cudaMemcpy(&nnzC, csrRowPtrC+M, sizeof(int), cudaMemcpyDeviceToHost);
//       cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
//       nnzC -= baseC;
//   }
//
//   // step 4: finish sparsity pattern and value of C
//   int *csrColIndC;
//   cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
//   double *csrValC;
//   cudaMalloc((void**)&csrValC, sizeof(double)*nnzC);
//   // Remark: set csrValC to null if only sparsity pattern is required.
//   cusparseScsrgemm2(handle, M, N, K, &alpha,
//           descrA, nnzA, csrValA, csrRowPtrA, csrColIndA,
//           descrB, nnzB, csrValB, csrRowPtrB, csrColIndB,
//           &beta,
//           descrD, nnzD, csrValD, csrRowPtrD, csrColIndD,
//           descrC, csrValC, csrRowPtrC, csrColIndC,
//           info, buffer);
//
//   cusparseScsr2dense(handle, M, N, descrC, csrValC, csrRowPtrC, csrColIndC, C, M);
//   cudaMemcpy(matrizC, C, M * N * sizeof(double), cudaMemcpyDeviceToHost);
//   // step 5: destroy the opaque structure
//   cusparseDestroyCsrgemm2Info(info);
// }

void reciclador_calCompSegunAncho_Normal_escritura(double* MC_imag, double* MC_real, char nombreDirPrin[], char* nombreDirSec, char nombreDirTer[], double ancho, double ancho_enDeltaU, int iterActual, double* u, double* v, double* w, double delta_u, double delta_v, long cantVisi, long N, double* matrizDeUnosTamN, int tamBloque, int numGPU)
{
  double inicioPorcenCompre = 0.0;
  double terminoPorcenCompre = 0.2;
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
  double* MV = calcularMV_Normal(v, delta_v, cantVisi, N, ancho, tamBloque, numGPU);
  tiempoCalculoMV = clock() - tiempoCalculoMV;
  double tiempoTotalCalculoMV = ((double)tiempoCalculoMV)/CLOCKS_PER_SEC;
  printf("Calculo de MV completado.\n");

   printf("...Comenzando calculo de MU...\n");
  clock_t tiempoCalculoMU;
  tiempoCalculoMU = clock();
  double* MU = calcularMV_Normal(u, delta_u, cantVisi, N, ancho, tamBloque, numGPU);
  tiempoCalculoMU = clock() - tiempoCalculoMU;
  double tiempoTotalCalculoMU = ((double)tiempoCalculoMU)/CLOCKS_PER_SEC;
  printf("Calculo de MU completado.\n");

  char* rutaADirecSec = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*sizeof(char)+sizeof(char)*3);
  strcpy(rutaADirecSec, nombreDirPrin);
  strcat(rutaADirecSec, "/");
  strcat(rutaADirecSec, nombreDirSec);
  // printf("Llegue: %s\n", rutaADirecSec);
  if(mkdir(rutaADirecSec, 0777) == -1)
  {
      printf("ERROR: No se pudo crear subdirectorio.\n");
      printf("%s\n", rutaADirecSec);
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  strcat(rutaADirecSec, "/");


   // ############### CALCULO NIVEL DE INFORMACION ##############
  clock_t tiempoInfo;
  tiempoInfo = clock();
  double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, numGPU);
  tiempoInfo = clock() - tiempoInfo;
  double tiempoTotalInfo = ((double)tiempoInfo)/CLOCKS_PER_SEC;
  cudaFree(MU);
  cudaFree(MV);


   // ############### RECONSTRUCCION DEL PLANO GRILLEADO Y ALMACENAMIENTO DE LA RECONSTRUCCION DE LA IMAGEN ##############
  char* nombreArchivoReconsImg = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreDirSec)*strlen(nombreArReconsImg)*sizeof(char)+sizeof(char)*3);
  strcpy(nombreArchivoReconsImg, rutaADirecSec);
  strcat(nombreArchivoReconsImg, nombreArReconsImg);
  clock_t tiempoCalculoMV_AF;
  tiempoCalculoMV_AF = clock();
  double* MV_AF = calcularMV_Normal_estFourier(ancho, N, delta_v, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMV_AF = clock() - tiempoCalculoMV_AF;
  double tiempoTotalCalculoMV_AF = ((double)tiempoCalculoMV_AF)/CLOCKS_PER_SEC;
  clock_t tiempoCalculoMU_AF;
  tiempoCalculoMU_AF = clock();
  double* MU_AF = calcularMV_Normal_estFourier(ancho, N, delta_u, matrizDeUnosTamN, tamBloque, numGPU);
  tiempoCalculoMU_AF = clock() - tiempoCalculoMU_AF;
  double tiempoTotalCalculoMU_AF = ((double)tiempoCalculoMU_AF)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartImag;
  tiempoReconsFourierPartImag = clock();
  double* estimacionFourier_ParteImag = estimacionDePlanoDeFourier(MV_AF, N, N, MC_imag, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartImag = clock() - tiempoReconsFourierPartImag;
  double tiempoTotalReconsFourierPartImag = ((double)tiempoReconsFourierPartImag)/CLOCKS_PER_SEC;
  clock_t tiempoReconsFourierPartReal;
  tiempoReconsFourierPartReal = clock();
  double* estimacionFourier_ParteReal = estimacionDePlanoDeFourier(MV_AF, N, N, MC_real, N, N, MU_AF, numGPU);
  tiempoReconsFourierPartReal = clock() - tiempoReconsFourierPartReal;
  double tiempoTotalReconsFourierPartReal = ((double)tiempoReconsFourierPartReal)/CLOCKS_PER_SEC;
  clock_t tiempoReconsTransInver;
  tiempoReconsTransInver = clock();
  escribirTransformadaInversaFourier2D(estimacionFourier_ParteImag, estimacionFourier_ParteReal, N, nombreArchivoReconsImg);
  tiempoReconsTransInver = clock() - tiempoReconsTransInver;
  double tiempoTotalReconsTransInver = ((double)tiempoReconsTransInver)/CLOCKS_PER_SEC;
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
  printf("...Comenzando calculo de compresiones...\n");
  clock_t tiempoCompresion;
  tiempoCompresion = clock();
  double* datosDelMin = calPSNRDeDistintasCompresiones(inicioPorcenCompre, cantPorcen, rutaADirecSec, rutaADirecTer, nombreArReconsCompreImg, MC_imag, MC_real, MV_AF, MU_AF, N, tamBloque, numGPU);
  tiempoCompresion = clock() - tiempoCompresion;
  double tiempoTotalCompresion = ((double)tiempoCompresion)/CLOCKS_PER_SEC;
  printf("Proceso de calculo de compresiones terminado.\n");
  free(rutaADirecTer);
  char* nombreArchivoInfoComp = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoCompresion)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoComp, nombreDirPrin);
  strcat(nombreArchivoInfoComp, "/");
  strcat(nombreArchivoInfoComp, nombreArInfoCompresion);
  double nivelDeCompresion = 1.0 - datosDelMin[4] * 1.0 / N*N;
  #pragma omp critical
  {
    FILE* archivo = fopen(nombreArchivoInfoComp, "a");
    fprintf(archivo, "%d %f %.12f %.12e %.12e %.12f %.12e %.12e %.12e %.12e %ld %.12e %.12e %.12e\n", iterActual, ancho_enDeltaU, ancho, medidasDeInfo[0], medidasDeInfo[1], nivelDeCompresion, datosDelMin[0], datosDelMin[1], datosDelMin[2], datosDelMin[3], (long) datosDelMin[4], datosDelMin[5], datosDelMin[6], datosDelMin[7]);
    fclose(archivo);
  }
  free(nombreArchivoInfoComp);
  free(medidasDeInfo);
  free(datosDelMin);
  cudaFree(MU_AF);
  cudaFree(MV_AF);

   // ############### ESCRITURA DE ARCHIVO CON TIEMPOS DE EJECUCION ##############
  char* nombreArchivoInfoTiemposEjecu = (char*) malloc(strlen(nombreDirPrin)*strlen(nombreArInfoTiemposEjecu)*sizeof(char)+sizeof(char)*2);
  strcpy(nombreArchivoInfoTiemposEjecu, nombreDirPrin);
  strcat(nombreArchivoInfoTiemposEjecu, "/");
  strcat(nombreArchivoInfoTiemposEjecu, nombreArInfoTiemposEjecu);
  #pragma omp critical
  {
    FILE* archivoInfoTiemposEjecu = fopen(nombreArchivoInfoTiemposEjecu, "a");
    fprintf(archivoInfoTiemposEjecu, "%d %.12f %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n", iterActual, ancho, tiempoTotalCalculoMV, tiempoTotalCalculoMU, tiempoTotalInfo, tiempoTotalCompresion, tiempoTotalCalculoMV_AF, tiempoTotalCalculoMU_AF, tiempoTotalReconsFourierPartImag, tiempoTotalReconsFourierPartReal, tiempoTotalReconsTransInver);
    fclose(archivoInfoTiemposEjecu);
  }
  free(nombreArchivoInfoTiemposEjecu);
  free(rutaADirecSec);
}

void reciclador_calculoDeInfoCompre_BaseNormal(char nombreArchivoConNombres[], char nombreDirPrin[], char nombreDirSec[], char nombreDirTer[], char nombreArchivoCoefs_imag[], char nombreArchivoCoefs_real[], int cantArchivos, int flag_multiThread, char nombreArchivoInfoCompre[], int maxIter, double* u, double* v, double* w, double delta_u, double delta_v, long cantVisi, long N, double* matrizDeUnosTamN, int tamBloque)
{
  if(mkdir(nombreDirPrin, 0777) == -1)
  {
      printf("ERROR: El directorio EXISTE, PELIGRO DE SOBREESCRITURA, por favor eliga otro nombre de directorio.\n");
      printf("PROGRAMA ABORTADO.\n");
      exit(0);
  }
  else
      printf("Directorio creado.\n");
  int* cantLineasPorArchivo;
  cudaMallocManaged(&cantLineasPorArchivo, cantArchivos*sizeof(int));
  int* largoDeNombres;
  cudaMallocManaged(&largoDeNombres, cantArchivos*sizeof(int));
  char** nombres = lecturaDeNombresDeArchivo(nombreArchivoConNombres, cantLineasPorArchivo, cantArchivos, largoDeNombres);
  int contadorDeIteraciones = -1;
  for(int numArchi=0; numArchi<cantArchivos; numArchi++)
  {
      int cantLineasActual = cantLineasPorArchivo[numArchi];
      char* nombreActual = (char*) malloc(sizeof(char)*(largoDeNombres[numArchi])+sizeof(char)*strlen(nombreArchivoInfoCompre)+sizeof(char)*5);
      strcpy(nombreActual, &(nombres[numArchi][0]));
      strcat(nombreActual, "/");
      strcat(nombreActual, nombreArchivoInfoCompre);

      // printf("%s\n", nombreActual);
      int* vectorDeNumItera;
      cudaMallocManaged(&vectorDeNumItera, cantLineasActual*sizeof(int));
      double* vectorDeAnchos_EnDeltaU;
      cudaMallocManaged(&vectorDeAnchos_EnDeltaU, cantLineasActual*sizeof(double));
      double* vectorDeAnchos_Real;
      cudaMallocManaged(&vectorDeAnchos_Real, cantLineasActual*sizeof(double));
      lecturaDeArchivo_infoCompre(nombreActual, vectorDeNumItera, vectorDeAnchos_EnDeltaU, vectorDeAnchos_Real, cantLineasActual);
      free(nombreActual);
      #pragma omp parallel num_threads(1)
      {
        #pragma omp for schedule(dynamic, 1)
        for(int numLinea=0; numLinea<cantLineasActual; numLinea++)
        {
          int numIteracion;
          if(flag_multiThread)
          {
            #pragma omp critical
            {
              contadorDeIteraciones++;
              numIteracion = contadorDeIteraciones;
            }
          }
          else
          {
            contadorDeIteraciones++;
          }
          int numIteAUsar = vectorDeNumItera[numLinea];
          char* numComoString_iterALeer = numAString(&numIteAUsar);
          sprintf(numComoString_iterALeer, "%d", numIteAUsar);
          char* nombreDirSecLeerCoefs = (char*) malloc(sizeof(char)*strlen(nombreDirSec)+sizeof(char)*strlen(numComoString_iterALeer)+sizeof(char)*5);
          strcpy(nombreDirSecLeerCoefs, nombreDirSec);
          strcat(nombreDirSecLeerCoefs, numComoString_iterALeer);
          free(numComoString_iterALeer);

          char* nombreArchivoActualCoefs_imag = (char*) malloc(sizeof(char)*(largoDeNombres[numArchi])+sizeof(char)*strlen(nombreArchivoCoefs_imag)+sizeof(char)*10);
          strcpy(nombreArchivoActualCoefs_imag, &(nombres[numArchi][0]));
          strcat(nombreArchivoActualCoefs_imag, "/");
          strcat(nombreArchivoActualCoefs_imag, nombreDirSecLeerCoefs);
          strcat(nombreArchivoActualCoefs_imag, "/");
          strcat(nombreArchivoActualCoefs_imag, nombreArchivoCoefs_imag);
          char* nombreArchivoActualCoefs_real = (char*) malloc(sizeof(char)*(largoDeNombres[numArchi])+sizeof(char)*strlen(nombreArchivoCoefs_real)+sizeof(char)*10);
          strcpy(nombreArchivoActualCoefs_real, &(nombres[numArchi][0]));
          strcat(nombreArchivoActualCoefs_real, "/");
          strcat(nombreArchivoActualCoefs_real, nombreDirSecLeerCoefs);
          strcat(nombreArchivoActualCoefs_real, "/");
          strcat(nombreArchivoActualCoefs_real, nombreArchivoCoefs_real);
          free(nombreDirSecLeerCoefs);
          double* MC_imag_actual, *MC_real_actual ;
          cudaMallocManaged(&MC_imag_actual, N*N*sizeof(double));
          cudaMallocManaged(&MC_real_actual, N*N*sizeof(double));
          #pragma omp critical
          {
            lecturaDeTXTDeCoefs(nombreArchivoActualCoefs_imag, MC_imag_actual, N, N);
            lecturaDeTXTDeCoefs(nombreArchivoActualCoefs_real, MC_real_actual, N, N);
          }
          free(nombreArchivoActualCoefs_imag);
          free(nombreArchivoActualCoefs_real);
          char* numComoString_iterAEscribir = numAString(&numIteracion);
          sprintf(numComoString_iterAEscribir, "%d", numIteracion);
          char* nombreNuevoDirSec = (char*) malloc(sizeof(char)*strlen(nombreDirSec)+sizeof(char)*strlen(numComoString_iterAEscribir)+sizeof(char)*5);
          strcpy(nombreNuevoDirSec, nombreDirSec);
          strcat(nombreNuevoDirSec, numComoString_iterAEscribir);
          free(numComoString_iterAEscribir);
          int thread_id = omp_get_thread_num();
          int deviceId = thread_id%4;
          cudaSetDevice(deviceId);
          af::setDevice(deviceId);
          reciclador_calCompSegunAncho_Normal_escritura(MC_imag_actual, MC_real_actual, nombreDirPrin, nombreNuevoDirSec, nombreDirTer, vectorDeAnchos_Real[numLinea], vectorDeAnchos_EnDeltaU[numLinea], numIteracion, u, v, w, delta_u, delta_v, cantVisi, N, matrizDeUnosTamN, tamBloque, deviceId);
          free(nombreNuevoDirSec);
          cudaFree(MC_imag_actual);
          cudaFree(MC_real_actual);
        }
      }
      cudaFree(vectorDeNumItera);
      cudaFree(vectorDeAnchos_EnDeltaU);
      cudaFree(vectorDeAnchos_Real);
  }
  cudaFree(cantLineasPorArchivo);
  cudaFree(largoDeNombres);
  for(int i=0; i<cantArchivos; i++)
    free(&(nombres[i][0]));
  free(nombres);
  exit(-1);
}

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

double* leerImagenFITS(char filename[])
{
    fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
    int status,  nfound, anynull;
    long naxes[2], fpixel, npixels;

    double nullval;

    status = 0;

    if (fits_open_file(&fptr, filename, READONLY, &status))
         printerror_cfitsio(status);

    /* read the NAXIS1 and NAXIS2 keyword to get image size */
    if (fits_read_keys_lng(fptr, "NAXIS", 1, 2, naxes, &nfound, &status))
         printerror_cfitsio(status);

    npixels  = naxes[0] * naxes[1];         /* number of pixels in the image */
    fpixel   = 1;
    nullval  = 0;                /* don't check for null values in the image */

    double* imagen;
    cudaMallocManaged(&imagen, npixels*sizeof(double));
    if (fits_read_img(fptr, TDOUBLE, fpixel, npixels, &nullval, imagen, &anynull, &status))
         printerror_cfitsio(status);
    if (fits_close_file(fptr, &status))
         printerror_cfitsio(status);
    return imagen;
}

void normalizarImagenFITS(double* imagen, int N)
{
  double epsilon = 0.1;
  af::array imagen_GPU(N*N, imagen);
  double maximo = af::max<double>(imagen_GPU);
  double minimo = af::min<double>(imagen_GPU);
  imagen_GPU = (imagen_GPU - minimo + epsilon)/(maximo - minimo + epsilon);
  af::eval(imagen_GPU);
  af::sync();
  double* auxiliar_imagen_GPU = imagen_GPU.device<double>();
  cudaMemcpy(imagen, auxiliar_imagen_GPU, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  imagen_GPU.unlock();
}

double compararImagenesFITS(char* nombreImagen, char* nombreIdeal, int N)
{
  double* imagen = leerImagenFITS(nombreImagen);
  double* imagenIdeal = leerImagenFITS(nombreIdeal);
  normalizarImagenFITS(imagen, N);
  normalizarImagenFITS(imagenIdeal, N);

  double* resultados;
  cudaMallocManaged(&resultados, N*N*sizeof(double));

  af::array imagen_GPU(N*N, imagen);
  af::array imagenIdeal_GPU(N*N, imagenIdeal);
  af::array resultados_GPU(N*N);
  resultados_GPU = abs(imagenIdeal_GPU - imagen_GPU);
  double total = af::sum<double>(resultados_GPU);
  af::eval(resultados_GPU);
  af::sync();
  double* auxiliar_resultados_GPU = resultados_GPU.device<double>();
  cudaMemcpy(resultados, auxiliar_resultados_GPU, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  imagen_GPU.unlock();
  imagenIdeal_GPU.unlock();
  resultados_GPU.unlock();

  // double *resultados;
  // cudaMallocManaged(&resultados, N*N*sizeof(double));
  // for(int i=0; i<N*N; i++)
  // {
  //   double numerador = abs(imagenIdeal[i] - imagen[i]);
  //   // int denominador = abs(imagenIdeal[i]);
  //   resultados[i] = numerador;
  // }
  // double total = 0.0;
  // for(int i=0; i<N*N; i++)
  // {
  //   total += resultados[i];
  // }

  // for(int i=0; i<N*N; i++)
  // {
  //   if(resultados[i] > 0.0)
  //     printf("%f\n", resultados[i]);
  // }

  // printf("El resultado es %f\n", total);
  return total;
}

void calcularListaDeMAPE(char nombreArchivoSalida[], char nombreDirectorioPrincipal_ImagenObt[], char nombreDirectorioSecundario_ImagenObt[], char nombre_ImagenObt[], char nombreDirectorio_ImagenIdeal[], char nombre_ImagenIdeal[], int cantCarpetas, int N)
{
  char* nombreDir_ImagenIdeal = (char*) malloc(sizeof(char)*(strlen(nombreDirectorio_ImagenIdeal)+strlen(nombre_ImagenIdeal)+3));
  strcpy(nombreDir_ImagenIdeal, nombreDirectorio_ImagenIdeal);
  strcat(nombreDir_ImagenIdeal, "/");
  strcat(nombreDir_ImagenIdeal, nombre_ImagenIdeal);
  FILE* archivoAEscribir = fopen(nombreArchivoSalida, "w");
  for(int i=0; i<cantCarpetas; i++)
  {
    char* numComoString = numAString(&i);
    sprintf(numComoString, "%d", i);
    char* nombreDir_ImagenObt = (char*) malloc(sizeof(char)*(strlen(nombreDirectorioPrincipal_ImagenObt)+strlen(nombreDirectorioSecundario_ImagenObt)+strlen(numComoString)+strlen(nombre_ImagenObt)+3));
    strcpy(nombreDir_ImagenObt, nombreDirectorioPrincipal_ImagenObt);
    strcat(nombreDir_ImagenObt, "/");
    strcat(nombreDir_ImagenObt, nombreDirectorioSecundario_ImagenObt);
    strcat(nombreDir_ImagenObt, numComoString);
    strcat(nombreDir_ImagenObt, "/");
    strcat(nombreDir_ImagenObt, nombre_ImagenObt);
    double errorActual = compararImagenesFITS(nombreDir_ImagenObt, nombreDir_ImagenIdeal, N);
    fprintf(archivoAEscribir, "%f\n", errorActual);
    free(numComoString);
    free(nombreDir_ImagenObt);
  }
  fclose(archivoAEscribir);
}


int main()
{
  // --csv --export-profile --print-gpu-summary

  // /usr/lib/cuda-10.0/bin/nvprof --csv --log-file %p /home/rarmijo/calCompreInfo
  // /usr/lib/cuda-10.0/bin/nvprof --csv --log-file %p /home/rarmijo/otro

  // char** nombresDeArchivos = (char**) malloc(sizeof(char*)*cantArchivos);
  // for(int i=0; i<cantArchivos; i++)
  // {
  //   nombreArchivos[i] = (char*) malloc(sizeof(char)*100);
  // }
  //
  // int largo = 150;
  // int* vectorDeNumItera;
  // cudaMallocManaged(&vectorDeNumItera, largo*sizeof(int));
  // double* vectorDeAnchos;
  // cudaMallocManaged(&vectorDeAnchos, largo*sizeof(double));
  // lecturaDeArchivo_infoCompre(nombreArchivito, vectorDeNumItera, vectorDeAnchos, largo);
  // double* MC_imag;
  // cudaMallocManaged(&MC_imag, N*N*sizeof(double));
  // double* MC_real;
  // cudaMallocManaged(&MC_real, N*N*sizeof(double));
  // char nombreUbiCoefsImag = "/home/rarmijo/experi_hd142_linspacevariable_Rect_visi153/ite34/coefs_imag.txt";
  // lecturaDeTXTDeCoefs(nombreUbiCoefsImag, MC_imag, N, N);
  // char nombreUbiCoefsReal = "/home/rarmijo/experi_hd142_linspacevariable_Rect_visi153/ite34/coefs_real.txt";
  // lecturaDeTXTDeCoefs(nombreUbiCoefsReal, MC_real, N, N);

  // 1/(1+(x/a)^2)
  // int largo = 5;
  // double *x;
  // cudaMallocManaged(&x, largo*sizeof(double));
  // x[0] = 1;
  // x[1] = 2;
  // x[2] = 6;
  // x[3] = 10;
  // x[4] = 20;
  // double* salida = hermite(x, largo, 3, 1024, 0);
  // imprimirMatrizColumna(salida, largo, 4);
  // exit(1);

  // PARAMETROS GENERALES
  // long cantVisi = 1000;
  // long inicio = 0;
  // long fin = 1000;

  // long cantVisi = 30000;
  // long inicio = 0;
  // long fin = 30000;

  // for i in {0..31}; do cp /disk2/tmp/experi_hd142_Normal_linspacevariable_visi153/ite$i/curvaPSNRSuavizada.txt /disk2/tmp/curvas_peque/curva$i.txt; done
  // sudo scp rarmijo@158.170.35.147:/disk2/tmp/curvas_peque/* /home/yoyisaurio/Desktop/pequeno/

  // for i in {0..950}; do cp /disk2/tmp/experi_hd142_Normal_reciclado_visi800/ite$i/curvaPSNRSuavizada.txt /disk2/tmp/curvas_experi_hd142_Normal_reciclado_visi800/curva$i.txt; done
  // sudo scp rarmijo@158.170.35.147:/disk2/tmp/curvas_experi_hd142_Normal_reciclado_visi800/* /home/yoyisaurio/Desktop/curvas_experi_hd142_Normal_reciclado_visi800/
  // sudo scp rarmijo@158.170.35.147:/disk2/tmp/experi_hd142_Normal_reciclado_visi800/infoCompre.txt /home/yoyisaurio/Desktop/curvas_experi_hd142_Normal_reciclado_visi800/

  // for i in {0..1200}; do cp /disk2/tmp/experi_hd142_Rect_reciclado_visi800/ite$i/curvaPSNRSuavizada.txt /disk2/tmp/curvas_experi_hd142_Rect_reciclado_visi800/curva$i.txt; done
  // sudo scp rarmijo@158.170.35.147:/disk2/tmp/curvas_experi_hd142_Rect_reciclado_visi800/* /home/yoyisaurio/Desktop/curvas_experi_hd142_Rect_reciclado_visi800/
  // sudo scp rarmijo@158.170.35.147:/disk2/tmp/experi_hd142_Rect_reciclado_visi800/infoCompre.txt /home/yoyisaurio/Desktop/curvas_experi_hd142_Rect_reciclado_visi800/

  // for i in {0..320}; do cp /disk2/tmp/experi_hd142_Normal_visi800_parte1/ite$i/curvaPSNRSuavizada.txt /disk2/tmp/curvas_hd142_normal/curva$i.txt; done
  // for i in {320..799}; do cp /disk2/tmp/experi_hd142_Normal_visi800_parte2/ite$i/curvaPSNRSuavizada.txt /disk2/tmp/curvas_hd142_normal/curva$i.txt; done
  // sudo scp /disk2/tmp/curvas_hd142_normal/* /home/yoyisaurio/Desktop/hd142_normal/


  // cp /disk1/rarmijo/experi_hd142_InvCuadra_linspacevariable_visi400y800/infoCompre.txt /disk1/rarmijo/curvas_experi_hd142_InvCuadra_linspacevariable_visi400y800/
  // sudo scp rarmijo@158.170.35.139:/disk1/rarmijo/curvas_experi_hd142_InvCuadra_linspacevariable_visi400y800/infoCompre.txt /home/yoyisaurio/Desktop/curvas_experi_hd142_InvCuadra_linspacevariable_visi400y800/
  // for i in {0..1202}; do cp /disk1/rarmijo/experi_hd142_InvCuadra_linspacevariable_visi400y800/ite$i/curvaPSNRSuavizada.txt /disk1/rarmijo/curvas_experi_hd142_InvCuadra_linspacevariable_visi400y800/curva$i.txt; done
  // sudo scp /disk1/rarmijo/curvas_experi_hd142_InvCuadra_linspacevariable_visi400y800/* /home/yoyisaurio/Desktop/curvas_experi_hd142_InvCuadra_linspacevariable_visi400y800/


  // cp /disk1/rarmijo/experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/infoCompre.txt /disk1/rarmijo/curvas_experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/
  // sudo scp rarmijo@158.170.35.139:/disk1/rarmijo/curvas_experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/infoCompre.txt /home/yoyisaurio/Desktop/curvas_experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/
  // for i in {0..1202}; do cp /disk1/rarmijo/experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/ite$i/curvaPSNRSuavizada.txt /disk1/rarmijo/curvas_experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/curva$i.txt; done
  // sudo scp /disk1/rarmijo/curvas_experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/* /home/yoyisaurio/Desktop/curvas_experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/


  // cp /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800/infoCompre.txt /srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Normal_linspacevariable_visi400y800/
  // sudo scp rarmijo@158.170.35.147:/srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Normal_linspacevariable_visi400y800/infoCompre.txt /home/yoyisaurio/Desktop/curvas_experi_hd142_b9_model_Normal_linspacevariable_visi400y800/
  // for i in {0..1202}; do cp /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800/ite$i/curvaPSNRSuavizada.txt /srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Normal_linspacevariable_visi400y800/curva$i.txt; done
  // sudo scp /srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Normal_linspacevariable_visi400y800/* /home/yoyisaurio/Desktop/curvas_experi_hd142_b9_model_Normal_linspacevariable_visi400y800/


  // cp /srv/nas01/rarmijo/experi_hd142_b9_model_Rect_linspacevariable_visi400y800/infoCompre.txt /srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Rect_linspacevariable_visi400y800/
  // sudo scp rarmijo@158.170.35.147:/srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Rect_linspacevariable_visi400y800/infoCompre.txt /home/yoyisaurio/Desktop/curvas_experi_hd142_b9_model_Rect_linspacevariable_visi400y800/
  // for i in {0..1202}; do cp /srv/nas01/rarmijo/experi_hd142_b9_model_Rect_linspacevariable_visi400y800/ite$i/curvaPSNRSuavizada.txt /srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Rect_linspacevariable_visi400y800/curva$i.txt; done
  // sudo scp rarmijo@158.170.35.147:/srv/nas01/rarmijo/curvas_experi_hd142_b9_model_Rect_linspacevariable_visi400y800/* /home/yoyisaurio/Desktop/curvas_experi_hd142_b9_model_Normal_linspacevariable_visi400y800/

  // scp -r rarmijo@158.170.35.139:/disk1/rarmijo/experi_hd142_b9_model_InvCuadra_linspacevariable_visi400y800/* /srv/nas01/rarmijo/resultados/experi_hd142_b9_model_InvCuadra


  // for i in {0..999}; do if [ ! -d ite$i ]; then echo "ite$i"; fi; done

  // for i in {0..950}; do if [ ! -d ite$i ]; then echo "ite$i"; fi; done

  // for i in {0..950}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/coefs_imag.txt ]; then echo "ite$i"; fi ; done
  // for i in {0..950}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/coefs_real.txt ]; then echo "ite$i"; fi ; done

  // for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/curvaPSNRSuavizada.txt ]; then echo "ite$i"; fi ; done

  // for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/idsCoefsCeroAporte.txt ]; then echo "ite$i"; fi ; done

  // for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/mejorTradeOffPSNRCompre.txt ]; then echo "ite$i"; fi ; done

  //for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/datosDeIteLegible.txt ]; then echo "ite$i"; fi ; done

  //for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/datosDeIte.txt ]; then echo "ite$i"; fi ; done

  //for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/minCoefs_imag.txt ]; then echo "ite$i"; fi ; done

  //for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/minCoefs_real.txt ]; then echo "ite$i"; fi ; done

  //for i in {0..900}; do if [ ! -e /srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4/ite$i/reconsImg.fit ]; then echo "ite$i"; fi ; done


  //for i in {680..1202}; do mv /srv/nas01/rarmijo/experi_hd142_b9_model_Rect_linspacevariable_visi400y800_parte2/ite$i /srv/nas01/rarmijo/experi_hd142_b9_model_Rect_linspacevariable_visi400y800 ; done

  // for i in {680..693}; do rm -r /srv/nas01/rarmijo/experi_hd142_b9_model_Rect_linspacevariable_visi400y800/ite$i; done


  long cantVisi = 15034;
  long inicio = 0;
  long fin = 15034;

  // long cantVisi = 1000;
  // long inicio = 0;
  // long fin = 1000;

  int tamBloque = 1024;
  int N = 512;
  // long N = 1600; //HLTau_B6cont.calavg.tav300s
  int maxIter = 1;

  double tolGrad = 1E-12;

  double delta_x = 0.02;
  // double delta_x = 0.005; //HLTau_B6cont.calavg.tav300s
  // double delta_x = 0.03; //co65
  double delta_x_rad = (delta_x * M_PI)/648000.0;
  double delta_u = 1.0/(N*delta_x_rad);
  double delta_v = 1.0/(N*delta_x_rad);

  //PARAMETROS PARTICULARES DE BASE RECT
  double estrechezDeBorde = 1000.0;

  // double frecuencia;
  // double *u, *v, *w, *visi_parteImaginaria, *visi_parteReal;
  // cudaMallocManaged(&u, cantVisi*sizeof(double));
  // cudaMallocManaged(&v, cantVisi*sizeof(double));
  // cudaMallocManaged(&w, cantVisi*sizeof(double));
  // cudaMallocManaged(&visi_parteImaginaria, cantVisi*sizeof(double));
  // cudaMallocManaged(&visi_parteReal, cantVisi*sizeof(double));
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

  // // ########### PC-LAB ##############
  // char nombreArchivo[] = "/home/rarmijo/hd142_b9cont_self_tav.ms";
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

  // // ########### BEAM ##############
  // char nombreArchivo[] = "./hd142_b9cont_self_tav.ms";
  // char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // // ########### BEAM ##############
  // char nombreArchivo[] = "/home/rarmijo/HLTau_Band6_CalibratedData/HLTau_B6cont.calavg";
  // char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // ########### BEAM ##############
  char nombreArchivo[] = "./hd142_b9_model";
  char comandoCasaconScript[] = "casa -c ./deMSaTXT.py";

  // char* comandoScriptMSaTXT = (char*) malloc(strlen(comandoCasaconScript)*strlen(nombreArchivo)*sizeof(char)+sizeof(char)*3);
  // strcpy(comandoScriptMSaTXT, comandoCasaconScript);
  // strcat(comandoScriptMSaTXT, " ");
  // strcat(comandoScriptMSaTXT, nombreArchivo);
  // system(comandoScriptMSaTXT);
  // free(comandoScriptMSaTXT);


  lectCantVisi(nombreArchivo, &cantVisi);
  double *u, *v, *w, *visi_parteImaginaria, *visi_parteReal;
  cudaMallocManaged(&u, cantVisi*sizeof(double));
  cudaMallocManaged(&v, cantVisi*sizeof(double));
  cudaMallocManaged(&w, cantVisi*sizeof(double));
  cudaMallocManaged(&visi_parteImaginaria, cantVisi*sizeof(double));
  cudaMallocManaged(&visi_parteReal, cantVisi*sizeof(double));
  lectDeTXTcreadoDesdeMS(nombreArchivo, u, v, w, visi_parteImaginaria, visi_parteReal);
  // lectDeTXTcreadoDesdeMSConLimite(nombreArchivo, u, v, w, visi_parteImaginaria, visi_parteReal, inicio, fin, cantVisi);

  double* matrizDeUnos, *matrizDeUnosTamN, *matrizDeUnosNxN;
  cudaMallocManaged(&matrizDeUnos, cantVisi*N*sizeof(double));
  for(long i=0; i<(cantVisi*N); i++)
  {
    matrizDeUnos[i] = 1.0;
  }
  cudaMallocManaged(&matrizDeUnosTamN, N*sizeof(double));
  for(long i=0; i<N; i++)
  {
    matrizDeUnosTamN[i] = 1.0;
  }
  cudaMallocManaged(&matrizDeUnosNxN, N*N*sizeof(double));
  for(long i=0; i<N*N; i++)
  {
    matrizDeUnosNxN[i] = 1.0;
  }

  // int cantParamEvaInfo = 23;
  // double limitesDeZonas[] = {0.001, 2.0, 3.0};
  // double cantPuntosPorZona[] = {10, 10};
  // int cantPtosLimites = 3;
  // double* paramEvaInfo_pre = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  // double* paramEvaInfo;
  // cudaMallocManaged(&paramEvaInfo, cantParamEvaInfo*sizeof(double));
  // combinacionLinealMatrices(delta_u, paramEvaInfo_pre, cantParamEvaInfo, 1, 0.0, paramEvaInfo, tamBloque, 0);
  // FILE* archivito = fopen("/home/rarmijo/info_hd142_rect.txt", "w");
  // for(int i=0; i<cantParamEvaInfo; i++)
  // {
  //   double* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, paramEvaInfo[i], matrizDeUnos, tamBloque, 0);
  //   double* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, paramEvaInfo[i], matrizDeUnos, tamBloque, 0);
  //   double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, tamBloque, 0);
  //   double medidaSumaDeLaDiagonal = medidasDeInfo[0];
  //   free(medidasDeInfo);
  //   cudaFree(MV);
  //   cudaFree(MU);
  //   fprintf(archivito, "%.12f %.12e\n", paramEvaInfo_pre[i], medidaSumaDeLaDiagonal);
  // }
  // cudaFree(paramEvaInfo_pre);
  // fclose(archivito);
  // exit(1);

  // int cantParamEvaInfo = 203;
  // double limitesDeZonas[] = {0.001, 2.0, 3.0};
  // double cantPuntosPorZona[] = {150, 50};
  // int cantPtosLimites = 3;
  // double* paramEvaInfo = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  // FILE* archivito = fopen("/home/rarmijo/Desktop/info_hd142_hermite.txt", "w");
  // int n = N-1;
  // double maxu = buscarMaximo(u, cantVisi);
  // double maxv = buscarMaximo(v, cantVisi);
  // double max_radius = maximoEntre2Numeros(maxu,maxv);
  // for(int i=0; i<cantParamEvaInfo; i++)
  // {
  //   double beta_u = paramEvaInfo[i]/max_radius;
  //   double K = beta_u * (sqrt(2*n+1)+1);
  //   double* x_samp = combinacionLinealMatrices_conretorno(K, u, cantVisi, 1, 0.0, u, tamBloque, 0);
  //   double* y_samp = combinacionLinealMatrices_conretorno(K, v, cantVisi, 1, 0.0, v, tamBloque, 0);
  //   double* MV = hermite(y_samp, cantVisi, n, tamBloque, 0);
  //   double* MU = hermite(x_samp, cantVisi, n, tamBloque, 0);
  //   cudaFree(x_samp);
  //   cudaFree(y_samp);
  //   double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, 1024, 0);
  //   double medidaSumaDeLaDiagonal = medidasDeInfo[0];
  //   free(medidasDeInfo);
  //   cudaFree(MV);
  //   cudaFree(MU);
  //   fprintf(archivito, "%.12f %.12e\n", paramEvaInfo[i], medidaSumaDeLaDiagonal);
  // }
  // fclose(archivito);
  // exit(1);


  // int cantParamEvaInfo = 203;
  // double limitesDeZonas[] = {0.001, 2.0, 3.0};
  // double cantPuntosPorZona[] = {150, 50};
  // int cantPtosLimites = 3;
  // double* paramEvaInfo_pre = linspaceNoEquiespaciadoMitad(limitesDeZonas, cantPuntosPorZona, cantPtosLimites);
  // double* paramEvaInfo;
  // cudaMallocManaged(&paramEvaInfo, cantParamEvaInfo*sizeof(double));
  // combinacionLinealMatrices(delta_u, paramEvaInfo_pre, cantParamEvaInfo, 1, 0.0, paramEvaInfo, tamBloque, 0);
  // FILE* archivito = fopen("/home/rarmijo/Desktop/info_hd142_normal.txt", "w");
  // for(int i=0; i<cantParamEvaInfo; i++)
  // {
  //   double* MV = calcularMV_Normal(v, delta_v, cantVisi, N, paramEvaInfo[i], 1024, 0);
  //   double* MU = calcularMV_Normal(u, delta_u, cantVisi, N, paramEvaInfo[i], 1024, 0);
  //   double* medidasDeInfo = calInfoFisherDiag(MV, cantVisi, N, MU, w, 1024, 0);
  //   double medidaSumaDeLaDiagonal = medidasDeInfo[0];
  //   free(medidasDeInfo);
  //   cudaFree(MV);
  //   cudaFree(MU);
  //   fprintf(archivito, "%.12f %.12e\n", paramEvaInfo_pre[i], medidaSumaDeLaDiagonal);
  // }
  // fclose(archivito);
  // cudaFree(paramEvaInfo_pre);
  // exit(1);


  // double* resultado1, * resultado2;
  // cudaMallocManaged(&resultado1, cantVisi*N*sizeof(double));
  // cudaMallocManaged(&resultado2, cantVisi*N*sizeof(double));
  //
  // printf("...Comenzando calculo de MV...\n");
  // double* MV = calcularMV_Rect(v, delta_v, cantVisi, N, estrechezDeBorde, delta_u, matrizDeUnos);
  // printf("Calculo de MV completado.\n");
  //
  // printf("...Comenzando calculo de MU...\n");
  // double* MU = calcularMV_Rect(u, delta_u, cantVisi, N, estrechezDeBorde, delta_u, matrizDeUnos);
  // printf("Calculo de MU completado.\n");
  //
  //
  // clock_t tiempoCalculo1;
  // tiempoCalculo1 = clock();
  // multMatrices(MV, cantVisi, N, MU, N, resultado1);
  // tiempoCalculo1 = clock() - tiempoCalculo1;
  // double tiempoTotalCalculo1 = ((double)tiempoCalculo1)/CLOCKS_PER_SEC;
  //
  // clock_t tiempoCalculo2;
  // tiempoCalculo2 = clock();
  // // multMatrices3(MV, cantVisi, N, MU, N, resultado2);
  // tiempoCalculo2 = clock() - tiempoCalculo2;
  // double tiempoTotalCalculo2 = ((double)tiempoCalculo2)/CLOCKS_PER_SEC;
  //
  // printf("La multiplicacion con cublas tomo %.12e segundos mientras que la dispersa tomo %.12e segundos.\n", tiempoTotalCalculo2, tiempoTotalCalculo1);
  //
  // exit(1);

  double cotaEnergia = 0.99;
  // char nombreDirPrin[] = "probando";
  // char nombreDirPrin[] = "/disk2/tmp/probando";
  // char nombreDirPrin[] = "double_calCompresion_baseNormal_cota";
  // char nombreDirPrin[] = "experi_hd142_Normal_visi800";
  // char nombreDirPrin[] = "/disk2/tmp/experi_hd142_linspacevariable_Rect_visi350";
  // char nombreDirPrin[] = "/disk2/tmp/experi_HLTau_Band6_CalibratedData_linspacevariable_Rect_UnMillon";
  // char nombreDirPrin[] = "/disk2/tmp/experi_hd142_Normal_linspacevariable_visi153";
  // char nombreDirPrin[] = "/disk2/tmp/experi_hd142_Hermite_linspacevariable_visi153";
  // char nombreDirPrin[] = "/disk2/tmp/experi_hd142_Normal_linspacevariable_visi400";
  // char nombreDirPrin[] = "/disk2/tmp/experi_hd142_Rect_linspacevariable_visi500";
  // char nombreDirPrin[] = "/srv/nas01/rarmijo/experi_hd142_InvCuadra_linspacevariable_visi400y800";
  // char nombreDirPrin[] = "/srv/nas01/rarmijo/experi_hd142_InvCuadra_linspacevariable_visi400y800_3";
  // char nombreDirPrin[] = "/srv/nas01/rarmijo/experi_hd142_b9_model_Normal_linspacevariable_visi400y800_4";  char nombreDirPrin[] = "/srv/nas01/rarmijo/experi_hd142_InvCuadra_linspacevariable_visi400y800";
  // char nombreDirPrin[] = "/srv/nas01/rarmijo/experimento_l1_hd142_b9_model_descartandonada";
  // char nombreDirSec[] = "ite";
  // char nombreDirTer[] = "compresiones";
  // char nombreArchivoTiempo[] = "tiempo.txt";
  // // int cantParamEvaInfo = 153;
  // int cantParamEvaInfo = 1203;
  // double inicioIntervalo = 0.001;
  // double finIntervalo = 3.0;
  // double tolGolden = 1E-12;
  // int iterActual = 0;

  // char nombreDirPrin[] = "/srv/nas01/rarmijo/resultados/experi_hd142_b9_model_Rect_reciclado_absEnImagen";
  char nombreDirPrin[] = "/srv/nas01/rarmijo/resultados/experi_hd142_b9_model_Rect_l1";
  char nombreDirSec[] = "ite";
  char nombreDirTer[] = "compresiones";
  char nombreArchivoTiempo[] = "tiempo.txt";
  // char nombreArchivoConNombres[] = "nombredeprueba.txt";
  char nombreArchivoConNombres[] = "datosAJuntar_Rect.txt";
  char nombreArchivoCoefs_imag[] = "coefs_imag.txt";
  char nombreArchivoCoefs_real[] = "coefs_real.txt";
  int cantArchivos = 1;
  int flag_multiThread = 1;
  char nombreArchivoInfoCompre[] = "infoCompre.txt";

  // clock_t t;
  // t = clock();
  double iStart = cpuSecond();
  // calculoDeInfoCompre_BaseHermite(nombreArchivo, maxIter, tolGrad, tolGolden, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, cotaEnergia, nombreDirPrin, nombreDirSec, nombreDirTer, cantParamEvaInfo, inicioIntervalo, finIntervalo, matrizDeUnosTamN, estrechezDeBorde, tamBloque);
  // calculoDeInfoCompre_BaseNormal(nombreArchivo, maxIter, tolGrad, tolGolden, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, cotaEnergia, nombreDirPrin, nombreDirSec, nombreDirTer, cantParamEvaInfo, inicioIntervalo, finIntervalo, matrizDeUnosTamN, estrechezDeBorde, tamBloque);
  // calculoDeInfoCompre_BaseInvCuadra(nombreArchivo, maxIter, tolGrad, tolGolden, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, cantVisi, N, cotaEnergia, nombreDirPrin, nombreDirSec, nombreDirTer, cantParamEvaInfo, inicioIntervalo, finIntervalo, matrizDeUnosTamN, estrechezDeBorde, tamBloque);
  // calculoDeInfoCompre_BaseRect(nombreArchivo, maxIter, tolGrad, tolGolden, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, cotaEnergia, nombreDirPrin, nombreDirSec, nombreDirTer, cantParamEvaInfo, inicioIntervalo, finIntervalo, matrizDeUnosTamN, estrechezDeBorde, tamBloque, matrizDeUnosNxN);

  calculoDeInfoCompre_l1_BaseRect(nombreArchivo, maxIter, tolGrad, u, v, w, visi_parteImaginaria, visi_parteReal, delta_u, delta_v, matrizDeUnos, cantVisi, N, cotaEnergia, nombreDirPrin, nombreDirSec, nombreDirTer, matrizDeUnosTamN, estrechezDeBorde, tamBloque, matrizDeUnosNxN);

  // reciclador_calculoDeInfoCompre_BaseNormal(nombreArchivoConNombres, nombreDirPrin, nombreDirSec, nombreDirTer, nombreArchivoCoefs_imag, nombreArchivoCoefs_real, cantArchivos, flag_multiThread, nombreArchivoInfoCompre, maxIter, u, v, w, delta_u, delta_v, cantVisi, N, matrizDeUnosTamN, tamBloque);
  // reciclador_calculoDeInfoCompre_BaseRect(nombreArchivoConNombres, nombreDirPrin, nombreDirSec, nombreDirTer, nombreArchivoCoefs_imag, nombreArchivoCoefs_real, cantArchivos, flag_multiThread, nombreArchivoInfoCompre, maxIter, u, v, w, delta_u, delta_v, matrizDeUnos, cantVisi, N, matrizDeUnosTamN, tamBloque, matrizDeUnosNxN, estrechezDeBorde);
  double time_taken = cpuSecond() - iStart;
  // t = clock() - t;
  // double time_taken = ((double)t)/CLOCKS_PER_SEC;
  char* nombreCompletoArchivoTiempo = (char*) malloc(sizeof(char)*(strlen(nombreArchivoTiempo)+strlen(nombreDirPrin))+sizeof(char)*3);
  strcpy(nombreCompletoArchivoTiempo, nombreDirPrin);
  strcat(nombreCompletoArchivoTiempo, "/");
  strcat(nombreCompletoArchivoTiempo, nombreArchivoTiempo);
  FILE* archivoTiempo = fopen(nombreCompletoArchivoTiempo, "w");
  double minutitos = time_taken/60;
  double horas = minutitos/60;
  printf("El tiempo de ejecucion fue %.12e segundos o %.12e minutos o %.12e horas.\n", time_taken, minutitos, horas);
  fprintf(archivoTiempo, "El tiempo de ejecucion fue %.12e segundos o %.12e minutos o %.12e horas.\n", time_taken, minutitos, horas);
  fclose(archivoTiempo);
  free(nombreCompletoArchivoTiempo);

  cudaFree(u);
  cudaFree(v);
  cudaFree(w);
  cudaFree(visi_parteImaginaria);
  cudaFree(visi_parteReal);
  cudaFree(matrizDeUnos);
  cudaFree(matrizDeUnosTamN);
}


// int main
// {
//   // char nombreArchivoSalida[] = "/home/rarmijo/listaDeErrores_experi_hd142_b9_model_Rect.txt";
//   // char nombreDirectorioPrincipal_ImagenObt[] = "/srv/nas01/rarmijo/resultados/experi_hd142_b9_model_Rect";
//
//   // char nombreArchivoSalida[] = "/home/rarmijo/listaDeErrores_experi_hd142_Rect.txt";
//   // char nombreDirectorioPrincipal_ImagenObt[] = "/srv/nas01/rarmijo/resultados/experi_hd142_Rect";
//
//   // char nombreArchivoSalida[] = "/home/rarmijo/listaDeErrores_experi_hd142_b9_model_Normal.txt";
//   // char nombreDirectorioPrincipal_ImagenObt[] = "/srv/nas01/rarmijo/resultados/experi_hd142_b9_model_Normal";
//
//   // char nombreArchivoSalida[] = "/home/rarmijo/listaDeErrores_experi_hd142_Normal.txt";
//   // char nombreDirectorioPrincipal_ImagenObt[] = "/srv/nas01/rarmijo/resultados/experi_hd142_Normal";
//
//   // char nombreArchivoSalida[] = "/home/rarmijo/listaDeErrores_experi_hd142_b9_model_InvCuadra.txt";
//   // char nombreDirectorioPrincipal_ImagenObt[] = "/srv/nas01/rarmijo/resultados/experi_hd142_b9_model_InvCuadra";
//
//   char nombreArchivoSalida[] = "/home/rarmijo/listaDeErrores_experi_hd142_InvCuadra.txt";
//   char nombreDirectorioPrincipal_ImagenObt[] = "/srv/nas01/rarmijo/resultados/experi_hd142_InvCuadra";
//
//   char nombreDirectorioSecundario_ImagenObt[] = "ite";
//   char nombre_ImagenObt[] = "reconsImg.fit";
//   char nombreDirectorio_ImagenIdeal[] = "/home/rarmijo/imagenesAComparar";
//   char nombre_ImagenIdeal[] = "imagenIdeal.fits";
//   int cantCarpetas = 1202;
//   calcularListaDeMAPE(nombreArchivoSalida, nombreDirectorioPrincipal_ImagenObt, nombreDirectorioSecundario_ImagenObt, nombre_ImagenObt, nombreDirectorio_ImagenIdeal, nombre_ImagenIdeal, cantCarpetas, 512);
//   exit(0);
//
//   // char nombreArchivoActualCoefs_imag[] = "/srv/nas01/rarmijo/experi_hd142_b9_model_Rect_linspacevariable_visi400y800/ite195/coefs_imag.txt";
//   // double* MC;
//   // cudaMallocManaged(&MC, 512*512*sizeof(double));
//   // lecturaDeTXTDeCoefs(nombreArchivoActualCoefs_imag, MC, 512, 512);
//   // for(int i=0; i<512*512; i++)
//   // {
//   //   if(MC[i] != 0.0)
//   //     printf("%.12e\n", MC[i]);
//   // }
//   // exit(-1);
//
//   char nombreImagen1[] = "/home/rarmijo/imagenesAComparar/cerocomacuarentayochodeltau_ite195_experi_hd142_b9_model_Rect_linspacevariable_visi400y800.fit";
//   char nombreImagenIdeal[] = "/home/rarmijo/imagenesAComparar/imagenIdeal.fits";
//   compararImagenesFITS(nombreImagen1, nombreImagenIdeal, 512);
//
//   char nombreImagen2[] = "/home/rarmijo/imagenesAComparar/cerocomanuevedeltau_ite363_experi_hd142_b9_model_Rect_linspacevariable_visi400y800.fit";
//   compararImagenesFITS(nombreImagen2, nombreImagenIdeal, 512);
//
//   char nombreImagen3[] = "/home/rarmijo/imagenesAComparar/undeltau_ite401_experi_hd142_b9_model_Rect_linspacevariable_visi400y800.fit";
//   compararImagenesFITS(nombreImagen3, nombreImagenIdeal, 512);
//   exit(0);
// }
