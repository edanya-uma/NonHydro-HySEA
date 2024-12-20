#ifndef _CONSTANTES_H_
#define _CONSTANTES_H_

#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#define EPSILON           DBL_EPSILON
#define EARTH_RADIUS      6378136.6
#define MAX_LEVELS        6      // Maximum number of levels
#define MAX_GRIDS_LEVEL   200    // Maximum number of grids per level
#define MAX_FAULTS        5000   // Maximum number of Okada faults
#define GPUS_PER_NODE     8      // Number of GPUs per node
#define FILE_WRITING_MODE 1      // 1: synchronous, 2: asynchronous
#define SPONGE_SIZE       8      // Size of the sponge layer (>= 0)
#define MIN_H_METRICS     1e-3   // Minimum h in meters to compute the global metrics. If -1, epsilon_h is used
#define SEA_LEVEL         0.0    // Sea level in meters. Used in sponge layer
#define PREPROCESS_POINTS 0      // 1: remove repeated points in the time series, 0: do not remove
#define DEFLATE_LEVEL     5      // Level of compression of the NetCDF files (0-9)

#define OrdenTiempo 2      // 1-3
#define HVEL        1e-10
#define GAMMA       2.0    // 2.0: Jacques Sainte-Marie model; 1.5: Serre Green-Naghdi model
#define ERROR_NH    1e-5   // Tolerance for the resolution of the linear systems
#define ITER_MAX_NH 20000
#define BREAKING

// Algorithm to process the nested meshes
// 1: assign fluctuations of state values; 2: assign state values
#define NESTED_ALGORITHM  2

// Values of the initialization flag
#define SEA_SURFACE_FROM_FILE      0
#define OKADA_STANDARD             1
#define OKADA_STANDARD_FROM_FILE   2
#define OKADA_TRIANGULAR           3
#define OKADA_TRIANGULAR_FROM_FILE 4
#define DEFORMATION_FROM_FILE      5
#define DYNAMIC_DEFORMATION        6
#define GAUSSIAN                   7

// Values of the friction type flag
#define FIXED_FRICTION        0
#define VARIABLE_FRICTION_0   1
#define VARIABLE_FRICTION_ALL 2

using namespace std;

// Punteros a datos de un cluster
typedef struct {
	// inix, iniy: Coordenadas x e y de la submalla donde empiezan los datos del cluster,
	// sin contar los volúmenes de comunicación de los clusters superior e izquierdo
	int inix, iniy;
	int numVolx, numVoly;
	int numVolumenes;
	int iniLeerX, numVolsLeerX;
	int iniLeerY, numVolsLeerY;

	// Punteros que apuntan al principio de los volúmenes de comunicación
	// izquierdos y derechos del cluster
	double2 *datosVolumenesComClusterIzq_1, *datosVolumenesComClusterDer_1;
	double3 *datosVolumenesComClusterIzq_2, *datosVolumenesComClusterDer_2;
	// Punteros que apuntan al principio de los volúmenes de comunicación
	// superiores e inferiores del cluster
	double2 *datosVolumenesComClusterSup_1, *datosVolumenesComClusterInf_1;
	double3 *datosVolumenesComClusterSup_2, *datosVolumenesComClusterInf_2;
	// Punteros que apuntan al principio de los volúmenes de comunicación
	// del cluster adyacente izquierdo y derecho
	double2 *datosVolumenesComOtroClusterIzq_1, *datosVolumenesComOtroClusterDer_1;
	double3 *datosVolumenesComOtroClusterIzq_2, *datosVolumenesComOtroClusterDer_2;
	// Punteros que apuntan al principio de los volúmenes de comunicación
	// del cluster adyacente superior e inferior
	double2 *datosVolumenesComOtroClusterSup_1, *datosVolumenesComOtroClusterInf_1;
	double3 *datosVolumenesComOtroClusterSup_2, *datosVolumenesComOtroClusterInf_2;
	// Punteros que apuntan al principio de los vértices de comunicación
	// izquierdos y derechos del cluster
	double *datosVerticesComClusterIzq_P, *datosVerticesComClusterDer_P;
	// Punteros que apuntan al principio de los vértices de comunicación
	// superiores e inferiores del cluster
	double *datosVerticesComClusterSup_P, *datosVerticesComClusterInf_P;
	// Punteros que apuntan al principio de los vértices de comunicación
	// del cluster adyacente izquierdo y derecho
	double *datosVerticesComOtroClusterIzq_P, *datosVerticesComOtroClusterDer_P;
	// Punteros que apuntan al principio de los vértices de comunicación
	// del cluster adyacente superior e inferior
	double *datosVerticesComOtroClusterSup_P, *datosVerticesComOtroClusterInf_P;
} TDatosClusterCPU;

typedef struct {
	double2 *d_datosVolumenesComClusterIzq_1, *d_datosVolumenesComClusterDer_1;
	double3 *d_datosVolumenesComClusterIzq_2, *d_datosVolumenesComClusterDer_2;

	double2 *d_datosVolumenesComClusterSup_1, *d_datosVolumenesComClusterInf_1;
	double3 *d_datosVolumenesComClusterSup_2, *d_datosVolumenesComClusterInf_2;

	double2 *d_datosVolumenesComOtroClusterIzq_1, *d_datosVolumenesComOtroClusterDer_1;
	double3 *d_datosVolumenesComOtroClusterIzq_2, *d_datosVolumenesComOtroClusterDer_2;

	double2 *d_datosVolumenesComOtroClusterSup_1, *d_datosVolumenesComOtroClusterInf_1;
	double3 *d_datosVolumenesComOtroClusterSup_2, *d_datosVolumenesComOtroClusterInf_2;

	double *d_datosVerticesComClusterIzq_P, *d_datosVerticesComClusterDer_P;
	double *d_datosVerticesComClusterSup_P, *d_datosVerticesComClusterInf_P;
	double *d_datosVerticesComOtroClusterIzq_P, *d_datosVerticesComOtroClusterDer_P;
	double *d_datosVerticesComOtroClusterSup_P, *d_datosVerticesComOtroClusterInf_P;
} TDatosClusterGPU;

// Tamaño de un bloque en el procesamiento de las aristas
#define NUM_HEBRAS_ANCHO_ARI 8
#define NUM_HEBRAS_ALTO_ARI  8
// Ancho de un bloque en el procesamiento de las aristas de comunicación (alto==2)
#define NUM_HEBRAS_ANCHO_ARI_COM 32
// Alto de un bloque en el procesamiento de las aristas de comunicación (ancho==2)
#define NUM_HEBRAS_ALTO_ARI_COM 32
// Tamaño de un bloque en el cálculo del nuevo estado y el delta_T
#define NUM_HEBRAS_ANCHO_EST 16
#define NUM_HEBRAS_ALTO_EST  16
// Tamaño de un bloque en el guardado de puntos
#define NUM_HEBRAS_PUNTOS    128

// Redondea a / b al mayor entero más cercano
#define iDivUp(a,b)  (((a)%(b) != 0) ? ((a)/(b) + 1) : ((a)/(b)))

// Redondea a / b al menor entero más cercano
#define iDivDown(a,b)  ((a)/(b))

// Alinea a al mayor entero múltiplo de b más cercano
#define iAlignUp(a,b)  (((a)%(b) != 0) ? ((a)-(a) % (b)+(b)) : (a))

// Alinea a al menor entero múltiplo de b más cercano
#define iAlignDown(a,b)  ((a)-(a) % (b))

typedef struct {
	double dx, dy;
	double *vcos;
	double *vccos;
	double *vtan;
	double *longitud;
	double *latitud;
} tipoDatosSubmalla;

typedef struct {
	int eta;
	int eta_max;
	int velocidades;
	int velocidades_max;
	int modulo_velocidades;
	int modulo_velocidades_max;
	int modulo_caudales_max;
	int presion_no_hidrostatica;
	int flujo_momento;
	int flujo_momento_max;
	int tiempos_llegada;
} VariablesGuardado;

#endif
