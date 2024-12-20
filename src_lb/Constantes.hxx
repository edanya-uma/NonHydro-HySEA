#ifndef _CONSTANTES_H_
#define _CONSTANTES_H_

#include <iostream>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>
#include <cstdint>

#define EPSILON           DBL_EPSILON
#define EARTH_RADIUS      6378136.6
#define MAX_LEVELS        6      // Maximum number of levels
#define MAX_GRIDS_LEVEL   200    // Maximum number of grids per level
#define MAX_FAULTS        5000   // Maximum number of Okada faults

// Value of gamma in the load balancing algorithm
// (0: cumulative ratio; 1: 2^level)
#define GAMMA 1

// Load balancing weights
// Mediterranean weights:
#define PESO_ARI_2S         0.90  // if ((h0 > hpos) && (h1 > hpos) && (! alfa_cero))
#define PESO_ARI_WAF_MOJADA 1.0   // else if ((h0 > EPSILON) || (h1 > EPSILON))
#define PESO_ARI_WAF_SECA   0.35  // else
#define PESO_VOL_MOJADO     0.27  // if (h > hpos)
#define PESO_VOL_SECO       0.28  // else

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

typedef struct {
	double x, y;
} double2;

typedef struct {
	int x, y;
} int2;

typedef struct int4 {
	int x, y, z, w;
} int4;

typedef struct {
	double dx, dy;
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
	int flujo_momento;
	int flujo_momento_max;
	int tiempos_llegada;
} VariablesGuardado;

#endif

