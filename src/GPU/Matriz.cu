#ifndef _MATRIZ_H_
#define _MATRIZ_H_

#include "Constantes.hxx"

/***********************/
/* Definición de tipos */
/***********************/

// Tipo vector 3x1
typedef struct TVec3 {
	double x, y, z;
} TVec3;

// Tipo vector 2x1
typedef double2 TVec2;

/******************************/
/* Inicialización de vectores */
/******************************/

// Copia el vector in en out
__device__ void v_copy2(TVec2 *in, TVec2 *out) {
	out->x = in->x;
	out->y = in->y;
}

// Copia el vector in en out
__device__ void v_copy3(TVec3 *in, TVec3 *out) {
	out->x = in->x;
	out->y = in->y;
	out->z = in->z;
}

// Pone todos los elementos del vector v a cero
__device__ void v_zero2(TVec2 *v) {
	v->x = 0.0;
	v->y = 0.0;
}

// Pone todos los elementos del vector v a cero
__device__ void v_zero3(TVec3 *v) {
	v->x = 0.0;
	v->y = 0.0;
	v->z = 0.0;
}

/***************/
/* Operaciones */
/***************/

__device__ double pow4(double x) {
    return x*x*x*x;
}


#endif
