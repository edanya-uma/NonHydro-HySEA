#ifndef _ARISTA_COMMON_H_
#define _ARISTA_COMMON_H_

#include <stdio.h>
#include "Volumen_kernel.cu"
#include "Matriz.cu"

/****************************/
/* Procesamiento de aristas */
/****************************/

__device__ void cond_contorno(double &h0, double &H0, double3 &u0, double &h1, double &H1, double3 &u1, int frontera,
				double borde_sup, double borde_inf, double borde_izq, double borde_der)
{
	h1 = h0;
	H1 = H0;

	if (frontera == 4)
		u1.x = u0.x*borde_izq;
	else if (frontera == 2)
		u1.x = u0.x*borde_der;
	else if (frontera == 1)
		u1.x = u0.x*borde_inf;
	else if (frontera == 3)
		u1.x = u0.x*borde_sup;
}

__device__ void procesarArista(double cosa, double h0, double H0, double3 u0, double h1, double H1, double3 u1, double normal_x,
				double normal_y, double longitud, double delta_T, double2 *d_acumuladorh, double3 *d_acumuladorq, int pos_vol0,
				int pos_vol1, double epsilon_h, int frontera, double borde_sup, double borde_inf, double borde_izq, double borde_der)
{
	double hm, um;
	double max_autovalor;
	double aux;
	double sqrt_h0, sqrt_h1;
    double alfa0, alfa1;
    double SL, SR;
    double3 U0, U1;
    double hh0, HH0, hh1, HH1;
    double dhp,dhm,dqpx,dqmx,dqpy,dqmy,dqpz,dqmz,deta,deh,deq,vh,vq,q1,q0;

    U0.x = u0.x*normal_x+u0.y*normal_y;
    U0.y = u0.y*normal_x-u0.x*normal_y;
    U0.z = u0.z;

    U1.x = u1.x*normal_x+u1.y*normal_y;
    U1.y = u1.y*normal_x-u1.x*normal_y;
    U1.z = u1.z;

    if (frontera > 0) {
        cond_contorno(h0, H0, U0, h1, H1, U1, frontera, borde_sup, borde_inf, borde_izq, borde_der);
    }
    HH0 = H0/cosa;
    HH1 = H1/cosa;
    hh0 = h0/cosa;
    hh1 = h1/cosa;

    double Hm = min(HH0,HH1);
    hh1 = max(hh1-HH1+Hm,0.0);
    hh0 = max(hh0-HH0+Hm,0.0);
    deta = hh1-hh0;

    q0 = hh0*cosa*U0.x;
    q1 = hh1*cosa*U1.x;
    //deta /= cosa;
    
    hm = 0.5*(hh0+hh1);
    sqrt_h0 = sqrt(hh0);
    sqrt_h1 = sqrt(hh1);

    // Esto es para definir el polinomio de viscosidad
    um = (sqrt_h0*U0.x + sqrt_h1*U1.x)/(sqrt_h0 + sqrt_h1 + EPSILON);
    SL = um-sqrt(hm);
    SR = um+sqrt(hm);

    max_autovalor = max(fabs(SR),fabs(SL));
    max_autovalor = max(max_autovalor,fabs(U0.x)+sqrt_h0);
    max_autovalor = max(max_autovalor,fabs(U1.x)+sqrt_h1);

    aux = SR-SL;
    if (fabs(aux) > epsilon_h) {
        double aSR = fabs(SR);
        double aSL = fabs(SL);
        alfa0 = (SR*aSL - SL*aSR)/aux;
        alfa1 =(aSR-aSL)/aux;
    }
    else {
        alfa0 = max_autovalor;
        alfa1 = 0.0;
    }
    deh = q1-q0;
    deq = U1.x*q1 - U0.x*q0 + cosa*hm*deta;
    vh = alfa0*deta*cosa + alfa1*deh;
    vq = alfa0*deh + alfa1*deq;

    dhm = 0.5*(deh-vh)+q0;
    dhp = 0.5*(deh+vh)-q1;
    dqmx = 0.5*(deq-vq)+U0.x*q0;
    dqpx = 0.5*(deq+vq)-U1.x*q1;

    aux = -1.0*(dhm<0.0) + 1.0*(dhm>0.0);  // sgn(F)
    double ut = 0.5*(U0.y+U1.y) - 0.5*aux*(U1.y-U0.y);
    double uz = 0.5*(U0.z+U1.z) - 0.5*aux*(U1.z-U0.z);

    dqmy = dhm*ut;
    dqpy = -dqmy;
    dqmz = dhm*uz;    
    dqpz = -dqmz;

    if (max_autovalor < HVEL)
        max_autovalor += HVEL;

    aux = max_autovalor/longitud;
    double dd = delta_T/longitud;

    d_acumuladorh[pos_vol0].x -= dd*dhm; 
    d_acumuladorh[pos_vol0].y += aux;
    d_acumuladorq[pos_vol0].x -= dd*(normal_x*dqmx - normal_y*dqmy);
    d_acumuladorq[pos_vol0].y -= dd*(normal_y*dqmx + normal_x*dqmy);
    d_acumuladorq[pos_vol0].z -= dd*dqmz;
    
    if (pos_vol1 != -1) {
        d_acumuladorh[pos_vol1].x -= dd*dhp;
        d_acumuladorh[pos_vol1].y += aux;
        d_acumuladorq[pos_vol1].x -= dd*(normal_x*dqpx - normal_y*dqpy);
        d_acumuladorq[pos_vol1].y -= dd*(normal_y*dqpx + normal_x*dqpy);
	    d_acumuladorq[pos_vol1].z -= dd*dqpz;
    }
}


// ini borrar
__device__ double limitador(double p0m, double p0, double p1, double p1p, double S)
{
	double chi;
	double vs;
	double c0,c1;
	double aux;

	c0 = fabs(p1-p0);
	c1 = fabs((p1p-p1)*(S<0.0) + (p0-p0m)*(S>0.0) + 0.5*(p1p-p0m)*(S==0.0));
   
	if (c0 < EPSILON)
		chi = 1.0;
	else {
		aux = fabs((c1-c0)/(c1+c0));
		c1 = 0.5*(c1+c0)*(1.0-aux*aux*aux);
		vs = (c1/(c0+EPSILON))*(c0>c1) + (c0/(c1+EPSILON))*(c1>=c0);
		chi = vs*(1.0+vs)/(1.0+vs*vs);
	}

	return chi;
}

__device__ double limitador_monotono(double p0m, double p0, double p1, double p1p, double S)
{
	double chi;
	double vs;
	double c0,c1;
	double aux,sg;

	c0 = p1-p0;
	c1 = (p1p-p1)*(S<0.0) + (p0-p0m)*(S>0.0) + 0.5*(p1p-p0m)*(S==0.0);
	sg = 1.0*(c0*c1>=0.0);
	c0 = fabs(c0);
	c1 = fabs(c1);
   
	if (c0 < EPSILON)
		chi = 1.0;
	else {
		aux = fabs((c1-c0)/(c1+c0));
		c1 = 0.5*(c1+c0)*(1.0-aux*aux*aux);
		vs = (c1/(c0+EPSILON))*(c0>c1) + (c0/(c1+EPSILON))*(c1>=c0);
		chi = sg*vs*(1.0+vs)/(1.0+vs*vs);
	}

	return chi;
}

__device__ void tratamientoSecoMojado(double h0, double h1, double eta0, double eta1, double H0, double H1,
				double epsilon_h, double *q0n, double *q1n)
{
	if ((h0 < epsilon_h) && (-eta1 > H0))
		*q1n = 0.0;
	if ((h1 < epsilon_h) && (-eta0 > H1))
		*q0n = 0.0;
}

__device__ void positividad(double h0, double h1, double delta_T, double area0, double area1, double *Fmenos)
{
	double dt0;
	double factor = 0.25;
	double alpha;

	if (*Fmenos > 0.0) {
		dt0 = factor*h0*area0/((*Fmenos) + EPSILON);
	}
	else {
		dt0 = factor*h1*area1/(-(*Fmenos) + EPSILON);
	}
	if (delta_T > dt0) {
		alpha = dt0/(delta_T + EPSILON);
		*Fmenos *= alpha;
	}
}

__device__ void positividad_H(double h0, double h1, double cosPhi0, double cosPhi1, double delta_T,
				double area0, double area1, double *Fmenos)
{
	double dt0;
	double factor = 0.25;
	double alpha;

	if (*Fmenos > 0.0) {
		dt0 = factor*h0*cosPhi0*area0/((*Fmenos) + EPSILON);
	}
	else {
		dt0 = factor*h1*cosPhi1*area1/(-(*Fmenos) + EPSILON);
	}
	if (delta_T > dt0) {
		alpha = dt0/(delta_T + EPSILON);
		*Fmenos *= alpha;
	}
}
// fin borrar


/**************************/
/* Correcciones de flujos */
/**************************/

__device__ double atomicAdd2(double *address, double val)
{
	unsigned long long int *address_as_ull = (unsigned long long int *) address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);

	return __longlong_as_double(old);
}

__device__ void escribirCorreccionVol0DesdeNivel0Paso1(double *d_correccionEta, int pos_vol0,
				double Fmenos, double deltaTNivel0, double area0)
{
	double val = deltaTNivel0/area0;

	atomicAdd2(&(d_correccionEta[pos_vol0]), val*Fmenos);
}

__device__ void escribirCorreccionVol1DesdeNivel0Paso1(double *d_correccionEta, int pos_vol1,
				double Fmas, double deltaTNivel0, double area1)
{
	double val = deltaTNivel0/area1;

	atomicAdd2(&(d_correccionEta[pos_vol1]), val*Fmas);
}

__device__ void escribirCorreccionVol0DesdeNivel0Paso2(double2 *d_correccion_2, int pos_vol0,
				TVec2 *Fmenos2, double deltaTNivel0, double area0)
{
	double val = deltaTNivel0/area0;

	atomicAdd2(&(d_correccion_2[pos_vol0].x), val*Fmenos2->x);
	atomicAdd2(&(d_correccion_2[pos_vol0].y), val*Fmenos2->y);
}

__device__ void escribirCorreccionVol1DesdeNivel0Paso2(double2 *d_correccion_2, int pos_vol1,
				TVec2 *Fmas2, double deltaTNivel0, double area1)
{
	double val = deltaTNivel0/area1;

	atomicAdd2(&(d_correccion_2[pos_vol1].x), val*Fmas2->x);
	atomicAdd2(&(d_correccion_2[pos_vol1].y), val*Fmas2->y);
}

__device__ void escribirCorreccionDesdeNivel1Paso1(double *d_correccionEta, int pos_vol1, double Fmas,
												double deltaTNivel1, double area1Nivel0)
{
	double val = deltaTNivel1/area1Nivel0;

	atomicAdd2(&(d_correccionEta[pos_vol1]), -val*Fmas);
}

__device__ void escribirCorreccionDesdeNivel1Paso2(double2 *d_correccion_2, int pos_vol1, TVec2 *Fmas2,
												double deltaTNivel1, double area1Nivel0)
{
	double val = deltaTNivel1/area1Nivel0;

	atomicAdd2(&(d_correccion_2[pos_vol1].x), -val*Fmas2->x);
	atomicAdd2(&(d_correccion_2[pos_vol1].y), -val*Fmas2->y);
}

#endif
