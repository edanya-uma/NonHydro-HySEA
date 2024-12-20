#ifndef _METRICS_H_
#define _METRICS_H

#include <stdio.h>
#include "Constantes.hxx"

// AMPLITUD MÁXIMA

__global__ void inicializarEta1MaximaNivelGPU(double2 *d_datosVolumenes_1, double *d_eta1_maxima,
				int num_volx, int num_voly, int iniySubmallaCluster, double *vccos, double hmin_metrics)
{
	double2 W;
    int j_global;
	int pos, pos_eta;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_eta = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;

		W = d_datosVolumenes_1[pos];
		d_eta1_maxima[pos_eta] = ((W.x < hmin_metrics*vccos[j_global]) ? -1e30 : W.x - W.y);
	}
}

__global__ void actualizarEta1MaximaNivelGPU(double2 *d_datosVolumenes_1, double *d_eta1_maxima,
				int num_volx, int num_voly, int iniySubmallaCluster, double *vccos, double hmin_metrics)
{
	double2 W;
	double val, eta1;
    int j_global;
	int pos, pos_eta;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_eta = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;

		eta1 = d_eta1_maxima[pos_eta];
		W = d_datosVolumenes_1[pos];
		val = W.x - W.y;
		if ((val > eta1) && (W.x > hmin_metrics*vccos[j_global]))
			d_eta1_maxima[pos_eta] = val;
	}
}

void inicializarEta1Maxima(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double **d_eta1MaximaNivel, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int *numSubmallasNivel, tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				inicializarEta1MaximaNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					d_datosVolumenesNivel_1[l]+pos, d_eta1MaximaNivel[l]+pos, nvx, nvy, datosClusterCPU[l][i].iniy,
					d_datosNivel[l][i].vccos, hmin_metrics);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

void actualizarEta1Maxima(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double **d_eta1MaximaNivel, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int *numSubmallasNivel, tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				actualizarEta1MaximaNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					d_datosVolumenesNivel_1[l]+pos, d_eta1MaximaNivel[l]+pos, nvx, nvy, datosClusterCPU[l][i].iniy,
					d_datosNivel[l][i].vccos, hmin_metrics);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

// VELOCIDADES MAXIMAS

__global__ void inicializarVelocidadesMaximasNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double3 *d_velocidades_maximas, int num_volx, int num_voly, int iniySubmallaCluster, double epsilon_h,
				double *vccos, double hmin_metrics)
{
	double h;
	double3 q, u;
    int j_global;
	double val_cos;
	int pos, pos_vel;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_vel = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		q = d_datosVolumenes_2[pos];
		if (h > hmin_metrics*val_cos) {
			u.x = (q.x/(h + EPSILON))*(h > HVEL*val_cos);
			u.y = (q.y/(h + EPSILON))*(h > HVEL*val_cos);
			if (no_hidros) {
				u.z = (q.z/(h + EPSILON))*(h > HVEL*val_cos);
			}
			d_velocidades_maximas[pos_vel] = u;
		}
		else {
			// Inicializamos a cero
			u.x = u.y = 0.0;
			d_velocidades_maximas[pos_vel] = u;
		}
	}
}

__global__ void actualizarVelocidadesMaximasNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double3 *d_velocidades_maximas, int num_volx, int num_voly, int iniySubmallaCluster, double epsilon_h,
				double *vccos, double hmin_metrics)
{
	double h;
	double3 q, u, val;
    int j_global;
	double val_cos;
	int pos, pos_vel;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_vel = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		if (h > hmin_metrics*val_cos) {
			q = d_datosVolumenes_2[pos];
			u = d_velocidades_maximas[pos_vel];
			val.x = max(u.x, (q.x/(h + EPSILON))*(h > HVEL*val_cos));
			val.y = max(u.y, (q.y/(h + EPSILON))*(h > HVEL*val_cos));
			if (no_hidros) {
				val.z = max(u.z, (q.z/(h + EPSILON))*(h > HVEL*val_cos));
			}
			d_velocidades_maximas[pos_vel] = val;
		}
	}
}

void inicializarVelocidadesMaximas(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double3 **d_velocidadesMaximasNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double epsilon_h,
			tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst,
			cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				inicializarVelocidadesMaximasNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_velocidadesMaximasNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, epsilon_h, d_datosNivel[l][i].vccos, hmin_metrics);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

void actualizarVelocidadesMaximas(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double3 **d_velocidadesMaximasNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double epsilon_h,
			tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst,
			cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				actualizarVelocidadesMaximasNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_velocidadesMaximasNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, epsilon_h, d_datosNivel[l][i].vccos, hmin_metrics);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

// MODULO DE VELOCIDAD MAXIMA

__global__ void inicializarModuloVelocidadMaximaNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double *d_velocidad_maxima, int num_volx, int num_voly, int iniySubmallaCluster, double epsilon_h,
				double *vccos, double hmin_metrics, double H, double U, double T)
{
	double h, ux, uy, uz;
	double u = 0.0;
	double3 q;
    int j_global;
	double val_cos;
	int pos, pos_vel;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_vel = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		if (h > hmin_metrics*val_cos) {
			q = d_datosVolumenes_2[pos];
			if (no_hidros) {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					uz = (q.z/(h + EPSILON))*H/T;
					u = sqrt(ux*ux + uy*uy + uz*uz);
				}
			}
			else {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					u = sqrt(ux*ux + uy*uy);
				}
			}
			d_velocidad_maxima[pos_vel] = u;
		}
		else {
			// Inicializamos a cero
			d_velocidad_maxima[pos_vel] = 0.0;
		}
	}
}

__global__ void actualizarModuloVelocidadMaximaNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double *d_velocidad_maxima, int num_volx, int num_voly, int iniySubmallaCluster, double epsilon_h,
				double *vccos, double hmin_metrics, double H, double U, double T)
{
	double h, ux, uy, uz, u;
	double val = 0.0;
	double3 q;
    int j_global;
	double val_cos;
	int pos, pos_vel;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_vel = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		if (h > hmin_metrics*val_cos) {
			q = d_datosVolumenes_2[pos];
			u = d_velocidad_maxima[pos_vel];
			if (no_hidros) {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					uz = (q.z/(h + EPSILON))*H/T;
					val = sqrt(ux*ux + uy*uy + uz*uz);
				}
			}
			else {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					val = sqrt(ux*ux + uy*uy);
				}
			}
			if (val > u)
				d_velocidad_maxima[pos_vel] = val;
		}
	}
}

void inicializarModuloVelocidadMaxima(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double **d_moduloVelocidadMaximaNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double epsilon_h,
			tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, double H, double U,
			double T, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				inicializarModuloVelocidadMaximaNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_moduloVelocidadMaximaNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, epsilon_h, d_datosNivel[l][i].vccos, hmin_metrics, H, U, T);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

void actualizarModuloVelocidadMaxima(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double **d_moduloVelocidadMaximaNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double epsilon_h,
			tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, double H, double U,
			double T, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				actualizarModuloVelocidadMaximaNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_moduloVelocidadMaximaNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, epsilon_h, d_datosNivel[l][i].vccos, hmin_metrics, H, U, T);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

// MODULO DE CAUDAL MAXIMO

__global__ void inicializarModuloCaudalMaximoNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double *d_caudal_maximo, int num_volx, int num_voly, int iniySubmallaCluster, double *vccos, double hmin_metrics,
				double H, double U, double T)
{
	double h, caudal;
	double3 q;
    int j_global;
	double val_cos;
	int pos, pos_caudal;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_caudal = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		if (h > hmin_metrics*val_cos) {
			q = d_datosVolumenes_2[pos];
			q.x = (q.x/val_cos)*H*U;
			q.y = (q.y/val_cos)*H*U;
			if (no_hidros) {
				q.z = (q.z/val_cos)*H*H/T;
				caudal = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
			}
			else {
				caudal = sqrt(q.x*q.x + q.y*q.y);
			}
			d_caudal_maximo[pos_caudal] = caudal;
		}
		else {
			// Inicializamos a cero
			d_caudal_maximo[pos_caudal] = 0.0;
		}
	}
}

__global__ void actualizarModuloCaudalMaximoNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double *d_caudal_maximo, int num_volx, int num_voly, int iniySubmallaCluster, double *vccos, double hmin_metrics,
				double H, double U, double T)
{
	double h, caudal;
	double val, val_cos;
	double3 q;
    int j_global;
	int pos, pos_caudal;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_caudal = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		if (h > hmin_metrics*val_cos) {
			q = d_datosVolumenes_2[pos];
			caudal = d_caudal_maximo[pos_caudal];
			q.x = (q.x/val_cos)*H*U;
			q.y = (q.y/val_cos)*H*U;
			if (no_hidros) {
				q.z = (q.z/val_cos)*H*H/T;
				val = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
			}
			else {
				val = sqrt(q.x*q.x + q.y*q.y);
			}
			if (val > caudal)
				d_caudal_maximo[pos_caudal] = val;
		}
	}
}

void inicializarModuloCaudalMaximo(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double **d_moduloCaudalMaximoNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double hmin_metrics, dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, double H, double U, double T,
			cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				inicializarModuloCaudalMaximoNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_moduloCaudalMaximoNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, d_datosNivel[l][i].vccos, hmin_metrics, H, U, T);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

void actualizarModuloCaudalMaximo(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double **d_moduloCaudalMaximoNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double hmin_metrics, dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, double H, double U, double T,
			cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				actualizarModuloCaudalMaximoNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_moduloCaudalMaximoNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, d_datosNivel[l][i].vccos, hmin_metrics, H, U, T);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

// FLUJO DE MOMENTO MAXIMO

__global__ void inicializarFlujoMomentoMaximoNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double *d_flujo_maximo, int num_volx, int num_voly, int iniySubmallaCluster, double epsilon_h,
				double *vccos, double hmin_metrics, double H, double U, double T)
{
	double h, ux, uy, uz;
	double val_cos;
	double flujo = 0.0;
	double3 q;
    int j_global;
	int pos, pos_flujo;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_flujo = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		if (h > hmin_metrics*val_cos) {
			q = d_datosVolumenes_2[pos];
			if (no_hidros) {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					uz = (q.z/(h + EPSILON))*H/T;
					q.x = (q.x/val_cos)*H*U;
					q.y = (q.y/val_cos)*H*U;
					q.z = (q.z/val_cos)*H*H/T;
					flujo = q.x*ux + q.y*uy + q.z*uz;
				}
			}
			else {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					q.x = (q.x/val_cos)*H*U;
					q.y = (q.y/val_cos)*H*U;
					flujo = q.x*ux + q.y*uy;
				}
			}
			d_flujo_maximo[pos_flujo] = flujo;
		}
		else {
			// Inicializamos a cero
			d_flujo_maximo[pos_flujo] = 0.0;
		}
	}
}

__global__ void actualizarFlujoMomentoMaximoNivelGPU(int no_hidros, double2 *d_datosVolumenes_1, double3 *d_datosVolumenes_2,
				double *d_flujo_maximo, int num_volx, int num_voly, int iniySubmallaCluster, double epsilon_h,
				double *vccos, double hmin_metrics, double H, double U, double T)
{
	double h, ux, uy, uz;
	double flujo;
	double val = 0.0;
	double3 q;
    int j_global;
    double val_cos;
	int pos, pos_flujo;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos_flujo = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;
        val_cos = vccos[j_global];

		h = d_datosVolumenes_1[pos].x;
		if (h > hmin_metrics*val_cos) {
			q = d_datosVolumenes_2[pos];
			flujo = d_flujo_maximo[pos_flujo];
			if (no_hidros) {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					uz = (q.z/(h + EPSILON))*H/T;
					q.x = (q.x/val_cos)*H*U;
					q.y = (q.y/val_cos)*H*U;
					q.z = (q.z/val_cos)*H*H/T;
					val = q.x*ux + q.y*uy + q.z*uz;
				}
			}
			else {
				if (h > HVEL*val_cos) {
					ux = (q.x/(h + EPSILON))*U;
					uy = (q.y/(h + EPSILON))*U;
					q.x = (q.x/val_cos)*H*U;
					q.y = (q.y/val_cos)*H*U;
					val = q.x*ux + q.y*uy;
				}
			}
			if (val > flujo) {
				d_flujo_maximo[pos_flujo] = val;
			}
		}
	}
}

void inicializarFlujoMomentoMaximo(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double **d_flujoMomentoMaximoNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double epsilon_h,
			tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, double H, double U,
			double T, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				inicializarFlujoMomentoMaximoNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_flujoMomentoMaximoNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, epsilon_h, d_datosNivel[l][i].vccos, hmin_metrics, H, U, T);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

void actualizarFlujoMomentoMaximo(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double3 **d_datosVolumenesNivel_2, double **d_flujoMomentoMaximoNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double epsilon_h,
			tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, double H, double U,
			double T, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				actualizarFlujoMomentoMaximoNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					no_hidros, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, d_flujoMomentoMaximoNivel[l]+pos,
					nvx, nvy, datosClusterCPU[l][i].iniy, epsilon_h, d_datosNivel[l][i].vccos, hmin_metrics, H, U, T);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

// TIEMPOS DE LLEGADA

__global__ void inicializarTiemposLlegadaNivelGPU(double *d_tiempos_llegada, int num_volx, int num_voly)
{
	int pos, pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		d_tiempos_llegada[pos] = -1.0;
	}
}

__global__ void actualizarTiemposLlegadaNivelGPU(double2 *d_datosVolumenes_1, double *d_tiempos_llegada,
				double *d_eta1_inicial, int num_volx, int num_voly, int iniySubmallaCluster,
				double tiempo_act, double *vccos, double hmin_metrics, double difh_at)
{
	double2 W;
	double eta_act, eta_ini, t;
	int j_global;
	int pos, pos2;
	int pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = (pos_y_hebra+2)*(num_volx + 4) + pos_x_hebra+2;
		pos2 = pos_y_hebra*num_volx + pos_x_hebra;
        j_global = iniySubmallaCluster + pos_y_hebra;

		t = d_tiempos_llegada[pos2];
		eta_ini = d_eta1_inicial[pos2];
		W = d_datosVolumenes_1[pos];
		eta_act = W.x - W.y;
		if ((t < 0.0) && (fabs(eta_act - eta_ini) > difh_at) && (W.x > hmin_metrics*vccos[j_global]))
			d_tiempos_llegada[pos2] = tiempo_act;
	}
}

void inicializarTiemposLlegada(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double **d_tiemposLlegadaNivel, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int *numSubmallasNivel, dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			dim3 threadBlockEst, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				inicializarTiemposLlegadaNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					d_tiemposLlegadaNivel[l]+pos, nvx, nvy);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

void actualizarTiemposLlegada(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 **d_datosVolumenesNivel_1, double **d_tiemposLlegadaNivel, double **d_eta1InicialNivel,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double tiempo_act,
			tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double hmin_metrics, double difh_at,
			dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, cudaStream_t *streams, int nstreams)
{
	int i, l, pos;
	int nvx, nvy;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				actualizarTiemposLlegadaNivelGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					d_datosVolumenesNivel_1[l]+pos, d_tiemposLlegadaNivel[l]+pos, d_eta1InicialNivel[l]+pos, nvx, nvy,
					datosClusterCPU[l][i].iniy, tiempo_act, d_datosNivel[l][i].vccos, hmin_metrics, difh_at);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

#endif
