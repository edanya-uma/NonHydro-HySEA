#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "Timestep.cu"
#include "TimestepNoHidro.cu"
#include "Metrics.cu"
#include "Deformacion.cu"
#include "DeformacionDinamica.cu"
#include "FriccionVariable.cu"
#include "netcdf.cu"

/*******************************/
/* Hebra de guardado en NetCDF */
/*******************************/

std::thread hebraNetCDF;
std::mutex mtx;
bool terminarHebraNetCDF;
double tiempoAGuardar;
// Si los datos que se van a escribir en NetCDF ya están en datosVolumenesNivel
bool datosPreparados;
std::condition_variable cv_datosPreparados;
// Si ya se han terminado de escribir los datos en el fichero NetCDF
bool escrituraTerminada;
std::condition_variable cv_escrituraTerminada;

void guardarNetCDFNivel(int no_hidros, int nivel, int submalla, VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double2 *datosVolumenes_1, double3 *datosVolumenes_2, double *datosPnh, double *vccos, float *vec1, float *vec2,
			float *vec3, int num, int num_volx, int nx_nc, int ny_nc, int inix, int iniy, int inix_nc, int iniy_nc, int npics,
			int iniySubmallaCluster, double Hmin, double tiempo_act, double epsilon_h, double H, double U, double T)
{
	double tiempo = tiempo_act*T;
	double h, ux, uy, uz;
	double val_cos;
	double2 datos;
	double3 q;
	VariablesGuardado *var;
	int i, j;
	int j_global;
	int pos, ind;
	int desp, desp_vert;

	num_volx = num_volx + 4;
	desp = 2*num_volx;
	writeTimeNC(nivel, submalla, num, tiempo);
	var = &(guardarVariables[nivel][submalla]);
	// eta
	if (var->eta != 0) {
		for (j=0; j<ny_nc; j++) {
			pos = desp + (iniy + j*npics)*num_volx + 2;  // El último 2 es el offsetX
			j_global = iniySubmallaCluster + iniy + j*npics;
			val_cos = vccos[j_global];
			for (i=0; i<nx_nc; i++) {
				datos = datosVolumenes_1[pos + (inix + i*npics)];
				vec1[j*nx_nc + i] = (float) ((datos.x < epsilon_h*val_cos) ? -9999.0 : (datos.x - datos.y - Hmin)*H/val_cos);
			}
		}
		writeEtaNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec1);
	}
	// Velocidades
	if ((var->velocidades != 0) || (var->modulo_velocidades != 0) || (var->flujo_momento != 0)) {
		if (no_hidros) {
			// Guardamos ux, uy, uz
			for (j=0; j<ny_nc; j++) {
				pos = desp + (iniy + j*npics)*num_volx + 2;
				j_global = iniySubmallaCluster + iniy + j*npics;
				val_cos = vccos[j_global];
				for (i=0; i<nx_nc; i++) {
					ind = pos + (inix + i*npics);
					h = datosVolumenes_1[ind].x;
					q = datosVolumenes_2[ind];
					if (h > HVEL*val_cos) {
						ux = (q.x/(h + EPSILON))*U;
						uy = (q.y/(h + EPSILON))*U;
						uz = (q.z/(h + EPSILON))*H/T;
					}
					else {
						ux = uy = uz = 0.0;
					}
					vec1[j*nx_nc + i] = (float) ux;
					vec2[j*nx_nc + i] = (float) uy;
					vec3[j*nx_nc + i] = (float) uz;
				}
			}
		}
		else {
			// Guardamos ux, uy
			for (j=0; j<ny_nc; j++) {
				pos = desp + (iniy + j*npics)*num_volx + 2;
				j_global = iniySubmallaCluster + iniy + j*npics;
				val_cos = vccos[j_global];
				for (i=0; i<nx_nc; i++) {
					ind = pos + (inix + i*npics);
					h = datosVolumenes_1[ind].x;
					q = datosVolumenes_2[ind];
					if (h > HVEL*val_cos) {
						ux = (q.x/(h + EPSILON))*U;
						uy = (q.y/(h + EPSILON))*U;
					}
					else {
						ux = uy = 0.0;
					}
					vec1[j*nx_nc + i] = (float) ux;
					vec2[j*nx_nc + i] = (float) uy;
				}
			}
		}
	}
	if (var->velocidades != 0) {
		writeUxNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec1);
		writeUyNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec2);
		if (no_hidros) {
			writeUzNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec3);
		}
	}
	// Módulo de la velocidad
	if (var->modulo_velocidades != 0) {
		if (no_hidros) {
			for (j=0; j<ny_nc; j++) {
				for (i=0; i<nx_nc; i++) {
					ind = j*nx_nc + i;
					ux = (double) vec1[ind];
					uy = (double) vec2[ind];
					uz = (double) vec3[ind];
					vec3[ind] = (float) sqrt(ux*ux + uy*uy + uz*uz);
				}
			}
		}
		else {
			for (j=0; j<ny_nc; j++) {
				for (i=0; i<nx_nc; i++) {
					ind = j*nx_nc + i;
					ux = (double) vec1[ind];
					uy = (double) vec2[ind];
					vec3[ind] = (float) sqrt(ux*ux + uy*uy);
				}
			}
		}
		writeUNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec3);
	}
	// Presión no hidrostática
	if (var->presion_no_hidrostatica != 0) {
		int posNHSW, posNHSE, posNHNW, posNHNE;
		desp_vert = num_volx-1;
		for (j=0; j<ny_nc; j++) {
			pos = desp_vert + (iniy + j*npics)*(num_volx-1) + 1;
			for (i=0; i<nx_nc; i++) {
				posNHSW = pos + (inix + i*npics);
				posNHSE = posNHSW + 1;
				posNHNW = posNHSW + (num_volx-1);
				posNHNE = posNHNW + 1;
				vec3[j*nx_nc + i] = (float) (0.25*(datosPnh[posNHSW] + datosPnh[posNHSE] + datosPnh[posNHNW] + datosPnh[posNHNE])*U*U);
			}
		}
		writePresionNH(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec3);
	}
	// Flujo de momento
	if (var->flujo_momento != 0) {
		if (no_hidros) {
			// Para calcular el flujo de momento vertical volvemos a calcular uz porque antes se ha
			// machacado vec3 si se ha guardado el módulo de la velocidad del agua o la presión
			for (j=0; j<ny_nc; j++) {
				pos = desp + (iniy + j*npics)*num_volx + 2;
				j_global = iniySubmallaCluster + iniy + j*npics;
				val_cos = vccos[j_global];
				for (i=0; i<nx_nc; i++) {
					ind = pos + (inix + i*npics);
					h = datosVolumenes_1[ind].x;
					q = datosVolumenes_2[ind];
					if (h > HVEL*val_cos) {
						uz = (q.z/(h + EPSILON))*H/T;
					}
					else {
						uz = 0.0;
					}
					vec1[j*nx_nc + i] *= (q.x/val_cos)*H*U;
					vec2[j*nx_nc + i] *= (q.y/val_cos)*H*U;
					vec3[j*nx_nc + i] = uz*(q.z/val_cos)*H*H/T;
				}
			}
			writeFluxxNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec1);
			writeFluxyNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec2);
			writeFluxzNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec3);
		}
		else {
			for (j=0; j<ny_nc; j++) {
				pos = desp + (iniy + j*npics)*num_volx + 2;
				for (i=0; i<nx_nc; i++) {
					q = datosVolumenes_2[pos + (inix + i*npics)];
					vec1[j*nx_nc + i] *= (q.x/val_cos)*H*U;
					vec2[j*nx_nc + i] *= (q.y/val_cos)*H*U;
				}
			}
			writeFluxxNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec1);
			writeFluxyNC(nivel, submalla, nx_nc, ny_nc, inix_nc, iniy_nc, num, vec2);
		}
	}
}

void funcionHebraNetCDF(int no_hidros, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], int numNiveles, double2 **datosVolumenesNivel_1,
			double3 **datosVolumenesNivel_2, double **datosPnh, tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int *numSubmallasNivel, bool crearFicheroNetCDF[MAX_LEVELS][MAX_GRIDS_LEVEL], VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL],
			bool *guardarEstadoActualNetCDF, float *vec1, float *vec2, float *vec3, int *num, int nx_nc[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int ny_nc[MAX_LEVELS][MAX_GRIDS_LEVEL], int inix[MAX_LEVELS][MAX_GRIDS_LEVEL], int iniy[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int inix_nc[MAX_LEVELS][MAX_GRIDS_LEVEL], int iniy_nc[MAX_LEVELS][MAX_GRIDS_LEVEL], int npics, double Hmin,
			double epsilon_h, double H, double U, double T)
{
	int l, i;
	int pos_vol, pos_ver;
	int nvx, nvy;

#if (FILE_WRITING_MODE == 1)
	// Escritura síncrona de ficheros NetCDF
	for (l=0; l<numNiveles; l++) {
		if (guardarEstadoActualNetCDF[l]) {
			pos_vol = pos_ver = 0;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				if (datosClusterCPU[l][i].iniy != -1) {
					nvx = datosClusterCPU[l][i].numVolx;
					nvy = datosClusterCPU[l][i].numVoly;
					if (crearFicheroNetCDF[l][i]) {
						guardarNetCDFNivel(no_hidros, l, i, guardarVariables, datosVolumenesNivel_1[l]+pos_vol, datosVolumenesNivel_2[l]+pos_vol,
							datosPnh[l]+pos_ver, datosNivel[l][i].vccos, vec1, vec2, vec3, num[l], nvx, nx_nc[l][i], ny_nc[l][i], inix[l][i],
							iniy[l][i], inix_nc[l][i], iniy_nc[l][i], npics, datosClusterCPU[l][i].iniy, Hmin, tiempoAGuardar, epsilon_h, H, U, T);
					}
					pos_vol += (nvx+4)*(nvy+4);
					pos_ver += (nvx+3)*(nvy+3);
				}
			}
			num[l] = num[l] + 1;
		}
	}
#else
	// Escritura asíncrona de ficheros NetCDF
	while (! terminarHebraNetCDF) {
		std::unique_lock<std::mutex> lock(mtx);
		cv_datosPreparados.wait(lock, [](){ return datosPreparados; });
		datosPreparados = false;

		if (! terminarHebraNetCDF) {
			for (l=0; l<numNiveles; l++) {
				if (guardarEstadoActualNetCDF[l]) {
					pos_vol = pos_ver = 0;
					for (i=0; i<numSubmallasNivel[l]; i++) {
						if (datosClusterCPU[l][i].iniy != -1) {
							nvx = datosClusterCPU[l][i].numVolx;
							nvy = datosClusterCPU[l][i].numVoly;
							if (crearFicheroNetCDF[l][i]) {
								guardarNetCDFNivel(no_hidros, l, i, guardarVariables, datosVolumenesNivel_1[l]+pos_vol, datosVolumenesNivel_2[l]+pos_vol,
									datosPnh[l]+pos_ver, datosNivel[l][i].vccos, vec1, vec2, vec3, num[l], nvx, nx_nc[l][i], ny_nc[l][i], inix[l][i],
									iniy[l][i], inix_nc[l][i], iniy_nc[l][i], npics, datosClusterCPU[l][i].iniy, Hmin, tiempoAGuardar, epsilon_h, H, U, T);
							}
							pos_vol += (nvx+4)*(nvy+4);
							pos_ver += (nvx+3)*(nvy+3);
						}
					}
					num[l] = num[l] + 1;
				}
			}
		}

		escrituraTerminada = true;
		lock.unlock();
		cv_escrituraTerminada.notify_one();
	}
#endif
}

bool hayQueGuardarEstadoActualEnAlgunNivel(int numNiveles, double tiempoAct, double *sigTiempoGuardarNetCDF,
			double *tiempoGuardarNetCDF, bool hayQueCrearAlgunFicheroNetCDF)
{
	bool guardar = false;
	int l;

	if (hayQueCrearAlgunFicheroNetCDF) {
		l = 0;
		while ((l < numNiveles) && (! guardar)) {
			if ((tiempoGuardarNetCDF[l] >= 0.0) && (tiempoAct >= sigTiempoGuardarNetCDF[l]))
				guardar = true;
			else
				l++;
		}
	}

	return guardar;
}

void asignarGuardarEstadoActualNetCDF(int numNiveles, double tiempoAct, double *sigTiempoGuardarNetCDF,
			double *tiempoGuardarNetCDF, bool *guardarEstadoActualNetCDF)
{
	int l;

	for (l=0; l<numNiveles; l++) {
		if ((tiempoGuardarNetCDF[l] >= 0.0) && (tiempoAct >= sigTiempoGuardarNetCDF[l])) {
			sigTiempoGuardarNetCDF[l] += tiempoGuardarNetCDF[l];
			guardarEstadoActualNetCDF[l] = true;
		}
		else {
			guardarEstadoActualNetCDF[l] = false;
		}
	}
}

/*********************/
/* Series de tiempos */
/*********************/

// Formato del fichero NetCDF de serie de tiempos hidrostático:
// Para cada punto:
//   <longitud>
//   <latitud>
//   <batimetría (original si okada_flag es SEA_SURFACE_FROM_FILE o GAUSSIAN; deformada si okada_flag es OKADA_STANDARD, OKADA_STANDARD_FROM_FILE,
//                OKADA_TRIANGULAR, OKADA_TRIANGULAR_FROM_FILE, DEFORMATION_FROM_FILE o DYNAMIC_DEFORMATION)
//   <eta mínima>
//   <eta máxima>
// Para cada tiempo:
//   Para cada punto:
//     <eta punto 1> <u punto 1> <v punto 1> ... <eta punto n> <u punto n> <v punto n>
//
// Formato del fichero NetCDF de serie de tiempos no hidrostático:
// Para cada punto:
//   <longitud>
//   <latitud>
//   <batimetría (original si okada_flag es SEA_SURFACE_FROM_FILE o GAUSSIAN; deformada si okada_flag es OKADA_STANDARD, OKADA_STANDARD_FROM_FILE,
//                OKADA_TRIANGULAR, OKADA_TRIANGULAR_FROM_FILE, DEFORMATION_FROM_FILE o DYNAMIC_DEFORMATION)
//   <eta mínima>
//   <eta máxima>
// Para cada tiempo:
//   Para cada punto:
//     <eta punto 1> <u punto 1> <v punto 1> <z punto 1> ... <eta punto n> <u punto n> <v punto n> <z punto n>
void obtenerBatimetriaParaSerieTiempos(TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 *datosVolumenesNivel_1,
			int numPuntosGuardar, int4 *posicionesVolumenesGuardado, float *profPuntosGuardado,
			float *profPuntosGuardadoGlobal, double Hmin, double H, int id_hebra, MPI_Comm comm_cartesiano)
{
	float val = -9999.0f;
	int4 pos;
	int i, j_global;
	double val_cos;

	for (i=0; i<numPuntosGuardar; i++) {
		pos = posicionesVolumenesGuardado[i];
		if (pos.x != -1) {
			// pos.x: nivel, pos.z: submalla, pos.w: coordenada y del punto en la submalla
			j_global = datosClusterCPU[pos.x][pos.z].iniy + pos.w;
			val_cos = datosNivel[pos.x][pos.z].vccos[j_global];

			profPuntosGuardado[i] = (float) (((datosVolumenesNivel_1[i].y + Hmin)/val_cos)*H);
		}
	}

	// La hebra 0 obtiene la batimetría de cada punto de guardado y actualiza el fichero
	MPI_Reduce(profPuntosGuardado, profPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);

	if (id_hebra == 0) {
		for (i=0; i<numPuntosGuardar; i++) {
			if (profPuntosGuardadoGlobal[i] > 1e20f)
				profPuntosGuardadoGlobal[i] = val;
		}
	}
}

void guardarSerieTiemposHidrosNivel0(TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 *datosVolumenesNivel_1,
			double3 *datosVolumenesNivel_2, int numPuntosGuardar, int4 *posicionesVolumenesGuardado,
			float *etaPuntosGuardado, float *etaPuntosGuardadoGlobal, float *uPuntosGuardado,
			float *uPuntosGuardadoGlobal, float *vPuntosGuardado, float *vPuntosGuardadoGlobal,
			float *etaMinPuntosGuardadoGlobal, float *etaMaxPuntosGuardadoGlobal, double Hmin, int num_ts,
			double tiempo_act, double epsilon_h, double H, double U, double T, int id_hebra, MPI_Comm comm_cartesiano)
{
	double tiempo = tiempo_act*T;
	float val = -9999.0f;
	double h, ux, uy;
	double val_cos;
	int4 pos;
	int i, j_global;

	for (i=0; i<numPuntosGuardar; i++) {
		pos = posicionesVolumenesGuardado[i];
		if (pos.x != -1) {
			// pos.x: nivel, pos.z: submalla, pos.w: coordenada y del punto en la submalla
			j_global = datosClusterCPU[pos.x][pos.z].iniy + pos.w;
			val_cos = datosNivel[pos.x][pos.z].vccos[j_global];

			h = datosVolumenesNivel_1[i].x;
			// Convertimos a velocidades porque en datosVolumenesNivel_2 tenemos los caudales (ver escribirVolumenesGuardadoGPU)
			if (h > HVEL*val_cos) {
				ux = (datosVolumenesNivel_2[i].x/(h + EPSILON))*U;
				uy = (datosVolumenesNivel_2[i].y/(h + EPSILON))*U;
			}
			else {
				ux = uy = 0.0;
			}

			etaPuntosGuardado[i] = (float) (((h - datosVolumenesNivel_1[i].y - Hmin)/val_cos)*H);
			uPuntosGuardado[i] = (float) ux;
			vPuntosGuardado[i] = (float) uy;
		}
	}

	// La hebra 0 obtiene la eta, u y v de cada punto de guardado y actualiza el fichero, etaMinPuntosGuardadoGlobal
	// y etaMaxPuntosGuardadoGlobal
	MPI_Reduce(etaPuntosGuardado, etaPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);
	MPI_Reduce(uPuntosGuardado, uPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);
	MPI_Reduce(vPuntosGuardado, vPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);

	if (id_hebra == 0) {
		for (i=0; i<numPuntosGuardar; i++) {
			if (etaPuntosGuardadoGlobal[i] < 1e20f) {
				etaMinPuntosGuardadoGlobal[i] = min(etaPuntosGuardadoGlobal[i], etaMinPuntosGuardadoGlobal[i]);
				etaMaxPuntosGuardadoGlobal[i] = max(etaPuntosGuardadoGlobal[i], etaMaxPuntosGuardadoGlobal[i]);
			}
			else {
				etaPuntosGuardadoGlobal[i] = val;
				uPuntosGuardadoGlobal[i] = val;
				vPuntosGuardadoGlobal[i] = val;
				etaMinPuntosGuardadoGlobal[i] = val;
				etaMaxPuntosGuardadoGlobal[i] = val;
			}
		}
		writeStateTimeSeriesHidrosNC(num_ts, (float) tiempo, numPuntosGuardar, etaPuntosGuardadoGlobal,
			uPuntosGuardadoGlobal, vPuntosGuardadoGlobal);
	}
}

void guardarSerieTiemposNoHidrosNivel0(TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 *datosVolumenesNivel_1,
			double3 *datosVolumenesNivel_2, int numPuntosGuardar, int4 *posicionesVolumenesGuardado,
			float *etaPuntosGuardado, float *etaPuntosGuardadoGlobal, float *uPuntosGuardado,
			float *uPuntosGuardadoGlobal, float *vPuntosGuardado, float *vPuntosGuardadoGlobal,
			float *zPuntosGuardado, float *zPuntosGuardadoGlobal, float *etaMinPuntosGuardadoGlobal,
			float *etaMaxPuntosGuardadoGlobal, double Hmin, int num_ts, double tiempo_act,
			double epsilon_h, double H, double U, double T, int id_hebra, MPI_Comm comm_cartesiano)
{
	double tiempo = tiempo_act*T;
	float val = -9999.0f;
	double h, ux, uy, uz;
	double val_cos;
	int4 pos;
	int i, j_global;

	for (i=0; i<numPuntosGuardar; i++) {
		pos = posicionesVolumenesGuardado[i];
		if (pos.x != -1) {
			// pos.x: nivel, pos.z: submalla, pos.w: coordenada y del punto en la submalla
			j_global = datosClusterCPU[pos.x][pos.z].iniy + pos.w;
			val_cos = datosNivel[pos.x][pos.z].vccos[j_global];

			h = datosVolumenesNivel_1[i].x;
			// Convertimos a velocidades porque en datosVolumenesNivel_2 tenemos los caudales (ver escribirVolumenesGuardadoGPU)
			if (h > HVEL*val_cos) {
				ux = (datosVolumenesNivel_2[i].x/(h + EPSILON))*U;
				uy = (datosVolumenesNivel_2[i].y/(h + EPSILON))*U;
				uz = (datosVolumenesNivel_2[i].z/(h + EPSILON))*H/T;
			}
			else {
				ux = uy = uz = 0.0;
			}

			etaPuntosGuardado[i] = (float) (((h - datosVolumenesNivel_1[i].y - Hmin)/val_cos)*H);
			uPuntosGuardado[i] = (float) ux;
			vPuntosGuardado[i] = (float) uy;
			zPuntosGuardado[i] = (float) uz;
		}
	}

	// La hebra 0 obtiene la eta, u y v de cada punto de guardado y actualiza el fichero, etaMinPuntosGuardadoGlobal
	// y etaMaxPuntosGuardadoGlobal
	MPI_Reduce(etaPuntosGuardado, etaPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);
	MPI_Reduce(uPuntosGuardado, uPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);
	MPI_Reduce(vPuntosGuardado, vPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);
	MPI_Reduce(zPuntosGuardado, zPuntosGuardadoGlobal, numPuntosGuardar, MPI_FLOAT, MPI_MIN, 0, comm_cartesiano);

	if (id_hebra == 0) {
		for (i=0; i<numPuntosGuardar; i++) {
			if (etaPuntosGuardadoGlobal[i] < 1e20f) {
				etaMinPuntosGuardadoGlobal[i] = min(etaPuntosGuardadoGlobal[i], etaMinPuntosGuardadoGlobal[i]);
				etaMaxPuntosGuardadoGlobal[i] = max(etaPuntosGuardadoGlobal[i], etaMaxPuntosGuardadoGlobal[i]);
			}
			else {
				etaPuntosGuardadoGlobal[i] = val;
				uPuntosGuardadoGlobal[i] = val;
				vPuntosGuardadoGlobal[i] = val;
				zPuntosGuardadoGlobal[i] = val;
				etaMinPuntosGuardadoGlobal[i] = val;
				etaMaxPuntosGuardadoGlobal[i] = val;
			}
		}
		writeStateTimeSeriesNoHidrosNC(num_ts, (float) tiempo, numPuntosGuardar, etaPuntosGuardadoGlobal,
			uPuntosGuardadoGlobal, vPuntosGuardadoGlobal, zPuntosGuardadoGlobal);
	}
}

/*********************/
/* Tamaños de bloque */
/*********************/

void obtenerTamBloquesKernel(int num_volx, int num_voly, dim3 *blockGridVer1, dim3 *blockGridVer2, dim3 *blockGridHor1,
			dim3 *blockGridHor2, dim3 *blockGridEst, dim3 *blockGridVert)
{
	int num_aristas_ver1, num_aristas_ver2;
	int num_aristas_hor1, num_aristas_hor2;

	// Número de aristas verticales y horizontales
	num_aristas_ver1 = (num_volx/2 + 1)*num_voly;
	num_aristas_ver2 = ((num_volx&1) == 0) ? num_volx*num_voly/2 : num_aristas_ver1;
	num_aristas_hor1 = (num_voly/2 + 1)*num_volx;
	num_aristas_hor2 = ((num_voly&1) == 0) ? num_volx*num_voly/2 : num_aristas_hor1;

	blockGridVer1->x = iDivUp(num_aristas_ver1/num_voly, NUM_HEBRAS_ANCHO_ARI);
	blockGridVer1->y = iDivUp(num_voly, NUM_HEBRAS_ALTO_ARI);
	blockGridVer2->x = iDivUp(num_aristas_ver2/num_voly, NUM_HEBRAS_ANCHO_ARI);
	blockGridVer2->y = iDivUp(num_voly, NUM_HEBRAS_ALTO_ARI);

	blockGridHor1->x = iDivUp(num_volx, NUM_HEBRAS_ANCHO_ARI);
	blockGridHor1->y = iDivUp(num_aristas_hor1/num_volx, NUM_HEBRAS_ALTO_ARI);
	blockGridHor2->x = iDivUp(num_volx, NUM_HEBRAS_ANCHO_ARI);
	blockGridHor2->y = iDivUp(num_aristas_hor2/num_volx, NUM_HEBRAS_ALTO_ARI);

	// Tamaño del grid en el cálculo del nuevo estado
	blockGridEst->x = iDivUp(num_volx, NUM_HEBRAS_ANCHO_EST);
	blockGridEst->y = iDivUp(num_voly, NUM_HEBRAS_ALTO_EST);

	// Tamaño del grid en el procesamiento de los vértices (parte no hidrostática)
	blockGridVert->x = iDivUp(num_volx+1, NUM_HEBRAS_ANCHO_EST);
	blockGridVert->y = iDivUp(num_voly+1, NUM_HEBRAS_ALTO_EST);
}

void obtenerTamBloqueKernelFan(int num_volx, int num_voly, dim3 *blockGridFan)
{
	// Tamaño del grid en el cálculo de los volúmenes fantasma del nivel 1
	blockGridFan->x = iDivUp(num_volx, NUM_HEBRAS_ANCHO_EST);
	blockGridFan->y = iDivUp(num_voly, NUM_HEBRAS_ALTO_EST);
}

/*************/
/* Gaussiana */
/*************/

void aplicarGaussiana(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double2 **d_datosVolumenesNivel_1,
		double lonGauss, double latGauss, double heightGauss, double sigmaGauss, dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		dim3 threadBlockEst, double H, cudaStream_t *streams, int nstreams)
{
	int i, l;
	int pos, nvx, nvy;
	int inix, iniy;
	tipoDatosSubmalla *tds;

	for (l=0; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			inix = datosClusterCPU[l][i].inix;
			iniy = datosClusterCPU[l][i].iniy;
			if (iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				tds = &(datosNivel[l][i]);
				aplicarGaussianaGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(d_datosVolumenesNivel_1[l]+pos,
					nvx, nvy, tds->longitud[inix], tds->longitud[inix+nvx-1], tds->latitud[iniy], tds->latitud[iniy+nvy-1],
					lonGauss, latGauss, heightGauss, sigmaGauss, d_datosNivel[l][i].vccos, iniy, H);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

/************************/
/* Continuar simulación */
/************************/

void cargarDatosMetricasGPU(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], int okada_flag, double *datosGRD,
		double3 **datosVolumenesNivel_2, float *vec1, float *vec2, double **d_eta1MaximaNivel, double3 **d_velocidadesMaximasNivel,
		double **d_moduloVelocidadMaximaNivel, double **d_moduloCaudalMaximoNivel, double **d_flujoMomentoMaximoNivel, double **d_eta1InicialNivel,
		double **d_tiemposLlegadaNivel, double Hmin, double H, double U, double T, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int *numSubmallasNivel, bool guardarEtaMax, bool guardarVelocidadesMax, bool guardarModuloVelocidadMax, bool guardarModuloCaudalMax,
		bool guardarFlujoMomentoMax, bool guardarTiemposLlegada, string prefijo[MAX_LEVELS][MAX_GRIDS_LEVEL],
		VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL], int64_t *tam_datosVolDoubleNivel, int64_t *tam_datosVolDouble2Nivel,
		dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], MPI_Comm comunicadores[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst,
		cudaStream_t *streams, int nstreams)
{
}

/*************************/
/* Aplicar deformaciones */
/*************************/

void comprobarYAplicarDeformaciones(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		double2 **d_datosVolumenesNivel_1, double **d_eta1InicialNivel, double **d_deformacionNivel0, double **d_deformacionAcumuladaNivel,
		double *d_deltaTVolumenesNivel, double **d_mediaEtaNivel, double tiempoAntSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL],
		double tiempoActSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel, double *factorCorreccionNivel,
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, int4 *submallasDeformacion,
		int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int okada_flag, int numFaults, int kajiura_flag, double depth_kajiura,
		cuDoubleComplex *d_datosKajiura, double **d_F2Sx, double **d_F2Sy, int *fallaOkada, double *defTime, int numEstadosDefDinamica,
		int *indiceEstadoSigDefDim, double *LON_C, double *LAT_C, double *DEPTH_C, double *FAULT_L, double *FAULT_W, double *STRIKE,
		double *DIP, double *RAKE, double *SLIP, double2 **d_vc, double **d_vz, double *LONCTRI, double *LATCTRI, double **d_SLIPVEC,
		dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 blockGridFanNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst,
		dim3 blockGridOkada[MAX_FAULTS], cudaStream_t *streams, int nstreams, double H, double T, MPI_Datatype tipo_int2,
		int id_hebra, MPI_Comm comm_cartesiano)
{
	bool okada_aplicado, encontrado;
	// Variables para los warnings de los valores de la deformación
	int2 avisoNaNEnDef, avisoValorGrandeEnDef;
	dim3 blockGridWarning(1,1);
	dim3 threadBlockWarning(1,1);

	// Comprobamos si hay que aplicar alguna deformación en el tiempo actual y, si es así, la aplicamos
	if (*fallaOkada < numFaults) {
		if (okada_flag != DYNAMIC_DEFORMATION) {
			okada_aplicado = false;
			encontrado = true;
			inicializarFlagsWarningsDef<<<blockGridWarning, threadBlockWarning>>>();
			while ((*fallaOkada < numFaults) && encontrado) {
				if (tiempoActSubmalla[0][0]*T >= defTime[*fallaOkada]) {
					if (id_hebra == 0)
						fprintf(stdout, "Applying Okada with fault %d\n", (*fallaOkada)+1);
					aplicarOkada(numNiveles, okada_flag, *fallaOkada, kajiura_flag, depth_kajiura, datosClusterCPU, datosNivel, d_datosNivel,
						submallasNivel, numSubmallasNivel, submallasDeformacion[*fallaOkada], d_datosVolumenesNivel_1, d_eta1InicialNivel,
						d_deformacionNivel0, d_deformacionAcumuladaNivel, d_datosKajiura, d_F2Sx[*fallaOkada], d_F2Sy[*fallaOkada],
						submallaNivelSuperior, ratioRefNivel, LON_C[*fallaOkada], LAT_C[*fallaOkada], DEPTH_C[*fallaOkada], FAULT_L[*fallaOkada],
						FAULT_W[*fallaOkada], STRIKE[*fallaOkada], DIP[*fallaOkada], RAKE[*fallaOkada], SLIP[*fallaOkada], d_vc[*fallaOkada],
						d_vz[*fallaOkada], LONCTRI[*fallaOkada], LATCTRI[*fallaOkada], d_SLIPVEC[*fallaOkada], blockGridOkada[*fallaOkada],
						blockGridEstNivel, blockGridFanNivel, threadBlockEst, H, streams, nstreams);
					okada_aplicado = true;
					(*fallaOkada)++;
				}
				else encontrado = false;
			}
			if (okada_aplicado) {
#if (NESTED_ALGORITHM == 1)
				obtenerMediasEta(numNiveles, datosClusterCPU, submallasNivel, numSubmallasNivel, d_datosVolumenesNivel_1,
					d_mediaEtaNivel, ratioRefNivel, factorCorreccionNivel, blockGridEstNivel, threadBlockEst, streams, nstreams);
#endif
				// Comprobación de warnings de los valores de la deformación
				cudaMemcpyFromSymbol(&avisoNaNEnDef, d_avisoNaNEnDef, sizeof(int2), 0, cudaMemcpyDeviceToHost);
				cudaMemcpyFromSymbol(&avisoValorGrandeEnDef, d_avisoValorGrandeEnDef, sizeof(int2), 0, cudaMemcpyDeviceToHost);
				MPI_Bcast(&avisoNaNEnDef, 1, tipo_int2, 0, comm_cartesiano);
				MPI_Bcast(&avisoValorGrandeEnDef, 1, tipo_int2, 0, comm_cartesiano);
				if (id_hebra == 0) {
					if (avisoNaNEnDef.x)
						fprintf(stdout, "Warning: NaN set to zero in fault %d\n", avisoNaNEnDef.y+1);
					if (avisoValorGrandeEnDef.x)
						fprintf(stdout, "Warning: Value out of [-500,500] meters interval in fault %d\n", avisoValorGrandeEnDef.y+1);
				}
			}
		}
		else {
			// Deformación dinámica
			if ((tiempoActSubmalla[0][0]*T >= defTime[0]) && (tiempoActSubmalla[0][0]*T <= defTime[numEstadosDefDinamica-1])) {
				if (id_hebra == 0)
					fprintf(stdout, "Applying dynamic deformation %d\n", (*fallaOkada)+1);
				aplicarDefDinamica(numNiveles, *indiceEstadoSigDefDim, kajiura_flag, depth_kajiura, datosClusterCPU, d_datosNivel,
					submallasNivel, numSubmallasNivel, submallasDeformacion[*fallaOkada], d_datosVolumenesNivel_1, d_eta1InicialNivel,
					d_deformacionNivel0, d_deformacionAcumuladaNivel, d_deltaTVolumenesNivel, defTime, tiempoAntSubmalla[0][0]*T,
					tiempoActSubmalla[0][0]*T, d_datosKajiura, d_F2Sx[*fallaOkada], d_F2Sy[*fallaOkada], submallaNivelSuperior,
					ratioRefNivel, blockGridOkada[*fallaOkada], blockGridEstNivel, blockGridFanNivel, threadBlockEst, streams, nstreams);
				if (tiempoActSubmalla[0][0]*T >= defTime[*indiceEstadoSigDefDim]) {
					// Estamos en el estado indiceEstadoSigDefDim de la deformación dinámica. Incrementamos el índice
					*indiceEstadoSigDefDim = min((*indiceEstadoSigDefDim)+1, numEstadosDefDinamica-1);
				}
				if (tiempoActSubmalla[0][0]*T >= defTime[numEstadosDefDinamica-1]) {
					// Hemos llegado al final de la deformación dinámica
					(*fallaOkada)++;
				}
#if (NESTED_ALGORITHM == 1)
				obtenerMediasEta(numNiveles, datosClusterCPU, submallasNivel, numSubmallasNivel, d_datosVolumenesNivel_1,
					d_mediaEtaNivel, ratioRefNivel, factorCorreccionNivel, blockGridEstNivel, threadBlockEst, streams, nstreams);
#endif
			}
		}
	}
}

/****************/
/* Aplicar paso */
/****************/

/*void aplicarPaso(int l, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		TDatosClusterGPU datosClusterGPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 **d_datosVolumenesNivel_1, double2 **d_datosVolumenesNivel_2,
		double2 **d_datosVolumenesNivelSig_1, double2 **d_datosVolumenesNivelSig_2, bool **d_refinarNivel, double Hmin,
		tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], double *d_deltaTVolumenesNivel, double **d_correccionEtaNivel,
		double2 **d_correccionNivel_2, double2 *d_acumuladorNivel_1, double2 *d_acumuladorNivel_2, double **d_mediaEtaNivel,
		double2 **d_mediaNivel_2, double borde_sup, double borde_inf, double borde_izq, double borde_der, int tam_spongeSup,
		int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer, double sea_level, double tiempoAntSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL],
		double tiempoActSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], double deltaTNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		double deltaTNivelSinTruncar[MAX_LEVELS][MAX_GRIDS_LEVEL], double **d_friccionesNivel, double vmax, double CFL, double epsilon_h,
		int *ratioRefNivel, int *ratioRefAcumNivel, double *factorCorreccionNivel, double L, double H, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int *numSubmallasNivel, int2 iniGlobalSubmallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int posSubmallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int **d_posCopiaNivel, bool *haySubmallasAdyacentesNivel,
		int64_t *tam_datosVolDoubleNivel, int64_t *tam_datosVolDouble2Nivel, bool finNivelSup, dim3 blockGridVer1Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		dim3 blockGridVer2Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 blockGridHor1Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		dim3 blockGridHor2Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockAri, dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		dim3 blockGridFanNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, cudaStream_t *streams, int nstreams, cudaStream_t streamMemcpy,
		double T, MPI_Datatype tipo_filas[MAX_LEVELS][MAX_GRIDS_LEVEL], MPI_Datatype tipo_columnas[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int id_hebra, bool ultimaHebraXSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], bool ultimaHebraYSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL],
		MPI_Comm comm_cartesiano, int hebra_izq, int hebra_der, int hebra_sup, int hebra_inf)
{
	bool fin;
	int i;

	obtenerDeltaTTruncados(l, datosClusterCPU[l], datosClusterGPU[l], deltaTNivel[l], deltaTNivelSinTruncar[l], tiempoActSubmalla,
		submallasNivel, numSubmallasNivel[l], submallaNivelSuperior[l], haySubmallasAdyacentesNivel[l], T, id_hebra, comm_cartesiano);
	if (l < numNiveles-1) {
		// l es un nivel intermedio
		cudaMemset(d_correccionEtaNivel[l], 0, tam_datosVolDoubleNivel[l]);
		cudaMemset(d_correccionNivel_2[l], 0, tam_datosVolDouble2Nivel[l]);
		fin = siguientePasoNivel(l, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
				d_datosVolumenesNivelSig_1[l-1], d_datosVolumenesNivelSig_2[l-1], d_datosVolumenesNivelSig_1[l], d_datosVolumenesNivelSig_2[l],
				d_datosNivel[l], d_refinarNivel, Hmin, d_deltaTVolumenesNivel, d_correccionEtaNivel, d_correccionNivel_2, d_acumuladorNivel_1,
				d_acumuladorNivel_2, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer,
				sea_level, tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel[l], deltaTNivelSinTruncar[l], d_friccionesNivel, vmax, CFL, epsilon_h,
				ratioRefNivel[l], ratioRefAcumNivel[l], L, H, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel, submallaNivelSuperior[l],
				posSubmallaNivelSuperior[l], d_posCopiaNivel[l], false, haySubmallasAdyacentesNivel[l], tam_datosVolDouble2Nivel[l],
				blockGridVer1Nivel[l], blockGridVer2Nivel[l], blockGridHor1Nivel[l], blockGridHor2Nivel[l], threadBlockAri, blockGridEstNivel[l],
				blockGridFanNivel[l], threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas[l], tipo_columnas[l], id_hebra,
				ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
		aplicarPaso(l+1, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosVolumenesNivelSig_1,
			d_datosVolumenesNivelSig_2, d_refinarNivel, Hmin, d_datosNivel, d_deltaTVolumenesNivel, d_correccionEtaNivel, d_correccionNivel_2,
			d_acumuladorNivel_1, d_acumuladorNivel_2, d_mediaEtaNivel, d_mediaNivel_2, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup,
			tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level, tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel, deltaTNivelSinTruncar,
			d_friccionesNivel, vmax, CFL, epsilon_h, ratioRefNivel, ratioRefAcumNivel, factorCorreccionNivel, L, H, submallasNivel, numSubmallasNivel,
			iniGlobalSubmallasNivel, submallaNivelSuperior, posSubmallaNivelSuperior, d_posCopiaNivel, haySubmallasAdyacentesNivel, tam_datosVolDoubleNivel,
			tam_datosVolDouble2Nivel, fin, blockGridVer1Nivel, blockGridVer2Nivel, blockGridHor1Nivel, blockGridHor2Nivel, threadBlockAri,
			blockGridEstNivel, blockGridFanNivel, threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas, tipo_columnas, id_hebra,
			ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
		while (! fin) {
			cudaMemset(d_correccionEtaNivel[l], 0, tam_datosVolDoubleNivel[l]);
			cudaMemset(d_correccionNivel_2[l], 0, tam_datosVolDouble2Nivel[l]);
			fin = siguientePasoNivel(l, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
					d_datosVolumenesNivelSig_1[l-1], d_datosVolumenesNivelSig_2[l-1], d_datosVolumenesNivelSig_1[l], d_datosVolumenesNivelSig_2[l],
					d_datosNivel[l], d_refinarNivel, Hmin, d_deltaTVolumenesNivel, d_correccionEtaNivel, d_correccionNivel_2, d_acumuladorNivel_1,
					d_acumuladorNivel_2, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer,
					sea_level, tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel[l], deltaTNivelSinTruncar[l], d_friccionesNivel, vmax, CFL, epsilon_h,
					ratioRefNivel[l], ratioRefAcumNivel[l], L, H, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel, submallaNivelSuperior[l],
					posSubmallaNivelSuperior[l], d_posCopiaNivel[l], true, haySubmallasAdyacentesNivel[l], tam_datosVolDouble2Nivel[l],
					blockGridVer1Nivel[l], blockGridVer2Nivel[l], blockGridHor1Nivel[l], blockGridHor2Nivel[l], threadBlockAri, blockGridEstNivel[l],
					blockGridFanNivel[l], threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas[l], tipo_columnas[l], id_hebra,
					ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
			aplicarPaso(l+1, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosVolumenesNivelSig_1,
				d_datosVolumenesNivelSig_2, d_refinarNivel, Hmin, d_datosNivel, d_deltaTVolumenesNivel, d_correccionEtaNivel, d_correccionNivel_2,
				d_acumuladorNivel_1, d_acumuladorNivel_2, d_mediaEtaNivel, d_mediaNivel_2, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup,
				tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level, tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel, deltaTNivelSinTruncar,
				d_friccionesNivel, vmax, CFL, epsilon_h, ratioRefNivel, ratioRefAcumNivel, factorCorreccionNivel, L, H, submallasNivel, numSubmallasNivel,
				iniGlobalSubmallasNivel, submallaNivelSuperior, posSubmallaNivelSuperior, d_posCopiaNivel, haySubmallasAdyacentesNivel, tam_datosVolDoubleNivel,
				tam_datosVolDouble2Nivel, fin, blockGridVer1Nivel, blockGridVer2Nivel, blockGridHor1Nivel, blockGridHor2Nivel, threadBlockAri,
				blockGridEstNivel, blockGridFanNivel, threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas, tipo_columnas, id_hebra,
				ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
		}
	}
	else {
		// l es el último nivel
		fin = siguientePasoNivel(l, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
				d_datosVolumenesNivelSig_1[l-1], d_datosVolumenesNivelSig_2[l-1], d_datosVolumenesNivel_1[l], d_datosVolumenesNivel_2[l],
				d_datosNivel[l], d_refinarNivel, Hmin, d_deltaTVolumenesNivel, d_correccionEtaNivel, d_correccionNivel_2, d_acumuladorNivel_1,
				d_acumuladorNivel_2, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer,
				sea_level, tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel[l], deltaTNivelSinTruncar[l], d_friccionesNivel, vmax, CFL, epsilon_h,
				ratioRefNivel[l], ratioRefAcumNivel[l], L, H, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel, submallaNivelSuperior[l],
				posSubmallaNivelSuperior[l], d_posCopiaNivel[l], false, haySubmallasAdyacentesNivel[l], tam_datosVolDouble2Nivel[l],
				blockGridVer1Nivel[l], blockGridVer2Nivel[l], blockGridHor1Nivel[l], blockGridHor2Nivel[l], threadBlockAri, blockGridEstNivel[l],
				blockGridFanNivel[l], threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas[l], tipo_columnas[l], id_hebra,
				ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
		while (! fin) {
			fin = siguientePasoNivel(l, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
					d_datosVolumenesNivelSig_1[l-1], d_datosVolumenesNivelSig_2[l-1], d_datosVolumenesNivel_1[l], d_datosVolumenesNivel_2[l],
					d_datosNivel[l], d_refinarNivel, Hmin, d_deltaTVolumenesNivel, d_correccionEtaNivel, d_correccionNivel_2, d_acumuladorNivel_1,
					d_acumuladorNivel_2, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer,
					sea_level, tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel[l], deltaTNivelSinTruncar[l], d_friccionesNivel, vmax, CFL, epsilon_h,
					ratioRefNivel[l], ratioRefAcumNivel[l], L, H, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel, submallaNivelSuperior[l],
					posSubmallaNivelSuperior[l], d_posCopiaNivel[l], true, haySubmallasAdyacentesNivel[l], tam_datosVolDouble2Nivel[l],
					blockGridVer1Nivel[l], blockGridVer2Nivel[l], blockGridHor1Nivel[l], blockGridHor2Nivel[l], threadBlockAri, blockGridEstNivel[l],
					blockGridFanNivel[l], threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas[l], tipo_columnas[l], id_hebra,
					ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
		}
	}
#if (NESTED_ALGORITHM == 1)
	corregirSolucionDiferenciasNivelGPU(l-1, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1[l-1], d_datosVolumenesNivel_2[l-1],
		d_datosVolumenesNivelSig_1[l-1], d_datosVolumenesNivelSig_2[l-1], d_mediaEtaNivel[l], d_mediaNivel_2[l], d_correccionEtaNivel[l-1],
		d_correccionNivel_2[l-1], d_datosVolumenesNivel_1[l], d_datosVolumenesNivel_2[l], ratioRefNivel[l], factorCorreccionNivel[l],
		epsilon_h, submallasNivel, numSubmallasNivel, submallaNivelSuperior, blockGridEstNivel, blockGridFanNivel, threadBlockEst,
		streams, nstreams, ultimaHebraXSubmalla[l], ultimaHebraYSubmalla[l]);
#else
	corregirSolucionValoresNivelGPU(l-1, datosClusterCPU, d_datosVolumenesNivelSig_1[l-1], d_datosVolumenesNivelSig_2[l-1],
		d_correccionEtaNivel[l-1], d_correccionNivel_2[l-1], d_datosVolumenesNivel_1[l], d_datosVolumenesNivel_2[l], ratioRefNivel[l],
		factorCorreccionNivel[l], epsilon_h, submallasNivel, numSubmallasNivel, submallaNivelSuperior, blockGridEstNivel,
		threadBlockEst, streams, nstreams);
#endif

	// Copiamos el siguiente estado en el actual
	cudaMemcpy(d_datosVolumenesNivel_1[l-1], d_datosVolumenesNivelSig_1[l-1], tam_datosVolDouble2Nivel[l-1], cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_datosVolumenesNivel_2[l-1], d_datosVolumenesNivelSig_2[l-1], tam_datosVolDouble2Nivel[l-1], cudaMemcpyDeviceToDevice);

	if (! finNivelSup) {
		// Copiamos los volúmenes de comunicación del nivel l-1 a memoria CPU de forma asíncrona
		for (i=0; i<numSubmallasNivel[l-1]; i++) {
			if (datosClusterCPU[l-1][i].iniy != -1) {
				copiarVolumenesComAsincACPU(&(datosClusterCPU[l-1][i]), &(datosClusterGPU[l-1][i]), streamMemcpy,
					datosClusterCPU[l-1][i].inix, datosClusterCPU[l-1][i].iniy, ultimaHebraXSubmalla[l-1][i], ultimaHebraYSubmalla[l-1][i]);
			}
		}
	}
}*/

/*******************/
/* Liberar memoria */
/*******************/

void liberarMemoria(int no_hidros, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 **d_datosVolumenesNivel_1,
		double3 **d_datosVolumenesNivel_2, double **d_friccionesNivel, tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		double2 **d_datosVolumenesNivelSig_1, double3 **d_datosVolumenesNivelSig_2, int *numSubmallasNivel, int **d_posCopiaNivel,
		bool *d_activacionMallasAnidadas, double *d_deltaTVolumenesNivel, double2 *d_acumuladorNivel_1, double3 *d_acumuladorNivel_2,
		double *d_aristaReconstruido, bool **d_refinarNivel, double **d_correccionEtaNivel, double2 **d_correccionNivel_2,
		double **d_mediaEtaNivel, double2 **d_mediaNivel_2, int leer_fichero_puntos, int4 *d_posicionesVolumenesGuardado,
		double2 *d_datosVolumenesGuardado_1, double3 *d_datosVolumenesGuardado_2, double2 **d_datosVolumenesNivelGPU_1,
		double3 **d_datosVolumenesNivelGPU_2, bool guardarEtaMax, bool guardarVelocidadesMax, bool guardarModuloVelocidadMax,
		bool guardarModuloCaudalMax, bool guardarFlujoMomentoMax, bool guardarTiemposLlegada, double **d_eta1InicialNivel,
		double **d_eta1MaximaNivel, double3 **d_velocidadesMaximasNivel, double **d_moduloVelocidadMaximaNivel,
		double **d_moduloCaudalMaximoNivel, double **d_flujoMomentoMaximoNivel, double **d_tiemposLlegadaNivel, int okada_flag,
		int kajiura_flag, int numFaults, int numEstadosDefDinamica, double **d_deformacionNivel0, double **d_deformacionAcumuladaNivel,
		double2 **d_vc, double **d_vz, double **d_SLIPVEC, cuDoubleComplex *d_datosKajiura, double **d_F2Sx, double **d_F2Sy,
		double *d_RHS_dt, double *d_dtHs, double *d_CoefPE, double *d_CoefPW, double *d_CoefPN, double *d_CoefPS, double **d_Pnh0, double **d_Pnh1)
{
	int l, i;

	for (l=0; l<numNiveles; l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				cudaFree(d_datosNivel[l][i].vcos);
				cudaFree(d_datosNivel[l][i].vccos);
				cudaFree(d_datosNivel[l][i].vtan);
			}
		}
		cudaFree(d_datosVolumenesNivel_1[l]);
		cudaFree(d_datosVolumenesNivel_2[l]);
		cudaFree(d_datosVolumenesNivelSig_1[l]);
		cudaFree(d_datosVolumenesNivelSig_2[l]);
		cudaFree(d_friccionesNivel[l]);
		cudaFree(d_eta1InicialNivel[l]);
		if (l > 0) {
			cudaFree(d_posCopiaNivel[l]);
			cudaFree(d_mediaEtaNivel[l]);
			cudaFree(d_mediaNivel_2[l]);
		}
		if ((numNiveles > 1) && (l < numNiveles-1)) {
			cudaFree(d_refinarNivel[l]);
			cudaFree(d_correccionEtaNivel[l]);
			cudaFree(d_correccionNivel_2[l]);
		}
		if (l < numNiveles-1)
			cudaFree(d_deformacionAcumuladaNivel[l]);
		if (guardarEtaMax)
			cudaFree(d_eta1MaximaNivel[l]);
		if (guardarVelocidadesMax)
			cudaFree(d_velocidadesMaximasNivel[i]);
		if (guardarModuloVelocidadMax)
			cudaFree(d_moduloVelocidadMaximaNivel[l]);
		if (guardarModuloCaudalMax)
			cudaFree(d_moduloCaudalMaximoNivel[l]);
		if (guardarFlujoMomentoMax)
			cudaFree(d_flujoMomentoMaximoNivel[l]);
		if (guardarTiemposLlegada)
			cudaFree(d_tiemposLlegadaNivel[l]);
		if (no_hidros) {
			cudaFree(d_Pnh0[l]);
			cudaFree(d_Pnh1[l]);
		}
	}
	cudaFree(d_deltaTVolumenesNivel);
	cudaFree(d_acumuladorNivel_1);
	cudaFree(d_acumuladorNivel_2);
	cudaFree(d_aristaReconstruido);
	if (no_hidros) {
		cudaFree(d_RHS_dt);
		cudaFree(d_dtHs);
		cudaFree(d_CoefPE);
		cudaFree(d_CoefPW);
		cudaFree(d_CoefPN);
		cudaFree(d_CoefPS);
	}
	if (numNiveles > 1) {
		cudaFree(d_activacionMallasAnidadas);
	}
	if (leer_fichero_puntos == 1) {
		cudaFree(d_posicionesVolumenesGuardado);
		cudaFree(d_datosVolumenesGuardado_1);
		cudaFree(d_datosVolumenesGuardado_2);
		cudaFree(d_datosVolumenesNivelGPU_1);
		cudaFree(d_datosVolumenesNivelGPU_2);
	}
	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE)) {
		cudaFree(d_deformacionNivel0[0]);
	}
	else if (okada_flag == DEFORMATION_FROM_FILE) {
		for (i=0; i<numFaults; i++)
			cudaFree(d_deformacionNivel0[i]);
	}
	else if (okada_flag == DYNAMIC_DEFORMATION) {
		for (i=0; i<numEstadosDefDinamica; i++)
			cudaFree(d_deformacionNivel0[i]);
	}
	else if ((okada_flag == OKADA_TRIANGULAR) || (okada_flag == OKADA_TRIANGULAR_FROM_FILE)) {
		cudaFree(d_deformacionNivel0[0]);
		for (i=0; i<numFaults; i++) {
			cudaFree(d_vc[i]);
			cudaFree(d_vz[i]);
			cudaFree(d_SLIPVEC[i]);
		}
	}
	if (kajiura_flag == 1) {
		cudaFree(d_datosKajiura);
		if (okada_flag == DYNAMIC_DEFORMATION) {
			for (i=0; i<numEstadosDefDinamica; i++) {
				cudaFree(d_F2Sx[i]);
				cudaFree(d_F2Sy[i]);
			}
		}
		else {
			for (i=0; i<numFaults; i++) {
				cudaFree(d_F2Sx[i]);
				cudaFree(d_F2Sy[i]);
			}
		}
	}
}


/*********************/
/* Función principal */
/*********************/

// Devuelve 0 si todo ha ido bien, 1 si no hay memoria GPU suficiente, y 2 si no hay memoria CPU suficiente
extern "C" int shallowWater(TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], int no_hidros, int numPesosJacobi, int numNiveles,
			int okada_flag, int kajiura_flag, double depth_kajiura, char *fich_okada, int numFaults, int numEstadosDefDinamica, double *LON_C,
			double *LAT_C, double *DEPTH_C, double *FAULT_L, double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *defTime,
			double DEPTH_v[MAX_FAULTS][4], double2 vc[MAX_FAULTS][4], double *LONCTRI, double *LATCTRI, double SLIPVEC[MAX_FAULTS][3],
			double **deformacionNivel0, string *fich_def, double lonGauss, double latGauss, double heightGauss, double sigmaGauss,
			float *batiOriginal[MAX_LEVELS], double2 **datosVolumenesNivel_1, double3 **datosVolumenesNivel_2, double **datosPnh, double *datosGRD,
			tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int **posCopiaNivel, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int *numSubmallasNivel, int2 iniGlobalSubmallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int4 *submallasDeformacion, bool **refinarNivel,
			int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int posSubmallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int leer_fichero_puntos,
			int4 *posicionesVolumenesGuardado, int numPuntosGuardarAnt, int numPuntosGuardarTotal, double *lonPuntos, double *latPuntos,
			int numVolxTotalNivel0, int numVolyTotalNivel0, int64_t *numVolumenesNivel, int64_t *numVerticesNivel, double Hmin, char *nombre_bati,
			string prefijo[MAX_LEVELS][MAX_GRIDS_LEVEL], double borde_sup, double borde_inf, double borde_izq, double borde_der,
			int tam_spongeSup, int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer, double tiempo_tot, double *tiempoGuardarNetCDF,
			VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL], double tiempoGuardarSeries, double CFL, int tipo_friccion,
			string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL], double **friccionesNivel, double vmax, double epsilon_h, int continuar_simulacion,
			double tiempo_continuar, int *numEstadoNetCDF, int *ratioRefNivel, int *ratioRefAcumNivel, bool *haySubmallasAdyacentesNivel,
			double difh_at, double L, double H, double U, double T, int num_procs, int num_procsX, int num_procsY, int id_hebra,
			MPI_Comm comm_cartesiano, char *version, double *tiempo)
{
	double2 *d_datosVolumenesNivel_1[MAX_LEVELS];
	double3 *d_datosVolumenesNivel_2[MAX_LEVELS];
	double2 *d_datosVolumenesNivelSig_1[MAX_LEVELS];
	double3 *d_datosVolumenesNivelSig_2[MAX_LEVELS];
	double *d_friccionesNivel[MAX_LEVELS];
	double2 *d_acumuladorNivel_1;
	double3 *d_acumuladorNivel_2;
	double *d_aristaReconstruido;
	double  *d_correccionEtaNivel[MAX_LEVELS];
	double2 *d_correccionNivel_2[MAX_LEVELS];
	double  *d_mediaEtaNivel[MAX_LEVELS];
	double2 *d_mediaNivel_2[MAX_LEVELS];
	tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int *d_posCopiaNivel[MAX_LEVELS];
	bool *d_refinarNivel[MAX_LEVELS];
	double factorCorreccionNivel[MAX_LEVELS];
	double *d_deltaTVolumenesNivel;
	int4 *d_posicionesVolumenesGuardado;
	// Vectores para la simulación no hidrostática
	double *d_RHS_dt, *d_dtHs;
	double *d_CoefPE, *d_CoefPW, *d_CoefPN, *d_CoefPS;
	double *d_Pnh0[MAX_LEVELS];
	double *d_Pnh1[MAX_LEVELS];
	double omega[numPesosJacobi];
	int ord[numPesosJacobi];
	// Vectores para obtener datos de series de tiempos y valores máximos que se guardarán en NetCDF
	double2 *d_datosVolumenesGuardado_1;
	double3 *d_datosVolumenesGuardado_2;
	double *d_eta1MaximaNivel[MAX_LEVELS];
	double3 *d_velocidadesMaximasNivel[MAX_LEVELS];
	double *d_moduloVelocidadMaximaNivel[MAX_LEVELS];
	double *d_moduloCaudalMaximoNivel[MAX_LEVELS];
	double *d_flujoMomentoMaximoNivel[MAX_LEVELS];
	double *d_tiemposLlegadaNivel[MAX_LEVELS];
	double *d_eta1InicialNivel[MAX_LEVELS];
	// Vector y variables para la activación del procesamiento de las mallas anidadas
	bool *d_activacionMallasAnidadas;
//	int mallasAnidadasActivadas = 0;  // Descomentar para mallas anidadas
//	int mallasAnidadasActivadas_global = 0;  // Descomentar para mallas anidadas
	// Vectores y variables para comprobar el guardado de datos
	// en NetCDF para cada submalla y variable
	bool crearFicheroNetCDF[MAX_LEVELS][MAX_GRIDS_LEVEL];
	bool guardarEstadoActualNetCDF[MAX_LEVELS];
	bool hayQueGuardarEstadoAct;
	bool guardarEtaMax;
	bool guardarVelocidadesMax;
	bool guardarModuloVelocidadMax;
	bool guardarModuloCaudalMax;
	bool guardarFlujoMomentoMax;
	bool guardarTiemposLlegada;
	VariablesGuardado *p_var;
	// Vectores para las series de tiempos
	float *etaPuntosGuardado, *etaPuntosGuardadoGlobal;
	float *uPuntosGuardado, *uPuntosGuardadoGlobal;
	float *vPuntosGuardado, *vPuntosGuardadoGlobal;
	float *zPuntosGuardado, *zPuntosGuardadoGlobal;
	float *etaMinPuntosGuardadoGlobal;
	float *etaMaxPuntosGuardadoGlobal;
	double2 *datosGRD_double2 = (double2 *) datosGRD;
	double3 *datosGRD_puntosQ = (double3 *) (datosGRD_double2 + numPuntosGuardarTotal);
	// Otros vectores
	float *vec1, *vec2, *vec3;
	double *p_eta, *p_flujo;
	cudaError_t err_cuda;
	// Vectores usados en el guardado de puntos. d_datosVolumenesNivelGPU_i
	// es una copia en GPU del vector d_datosVolumenesNivel_i
	double2 **d_datosVolumenesNivelGPU_1;
	double3 **d_datosVolumenesNivelGPU_2;
	// Número del estado que se va guardando en el fichero NetCDF de series de tiempos
	int num_ts = 0;
	// Variables para Okada estándar
	int fallaOkada = 0;
	bool encontrado;
	double *d_deformacionNivel0[MAX_FAULTS];
	double *d_deformacionAcumuladaNivel[MAX_LEVELS];
	// Variables para Kajiura
	cuDoubleComplex *d_datosKajiura;
	double *d_F2Sx[MAX_FAULTS], *d_F2Sy[MAX_FAULTS];
	double *F2Sx[MAX_FAULTS], *F2Sy[MAX_FAULTS];
	// Variables para Okada triangular
	double2 *d_vc[MAX_FAULTS];
	double *d_vz[MAX_FAULTS];
	double *d_SLIPVEC[MAX_FAULTS];
	// Variables para la deformación dinámica
	int indiceEstadoSigDefDim;
	bool saltar_deformacion = false;
	// nvolx y nvoly que se guardan en NetCDF
	int nx_nc[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int ny_nc[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int npics = 1;
	double sea_level = SEA_LEVEL/H;
	// cfb solo se usa si la fricción es fija para guardar su valor en los ficheros NetCDF
	double cfb = (friccionesNivel[0][0])*pow(H,4.0/3.0)/L;
	// Variables usadas en la resolución del sistema
	int iter_sistema;
	double error_sistema;
	// Mínima h para el cálculo de las métricas globales
	double hmin_metrics = ((MIN_H_METRICS < 0.0) ? epsilon_h : MIN_H_METRICS/H);
	// inix[l][i], iniy[l][i]: coordenadas x, y locales a datosVolumenesNivel[l] a partir de las que se guardarán
	// puntos de la submalla (por si npics != 1)
	int inix[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int iniy[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int inix_nc[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int iniy_nc[MAX_LEVELS][MAX_GRIDS_LEVEL];
	// Tamaños del grid y de bloque para cada nivel
	dim3 blockGridOkada[MAX_FAULTS];
	dim3 blockGridVer1Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	dim3 blockGridVer2Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	dim3 blockGridHor1Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	dim3 blockGridHor2Nivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	dim3 blockGridFanNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	dim3 blockGridVertNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	dim3 threadBlockAri(NUM_HEBRAS_ANCHO_ARI, NUM_HEBRAS_ALTO_ARI);
	dim3 threadBlockEst(NUM_HEBRAS_ANCHO_EST, NUM_HEBRAS_ALTO_EST);
	// Tamaños del grid y de bloque para el guardado de puntos
	dim3 blockGridPuntos(iDivUp(numPuntosGuardarTotal, NUM_HEBRAS_PUNTOS), 1);
	dim3 threadBlockPuntos(NUM_HEBRAS_PUNTOS, 1);
	// Streams para procesar las submallas
	cudaStream_t streams[32];
	// Stream para las transferencias asíncronas de GPU a CPU
	cudaStream_t streamMemcpy;
	int nstreams;
	TDatosClusterGPU datosClusterGPU[MAX_LEVELS][MAX_GRIDS_LEVEL];
	// Variables MPI
	int err, err_total;
	MPI_Datatype tipoFilasDouble2[MAX_LEVELS][MAX_GRIDS_LEVEL];
	MPI_Datatype tipoFilasDouble3[MAX_LEVELS][MAX_GRIDS_LEVEL];
	MPI_Datatype tipoColumnasDouble2[MAX_LEVELS][MAX_GRIDS_LEVEL];
	MPI_Datatype tipoColumnasDouble3[MAX_LEVELS][MAX_GRIDS_LEVEL];
	MPI_Datatype tipoColumnasPresion[MAX_LEVELS][MAX_GRIDS_LEVEL];
	MPI_Datatype tipo_double2, tipo_double3, tipo_int2;
	MPI_Datatype tipo[3];
	int blocklen[3] = {1, 1, 1};
	MPI_Aint disp[3];
	bool ultimaHebraXSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL];
	bool ultimaHebraYSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL];
	MPI_Comm comunicadores[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int hebra_izq, hebra_der;
	int hebra_sup, hebra_inf;
	// Coordenadas de la hebra en X e Y
	int coord_proc[2];
	int id_hebraX, id_hebraY;

	int64_t tam_datosVolDoubleNivel[MAX_LEVELS];
	int64_t tam_datosVolDouble2Nivel[MAX_LEVELS];
	int64_t tam_datosVolDouble3Nivel[MAX_LEVELS];
	int64_t tam_datosVertDoubleNivel[MAX_LEVELS];
	int64_t tam_datosCopiaNivel[MAX_LEVELS];
	int64_t tam_datosRefinarNivel[MAX_LEVELS];
	int64_t tam_datosDeltaTDouble;
	int64_t tam_datosAcumDouble2Nivel;
	int64_t tam_datosAcumDouble3Nivel;
	int64_t tam_datosCoefPDouble;
	int64_t tam_datosVolGuardadoDouble2 = ((int64_t) numPuntosGuardarTotal)*sizeof(double2);
	int64_t tam_datosVolGuardadoDouble3 = ((int64_t) numPuntosGuardarTotal)*sizeof(double3);
	double tiempo_ini, tiempo_fin;
//	double deltaTNivelSinTruncar[MAX_LEVELS][MAX_GRIDS_LEVEL];  // Descomentar para mallas anidadas
	double deltaTNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	double tiempoAntSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL];
	double tiempoActSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL];
	double sigTiempoGuardarNetCDF[MAX_LEVELS];
	double sigTiempoGuardarSeries = 0.0;
	bool hayQueCrearAlgunFicheroNetCDF = false;
	int l, i, j, k;
	int nvx, nvy;
//	int nvxNivelSup, nvyNivelSup, pos_sup;  // Descomentar para mallas anidadas
	int pos, iter;
	// Usada solo como índice en la asignación de punteros GPU y en las últimas escrituras en NetCDF
	int num;
	// Para la denormalización de las métricas globales
	int j_global;
	double val_cos;

	// El cudaSetDevice se hizo al reservar memoria en Problema.cxx
	// Tipos tipo_double2 y tipo_double3
	tipo[0] = MPI_DOUBLE;
	tipo[1] = MPI_DOUBLE;
	tipo[2] = MPI_DOUBLE;
	disp[0] = 0;
	disp[1] = sizeof(double);
	disp[2] = 2*sizeof(double);
	MPI_Type_create_struct(2, blocklen, disp, tipo, &tipo_double2);
	MPI_Type_create_struct(3, blocklen, disp, tipo, &tipo_double3);
	MPI_Type_commit(&tipo_double2);
	MPI_Type_commit(&tipo_double3);

	// Tipo tipo_int2 (el dato que se envía para los warnings de los valores de la deformación)
	tipo[0] = MPI_INT;
	tipo[1] = MPI_INT;
	disp[0] = 0;
	disp[1] = sizeof(int);
	MPI_Type_create_struct(2, blocklen, disp, tipo, &tipo_int2);
	MPI_Type_commit(&tipo_int2);

	// Tipos tipoFilas* y tipoColumnas* para cada submalla (los datos que se envían con MPI)
	for (l=0; l<numNiveles; l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				// double2
				MPI_Type_vector(1, 2*(nvx+4), nvx+4, tipo_double2, &(tipoFilasDouble2[l][i]));
				MPI_Type_commit(&(tipoFilasDouble2[l][i]));
				MPI_Type_vector(nvy, 2, 8, tipo_double2, &(tipoColumnasDouble2[l][i]));
				MPI_Type_commit(&(tipoColumnasDouble2[l][i]));
				// double3
				MPI_Type_vector(1, 2*(nvx+4), nvx+4, tipo_double3, &(tipoFilasDouble3[l][i]));
				MPI_Type_commit(&(tipoFilasDouble3[l][i]));
				MPI_Type_vector(nvy, 2, 8, tipo_double3, &(tipoColumnasDouble3[l][i]));
				MPI_Type_commit(&(tipoColumnasDouble3[l][i]));
				// double (presiones)
				MPI_Type_vector(nvy+1, 1, 4, MPI_DOUBLE, &(tipoColumnasPresion[l][i]));
				MPI_Type_commit(&(tipoColumnasPresion[l][i]));
			}
		}
	}

	// Obtenemos id_hebraX, id_hebraY y los identificadores de las hebras adyacentes
	MPI_Cart_coords(comm_cartesiano, id_hebra, 2, coord_proc);
	id_hebraX = coord_proc[0];
	id_hebraY = coord_proc[1];
	MPI_Cart_shift(comm_cartesiano, 0, 1, &hebra_izq, &hebra_der);
	MPI_Cart_shift(comm_cartesiano, 1, 1, &hebra_sup, &hebra_inf);

	// Asignamos el número de streams y los creamos
	k = 1;
	for (l=1; l<numNiveles; l++)
		k = max(k, numSubmallasNivel[l]);
	if (k > 8) nstreams = 16;
	else if (k > 4) nstreams = 8;
	else if (k > 2) nstreams = 4;
	else if (k > 1) nstreams = 2;
	else nstreams = 1;
	cudaStreamCreate(&streamMemcpy);
	for (i=0; i<nstreams; i++)
		cudaStreamCreate(streams+i);
	nstreams--;  // Para que funcione el AND bitwise (potencia de 2 menos 1)

	// Creamos los comunicadores de las submallas
	comunicadores[0][0] = comm_cartesiano;
	for (l=1; l<numNiveles; l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			k = ((datosClusterCPU[l][i].iniy != -1) ? 1 : 0);
			MPI_Comm_split(comm_cartesiano, k, id_hebra, &(comunicadores[l][i]));
		}
	}

	ultimaHebraXSubmalla[0][0] = ((id_hebraX == num_procsX-1) ? true : false);
	ultimaHebraYSubmalla[0][0] = ((id_hebraY == num_procsY-1) ? true : false);
	if (okada_flag == DYNAMIC_DEFORMATION) {
		indiceEstadoSigDefDim = 0;
		// blockGridOkada[i] corresponde al estado de la deformación dinámica en el índice de tiempo i
		// Tiene el mismo valor para todos los estados de la deformación dinámica
		for (i=0; i<numEstadosDefDinamica; i++) {
			blockGridOkada[i].x = iDivUp(submallasDeformacion[0].z, NUM_HEBRAS_ANCHO_EST);
			blockGridOkada[i].y = iDivUp(submallasDeformacion[0].w, NUM_HEBRAS_ALTO_EST);
		}
	}
	else {
		// blockGridOkada[i] corresponde a la deformación i-ésima
		for (i=0; i<numFaults; i++) {
			blockGridOkada[i].x = iDivUp(submallasDeformacion[i].z, NUM_HEBRAS_ANCHO_EST);
			blockGridOkada[i].y = iDivUp(submallasDeformacion[i].w, NUM_HEBRAS_ALTO_EST);
		}
	}
	nvx = datosClusterCPU[0][0].numVolx;
	nvy = datosClusterCPU[0][0].numVoly;
	obtenerTamBloquesKernel(nvx, nvy, &(blockGridVer1Nivel[0][0]), &(blockGridVer2Nivel[0][0]), &(blockGridHor1Nivel[0][0]),
		&(blockGridHor2Nivel[0][0]), &(blockGridEstNivel[0][0]), &(blockGridVertNivel[0][0]));
	obtenerTamBloqueKernelFan(nvx+4, nvy+4, &(blockGridFanNivel[0][0]));
	for (l=1; l<numNiveles; l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				if (datosClusterCPU[l][i].inix + nvx == submallasNivel[l][i].z)
					ultimaHebraXSubmalla[l][i] = true;
				else
					ultimaHebraXSubmalla[l][i] = false;
				if (datosClusterCPU[l][i].iniy + nvy == submallasNivel[l][i].w)
					ultimaHebraYSubmalla[l][i] = true;
				else
					ultimaHebraYSubmalla[l][i] = false;
				obtenerTamBloquesKernel(nvx, nvy, &(blockGridVer1Nivel[l][i]), &(blockGridVer2Nivel[l][i]), &(blockGridHor1Nivel[l][i]),
					&(blockGridHor2Nivel[l][i]), &(blockGridEstNivel[l][i]), &(blockGridVertNivel[l][i]));
				obtenerTamBloqueKernelFan(nvx+4, nvy+4, &(blockGridFanNivel[l][i]));
			}
			else {
//				deltaTNivelSinTruncar[l][i] = DBL_MAX;  // Descomentar para mallas anidadas
				deltaTNivel[l][i] = DBL_MAX;
			}
		}
	}

	// Inicializamos sigTiempoGuardarNetCDF y comprobamos si hay que crear el fichero NetCDF para cada submalla
	guardarEtaMax = false;
	guardarVelocidadesMax = false;
	guardarModuloVelocidadMax = false;
	guardarModuloCaudalMax = false;
	guardarFlujoMomentoMax = false;
	guardarTiemposLlegada = false;
	for (l=0; l<numNiveles; l++) {
		sigTiempoGuardarNetCDF[l] = 0.0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			crearFicheroNetCDF[l][i] = false;
			if ((datosClusterCPU[l][i].iniy != -1) && (tiempoGuardarNetCDF[l] >= 0.0)) {
				hayQueCrearAlgunFicheroNetCDF = true;
				p_var = &(guardarVariables[l][i]);
				if ((p_var->eta != 0) || (p_var->eta_max != 0) || (p_var->velocidades != 0) || (p_var->velocidades_max != 0) ||
					(p_var->modulo_velocidades != 0) || (p_var->flujo_momento != 0) || (p_var->modulo_velocidades_max != 0) ||
					(p_var->modulo_caudales_max != 0) || (p_var->presion_no_hidrostatica != 0) || (p_var->flujo_momento_max != 0) ||
					(p_var->tiempos_llegada != 0)) {
					crearFicheroNetCDF[l][i] = true;
				}
				if (p_var->eta_max != 0)
					 guardarEtaMax = true;
				if (p_var->velocidades_max != 0)
					guardarVelocidadesMax = true;
				if (p_var->modulo_velocidades_max != 0)
					 guardarModuloVelocidadMax = true;
				if (p_var->modulo_caudales_max != 0)
					 guardarModuloCaudalMax = true;
				if (p_var->flujo_momento_max != 0)
					 guardarFlujoMomentoMax = true;
				if (p_var->tiempos_llegada != 0)
					 guardarTiemposLlegada = true;
			}
		}
	}

	// Reservamos memoria en GPU
	err = 0;
	tam_datosDeltaTDouble = 0;
	tam_datosAcumDouble2Nivel = 0;
	tam_datosAcumDouble3Nivel = 0;
	tam_datosCoefPDouble = 0;
	for (l=0; l<numNiveles; l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvy = submallasNivel[l][i].w;
				d_datosNivel[l][i].dx = datosNivel[l][i].dx;
				d_datosNivel[l][i].dy = datosNivel[l][i].dy;
				cudaMalloc((void **)&(d_datosNivel[l][i].vcos), (2*nvy+1)*sizeof(double));
				cudaMalloc((void **)&(d_datosNivel[l][i].vccos), nvy*sizeof(double));
				cudaMalloc((void **)&(d_datosNivel[l][i].vtan), nvy*sizeof(double));
			}
		}
		factorCorreccionNivel[l] = 1.0/(ratioRefNivel[l]*ratioRefNivel[l]);
		tam_datosVolDoubleNivel[l] = numVolumenesNivel[l]*sizeof(double);
		tam_datosVolDouble2Nivel[l] = numVolumenesNivel[l]*sizeof(double2);
		tam_datosVolDouble3Nivel[l] = numVolumenesNivel[l]*sizeof(double3);
		tam_datosVertDoubleNivel[l] = numVerticesNivel[l]*sizeof(double);
		tam_datosDeltaTDouble = max(tam_datosDeltaTDouble, tam_datosVolDoubleNivel[l]);
		tam_datosAcumDouble2Nivel = max(tam_datosAcumDouble2Nivel, tam_datosVolDouble2Nivel[l]);
		tam_datosAcumDouble3Nivel = max(tam_datosAcumDouble3Nivel, tam_datosVolDouble3Nivel[l]);
		tam_datosCoefPDouble = max(tam_datosCoefPDouble, tam_datosVertDoubleNivel[l]);
		cudaMalloc((void **)&(d_datosVolumenesNivel_1[l]), tam_datosVolDouble2Nivel[l]);
		cudaMalloc((void **)&(d_datosVolumenesNivel_2[l]), tam_datosVolDouble3Nivel[l]);
		cudaMalloc((void **)&(d_datosVolumenesNivelSig_1[l]), tam_datosVolDouble2Nivel[l]);
		cudaMalloc((void **)&(d_datosVolumenesNivelSig_2[l]), tam_datosVolDouble3Nivel[l]);
		cudaMalloc((void **)&(d_friccionesNivel[l]), tam_datosVolDoubleNivel[l]);
		cudaMalloc((void **)&(d_eta1InicialNivel[l]), tam_datosVolDoubleNivel[l]);
		if (l > 0) {
			tam_datosCopiaNivel[l] = numVolumenesNivel[l]*sizeof(int);
			cudaMalloc((void **)&(d_posCopiaNivel[l]), tam_datosCopiaNivel[l]);
#if (NESTED_ALGORITHM == 1)
			cudaMalloc((void **)&(d_mediaEtaNivel[l]), numVolumenesNivel[l]*sizeof(double)/(ratioRefNivel[l]*ratioRefNivel[l]));
			cudaMalloc((void **)&(d_mediaNivel_2[l]), numVolumenesNivel[l]*sizeof(double2)/(ratioRefNivel[l]*ratioRefNivel[l]));
#else
			d_mediaEtaNivel[l] = 0;
			d_mediaNivel_2[l] = 0;
#endif
		}
		if ((numNiveles > 1) && (l < numNiveles-1)) {
			tam_datosRefinarNivel[l] = numVolumenesNivel[l]*sizeof(bool);
			cudaMalloc((void **)&(d_refinarNivel[l]), tam_datosRefinarNivel[l]);
			cudaMalloc((void **)&(d_correccionEtaNivel[l]), tam_datosVolDoubleNivel[l]);
			cudaMalloc((void **)&(d_correccionNivel_2[l]), tam_datosVolDouble2Nivel[l]);
		}
		if (l < numNiveles-1)
			cudaMalloc((void **)&(d_deformacionAcumuladaNivel[l]), tam_datosVolDoubleNivel[l]);
		if (guardarEtaMax)
			cudaMalloc((void **)&(d_eta1MaximaNivel[l]), tam_datosVolDoubleNivel[l]);
		if (guardarVelocidadesMax)
			cudaMalloc((void **)&(d_velocidadesMaximasNivel[l]), tam_datosVolDouble3Nivel[l]);
		if (guardarModuloVelocidadMax)
			cudaMalloc((void **)&(d_moduloVelocidadMaximaNivel[l]), tam_datosVolDoubleNivel[l]);
		if (guardarModuloCaudalMax)
			cudaMalloc((void **)&(d_moduloCaudalMaximoNivel[l]), tam_datosVolDoubleNivel[l]);
		if (guardarFlujoMomentoMax)
			cudaMalloc((void **)&(d_flujoMomentoMaximoNivel[l]), tam_datosVolDoubleNivel[l]);
		if (guardarTiemposLlegada)
			cudaMalloc((void **)&(d_tiemposLlegadaNivel[l]), tam_datosVolDoubleNivel[l]);
		if (no_hidros) {
			cudaMalloc((void **)&(d_Pnh0[l]), tam_datosVertDoubleNivel[l]);
			cudaMalloc((void **)&(d_Pnh1[l]), tam_datosVertDoubleNivel[l]);
		}
	}
	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE)) {
		cudaMalloc((void **)&(d_deformacionNivel0[0]), (int64_t) submallasDeformacion[0].z*submallasDeformacion[0].w*sizeof(double));
	}
	else if (okada_flag == DEFORMATION_FROM_FILE) {
		for (i=0; i<numFaults; i++)
			cudaMalloc((void **)&(d_deformacionNivel0[i]), (int64_t) submallasDeformacion[i].z*submallasDeformacion[i].w*sizeof(double));
	}
	else if (okada_flag == DYNAMIC_DEFORMATION) {
		for (i=0; i<numEstadosDefDinamica; i++)
			cudaMalloc((void **)&(d_deformacionNivel0[i]), (int64_t) submallasDeformacion[0].z*submallasDeformacion[0].w*sizeof(double));
	}
	else if ((okada_flag == OKADA_TRIANGULAR) || (okada_flag == OKADA_TRIANGULAR_FROM_FILE)) {
		cudaMalloc((void **)&(d_deformacionNivel0[0]), (int64_t) submallasDeformacion[0].z*submallasDeformacion[0].w*sizeof(double));
		for (i=0; i<numFaults; i++) {
			cudaMalloc((void **)&(d_vc[i]), 4*sizeof(double2));
			cudaMalloc((void **)&(d_vz[i]), 4*sizeof(double));
			cudaMalloc((void **)&(d_SLIPVEC[i]), 3*sizeof(double));
		}
	}
	if (kajiura_flag == 1) {
		int64_t tam_datosKajiura = 0;
		int64_t tam2;
		if (okada_flag == DYNAMIC_DEFORMATION) {
			tam2 = (int64_t) submallasDeformacion[0].z*submallasDeformacion[0].w*sizeof(cuDoubleComplex);
			tam_datosKajiura = max(tam_datosKajiura, tam2);
			for (i=0; i<numEstadosDefDinamica; i++) {
				cudaMalloc((void **)&(d_F2Sx[i]), submallasDeformacion[0].z*sizeof(double));
				cudaMalloc((void **)&(d_F2Sy[i]), submallasDeformacion[0].w*sizeof(double));
			}
		}
		else {
			for (i=0; i<numFaults; i++) {
				tam2 = (int64_t) submallasDeformacion[i].z*submallasDeformacion[i].w*sizeof(cuDoubleComplex);
				tam_datosKajiura = max(tam_datosKajiura, tam2);
				cudaMalloc((void **)&(d_F2Sx[i]), submallasDeformacion[i].z*sizeof(double));
				cudaMalloc((void **)&(d_F2Sy[i]), submallasDeformacion[i].w*sizeof(double));
			}
		}
		cudaMalloc((void **)&d_datosKajiura, tam_datosKajiura);
	}
	if (numNiveles > 1) {
		cudaMalloc((void **)&d_activacionMallasAnidadas, numVolumenesNivel[0]*sizeof(bool));
	}
	cudaMalloc((void **)&d_deltaTVolumenesNivel, tam_datosDeltaTDouble);
	cudaMalloc((void **)&d_acumuladorNivel_1, tam_datosAcumDouble2Nivel);
	cudaMalloc((void **)&d_acumuladorNivel_2, tam_datosAcumDouble3Nivel);
	if (no_hidros) {
		cudaMalloc((void **)&d_RHS_dt, tam_datosCoefPDouble);
		cudaMalloc((void **)&d_dtHs, tam_datosVolDoubleNivel[0]);
		cudaMalloc((void **)&d_CoefPE, tam_datosCoefPDouble);
		cudaMalloc((void **)&d_CoefPW, tam_datosCoefPDouble);
		cudaMalloc((void **)&d_CoefPN, tam_datosCoefPDouble);
		cudaMalloc((void **)&d_CoefPS, tam_datosCoefPDouble);
	}
	err_cuda = cudaMalloc((void **)&d_aristaReconstruido, tam_datosDeltaTDouble*4*5);  // 4 aristas por volumen y 5 puntos por arista
	if ((err_cuda != cudaErrorMemoryAllocation) && (kajiura_flag == 1)) {
		// Comprobamos si hay memoria suficiente para crear los planes para el Kajiura
		cufftHandle plan;
		if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE) || (okada_flag == OKADA_TRIANGULAR) ||
			(okada_flag == OKADA_TRIANGULAR_FROM_FILE) || (okada_flag == DYNAMIC_DEFORMATION)) {
			if (cufftPlan2d(&plan, submallasDeformacion[0].w, submallasDeformacion[0].z, CUFFT_Z2Z) != CUFFT_ALLOC_FAILED)
				cufftDestroy(plan);
			else
				err_cuda = cudaErrorMemoryAllocation;
		}
		else if (okada_flag == DEFORMATION_FROM_FILE) {
			encontrado = false;
			i = 0;
			while ((i < numFaults) && (! encontrado)) {
				if (cufftPlan2d(&plan, submallasDeformacion[i].w, submallasDeformacion[i].z, CUFFT_Z2Z) != CUFFT_ALLOC_FAILED) {
					cufftDestroy(plan);
					i++;
				}
				else {
					err_cuda = cudaErrorMemoryAllocation;
					encontrado = true;
				}
			}
		}
	}
	if (err_cuda == cudaErrorMemoryAllocation) {
		for (l=0; l<numNiveles; l++) {
			for (i=0; i<numSubmallasNivel[l]; i++) {
				if (datosClusterCPU[l][i].iniy != -1) {
					cudaFree(d_datosNivel[l][i].vcos);
					cudaFree(d_datosNivel[l][i].vccos);
					cudaFree(d_datosNivel[l][i].vtan);
				}
			}
			cudaFree(d_datosVolumenesNivel_1[l]);
			cudaFree(d_datosVolumenesNivel_2[l]);
			cudaFree(d_datosVolumenesNivelSig_1[l]);
			cudaFree(d_datosVolumenesNivelSig_2[l]);
			cudaFree(d_friccionesNivel[l]);
			cudaFree(d_eta1InicialNivel[l]);
			if (l > 0) {
				cudaFree(d_posCopiaNivel[l]);
				cudaFree(d_mediaEtaNivel[l]);
				cudaFree(d_mediaNivel_2[l]);
			}
			if ((numNiveles > 1) && (l < numNiveles-1)) {
				cudaFree(d_refinarNivel[l]);
				cudaFree(d_correccionEtaNivel[l]);
				cudaFree(d_correccionNivel_2[l]);
			}
			if (l < numNiveles-1)
				cudaFree(d_deformacionAcumuladaNivel[l]);
			if (guardarEtaMax)
				cudaFree(d_eta1MaximaNivel[l]);
			if (guardarVelocidadesMax)
				cudaFree(d_velocidadesMaximasNivel[l]);
			if (guardarModuloVelocidadMax)
				cudaFree(d_moduloVelocidadMaximaNivel[l]);
			if (guardarModuloCaudalMax)
				cudaFree(d_moduloCaudalMaximoNivel[l]);
			if (guardarFlujoMomentoMax)
				cudaFree(d_flujoMomentoMaximoNivel[l]);
			if (guardarTiemposLlegada)
				cudaFree(d_tiemposLlegadaNivel[l]);
			if (no_hidros) {
				cudaFree(d_Pnh0[l]);
				cudaFree(d_Pnh1[l]);
			}
		}
		if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE)) {
			cudaFree(d_deformacionNivel0[0]);
		}
		else if (okada_flag == DEFORMATION_FROM_FILE) {
			for (i=0; i<numFaults; i++)
				cudaFree(d_deformacionNivel0[i]);
		}
		else if (okada_flag == DYNAMIC_DEFORMATION) {
			for (i=0; i<numEstadosDefDinamica; i++)
				cudaFree(d_deformacionNivel0[i]);
		}
		else if ((okada_flag == OKADA_TRIANGULAR) || (okada_flag == OKADA_TRIANGULAR_FROM_FILE)) {
			cudaFree(d_deformacionNivel0[0]);
			for (i=0; i<numFaults; i++) {
				cudaFree(d_vc[i]);
				cudaFree(d_vz[i]);
				cudaFree(d_SLIPVEC[i]);
			}
		}
		if (kajiura_flag == 1) {
			cudaFree(d_datosKajiura);
			if (okada_flag == DYNAMIC_DEFORMATION) {
				for (i=0; i<numEstadosDefDinamica; i++) {
					cudaFree(d_F2Sx[i]);
					cudaFree(d_F2Sy[i]);
				}
			}
			else {
				for (i=0; i<numFaults; i++) {
					cudaFree(d_F2Sx[i]);
					cudaFree(d_F2Sy[i]);
				}
			}
		}
		if (numNiveles > 1) {
			cudaFree(d_activacionMallasAnidadas);
		}
		cudaFree(d_deltaTVolumenesNivel);
		cudaFree(d_acumuladorNivel_1);
		cudaFree(d_acumuladorNivel_2);
		if (no_hidros) {
			cudaFree(d_RHS_dt);
			cudaFree(d_dtHs);
			cudaFree(d_CoefPE);
			cudaFree(d_CoefPW);
			cudaFree(d_CoefPN);
			cudaFree(d_CoefPS);
		}
		err = 1;
	}
	if (leer_fichero_puntos == 1) {
		cudaMalloc((void **)&d_datosVolumenesNivelGPU_1, numNiveles*sizeof(double2 *));
		cudaMalloc((void **)&d_datosVolumenesNivelGPU_2, numNiveles*sizeof(double3 *));
		cudaMalloc((void **)&d_posicionesVolumenesGuardado, (int64_t) numPuntosGuardarTotal*sizeof(int4));
		cudaMalloc((void **)&d_datosVolumenesGuardado_1, tam_datosVolGuardadoDouble2);
		err_cuda = cudaMalloc((void **)&d_datosVolumenesGuardado_2, tam_datosVolGuardadoDouble3);
		if (err_cuda == cudaErrorMemoryAllocation) {
			cudaFree(d_datosVolumenesNivelGPU_1);
			cudaFree(d_datosVolumenesNivelGPU_2);
			cudaFree(d_posicionesVolumenesGuardado);
			cudaFree(d_datosVolumenesGuardado_1);
			liberarMemoria(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_friccionesNivel,
				d_datosNivel, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, numSubmallasNivel, d_posCopiaNivel, d_activacionMallasAnidadas,
				d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2, d_aristaReconstruido, d_refinarNivel, d_correccionEtaNivel,
				d_correccionNivel_2, d_mediaEtaNivel, d_mediaNivel_2, 0, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
				d_datosVolumenesGuardado_2, d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2, guardarEtaMax, guardarVelocidadesMax,
				guardarModuloVelocidadMax, guardarModuloCaudalMax, guardarFlujoMomentoMax, guardarTiemposLlegada, d_eta1InicialNivel,
				d_eta1MaximaNivel, d_velocidadesMaximasNivel, d_moduloVelocidadMaximaNivel, d_moduloCaudalMaximoNivel, d_flujoMomentoMaximoNivel,
				d_tiemposLlegadaNivel, okada_flag, kajiura_flag, numFaults, numEstadosDefDinamica, d_deformacionNivel0, d_deformacionAcumuladaNivel,
				d_vc, d_vz, d_SLIPVEC, d_datosKajiura, d_F2Sx, d_F2Sy, d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0, d_Pnh1);
			err = 1;
		}
	}

	// Comprobamos si se ha producido un error en algún proceso
	MPI_Allreduce(&err, &err_total, 1, MPI_INT, MPI_MAX, comm_cartesiano);

	// Asignamos los punteros de datosClusterGPU
	for (l=0; l<numNiveles; l++) {
		pos = num = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				// j = número de volúmenes en una fila contando los volúmenes fantasma
				j = nvx + 4;
				datosClusterGPU[l][i].d_datosVolumenesComClusterIzq_1 = d_datosVolumenesNivelSig_1[l] + pos + 2*j + 2;
				datosClusterGPU[l][i].d_datosVolumenesComClusterIzq_2 = d_datosVolumenesNivelSig_2[l] + pos + 2*j + 2;
				datosClusterGPU[l][i].d_datosVolumenesComClusterDer_1 = d_datosVolumenesNivelSig_1[l] + pos + 2*j + nvx;
				datosClusterGPU[l][i].d_datosVolumenesComClusterDer_2 = d_datosVolumenesNivelSig_2[l] + pos + 2*j + nvx;
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterIzq_1 = d_datosVolumenesNivelSig_1[l] + pos + 2*j + nvx + 2;
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterIzq_2 = d_datosVolumenesNivelSig_2[l] + pos + 2*j + nvx + 2;
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterDer_1 = d_datosVolumenesNivelSig_1[l] + pos + 2*j;
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterDer_2 = d_datosVolumenesNivelSig_2[l] + pos + 2*j;

				datosClusterGPU[l][i].d_datosVolumenesComClusterSup_1 = d_datosVolumenesNivelSig_1[l] + pos + 2*j;
				datosClusterGPU[l][i].d_datosVolumenesComClusterSup_2 = d_datosVolumenesNivelSig_2[l] + pos + 2*j;
				datosClusterGPU[l][i].d_datosVolumenesComClusterInf_1 = d_datosVolumenesNivelSig_1[l] + pos + j*nvy;
				datosClusterGPU[l][i].d_datosVolumenesComClusterInf_2 = d_datosVolumenesNivelSig_2[l] + pos + j*nvy;
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterSup_1 = d_datosVolumenesNivelSig_1[l] + pos + j*(nvy + 2);
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterSup_2 = d_datosVolumenesNivelSig_2[l] + pos + j*(nvy + 2);
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterInf_1 = d_datosVolumenesNivelSig_1[l] + pos;
				datosClusterGPU[l][i].d_datosVolumenesComOtroClusterInf_2 = d_datosVolumenesNivelSig_2[l] + pos;

				datosClusterGPU[l][i].d_datosVerticesComClusterIzq_P = d_Pnh0[l] + num + (nvx+3) + 1;
				datosClusterGPU[l][i].d_datosVerticesComClusterDer_P = d_Pnh0[l] + num + (nvx+3) + nvx+1;
				datosClusterGPU[l][i].d_datosVerticesComOtroClusterIzq_P = d_Pnh0[l] + num + (nvx+3) + nvx+2;
				datosClusterGPU[l][i].d_datosVerticesComOtroClusterDer_P = d_Pnh0[l] + num + (nvx+3);

				datosClusterGPU[l][i].d_datosVerticesComClusterSup_P = d_Pnh0[l] + num + 1*(nvx+3);
				datosClusterGPU[l][i].d_datosVerticesComClusterInf_P = d_Pnh0[l] + num + (nvy+1)*(nvx+3);
				datosClusterGPU[l][i].d_datosVerticesComOtroClusterSup_P = d_Pnh0[l] + num + (nvy+2)*(nvx+3);
				datosClusterGPU[l][i].d_datosVerticesComOtroClusterInf_P = d_Pnh0[l] + num;

				pos += (nvx + 4)*(nvy + 4);
				num += (nvx + 3)*(nvy + 3);
			}
		}
	}

	// Fijamos los tamaños relativos de la caché L1 y la memoria compartida
	cudaFuncSetCacheConfig(calcularCoeficientesReconstruccionNoCom, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(calcularCoeficientesReconstruccionCom, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(procesarFlujoLonGPU, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(procesarFlujoLatGPU, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(obtenerEstadoYDeltaTVolumenesGPU_RKTVD, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(compute_NH_coefficientsNoCom, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(compute_NH_coefficientsCom, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(compute_NH_pressureNoCom, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(compute_NH_pressureCom, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(NH_correction, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(obtenerMediaSubmallaNivel1GPU, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(corregirSolucionSubmallaValoresNivel1GPU, cudaFuncCachePreferShared);

	if (err_total != 1) {
		if (kajiura_flag == 1) {
			iter = crearDatosCPUKajiura(okada_flag, datosNivel, numFaults, numEstadosDefDinamica, submallasDeformacion, F2Sx, F2Sy, H);
			if (iter == 1) {
				liberarMemoria(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_friccionesNivel,
					d_datosNivel, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, numSubmallasNivel, d_posCopiaNivel, d_activacionMallasAnidadas,
					d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2, d_aristaReconstruido, d_refinarNivel, d_correccionEtaNivel,
					d_correccionNivel_2, d_mediaEtaNivel, d_mediaNivel_2, leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
					d_datosVolumenesGuardado_2, d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2, guardarEtaMax, guardarVelocidadesMax,
					guardarModuloVelocidadMax, guardarModuloCaudalMax, guardarFlujoMomentoMax, guardarTiemposLlegada, d_eta1InicialNivel,
					d_eta1MaximaNivel, d_velocidadesMaximasNivel, d_moduloVelocidadMaximaNivel, d_moduloCaudalMaximoNivel, d_flujoMomentoMaximoNivel,
					d_tiemposLlegadaNivel, okada_flag, kajiura_flag, numFaults, numEstadosDefDinamica, d_deformacionNivel0, d_deformacionAcumuladaNivel,
					d_vc, d_vz, d_SLIPVEC, d_datosKajiura, d_F2Sx, d_F2Sy, d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0, d_Pnh1);
				return 2;
			}
		}
		// Copiamos los datos de CPU a GPU
		for (l=0; l<numNiveles; l++) {
			for (i=0; i<numSubmallasNivel[l]; i++) {
				if (datosClusterCPU[l][i].iniy != -1) {
					nvy = submallasNivel[l][i].w;
					cudaMemcpy(d_datosNivel[l][i].vcos, datosNivel[l][i].vcos, (2*nvy+1)*sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpy(d_datosNivel[l][i].vccos, datosNivel[l][i].vccos, nvy*sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpy(d_datosNivel[l][i].vtan, datosNivel[l][i].vtan, nvy*sizeof(double), cudaMemcpyHostToDevice);
				}
			}
			cudaMemcpy(d_datosVolumenesNivel_1[l], datosVolumenesNivel_1[l], tam_datosVolDouble2Nivel[l], cudaMemcpyHostToDevice);
			cudaMemcpy(d_datosVolumenesNivel_2[l], datosVolumenesNivel_2[l], tam_datosVolDouble3Nivel[l], cudaMemcpyHostToDevice);
			if (l > 0)
				cudaMemcpy(d_posCopiaNivel[l], posCopiaNivel[l], tam_datosCopiaNivel[l], cudaMemcpyHostToDevice);
			if ((numNiveles > 1) && (l < numNiveles-1))
				cudaMemcpy(d_refinarNivel[l], refinarNivel[l], tam_datosRefinarNivel[l], cudaMemcpyHostToDevice);
			if (l < numNiveles-1)
				cudaMemset(d_deformacionAcumuladaNivel[l], 0, tam_datosVolDoubleNivel[l]);
			if (no_hidros) {
				cudaMemset(d_Pnh0[l], 0, tam_datosVertDoubleNivel[l]);
				cudaMemset(d_Pnh1[l], 0, tam_datosVertDoubleNivel[l]);
			}
		}
		if (okada_flag == DEFORMATION_FROM_FILE) {
			for (i=0; i<numFaults; i++) {
				int64_t tam2 = (int64_t) submallasDeformacion[i].z*submallasDeformacion[i].w*sizeof(double);
				cudaMemcpy(d_deformacionNivel0[i], deformacionNivel0[i], tam2, cudaMemcpyHostToDevice);
			}
		}
		else if (okada_flag == DYNAMIC_DEFORMATION) {
			int64_t tam2 = (int64_t) submallasDeformacion[0].z*submallasDeformacion[0].w*sizeof(double);
			for (i=0; i<numEstadosDefDinamica; i++) {
				cudaMemcpy(d_deformacionNivel0[i], deformacionNivel0[i], tam2, cudaMemcpyHostToDevice);
			}
		}
		else if ((okada_flag == OKADA_TRIANGULAR) || (okada_flag == OKADA_TRIANGULAR_FROM_FILE)) {
			for (i=0; i<numFaults; i++) {
				cudaMemcpy(d_vc[i], vc[i], 4*sizeof(double2), cudaMemcpyHostToDevice);
				cudaMemcpy(d_vz[i], DEPTH_v[i], 4*sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(d_SLIPVEC[i], SLIPVEC[i], 3*sizeof(double), cudaMemcpyHostToDevice);
			}
		}
		if (kajiura_flag == 1) {
			if (okada_flag == DYNAMIC_DEFORMATION) {
				for (i=0; i<numEstadosDefDinamica; i++) {
					cudaMemcpy(d_F2Sx[i], F2Sx[i], submallasDeformacion[0].z*sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpy(d_F2Sy[i], F2Sy[i], submallasDeformacion[0].w*sizeof(double), cudaMemcpyHostToDevice);
				}
			}
			else {
				for (i=0; i<numFaults; i++) {
					cudaMemcpy(d_F2Sx[i], F2Sx[i], submallasDeformacion[i].z*sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpy(d_F2Sy[i], F2Sy[i], submallasDeformacion[i].w*sizeof(double), cudaMemcpyHostToDevice);
				}
			}
		}
		if (no_hidros) {
		    cargarIndices(numPesosJacobi, omega, ord);
			cudaMemset(d_dtHs, 0, tam_datosVolDoubleNivel[0]);
		}
		if (leer_fichero_puntos == 1) {
			cudaMemcpy(d_datosVolumenesNivelGPU_1, d_datosVolumenesNivel_1, numNiveles*sizeof(double2 *), cudaMemcpyHostToDevice);
			cudaMemcpy(d_datosVolumenesNivelGPU_2, d_datosVolumenesNivel_2, numNiveles*sizeof(double3 *), cudaMemcpyHostToDevice);
			cudaMemcpy(d_posicionesVolumenesGuardado, posicionesVolumenesGuardado, (int64_t) numPuntosGuardarTotal*sizeof(int4), cudaMemcpyHostToDevice);
		}
		cudaMemcpy(d_friccionesNivel[0], friccionesNivel[0], tam_datosVolDoubleNivel[0], cudaMemcpyHostToDevice);
		if (tipo_friccion == VARIABLE_FRICTION_ALL) {
			for (l=1; l<numNiveles; l++)
				cudaMemcpy(d_friccionesNivel[l], friccionesNivel[l], tam_datosVolDoubleNivel[l], cudaMemcpyHostToDevice);
		}
		normalizarEInterpolarFricciones(numNiveles, datosClusterCPU, submallasNivel, numSubmallasNivel, tipo_friccion, d_friccionesNivel,
			submallaNivelSuperior, ratioRefNivel, blockGridEstNivel, blockGridFanNivel, threadBlockEst, L, H, streams, nstreams,
			id_hebraX, id_hebraY, ultimaHebraXSubmalla[0][0], ultimaHebraYSubmalla[0][0]);
		if (numNiveles > 1) {
			cudaMemset(d_activacionMallasAnidadas, 0, numVolumenesNivel[0]*sizeof(bool));
			inicializarActivacionMallasAnidadasGPU<<<blockGridEstNivel[0][0], threadBlockEst>>>(d_activacionMallasAnidadas,
				d_refinarNivel[0], datosClusterCPU[0][0].numVolx, datosClusterCPU[0][0].numVoly);
		}
		cudaDeviceSynchronize();

		if (continuar_simulacion == 1) {
			// Continuación de una simulación anterior
			// Asignamos el tiempo inicial (tiempo_continuar se asignó en Problema.cxx)
			for (l=0; l<numNiveles; l++) {
				for (i=0; i<numSubmallasNivel[l]; i++) {
					tiempoAntSubmalla[l][i] = tiempo_continuar;
					tiempoActSubmalla[l][i] = tiempo_continuar;
				}
			}

			if (id_hebra == 0) {
				fprintf(stdout, "\nNetCDF result files found with the specified stored variables and parameter values. ");
				fprintf(stdout, "Resuming the simulation at %g sec\n", tiempoActSubmalla[0][0]*T);
			}

			sigTiempoGuardarSeries = tiempoActSubmalla[0][0] + tiempoGuardarSeries;
			for (l=0; l<numNiveles; l++)
				sigTiempoGuardarNetCDF[l] = tiempoActSubmalla[0][0] + tiempoGuardarNetCDF[l];
			// Saltamos las fallas de Okada anteriores al tiempo actual
			while ((defTime[fallaOkada] <= 0.0) || (defTime[fallaOkada] < tiempoActSubmalla[0][0]*T))
				fallaOkada++;
		}
		else {
			// Nueva simulación
			// Asignamos el tiempo inicial
			for (l=0; l<numNiveles; l++) {
				for (i=0; i<numSubmallasNivel[l]; i++) {
					tiempoAntSubmalla[l][i] = 0.0;
					tiempoActSubmalla[l][i] = 0.0;
				}
			}

			if (id_hebra == 0) {
				if (continuar_simulacion == 2)
					fprintf(stdout, "\nNetCDF result files found but eta, ux or uy were not found in some of them. Starting a new simulation.\n");
				else if (continuar_simulacion == 3)
					fprintf(stdout, "\nNetCDF result files found but stored variables do not agree. Starting a new simulation.\n");
				else if (continuar_simulacion == 4)
					fprintf(stdout, "\nNetCDF result files found but the values of some parameters do not agree. Starting a new simulation.\n");
				else if (continuar_simulacion == 5)
					fprintf(stdout, "\nSome NetCDF result files not found. Starting a new simulation.\n");
			}

			inicializarEta1Maxima(numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_eta1InicialNivel, submallasNivel,
				numSubmallasNivel, d_datosNivel, hmin_metrics, blockGridEstNivel, threadBlockEst, streams, nstreams);
			if (guardarEtaMax) {
				inicializarEta1Maxima(numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_eta1MaximaNivel, submallasNivel,
					numSubmallasNivel, d_datosNivel, hmin_metrics, blockGridEstNivel, threadBlockEst, streams, nstreams);
			}
			if (guardarVelocidadesMax) {
				inicializarVelocidadesMaximas(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
					d_velocidadesMaximasNivel, submallasNivel, numSubmallasNivel, epsilon_h, d_datosNivel, hmin_metrics,
					blockGridEstNivel, threadBlockEst, streams, nstreams);
			}
			if (guardarModuloVelocidadMax) {
				inicializarModuloVelocidadMaxima(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
					d_moduloVelocidadMaximaNivel, submallasNivel, numSubmallasNivel, epsilon_h, d_datosNivel, hmin_metrics,
					blockGridEstNivel, threadBlockEst, H, U, T, streams, nstreams);
			}
			if (guardarModuloCaudalMax) {
				inicializarModuloCaudalMaximo(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
					d_moduloCaudalMaximoNivel, submallasNivel, numSubmallasNivel, d_datosNivel, hmin_metrics, blockGridEstNivel,
					threadBlockEst, H, U, T, streams, nstreams);
			}
			if (guardarFlujoMomentoMax) {
				inicializarFlujoMomentoMaximo(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
					d_flujoMomentoMaximoNivel, submallasNivel, numSubmallasNivel, epsilon_h, d_datosNivel, hmin_metrics,
					blockGridEstNivel, threadBlockEst, H, U, T, streams, nstreams);
			}
			if (guardarTiemposLlegada) {
				inicializarTiemposLlegada(numNiveles, datosClusterCPU, d_tiemposLlegadaNivel, submallasNivel, numSubmallasNivel,
					blockGridEstNivel, threadBlockEst, streams, nstreams);
			}

			// Aplicamos gaussiana si es necesario
			if (okada_flag == GAUSSIAN) {
				// Aplicamos la gaussiana inicial
				if (id_hebra == 0)
					fprintf(stdout, "Applying gaussian\n");
				aplicarGaussiana(numNiveles, datosClusterCPU, datosNivel, d_datosNivel, submallasNivel, numSubmallasNivel,
					d_datosVolumenesNivel_1, lonGauss, latGauss, heightGauss, sigmaGauss, blockGridEstNivel, threadBlockEst,
					H, streams, nstreams);
			}
			cudaDeviceSynchronize();
		}

/*		for (l=1; l<numNiveles; l++) {
			pos = k = 0;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				if (datosClusterCPU[l][i].iniy != -1) {
					nvx = datosClusterCPU[l][i].numVolx;
					nvy = datosClusterCPU[l][i].numVoly;
					j = submallaNivelSuperior[l][i];
					pos_sup = posSubmallaNivelSuperior[l][i];
					nvxNivelSup = submallasNivel[l-1][j].z;
					nvyNivelSup = submallasNivel[l-1][j].w;
					// Inicializamos el fondo de los volúmenes fantasma de las submallas del nivel 1
					obtenerFondoVolumenesFantasmaNivel1GPU<<<blockGridFanNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
						d_datosVolumenesNivel_1[l-1]+pos_sup, d_datosVolumenesNivel_1[l]+pos, nvxNivelSup, nvyNivelSup, submallasNivel[l][i].x,
						submallasNivel[l][i].y, nvx, nvy, d_posCopiaNivel[l]+pos, d_datosVolumenesNivel_1[l], ratioRefNivel[l],
						datosClusterCPU[l-1][j].inix, datosClusterCPU[l-1][j].iniy, datosClusterCPU[l-1][j].numVolx,
						datosClusterCPU[l-1][j].numVoly, datosClusterCPU[l][i].inix, datosClusterCPU[l][i].iniy,
						ultimaHebraXSubmalla[l-1][j], ultimaHebraYSubmalla[l-1][j], ultimaHebraXSubmalla[l][i],
						ultimaHebraYSubmalla[l][i], l-1);
#if (NESTED_ALGORITHM == 1)
					// Obtenemos los valores medios iniciales para cada celda gruesa
					obtenerMediaSubmallaNivel1GPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(d_mediaEtaNivel[l]+k,
						d_mediaNivel_2[l]+k, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos, nvx, nvy,
						ratioRefNivel[l], factorCorreccionNivel[l]);
#endif
					// Inicializamos las celdas fantasma de la submalla
					obtenerSiguientesDatosVolumenesFantasmaSecoMojadoNivel1GPU<<<blockGridFanNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
						d_datosVolumenesNivel_1[l-1]+pos_sup, d_datosVolumenesNivel_2[l-1]+pos_sup, d_datosVolumenesNivel_1[l-1]+pos_sup,
						d_datosVolumenesNivel_2[l-1]+pos_sup, d_datosVolumenesNivel_1[l]+pos, d_datosVolumenesNivel_2[l]+pos,
						nvxNivelSup, nvyNivelSup, submallasNivel[l][i].x, submallasNivel[l][i].y, nvx, nvy, d_posCopiaNivel[l]+pos,
						d_datosVolumenesNivel_1[l], d_datosVolumenesNivel_2[l], 1.0, ratioRefNivel[l], borde_sup, borde_inf,
						borde_izq, borde_der, datosClusterCPU[l-1][j].inix, datosClusterCPU[l-1][j].iniy, datosClusterCPU[l-1][j].numVolx,
						datosClusterCPU[l-1][j].numVoly, datosClusterCPU[l][i].inix, datosClusterCPU[l][i].iniy, ultimaHebraXSubmalla[l-1][j],
						ultimaHebraYSubmalla[l-1][j], ultimaHebraXSubmalla[l][i], ultimaHebraYSubmalla[l][i], l-1);
					pos += (nvx + 4)*(nvy + 4);
					k += (nvx*nvy)/(ratioRefNivel[l]*ratioRefNivel[l]);
				}
			}
		}*/
		// Copiamos los datos en datosVolumenesNivelSig. También es necesario para que el fondo esté inicializado
		// al interpolar los espesores de las celdas fantasma en siguientePasoNivel
		for (l=0; l<numNiveles; l++) {
			cudaMemcpy(d_datosVolumenesNivelSig_1[l], d_datosVolumenesNivel_1[l], tam_datosVolDouble2Nivel[l], cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_datosVolumenesNivelSig_2[l], d_datosVolumenesNivel_2[l], tam_datosVolDouble3Nivel[l], cudaMemcpyDeviceToDevice);
		}
		cudaDeviceSynchronize();
		MPI_Barrier(comm_cartesiano);

		// INICIO NETCDF
		if ((continuar_simulacion == 1) || hayQueCrearAlgunFicheroNetCDF) {
			vec1 = (float *) malloc(tam_datosDeltaTDouble);
			vec2 = (float *) malloc(tam_datosDeltaTDouble);
			vec3 = (float *) malloc(tam_datosDeltaTDouble);
			if (vec3 == NULL) {
				if (vec1 != NULL) free(vec1);
				if (vec2 != NULL) free(vec2);
				if (kajiura_flag == 1) {
					for (i=0; i<numFaults; i++) {
						free(F2Sx[i]);
						free(F2Sy[i]);
					}
				}
				liberarMemoria(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_friccionesNivel,
					d_datosNivel, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, numSubmallasNivel, d_posCopiaNivel, d_activacionMallasAnidadas,
					d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2, d_aristaReconstruido, d_refinarNivel, d_correccionEtaNivel,
					d_correccionNivel_2, d_mediaEtaNivel, d_mediaNivel_2, leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
					d_datosVolumenesGuardado_2, d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2, guardarEtaMax, guardarVelocidadesMax,
					guardarModuloVelocidadMax, guardarModuloCaudalMax, guardarFlujoMomentoMax, guardarTiemposLlegada, d_eta1InicialNivel,
					d_eta1MaximaNivel, d_velocidadesMaximasNivel, d_moduloVelocidadMaximaNivel, d_moduloCaudalMaximoNivel, d_flujoMomentoMaximoNivel,
					d_tiemposLlegadaNivel, okada_flag, kajiura_flag, numFaults, numEstadosDefDinamica, d_deformacionNivel0, d_deformacionAcumuladaNivel,
					d_vc, d_vz, d_SLIPVEC, d_datosKajiura, d_F2Sx, d_F2Sy, d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0, d_Pnh1);
				return 2;
			}
			// Creamos o abrimos los ficheros NetCDF
			if (continuar_simulacion == 1) {
				// Continuación de una simulación anterior
				// Cargamos eta1 máxima, la velocidad máxima, el caudal máximo, el flujo de momento máximo y los tiempos de llegada
				// del fichero NetCDF resultado, si es necesario
				if (guardarEtaMax || guardarModuloVelocidadMax || guardarModuloCaudalMax || guardarFlujoMomentoMax || guardarTiemposLlegada) {
					cargarDatosMetricasGPU(numNiveles, datosClusterCPU, okada_flag, datosGRD, datosVolumenesNivel_2, vec1, vec2, d_eta1MaximaNivel,
						d_velocidadesMaximasNivel, d_moduloVelocidadMaximaNivel, d_moduloCaudalMaximoNivel, d_flujoMomentoMaximoNivel,
						d_eta1InicialNivel, d_tiemposLlegadaNivel, Hmin, H, U, T, submallasNivel, numSubmallasNivel, guardarEtaMax,
						guardarVelocidadesMax, guardarModuloVelocidadMax, guardarModuloCaudalMax, guardarFlujoMomentoMax, guardarTiemposLlegada,
						prefijo, guardarVariables, tam_datosVolDoubleNivel, tam_datosVolDouble2Nivel, blockGridEstNivel, comunicadores,
						threadBlockEst, streams, nstreams);
				}

				// Abrimos los ficheros NetCDF
				for (l=0; l<numNiveles; l++) {
					for (i=0; i<numSubmallasNivel[l]; i++) {
						if (datosClusterCPU[l][i].iniy != -1) {
							if (crearFicheroNetCDF[l][i]) {
								readNC(l, i, guardarVariables, (char *) prefijo[l][i].c_str(), submallasNivel[l][i].z, submallasNivel[l][i].w,
									&(nx_nc[l][i]), &(ny_nc[l][i]), npics, okada_flag, tiempo_tot*T, comunicadores[l][i]);
							}
						}
					}
				}
			}
			else {
				// Nueva simulación. Creamos los ficheros NetCDF
				for (l=0; l<numNiveles; l++) {
					pos = 0;
					for (i=0; i<numSubmallasNivel[l]; i++) {
						if (datosClusterCPU[l][i].iniy != -1) {
							nvx = datosClusterCPU[l][i].numVolx;
							nvy = datosClusterCPU[l][i].numVoly;
							if (crearFicheroNetCDF[l][i]) {
								iter = initNC(no_hidros, numPesosJacobi, numNiveles, l, i, guardarVariables, nombre_bati, (char *) prefijo[l][i].c_str(),
											nvx, nvy, datosClusterCPU[l][i].inix, datosClusterCPU[l][i].iniy, submallasNivel[l][i].z, submallasNivel[l][i].w,
											&(nx_nc[l][i]), &(ny_nc[l][i]), npics, datosNivel[l][i].longitud, datosNivel[l][i].latitud, tiempo_tot*T,
											CFL, epsilon_h*H, tipo_friccion, fich_friccion, cfb, vmax*U, difh_at*H, borde_sup, borde_inf, borde_izq,
											borde_der, okada_flag, fich_okada, numFaults, defTime, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE,
											DIP, RAKE, SLIP, LONCTRI, LATCTRI, fich_def, kajiura_flag, depth_kajiura*H, lonGauss, latGauss, heightGauss,
											sigmaGauss, batiOriginal[l]+pos, version, comunicadores[l][i]);
								if (iter == 1) {
									free(vec1);
									free(vec2);
									free(vec3);
									if (kajiura_flag == 1) {
										for (j=0; j<numFaults; j++) {
											free(F2Sx[j]);
											free(F2Sy[j]);
										}
									}
									liberarMemoria(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_friccionesNivel,
										d_datosNivel, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, numSubmallasNivel, d_posCopiaNivel, d_activacionMallasAnidadas,
										d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2, d_aristaReconstruido, d_refinarNivel, d_correccionEtaNivel,
										d_correccionNivel_2, d_mediaEtaNivel, d_mediaNivel_2, leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
										d_datosVolumenesGuardado_2, d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2, guardarEtaMax, guardarVelocidadesMax,
										guardarModuloVelocidadMax, guardarModuloCaudalMax, guardarFlujoMomentoMax, guardarTiemposLlegada, d_eta1InicialNivel,
										d_eta1MaximaNivel, d_velocidadesMaximasNivel, d_moduloVelocidadMaximaNivel, d_moduloCaudalMaximoNivel, d_flujoMomentoMaximoNivel,
										d_tiemposLlegadaNivel, okada_flag, kajiura_flag, numFaults, numEstadosDefDinamica, d_deformacionNivel0, d_deformacionAcumuladaNivel,
										d_vc, d_vz, d_SLIPVEC, d_datosKajiura, d_F2Sx, d_F2Sy, d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0, d_Pnh1);
									return 2;
								}
							}
							pos += (nvx + 4)*(nvy + 4);
						}
					}
				}
			}
			// Reasignamos nx_nc y ny_nc para que sean locales al cluster, y asignamos inix, iniy, inix_nc e iniy_nc
			if (hayQueCrearAlgunFicheroNetCDF) {
				for (l=0; l<numNiveles; l++) {
					for (i=0; i<numSubmallasNivel[l]; i++) {
						if ((datosClusterCPU[l][i].iniy != -1) && (crearFicheroNetCDF[l][i])) {
							// Dimensión X
							nvx = datosClusterCPU[l][i].numVolx;
							for (inix[l][i]=datosClusterCPU[l][i].inix; (inix[l][i])%npics != 0; (inix[l][i])++);
							inix[l][i] = inix[l][i] - datosClusterCPU[l][i].inix;
							if (datosClusterCPU[l][i].inix == 0)
								inix_nc[l][i] = 0;
							else
								inix_nc[l][i] = (datosClusterCPU[l][i].inix-1)/npics + 1;
							nx_nc[l][i] = (datosClusterCPU[l][i].numVolx-1-inix[l][i])/npics + 1;
							// Dimensión Y
							nvy = datosClusterCPU[l][i].numVoly;
							for (iniy[l][i]=datosClusterCPU[l][i].iniy; (iniy[l][i])%npics != 0; (iniy[l][i])++);
							iniy[l][i] = iniy[l][i] - datosClusterCPU[l][i].iniy;
							if (datosClusterCPU[l][i].iniy == 0)
								iniy_nc[l][i] = 0;
							else
								iniy_nc[l][i] = (datosClusterCPU[l][i].iniy-1)/npics + 1;
							ny_nc[l][i] = (datosClusterCPU[l][i].numVoly-1-iniy[l][i])/npics + 1;
						}
					}
				}
			}
		}
		if (leer_fichero_puntos == 1) {
			etaPuntosGuardado = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			etaPuntosGuardadoGlobal = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			uPuntosGuardado = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			uPuntosGuardadoGlobal = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			vPuntosGuardado = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			vPuntosGuardadoGlobal = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			zPuntosGuardado = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			zPuntosGuardadoGlobal = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			etaMinPuntosGuardadoGlobal = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			etaMaxPuntosGuardadoGlobal = (float *) malloc(numPuntosGuardarTotal*sizeof(float));
			if (etaMaxPuntosGuardadoGlobal == NULL) {
				if (etaPuntosGuardado != NULL)			free(etaPuntosGuardado);
				if (etaPuntosGuardadoGlobal != NULL)	free(etaPuntosGuardadoGlobal);
				if (uPuntosGuardado != NULL)			free(uPuntosGuardado);
				if (uPuntosGuardadoGlobal != NULL)		free(uPuntosGuardadoGlobal);
				if (vPuntosGuardado != NULL)			free(vPuntosGuardado);
				if (vPuntosGuardadoGlobal != NULL)		free(vPuntosGuardadoGlobal);
				if (zPuntosGuardado != NULL)			free(zPuntosGuardado);
				if (zPuntosGuardadoGlobal != NULL)		free(zPuntosGuardadoGlobal);
				if (etaMinPuntosGuardadoGlobal != NULL) free(etaMinPuntosGuardadoGlobal);
				if ((continuar_simulacion == 1) || hayQueCrearAlgunFicheroNetCDF) {
					free(vec1);
					free(vec2);
					free(vec3);
				}
				if (kajiura_flag == 1) {
					for (i=0; i<numFaults; i++) {
						free(F2Sx[i]);
						free(F2Sy[i]);
					}
				}
				liberarMemoria(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_friccionesNivel,
					d_datosNivel, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, numSubmallasNivel, d_posCopiaNivel, d_activacionMallasAnidadas,
					d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2, d_aristaReconstruido, d_refinarNivel, d_correccionEtaNivel,
					d_correccionNivel_2, d_mediaEtaNivel, d_mediaNivel_2, leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
					d_datosVolumenesGuardado_2, d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2, guardarEtaMax, guardarVelocidadesMax,
					guardarModuloVelocidadMax, guardarModuloCaudalMax, guardarFlujoMomentoMax, guardarTiemposLlegada, d_eta1InicialNivel,
					d_eta1MaximaNivel, d_velocidadesMaximasNivel, d_moduloVelocidadMaximaNivel, d_moduloCaudalMaximoNivel, d_flujoMomentoMaximoNivel,
					d_tiemposLlegadaNivel, okada_flag, kajiura_flag, numFaults, numEstadosDefDinamica, d_deformacionNivel0, d_deformacionAcumuladaNivel,
					d_vc, d_vz, d_SLIPVEC, d_datosKajiura, d_F2Sx, d_F2Sy, d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0, d_Pnh1);
				return 2;
			}

			if (continuar_simulacion == 1) {
				// Continuación de una simulación anterior.
				// Renombramos el fichero de series de tiempos de la simulación anterior, creamos el nuevo fichero de series de tiempos,
				// traspasamos los datos del fichero antiguo al nuevo, y borramos el fichero antiguo.
				if (id_hebra == 0) {
					rename( ((prefijo[0][0])+"_ts.nc").c_str(), ((prefijo[0][0])+"_ts_old.nc").c_str() );
					initTimeSeriesNC(no_hidros, numPesosJacobi, numNiveles, nombre_bati, (char *) prefijo[0][0].c_str(), numPuntosGuardarTotal,
						lonPuntos, latPuntos, tiempo_tot*T, CFL, epsilon_h*H, tipo_friccion, fich_friccion, cfb, vmax*U, difh_at*H, borde_sup,
						borde_inf, borde_izq, borde_der, okada_flag, fich_okada, numFaults, defTime, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W,
						STRIKE, DIP, RAKE, SLIP, LONCTRI, LATCTRI, fich_def, kajiura_flag, depth_kajiura*H, lonGauss, latGauss, heightGauss,
						sigmaGauss, version);
				}
				MPI_Barrier(comm_cartesiano);
				abrirTimeSeriesOldNC(((prefijo[0][0])+"_ts_old.nc").c_str());
				num_ts = obtenerNumEstadosTimeSeriesOldNC();
				leerAmplitudesTimeSeriesOldNC(etaMinPuntosGuardadoGlobal, etaMaxPuntosGuardadoGlobal);
				if (id_hebra == 0) {
					traspasarDatosTimeSeriesNC(numPuntosGuardarAnt, numPuntosGuardarTotal, num_ts, etaPuntosGuardado, uPuntosGuardado, vPuntosGuardado);
				}
				cerrarTimeSeriesOldNC();
				MPI_Barrier(comm_cartesiano);
				if (id_hebra == 0) {
					remove( ((prefijo[0][0])+"_ts_old.nc").c_str() );
				}
			}
			else {
				// Nueva simulación
				// Creamos el fichero de series de tiempos
				if (id_hebra == 0) {
					initTimeSeriesNC(no_hidros, numPesosJacobi, numNiveles, nombre_bati, (char *) prefijo[0][0].c_str(), numPuntosGuardarTotal,
						lonPuntos, latPuntos, tiempo_tot*T, CFL, epsilon_h*H, tipo_friccion, fich_friccion, cfb, vmax*U, difh_at*H, borde_sup,
						borde_inf, borde_izq, borde_der, okada_flag, fich_okada, numFaults, defTime, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W,
						STRIKE, DIP, RAKE, SLIP, LONCTRI, LATCTRI, fich_def, kajiura_flag, depth_kajiura*H, lonGauss, latGauss, heightGauss,
						sigmaGauss, version);
				}
			}

			for (i=0; i<numPuntosGuardarTotal; i++) {
				if (posicionesVolumenesGuardado[i].x == -1) {
					etaPuntosGuardado[i] = FLT_MAX;
					uPuntosGuardado[i] = FLT_MAX;
					vPuntosGuardado[i] = FLT_MAX;
					zPuntosGuardado[i] = FLT_MAX;
				}
				if (i >= numPuntosGuardarAnt) {
					etaMinPuntosGuardadoGlobal[i] = 1e30f;
					etaMaxPuntosGuardadoGlobal[i] = -1e30f;
				}
			}
		}
		// FIN NETCDF

#if (FILE_WRITING_MODE == 2)
		// Lanzamos la hebra de guardado de datos en NetCDF
		if (hayQueCrearAlgunFicheroNetCDF) {
			terminarHebraNetCDF = false;
			datosPreparados = false;
			escrituraTerminada = true;
			hebraNetCDF = std::thread(funcionHebraNetCDF, no_hidros, datosClusterCPU, numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2,
							datosPnh, datosNivel, submallasNivel, numSubmallasNivel, crearFicheroNetCDF, guardarVariables, guardarEstadoActualNetCDF,
							vec1, vec2, vec3, numEstadoNetCDF, nx_nc, ny_nc, inix, iniy, inix_nc, iniy_nc, npics, Hmin, epsilon_h, H, U, T);
		}
#endif

		MPI_Barrier(comm_cartesiano);
		// Inicializamos los acumuladores del nivel 0
		cudaMemset(d_acumuladorNivel_1, 0, tam_datosVolDouble2Nivel[0]);
		cudaMemset(d_acumuladorNivel_2, 0, tam_datosVolDouble3Nivel[0]);
		// Copiamos los volúmenes de comunicación del nivel 0 a memoria CPU de forma asíncrona
		copiarVolumenesComAsincACPU(&(datosClusterCPU[0][0]), &(datosClusterGPU[0][0]), streamMemcpy, id_hebraX, id_hebraY,
			ultimaHebraXSubmalla[0][0], ultimaHebraYSubmalla[0][0]);

		// CÁLCULO DEL DELTA_T INICIAL
		tiempo_ini = MPI_Wtime();
		deltaTNivel[0][0] = obtenerDeltaTInicialNivel0(datosClusterCPU, d_datosVolumenesNivel_1[0], d_datosVolumenesNivel_2[0],
								&(d_datosNivel[0][0]), d_deltaTVolumenesNivel, CFL, epsilon_h, blockGridEstNivel[0][0],
								threadBlockEst, L, comm_cartesiano);
/*		for (l=1; l<numNiveles; l++) {
			obtenerDeltaTInicialNivel(l, datosClusterCPU[l], d_datosVolumenesNivel_1[l], d_datosVolumenesNivel_2[l], d_datosNivel[l],
				deltaTNivelSinTruncar[l], d_deltaTVolumenesNivel, d_acumuladorNivel_1, CFL, epsilon_h, submallasNivel[l],
				numSubmallasNivel[l], tam_datosVolDouble2Nivel[l], blockGridVer1Nivel[l], blockGridVer2Nivel[l], blockGridHor1Nivel[l],
				blockGridHor2Nivel[l], threadBlockAri, blockGridEstNivel[l], threadBlockEst, streams, nstreams, T);
		}*/
		if (id_hebra == 0) {
			fprintf(stdout, "\nInitial deltaT = %e sec\n", deltaTNivel[0][0]*T);
		}

		MPI_Barrier(comm_cartesiano);
		iter = 1;
		while (tiempoActSubmalla[0][0] < tiempo_tot) {
			// Comprobamos si hay que aplicar alguna deformación en el tiempo actual
			comprobarYAplicarDeformaciones(numNiveles, datosClusterCPU, datosNivel, d_datosNivel, d_datosVolumenesNivelSig_1,
				d_eta1InicialNivel, d_deformacionNivel0, d_deformacionAcumuladaNivel, d_deltaTVolumenesNivel, d_mediaEtaNivel,
				tiempoAntSubmalla, tiempoActSubmalla, ratioRefNivel, factorCorreccionNivel, submallasNivel, numSubmallasNivel,
				submallasDeformacion, submallaNivelSuperior, okada_flag, numFaults, kajiura_flag, depth_kajiura, d_datosKajiura,
				d_F2Sx, d_F2Sy, &fallaOkada, defTime, numEstadosDefDinamica, &indiceEstadoSigDefDim, LON_C, LAT_C, DEPTH_C,
				FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, d_vc, d_vz, LONCTRI, LATCTRI, d_SLIPVEC, blockGridEstNivel,
				blockGridFanNivel, threadBlockEst, blockGridOkada, streams, nstreams, H, T, tipo_int2, id_hebra, comm_cartesiano);

			// Guardamos el estado actual, si procede
			// INICIO NETCDF
			hayQueGuardarEstadoAct = hayQueGuardarEstadoActualEnAlgunNivel(numNiveles, tiempoActSubmalla[0][0], sigTiempoGuardarNetCDF,
										tiempoGuardarNetCDF, hayQueCrearAlgunFicheroNetCDF);
			if (hayQueGuardarEstadoAct) {
#if (FILE_WRITING_MODE == 1)
				// Escritura síncrona de ficheros NetCDF
				tiempoAGuardar = tiempoActSubmalla[0][0];
				asignarGuardarEstadoActualNetCDF(numNiveles, tiempoAGuardar, sigTiempoGuardarNetCDF, tiempoGuardarNetCDF, guardarEstadoActualNetCDF);
				for (l=0; l<numNiveles; l++) {
					if (guardarEstadoActualNetCDF[l]) {
						cudaMemcpy(datosVolumenesNivel_1[l], d_datosVolumenesNivelSig_1[l], tam_datosVolDouble2Nivel[l], cudaMemcpyDeviceToHost);
						cudaMemcpy(datosVolumenesNivel_2[l], d_datosVolumenesNivelSig_2[l], tam_datosVolDouble3Nivel[l], cudaMemcpyDeviceToHost);
						if (no_hidros) {
							cudaMemcpy(datosPnh[l], d_Pnh1[l], tam_datosVertDoubleNivel[l], cudaMemcpyDeviceToHost);
						}
					}
				}
				funcionHebraNetCDF(no_hidros, datosClusterCPU, numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, datosNivel,
					submallasNivel, numSubmallasNivel, crearFicheroNetCDF, guardarVariables, guardarEstadoActualNetCDF, vec1, vec2, vec3,
					numEstadoNetCDF, nx_nc, ny_nc, inix, iniy, inix_nc, iniy_nc, npics, Hmin, epsilon_h, H, U, T);
#else
				// Escritura asíncrona de ficheros NetCDF
				std::unique_lock<std::mutex> lock(mtx);
				cv_escrituraTerminada.wait(lock, [](){ return escrituraTerminada; });
				asignarGuardarEstadoActualNetCDF(numNiveles, tiempoActSubmalla[0][0], sigTiempoGuardarNetCDF, tiempoGuardarNetCDF, guardarEstadoActualNetCDF);
				for (l=0; l<numNiveles; l++) {
					if (guardarEstadoActualNetCDF[l]) {
						cudaMemcpy(datosVolumenesNivel_1[l], d_datosVolumenesNivelSig_1[l], tam_datosVolDouble2Nivel[l], cudaMemcpyDeviceToHost);
						cudaMemcpy(datosVolumenesNivel_2[l], d_datosVolumenesNivelSig_2[l], tam_datosVolDouble3Nivel[l], cudaMemcpyDeviceToHost);
						if (no_hidros) {
							cudaMemcpy(datosPnh[l], d_Pnh1[l], tam_datosVertDoubleNivel[l], cudaMemcpyDeviceToHost);
						}
					}
				}
				tiempoAGuardar = tiempoActSubmalla[0][0];
				escrituraTerminada = false;
				datosPreparados = true;
				lock.unlock();
				cv_datosPreparados.notify_one();
#endif
			}
			if ((tiempoGuardarSeries >= 0.0) && (tiempoActSubmalla[0][0] >= sigTiempoGuardarSeries)) {
				sigTiempoGuardarSeries += tiempoGuardarSeries;
				escribirVolumenesGuardadoGPU<<<blockGridPuntos, threadBlockPuntos>>>(d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2,
					d_datosVolumenesGuardado_1, d_datosVolumenesGuardado_2, d_posicionesVolumenesGuardado, numPuntosGuardarTotal, epsilon_h);
				cudaMemcpy(datosGRD_double2, d_datosVolumenesGuardado_1, tam_datosVolGuardadoDouble2, cudaMemcpyDeviceToHost);
				cudaMemcpy(datosGRD_puntosQ, d_datosVolumenesGuardado_2, tam_datosVolGuardadoDouble3, cudaMemcpyDeviceToHost);
				if (no_hidros) {
					guardarSerieTiemposNoHidrosNivel0(datosClusterCPU, datosNivel, datosGRD_double2, datosGRD_puntosQ, numPuntosGuardarTotal,
						posicionesVolumenesGuardado, etaPuntosGuardado, etaPuntosGuardadoGlobal, uPuntosGuardado, uPuntosGuardadoGlobal,
						vPuntosGuardado, vPuntosGuardadoGlobal, zPuntosGuardado, zPuntosGuardadoGlobal, etaMinPuntosGuardadoGlobal,
						etaMaxPuntosGuardadoGlobal, Hmin, num_ts, tiempoActSubmalla[0][0], epsilon_h, H, U, T, id_hebra, comm_cartesiano);
				}
				else {
					guardarSerieTiemposHidrosNivel0(datosClusterCPU, datosNivel, datosGRD_double2, datosGRD_puntosQ, numPuntosGuardarTotal,
						posicionesVolumenesGuardado, etaPuntosGuardado, etaPuntosGuardadoGlobal, uPuntosGuardado, uPuntosGuardadoGlobal,
						vPuntosGuardado, vPuntosGuardadoGlobal, etaMinPuntosGuardadoGlobal, etaMaxPuntosGuardadoGlobal, Hmin, num_ts,
						tiempoActSubmalla[0][0], epsilon_h, H, U, T, id_hebra, comm_cartesiano);
				}
				num_ts++;
				MPI_Barrier(comm_cartesiano);
			}
			// FIN NETCDF
			if (guardarEtaMax) {
				actualizarEta1Maxima(numNiveles, datosClusterCPU, d_datosVolumenesNivelSig_1, d_eta1MaximaNivel, submallasNivel,
					numSubmallasNivel, d_datosNivel, hmin_metrics, blockGridEstNivel, threadBlockEst, streams, nstreams);
			}
			if (guardarVelocidadesMax) {
				actualizarVelocidadesMaximas(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2,
					d_velocidadesMaximasNivel, submallasNivel, numSubmallasNivel, epsilon_h, d_datosNivel, hmin_metrics,
					blockGridEstNivel, threadBlockEst, streams, nstreams);
			}
			if (guardarModuloVelocidadMax) {
				actualizarModuloVelocidadMaxima(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2,
					d_moduloVelocidadMaximaNivel, submallasNivel, numSubmallasNivel, epsilon_h, d_datosNivel, hmin_metrics,
					blockGridEstNivel, threadBlockEst, H, U, T, streams, nstreams);
			}
			if (guardarModuloCaudalMax) {
				actualizarModuloCaudalMaximo(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2,
					d_moduloCaudalMaximoNivel, submallasNivel, numSubmallasNivel, d_datosNivel, hmin_metrics, blockGridEstNivel,
					threadBlockEst, H, U, T, streams, nstreams);
			}
			if (guardarFlujoMomentoMax) {
				actualizarFlujoMomentoMaximo(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2,
					d_flujoMomentoMaximoNivel, submallasNivel, numSubmallasNivel, epsilon_h, d_datosNivel, hmin_metrics,
					blockGridEstNivel, threadBlockEst, H, U, T, streams, nstreams);
			}
			if (guardarTiemposLlegada) {
				actualizarTiemposLlegada(numNiveles, datosClusterCPU, d_datosVolumenesNivelSig_1, d_tiemposLlegadaNivel,
					d_eta1InicialNivel, submallasNivel, numSubmallasNivel, tiempoActSubmalla[0][0], d_datosNivel,
					hmin_metrics, difh_at, blockGridEstNivel, threadBlockEst, streams, nstreams);
			}

			tiempoAntSubmalla[0][0] = tiempoActSubmalla[0][0];
			if (numNiveles == 1) {
				if (no_hidros) {
					deltaTNivel[0][0] = siguientePasoNivel0NoHidros(numPesosJacobi, omega, ord, numNiveles, datosClusterCPU, datosClusterGPU,
										d_datosVolumenesNivel_1[0], d_datosVolumenesNivel_2[0], &(d_datosNivel[0][0]), d_datosVolumenesNivelSig_1[0],
										d_datosVolumenesNivelSig_2[0], d_aristaReconstruido, d_refinarNivel[0], numVolxTotalNivel0, numVolyTotalNivel0,
										Hmin, d_deltaTVolumenesNivel, d_correccionEtaNivel[0], d_correccionNivel_2[0], d_acumuladorNivel_1, d_acumuladorNivel_2,
										borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level,
										&(tiempoActSubmalla[0][0]), CFL, deltaTNivel[0][0], d_friccionesNivel[0], vmax, epsilon_h, numSubmallasNivel,
										iniGlobalSubmallasNivel[0][0], d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0[0], d_Pnh1[0],
										&iter_sistema, &error_sistema, blockGridVer1Nivel[0][0], blockGridVer2Nivel[0][0], blockGridHor1Nivel[0][0],
										blockGridHor2Nivel[0][0], threadBlockAri, blockGridEstNivel[0][0], blockGridFanNivel[0][0], blockGridVertNivel[0][0],
										threadBlockEst, streamMemcpy, tipoFilasDouble2[0][0], tipoFilasDouble3[0][0], tipoColumnasDouble2[0][0],
										tipoColumnasDouble3[0][0], tipoColumnasPresion[0][0], tam_datosVertDoubleNivel[0], L, H, id_hebraX, id_hebraY,
										ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
				}
				else {
					deltaTNivel[0][0] = siguientePasoNivel0(numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1[0],
										d_datosVolumenesNivel_2[0], &(d_datosNivel[0][0]), d_datosVolumenesNivelSig_1[0], d_datosVolumenesNivelSig_2[0],
										d_aristaReconstruido, d_refinarNivel[0], numVolxTotalNivel0, numVolyTotalNivel0, Hmin, d_deltaTVolumenesNivel,
										d_correccionEtaNivel[0], d_correccionNivel_2[0], d_acumuladorNivel_1, d_acumuladorNivel_2, borde_sup, borde_inf,
										borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level, &(tiempoActSubmalla[0][0]),
										CFL, deltaTNivel[0][0], d_friccionesNivel[0], vmax, epsilon_h, numSubmallasNivel, iniGlobalSubmallasNivel[0][0],
										blockGridVer1Nivel[0][0], blockGridVer2Nivel[0][0], blockGridHor1Nivel[0][0], blockGridHor2Nivel[0][0], threadBlockAri,
										blockGridEstNivel[0][0], blockGridFanNivel[0][0], threadBlockEst, streamMemcpy, tipoFilasDouble2[0][0],
										tipoFilasDouble3[0][0], tipoColumnasDouble2[0][0], tipoColumnasDouble3[0][0], L, H, id_hebraX, id_hebraY,
										ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
				}
			}
/*			else {
				// Inicializamos los vectores de corrección de flujos
				cudaMemset(d_correccionEtaNivel[0], 0, tam_datosVolDoubleNivel[0]);
				cudaMemset(d_correccionNivel_2[0], 0, tam_datosVolDouble2Nivel[0]);

				deltaTNivel[0][0] = siguientePasoNivel0(numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1[0],
									d_datosVolumenesNivel_2[0], &(d_datosNivel[0][0]), d_datosVolumenesNivelSig_1[0], d_datosVolumenesNivelSig_2[0],
									d_refinarNivel[0], datosClusterCPU[0][0].numVolx, datosClusterCPU[0][0].numVoly, numVolxTotalNivel0, numVolyTotalNivel0,
									Hmin, d_deltaTVolumenesNivel, d_correccionEtaNivel[0], d_correccionNivel_2[0], d_acumuladorNivel_1, d_acumuladorNivel_2,
									borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level,
									&(tiempoActSubmalla[0][0]), CFL, deltaTNivel[0][0], d_friccionesNivel[0], vmax, epsilon_h, numSubmallasNivel,
									iniGlobalSubmallasNivel[0][0], blockGridVer1Nivel[0][0], blockGridVer2Nivel[0][0], blockGridHor1Nivel[0][0],
									blockGridHor2Nivel[0][0], threadBlockAri, blockGridVerComNivel[0][0], blockGridHorComNivel[0][0], threadBlockAriVerCom,
									threadBlockAriHorCom, blockGridEstNivel[0][0], threadBlockEst, streamMemcpy, tipo_filas[0][0], tipo_columnas[0][0],
									tam_datosVolDouble2Nivel[0], id_hebraX, id_hebraY, ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano,
									hebra_izq, hebra_der, hebra_sup, hebra_inf);

				if (mallasAnidadasActivadas_global) {
					aplicarPaso(1, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
						d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, d_refinarNivel, Hmin, d_datosNivel, d_deltaTVolumenesNivel,
						d_correccionEtaNivel, d_correccionNivel_2, d_acumuladorNivel_1, d_acumuladorNivel_2, d_mediaEtaNivel, d_mediaNivel_2,
						borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level,
						tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel, deltaTNivelSinTruncar, d_friccionesNivel, vmax, CFL, epsilon_h,
						ratioRefNivel, ratioRefAcumNivel, factorCorreccionNivel, L, H, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel,
						submallaNivelSuperior, posSubmallaNivelSuperior, d_posCopiaNivel, haySubmallasAdyacentesNivel, tam_datosVolDoubleNivel,
						tam_datosVolDouble2Nivel, false, blockGridVer1Nivel, blockGridVer2Nivel, blockGridHor1Nivel, blockGridHor2Nivel, threadBlockAri,
						blockGridEstNivel, blockGridFanNivel, threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas, tipo_columnas, id_hebra,
						ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
				}
				else {
					// Averiguamos si hay que procesar las mallas anidadas en la iteración actual
					obtenerActivacionMallasAnidadasGPU<<<blockGridEstNivel[0][0], threadBlockEst>>>(d_datosVolumenesNivelSig_1[0], d_eta1InicialNivel[0],
						d_activacionMallasAnidadas, datosClusterCPU[0][0].numVolx, datosClusterCPU[0][0].numVoly, epsilon_h);
					cudaMemcpyFromSymbol(&mallasAnidadasActivadas, d_mallasAnidadasActivadas, sizeof(int), 0, cudaMemcpyDeviceToHost);
					MPI_Allreduce(&mallasAnidadasActivadas, &mallasAnidadasActivadas_global, 1, MPI_INT, MPI_MAX, comm_cartesiano);

					if (mallasAnidadasActivadas_global) {
						// Sincronizamos los tiempos de las mallas anidadas con el tiempo del nivel 0 y procesamos las mallas anidadas
						for (l=1; l<numNiveles; l++) {
							for (i=0; i<numSubmallasNivel[l]; i++) {
								tiempoAntSubmalla[l][i] = tiempoAntSubmalla[0][0];
								tiempoActSubmalla[l][i] = tiempoAntSubmalla[l][i];
							}
						}
						aplicarPaso(1, numNiveles, datosClusterCPU, datosClusterGPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2,
							d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, d_refinarNivel, Hmin, d_datosNivel, d_deltaTVolumenesNivel,
							d_correccionEtaNivel, d_correccionNivel_2, d_acumuladorNivel_1, d_acumuladorNivel_2, d_mediaEtaNivel, d_mediaNivel_2,
							borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level,
							tiempoAntSubmalla, tiempoActSubmalla, deltaTNivel, deltaTNivelSinTruncar, d_friccionesNivel, vmax, CFL, epsilon_h,
							ratioRefNivel, ratioRefAcumNivel, factorCorreccionNivel, L, H, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel,
							submallaNivelSuperior, posSubmallaNivelSuperior, d_posCopiaNivel, haySubmallasAdyacentesNivel, tam_datosVolDoubleNivel,
							tam_datosVolDouble2Nivel, false, blockGridVer1Nivel, blockGridVer2Nivel, blockGridHor1Nivel, blockGridHor2Nivel, threadBlockAri,
							blockGridEstNivel, blockGridFanNivel, threadBlockEst, streams, nstreams, streamMemcpy, T, tipo_filas, tipo_columnas, id_hebra,
							ultimaHebraXSubmalla, ultimaHebraYSubmalla, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);
					}
					else {
						// Copiamos el siguiente estado del nivel 0 en el actual
						cudaMemcpy(d_datosVolumenesNivel_1[0], d_datosVolumenesNivelSig_1[0], tam_datosVolDouble2Nivel[0], cudaMemcpyDeviceToDevice);
						cudaMemcpy(d_datosVolumenesNivel_2[0], d_datosVolumenesNivelSig_2[0], tam_datosVolDouble2Nivel[0], cudaMemcpyDeviceToDevice);

						// Copiamos los volúmenes de comunicación del nivel 0 a memoria CPU de forma asíncrona
						copiarVolumenesComAsincACPU(&(datosClusterCPU[0][0]), &(datosClusterGPU[0][0]), streamMemcpy, id_hebraX, id_hebraY,
							ultimaHebraXSubmalla[0][0], ultimaHebraYSubmalla[0][0]);
					}
				}
			}*/
//			if (tiempoActSubmalla[0][0]+deltaTNivel[0][0] > sigTiempoGuardarNetCDF[0]) deltaTNivel[0][0] = sigTiempoGuardarNetCDF[0]-tiempoActSubmalla[0][0];
			if (! saltar_deformacion) {
				truncarDeltaTNivel0ParaSigDeformacion(okada_flag, numFaults, fallaOkada, tiempoActSubmalla[0][0], defTime,
					indiceEstadoSigDefDim, T, &(deltaTNivel[0][0]), &saltar_deformacion);
			}
			else {
				truncarDeltaTNivel0ParaSigDeformacionSaltando(okada_flag, numFaults, fallaOkada, tiempoActSubmalla[0][0], defTime,
					numEstadosDefDinamica, indiceEstadoSigDefDim, T, &(deltaTNivel[0][0]), &saltar_deformacion);
			}

			// Inicializamos los acumuladores del nivel 0 para la siguiente iteración
			cudaMemset(d_acumuladorNivel_1, 0, tam_datosVolDouble2Nivel[0]);
			cudaMemset(d_acumuladorNivel_2, 0, tam_datosVolDouble3Nivel[0]);

			if (id_hebra == 0) {
				fprintf(stdout, "Iteration %4d, deltaT = %e sec, Time = %g sec\n", iter, deltaTNivel[0][0]*T, tiempoActSubmalla[0][0]*T);
				iter++;
			}
		}
		cudaDeviceSynchronize();
		tiempo_fin = MPI_Wtime();
		*tiempo = tiempo_fin - tiempo_ini;  // En segundos

		// INICIO NETCDF
		if (hayQueCrearAlgunFicheroNetCDF) {
#if (FILE_WRITING_MODE == 2)
			// Sincronizamos con la hebra de guardado de datos en NetCDF
			terminarHebraNetCDF = true;
			datosPreparados = true;
			cv_datosPreparados.notify_one();
			if (hebraNetCDF.joinable())
				hebraNetCDF.join();
#endif

			for (l=0; l<numNiveles; l++) {
				// Amplitud máxima
				if (guardarEtaMax) {
					cudaMemcpy(datosVolumenesNivel_1[l], d_eta1MaximaNivel[l], tam_datosVolDoubleNivel[l], cudaMemcpyDeviceToHost);
					p_eta = (double *) datosVolumenesNivel_1[l];
					pos = 0;
					for (k=0; k<numSubmallasNivel[l]; k++) {
						if (datosClusterCPU[l][k].iniy != -1) {
							nvx = datosClusterCPU[l][k].numVolx;
							nvy = datosClusterCPU[l][k].numVoly;
							if (guardarVariables[l][k].eta_max != 0) {
								for (j=0; j<ny_nc[l][k]; j++) {
									num = pos + (iniy[l][k] + j*npics)*nvx;
									j_global = datosClusterCPU[l][k].iniy + iniy[l][k] + j*npics;
									val_cos = datosNivel[l][k].vccos[j_global];
									for (i=0; i<nx_nc[l][k]; i++) {
										int m = j*nx_nc[l][k] + i;
										double eta_max = p_eta[num + (inix[l][k] + i*npics)];
										if (eta_max < -1e20)
											vec1[m] = -9999.0f;
										else
											vec1[m] = (float) (((eta_max - Hmin)/val_cos)*H);
									}
								}
								guardarAmplitudMaximaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec1);
							}
							pos += (nvx+4)*(nvy+4);
						}
					}
				}
				// Velocidades máximas
				if (guardarVelocidadesMax) {
					cudaMemcpy(datosVolumenesNivel_2[l], d_velocidadesMaximasNivel[l], tam_datosVolDouble3Nivel[l], cudaMemcpyDeviceToHost);
					if (no_hidros) {
						pos = 0;
						for (k=0; k<numSubmallasNivel[l]; k++) {
							if (datosClusterCPU[l][k].iniy != -1) {
								nvx = datosClusterCPU[l][k].numVolx;
								nvy = datosClusterCPU[l][k].numVoly;
								if (guardarVariables[l][k].velocidades_max != 0) {
									for (j=0; j<ny_nc[l][k]; j++) {
										num = pos + (iniy[l][k] + j*npics)*nvx;
										for (i=0; i<nx_nc[l][k]; i++) {
											int ind_vec = j*nx_nc[l][k] + i;
											vec1[ind_vec] = (float) (datosVolumenesNivel_2[l][num + (inix[l][k] + i*npics)].x*U);
											vec2[ind_vec] = (float) (datosVolumenesNivel_2[l][num + (inix[l][k] + i*npics)].y*U);
											vec3[ind_vec] = (float) (datosVolumenesNivel_2[l][num + (inix[l][k] + i*npics)].z*H/T);
										}
									}
									guardarUxMaximaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec1);
									guardarUyMaximaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec2);
									guardarUzMaximaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec3);
								}
								pos += (nvx+4)*(nvy+4);
							}
						}
					}
					else {
						pos = 0;
						for (k=0; k<numSubmallasNivel[l]; k++) {
							if (datosClusterCPU[l][k].iniy != -1) {
								nvx = datosClusterCPU[l][k].numVolx;
								nvy = datosClusterCPU[l][k].numVoly;
								if (guardarVariables[l][k].velocidades_max != 0) {
									for (j=0; j<ny_nc[l][k]; j++) {
										num = pos + (iniy[l][k] + j*npics)*nvx;
										for (i=0; i<nx_nc[l][k]; i++) {
											int ind_vec = j*nx_nc[l][k] + i;
											vec1[ind_vec] = (float) (datosVolumenesNivel_2[l][num + (inix[l][k] + i*npics)].x*U);
											vec2[ind_vec] = (float) (datosVolumenesNivel_2[l][num + (inix[l][k] + i*npics)].y*U);
										}
									}
									guardarUxMaximaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec1);
									guardarUyMaximaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec2);
								}
								pos += (nvx+4)*(nvy+4);
							}
						}
					}
				}
				// Módulo de velocidad máxima
				if (guardarModuloVelocidadMax) {
					cudaMemcpy(datosVolumenesNivel_1[l], d_moduloVelocidadMaximaNivel[l], tam_datosVolDoubleNivel[l], cudaMemcpyDeviceToHost);
					p_flujo = (double *) datosVolumenesNivel_1[l];
					pos = 0;
					for (k=0; k<numSubmallasNivel[l]; k++) {
						if (datosClusterCPU[l][k].iniy != -1) {
							nvx = datosClusterCPU[l][k].numVolx;
							nvy = datosClusterCPU[l][k].numVoly;
							if (guardarVariables[l][k].modulo_velocidades_max != 0) {
								for (j=0; j<ny_nc[l][k]; j++) {
									num = pos + (iniy[l][k] + j*npics)*nvx;
									for (i=0; i<nx_nc[l][k]; i++)
										vec1[j*nx_nc[l][k] + i] = (float) (p_flujo[num + (inix[l][k] + i*npics)]);
								}
								guardarModuloVelocidadMaximaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec1);
							}
							pos += (nvx+4)*(nvy+4);
						}
					}
				}
				// Módulo de caudal máximo
				if (guardarModuloCaudalMax) {
					cudaMemcpy(datosVolumenesNivel_1[l], d_moduloCaudalMaximoNivel[l], tam_datosVolDoubleNivel[l], cudaMemcpyDeviceToHost);
					p_flujo = (double *) datosVolumenesNivel_1[l];
					pos = 0;
					for (k=0; k<numSubmallasNivel[l]; k++) {
						if (datosClusterCPU[l][k].iniy != -1) {
							nvx = datosClusterCPU[l][k].numVolx;
							nvy = datosClusterCPU[l][k].numVoly;
							if (guardarVariables[l][k].modulo_caudales_max != 0) {
								for (j=0; j<ny_nc[l][k]; j++) {
									num = pos + (iniy[l][k] + j*npics)*nvx;
									for (i=0; i<nx_nc[l][k]; i++)
										vec1[j*nx_nc[l][k] + i] = (float) (p_flujo[num + (inix[l][k] + i*npics)]);
								}
								guardarModuloCaudalMaximoNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec1);
							}
							pos += (nvx+4)*(nvy+4);
						}
					}
				}
				// Flujo de momento máximo
				if (guardarFlujoMomentoMax) {
					cudaMemcpy(datosVolumenesNivel_1[l], d_flujoMomentoMaximoNivel[l], tam_datosVolDoubleNivel[l], cudaMemcpyDeviceToHost);
					p_flujo = (double *) datosVolumenesNivel_1[l];
					pos = 0;
					for (k=0; k<numSubmallasNivel[l]; k++) {
						if (datosClusterCPU[l][k].iniy != -1) {
							nvx = datosClusterCPU[l][k].numVolx;
							nvy = datosClusterCPU[l][k].numVoly;
							if (guardarVariables[l][k].flujo_momento_max != 0) {
								for (j=0; j<ny_nc[l][k]; j++) {
									num = pos + (iniy[l][k] + j*npics)*nvx;
									for (i=0; i<nx_nc[l][k]; i++)
										vec1[j*nx_nc[l][k] + i] = (float) (p_flujo[num + (inix[l][k] + i*npics)]);
								}
								guardarFlujoMomentoMaximoNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec1);
							}
							pos += (nvx+4)*(nvy+4);
						}
					}
				}
				// Tiempos de llegada del tsunami
				if (guardarTiemposLlegada) {
					cudaMemcpy(datosVolumenesNivel_1[l], d_tiemposLlegadaNivel[l], tam_datosVolDoubleNivel[l], cudaMemcpyDeviceToHost);
					p_eta = (double *) datosVolumenesNivel_1[l];
					pos = 0;
					for (k=0; k<numSubmallasNivel[l]; k++) {
						if (datosClusterCPU[l][k].iniy != -1) {
							nvx = datosClusterCPU[l][k].numVolx;
							nvy = datosClusterCPU[l][k].numVoly;
							if (guardarVariables[l][k].tiempos_llegada != 0) {
								for (j=0; j<ny_nc[l][k]; j++) {
									num = pos + (iniy[l][k] + j*npics)*nvx;
									for (i=0; i<nx_nc[l][k]; i++) {
										double t = p_eta[num + (inix[l][k] + i*npics)]*T;
										vec1[j*nx_nc[l][k] + i] = (float) ((t < 0.0) ? -1.0 : t);
									}
								}
								guardarTiemposLlegadaNC(l, k, nx_nc[l][k], ny_nc[l][k], inix_nc[l][k], iniy_nc[l][k], vec1);
							}
							pos += (nvx+4)*(nvy+4);
						}
					}
				}
				// Batimetría deformada con Okada
				if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE) || (okada_flag == OKADA_TRIANGULAR) ||
					(okada_flag == OKADA_TRIANGULAR_FROM_FILE) || (okada_flag == DEFORMATION_FROM_FILE) || (okada_flag == DYNAMIC_DEFORMATION)) {
					if (l == numNiveles-1) {
						// Obtenemos la batimetría deformada directamente de d_datosVolumenesNivel
						cudaMemcpy(datosVolumenesNivel_1[l], d_datosVolumenesNivel_1[l], tam_datosVolDouble2Nivel[l], cudaMemcpyDeviceToHost);
						pos = 0;
						for (k=0; k<numSubmallasNivel[l]; k++) {
							if (datosClusterCPU[l][k].iniy != -1) {
								nvx = datosClusterCPU[l][k].numVolx;
								nvy = datosClusterCPU[l][k].numVoly;
								if (crearFicheroNetCDF[l][k]) {
									for (j=0; j<nvy; j++) {
										num = pos + (j+2)*(nvx+4);
										j_global = datosClusterCPU[l][k].iniy + j;
										val_cos = datosNivel[l][k].vccos[j_global];
										for (i=0; i<nvx; i++) {
											iter = num + (i + 2);  // El último 2 es el offsetX
											vec1[j*nvx+i] = (float) (((datosVolumenesNivel_1[l][iter].y + Hmin)/val_cos)*H);
										}
									}
									guardarBatimetriaModificadaNC(l, k, nvx, nvy, datosClusterCPU[l][k].inix, datosClusterCPU[l][k].iniy, vec1);
								}
								pos += (nvx+4)*(nvy+4);
							}
						}
					}
					else {
						// Obtenemos la batimetría deformada sumando batiOriginal y d_deformacionAcumuladaNivel
						cudaMemcpy(datosGRD, d_deformacionAcumuladaNivel[l], tam_datosVolDoubleNivel[l], cudaMemcpyDeviceToHost);
						pos = 0;
						for (k=0; k<numSubmallasNivel[l]; k++) {
							if (datosClusterCPU[l][k].iniy != -1) {
								nvx = datosClusterCPU[l][k].numVolx;
								nvy = datosClusterCPU[l][k].numVoly;
								if (crearFicheroNetCDF[l][k]) {
									for (j=0; j<nvy; j++) {
										num = pos + j*nvx;
										j_global = datosClusterCPU[l][k].iniy + j;
										val_cos = datosNivel[l][k].vccos[j_global];
										for (i=0; i<nvx; i++) {
											iter = num + i;
											vec1[j*nvx+i] = (float) (batiOriginal[l][iter] + (datosGRD[iter]/val_cos)*H);
										}
									}
									guardarBatimetriaModificadaNC(l, k, nvx, nvy, datosClusterCPU[l][k].inix, datosClusterCPU[l][k].iniy, vec1);
								}
								pos += (nvx+4)*(nvy+4);
							}
						}
					}
				}
				for (k=0; k<numSubmallasNivel[l]; k++) {
					if ((datosClusterCPU[l][k].iniy != -1) && (crearFicheroNetCDF[l][k]))
						closeNC(l, k);
				}
			}
			free(vec1);
			free(vec2);
			free(vec3);
		}
		else if (continuar_simulacion == 1) {
			for (l=0; l<numNiveles; l++) {
				for (k=0; k<numSubmallasNivel[l]; k++)
					closeNC(l, k);
			}
			free(vec1);
			free(vec2);
			free(vec3);
		}
		if (leer_fichero_puntos == 1) {
			// Batimetría (original si okada_flag es SEA_SURFACE_FROM_FILE o GAUSSIAN; deformada si okada_flag es OKADA_STANDARD, OKADA_STANDARD_FROM_FILE,
			// OKADA_TRIANGULAR, OKADA_TRIANGULAR_FROM_FILE, DEFORMATION_FROM_FILE o DYNAMIC_DEFORMATION)
			escribirVolumenesGuardadoGPU<<<blockGridPuntos, threadBlockPuntos>>>(d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2,
				d_datosVolumenesGuardado_1, d_datosVolumenesGuardado_2, d_posicionesVolumenesGuardado, numPuntosGuardarTotal, epsilon_h);
			cudaMemcpy(datosGRD_double2, d_datosVolumenesGuardado_1, tam_datosVolGuardadoDouble2, cudaMemcpyDeviceToHost);
			obtenerBatimetriaParaSerieTiempos(datosClusterCPU, datosNivel, datosGRD_double2, numPuntosGuardarTotal, posicionesVolumenesGuardado,
				etaPuntosGuardado, etaPuntosGuardadoGlobal, Hmin, H, id_hebra, comm_cartesiano);
			if (id_hebra == 0) {
				guardarBatimetriaModificadaTimeSeriesNC(etaPuntosGuardadoGlobal);
				guardarAmplitudesTimeSeriesNC(etaMinPuntosGuardadoGlobal, etaMaxPuntosGuardadoGlobal);
				closeTimeSeriesNC();
			}
			free(etaPuntosGuardado);
			free(etaPuntosGuardadoGlobal);
			free(uPuntosGuardado);
			free(uPuntosGuardadoGlobal);
			free(vPuntosGuardado);
			free(vPuntosGuardadoGlobal);
			free(zPuntosGuardado);
			free(zPuntosGuardadoGlobal);
			free(etaMinPuntosGuardadoGlobal);
			free(etaMaxPuntosGuardadoGlobal);
		}
		// FIN NETCDF

		if (kajiura_flag == 1) {
			for (i=0; i<numFaults; i++) {
				free(F2Sx[i]);
				free(F2Sy[i]);
			}
		}
		liberarMemoria(no_hidros, numNiveles, datosClusterCPU, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_friccionesNivel,
			d_datosNivel, d_datosVolumenesNivelSig_1, d_datosVolumenesNivelSig_2, numSubmallasNivel, d_posCopiaNivel, d_activacionMallasAnidadas,
			d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2, d_aristaReconstruido, d_refinarNivel, d_correccionEtaNivel,
			d_correccionNivel_2, d_mediaEtaNivel, d_mediaNivel_2, leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
			d_datosVolumenesGuardado_2, d_datosVolumenesNivelGPU_1, d_datosVolumenesNivelGPU_2, guardarEtaMax, guardarVelocidadesMax,
			guardarModuloVelocidadMax, guardarModuloCaudalMax, guardarFlujoMomentoMax, guardarTiemposLlegada, d_eta1InicialNivel,
			d_eta1MaximaNivel, d_velocidadesMaximasNivel, d_moduloVelocidadMaximaNivel, d_moduloVelocidadMaximaNivel, d_flujoMomentoMaximoNivel,
			d_tiemposLlegadaNivel, okada_flag, kajiura_flag, numFaults, numEstadosDefDinamica, d_deformacionNivel0, d_deformacionAcumuladaNivel,
			d_vc, d_vz, d_SLIPVEC, d_datosKajiura, d_F2Sx, d_F2Sy, d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0, d_Pnh1);
	}
	// Si err_total == 1, el proceso termina

	return err;
}

