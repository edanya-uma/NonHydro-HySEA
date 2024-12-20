#ifndef _TIMESTEP_H_
#define _TIMESTEP_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "Reduccion_kernel.cu"
#include "Volumen_kernel.cu"
#include "AristaVer_kernel.cu"
#include "AristaHor_kernel.cu"
#include "NestedMeshesDif.cu"
#include "NestedMeshesVal.cu"
#include "Reconstruccion.cu"

/******************/
/* DeltaT inicial */
/******************/

double obtenerDeltaTInicialNivel0(TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 *d_datosVolumenesNivel0_1,
		double3 *d_datosVolumenesNivel0_2, tipoDatosSubmalla *d_datosNivel0, double *d_deltaTVolumenes, double CFL,
		double epsilon_h, dim3 blockGridEst, dim3 threadBlockEst, double L, MPI_Comm comm_cartesiano)
{
	TDatosClusterCPU *datosClusterCPUNivel0 = &(datosClusterCPU[0][0]);
	int numVolxNivel0 = datosClusterCPUNivel0->numVolx;
	int numVolyNivel0 = datosClusterCPUNivel0->numVoly;
	int iniySubmallaCluster = datosClusterCPUNivel0->iniy;
	double dxR = (d_datosNivel0->dx)*EARTH_RADIUS/L;
	double dyR = (d_datosNivel0->dy)*EARTH_RADIUS/L;
	double dT_min, delta_T;

	// Obtenemos el delta T local de cada volumen
	obtenerDeltaTVolumenesGPU<<<blockGridEst, threadBlockEst>>>(d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
		d_datosNivel0->vccos, d_deltaTVolumenes, epsilon_h, dxR, dyR, numVolxNivel0, numVolyNivel0, CFL, iniySubmallaCluster);
	// Obtenemos el mínimo delta T del cluster aplicando un algoritmo de reducción
	dT_min = obtenerMinimoReduccion<double>(d_deltaTVolumenes, numVolxNivel0*numVolyNivel0);
	// Obtenemos el mínimo delta T de todos los clusters por reducción
	MPI_Allreduce (&dT_min, &delta_T, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	return delta_T;
}

/*void obtenerDeltaTInicialNivel(int l, TDatosClusterCPU *datosClusterCPU, double2 *d_datosVolumenesNivel1_1,
		double2 *d_datosVolumenesNivel1_2, tipoDatosSubmalla *d_datosNivel1, double *deltaTNivel1,
		double *d_deltaTVolumenesNivel1, double2 *d_acumulador_1, double CFL, double epsilon_h, int4 *submallasNivel1,
		int numSubmallasNivel1, int64_t tam_datosVolDouble2Nivel1, dim3 *blockGridVer1Nivel1, dim3 *blockGridVer2Nivel1,
		dim3 *blockGridHor1Nivel1, dim3 *blockGridHor2Nivel1, dim3 threadBlockAri, dim3 *blockGridEstNivel1,
		dim3 threadBlockEst, cudaStream_t *streams, int nstreams, double T)
{
	int j, pos;
	int nvx, nvy;

	// Inicializamos el acumulador
	cudaMemset(d_acumulador_1, 0, tam_datosVolDouble2Nivel1);

	pos = 0;
	for (j=0; j<numSubmallasNivel1; j++) {
		if (datosClusterCPU[j].iniy != -1) {
			nvx = datosClusterCPU[j].numVolx;
			nvy = datosClusterCPU[j].numVoly;
			// Procesamos las aristas verticales pares
			procesarAristasVerDeltaTInicialNivel1GPU<<<blockGridVer1Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(d_datosVolumenesNivel1_1+pos,
				d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi, nvx, nvy, datosClusterCPU[j].inix,
				d_datosNivel1[j].altoVolumenes, d_acumulador_1+pos, epsilon_h, 1);
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivel1; j++) {
		if (datosClusterCPU[j].iniy != -1) {
			nvx = datosClusterCPU[j].numVolx;
			nvy = datosClusterCPU[j].numVoly;
			// Procesamos las aristas verticales impares
			procesarAristasVerDeltaTInicialNivel1GPU<<<blockGridVer2Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(d_datosVolumenesNivel1_1+pos,
				d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi, nvx, nvy, datosClusterCPU[j].inix,
				d_datosNivel1[j].altoVolumenes, d_acumulador_1+pos, epsilon_h, 2);
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivel1; j++) {
		if (datosClusterCPU[j].iniy != -1) {
			nvx = datosClusterCPU[j].numVolx;
			nvy = datosClusterCPU[j].numVoly;
			// Procesamos las aristas horizontales pares
			procesarAristasHorDeltaTInicialNivel1GPU<<<blockGridHor1Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(d_datosVolumenesNivel1_1+pos,
				d_datosVolumenesNivel1_2+pos, nvx, nvy, datosClusterCPU[j].inix, datosClusterCPU[j].iniy,
				d_datosNivel1[j].anchoVolumenes, d_acumulador_1+pos, epsilon_h, 1);
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivel1; j++) {
		if (datosClusterCPU[j].iniy != -1) {
			nvx = datosClusterCPU[j].numVolx;
			nvy = datosClusterCPU[j].numVoly;
			// Procesamos las aristas horizontales impares
			procesarAristasHorDeltaTInicialNivel1GPU<<<blockGridHor2Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(d_datosVolumenesNivel1_1+pos,
				d_datosVolumenesNivel1_2+pos, nvx, nvy, datosClusterCPU[j].inix, datosClusterCPU[j].iniy,
				d_datosNivel1[j].anchoVolumenes, d_acumulador_1+pos, epsilon_h, 2);
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	pos = 0;
	for (j=0; j<numSubmallasNivel1; j++) {
		if (datosClusterCPU[j].iniy != -1) {
			nvx = datosClusterCPU[j].numVolx;
			nvy = datosClusterCPU[j].numVoly;
			// Obtenemos el delta T local de cada volumen
			obtenerDeltaTVolumenesGPU<<<blockGridEstNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(d_datosNivel1[j].areaYCosPhi,
				d_acumulador_1+pos, d_deltaTVolumenesNivel1+pos, nvx, nvy, CFL);
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivel1; j++) {
		deltaTNivel1[j] = DBL_MAX;
		if (datosClusterCPU[j].iniy != -1) {
			nvx = datosClusterCPU[j].numVolx;
			nvy = datosClusterCPU[j].numVoly;
			// Obtenemos el mínimo delta T aplicando un algoritmo de reducción
			deltaTNivel1[j] = obtenerMinimoReduccion<double>(d_deltaTVolumenesNivel1+pos, nvx*nvy);
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	// En este punto, cada hebra tiene en deltaTNivel[j] el deltaT local de su subdominio para la submalla j,
	// DBL_MAX si la hebra no tiene la submalla. La reducción de los deltaT se hará en obtenerDeltaTTruncados
}*/

/**********************/
/* Truncado de deltaT */
/**********************/

// Trunca deltaTNivel[0][0], si es necesario, para que el siguiente tiempo de la simulación
// coincida con el tiempo en el que hay que aplicar la siguiente deformación
void truncarDeltaTNivel0ParaSigDeformacion(int okada_flag, int numFaults, int fallaOkada, double tiempoActSubmallaNivel0,
		double *defTime, int indiceEstadoSigDefDim, double T, double *delta_T, bool *saltar_deformacion)
{
	if (fallaOkada < numFaults) {
		// Quedan deformaciones por aplicar
		if (okada_flag == DYNAMIC_DEFORMATION) {
			if ((tiempoActSubmallaNivel0 + (*delta_T))*T > defTime[indiceEstadoSigDefDim]) {
				*delta_T = defTime[indiceEstadoSigDefDim]/T - tiempoActSubmallaNivel0;
				// Hemos truncado deltaT y la deformación apuntada por indiceEstadoSigDefDim se aplicará dentro de dos iteraciones.
				// Activamos que, en la siguiente iteración, se compruebe la siguiente deformación a partir de la que hay que aplicar
				*saltar_deformacion = true;
			}
		}
		else {
			if ((tiempoActSubmallaNivel0 + (*delta_T))*T > defTime[fallaOkada]) {
				// Truncamos deltaT
				*delta_T = defTime[fallaOkada]/T - tiempoActSubmallaNivel0;
				*saltar_deformacion = true;
			}
		}
	}
}

// Igual que la anterior pero sin tener en cuenta todas las deformaciones con el mismo tiempo a partir de defTime[fallaOkada].
// Se llama a esta función cuando hay que aplicar una deformación de forma inminente en la siguiente iteración, para que
// no compruebe dicha deformación y compruebe la siguiente
void truncarDeltaTNivel0ParaSigDeformacionSaltando(int okada_flag, int numFaults, int fallaOkada, double tiempoActSubmallaNivel0,
		double *defTime, int numEstadosDefDinamica, int indiceEstadoSigDefDim, double T, double *delta_T, bool *saltar_deformacion)
{
	int i;
	bool encontrado = false;

	*saltar_deformacion = false;
	if (fallaOkada < numFaults) {
		// Quedan deformaciones por aplicar
		if (okada_flag == DYNAMIC_DEFORMATION) {
			// Saltamos la siguiente deformación (asumimos que no hay dos deformaciones consecutivas con el mismo tiempo)
			if (indiceEstadoSigDefDim+1 < numEstadosDefDinamica) {
				// Existe siguiente deformación
				if ((tiempoActSubmallaNivel0 + (*delta_T))*T > defTime[indiceEstadoSigDefDim+1]) {
					*delta_T = defTime[indiceEstadoSigDefDim+1]/T - tiempoActSubmallaNivel0;
					// Hemos truncado deltaT y la deformación apuntada por indiceEstadoSigDefDim se aplicará dentro de dos iteraciones.
					// Activamos que, en la siguiente iteración, se compruebe la siguiente deformación a partir de la que hay que aplicar
					*saltar_deformacion = true;
				}
			}
		}
		else {
			// Saltamos deformaciones con el mismo tiempo
			i = fallaOkada+1;
			while ((i < numFaults) && (! encontrado)) {
				if (fabs(defTime[i] - defTime[fallaOkada]) > EPSILON)
					encontrado = true;
				else
					i++;
			}
			if (encontrado) {
				// Existe siguiente deformación
				if ((tiempoActSubmallaNivel0 + (*delta_T))*T > defTime[i]) {
					// Truncamos deltaT
					*delta_T = defTime[i]/T - tiempoActSubmallaNivel0;
					*saltar_deformacion = true;
				}
			}
		}
	}
}

// Pone en deltaTNivel los deltaT truncados almacenados en deltaTNivelSinTruncar
/*void obtenerDeltaTTruncados(int l, TDatosClusterCPU *datosClusterCPU, TDatosClusterGPU *datosClusterGPU, double *deltaTNivel,
		double *deltaTNivelSinTruncar, double tiempoActSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int numSubmallasNivel, int *submallaNivelSuperior, bool haySubmallasAdyacentesNivel, double T, int id_hebra, MPI_Comm comm_cartesiano)
{
	int j, k;
	double tiempo_corte;
	double min_deltaT;

	// Truncamos los deltaT en deltaTNivelSinTruncar y los sobreescribimos
	for (j=0; j<numSubmallasNivel; j++) {
		if (datosClusterCPU[j].iniy != -1) {
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				if (tiempoActSubmalla[l][j] + deltaTNivelSinTruncar[j] > tiempo_corte)
					deltaTNivelSinTruncar[j] = tiempo_corte - tiempoActSubmalla[l][j];
			}
		}
	}

	// Ponemos en deltaTNivel los deltaT truncados globales de cada submalla
	MPI_Allreduce(deltaTNivelSinTruncar, deltaTNivel, numSubmallasNivel, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	if (haySubmallasAdyacentesNivel) {
		// Si hay submallas adyacentes, asignamos a todas las submallas el mínimo deltaT de las submallas
		min_deltaT = deltaTNivel[0];
		for (j=1; j<numSubmallasNivel; j++)
			min_deltaT = min(min_deltaT, deltaTNivel[j]);
		for (j=0; j<numSubmallasNivel; j++)
			deltaTNivel[j] = min_deltaT;
	}
	if (id_hebra == 0) {
		for (j=0; j<numSubmallasNivel; j++) {
			for (k=0; k<l; k++)
				fprintf(stdout, "  ");
			fprintf(stdout, "Level %d, Submesh %d, initial deltaT = %e sec\n", l, j+1, deltaTNivel[j]*T);
		}
	}
}*/

/**************************/
/* Siguiente paso nivel 0 */
/**************************/

// El nuevo estado se escribe en d_datosVolumenesNivel0Sig_1 y d_datosVolumenesNivel0Sig_2.
double siguientePasoNivel0(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		TDatosClusterGPU datosClusterGPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 *d_datosVolumenesNivel0_1, double3 *d_datosVolumenesNivel0_2,
		tipoDatosSubmalla *d_datosNivel0, double2 *d_datosVolumenesNivel0Sig_1, double3 *d_datosVolumenesNivel0Sig_2, double *d_aristaReconstruido,
		bool *d_refinarNivel0, int numVolxTotalNivel0, int numVolyTotalNivel0, double Hmin, double *d_deltaTVolumenes, double *d_correccionEtaNivel0,
		double2 *d_correccionNivel0_2, double2 *d_acumulador_1, double3 *d_acumulador_2, double borde_sup, double borde_inf, double borde_izq,
		double borde_der, int tam_spongeSup, int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer, double sea_level, double *tiempo_act,
		double CFL, double delta_T, double *d_friccionesNivel0, double vmax, double epsilon_h, int *numSubmallasNivel, int2 iniGlobalSubmallaNivel0,
		dim3 blockGridVer1, dim3 blockGridVer2, dim3 blockGridHor1, dim3 blockGridHor2, dim3 threadBlockAri, dim3 blockGridEst, dim3 blockGridFan,
		dim3 threadBlockEst, cudaStream_t streamMemcpy, MPI_Datatype tipoFilasDouble2, MPI_Datatype tipoFilasDouble3, MPI_Datatype tipoColumnasDouble2,
		MPI_Datatype tipoColumnasDouble3, double L, double H, int id_hebraX, int id_hebraY, bool ultimaHebraXSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL],
		bool ultimaHebraYSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], MPI_Comm comm_cartesiano, int hebra_izq, int hebra_der, int hebra_sup, int hebra_inf)
{
	TDatosClusterCPU *datosClusterCPUNivel0 = &(datosClusterCPU[0][0]);
	TDatosClusterGPU *datosClusterGPUNivel0 = &(datosClusterGPU[0][0]);
	int nvx = datosClusterCPUNivel0->numVolx;
	int nvy = datosClusterCPUNivel0->numVoly;
	bool ultima_hebraX = ultimaHebraXSubmalla[0][0];
	bool ultima_hebraY = ultimaHebraYSubmalla[0][0];
	double radio_tierra = EARTH_RADIUS/L;
	double dxR = (d_datosNivel0->dx)*radio_tierra;
	double dyR = (d_datosNivel0->dy)*radio_tierra;
	int inixSubmallaCluster = datosClusterCPUNivel0->inix;
	int iniySubmallaCluster = datosClusterCPUNivel0->iniy;
	int tam_pitchDouble2 = (nvx+4)*sizeof(double2);
	int tam_pitchDouble2_2 = 8*sizeof(double2);
	int tam_pitchDouble3 = (nvx+4)*sizeof(double3);
	int tam_pitchDouble3_2 = 8*sizeof(double3);
	MPI_Request reqSend_izq[2], reqSend_der[2];
	MPI_Request reqSend_sup[2], reqSend_inf[2];
	MPI_Request reqRecv_izq[2], reqRecv_der[2];
	MPI_Request reqRecv_sup[2], reqRecv_inf[2];
	double coef1, coef2, coef3;
	double dT_min;
	int j, k;

	for (k=1; k<=OrdenTiempo; k++) {
		// ENVIOS MPI
		// Esperamos a que lleguen los volúmenes de comunicación a CPU y los enviamos al cluster que corresponda
		cudaStreamSynchronize(streamMemcpy);
		if (! ultima_hebraX) {
			// Recibimos los volúmenes de comunicación izquierdos del cluster derecho
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterIzq_1, 1, tipoColumnasDouble2, hebra_der, 21, comm_cartesiano, reqRecv_der);
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterIzq_2, 1, tipoColumnasDouble3, hebra_der, 22, comm_cartesiano, reqRecv_der+1);
			// Enviamos los volúmenes de comunicación derechos al cluster derecho
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterDer_1, 1, tipoColumnasDouble2, hebra_der, 23, comm_cartesiano, reqSend_der);
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterDer_2, 1, tipoColumnasDouble3, hebra_der, 24, comm_cartesiano, reqSend_der+1);
		}
		if (id_hebraX != 0) {
			// Recibimos los volúmenes de comunicación derechos del cluster izquierdo
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterDer_1, 1, tipoColumnasDouble2, hebra_izq, 23, comm_cartesiano, reqRecv_izq);
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterDer_2, 1, tipoColumnasDouble3, hebra_izq, 24, comm_cartesiano, reqRecv_izq+1);
			// Enviamos los volúmenes de comunicación izquierdos al cluster izquierdo
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterIzq_1, 1, tipoColumnasDouble2, hebra_izq, 21, comm_cartesiano, reqSend_izq);
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterIzq_2, 1, tipoColumnasDouble3, hebra_izq, 22, comm_cartesiano, reqSend_izq+1);
		}
		if (! ultima_hebraY) {
			// Recibimos los volúmenes de comunicación superiores del cluster inferior
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterSup_1, 1, tipoFilasDouble2, hebra_inf, 31, comm_cartesiano, reqRecv_inf);
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterSup_2, 1, tipoFilasDouble3, hebra_inf, 32, comm_cartesiano, reqRecv_inf+1);
			// Enviamos los volúmenes de comunicación inferiores al cluster inferior
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterInf_1, 1, tipoFilasDouble2, hebra_inf, 33, comm_cartesiano, reqSend_inf);
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterInf_2, 1, tipoFilasDouble3, hebra_inf, 34, comm_cartesiano, reqSend_inf+1);
		}
		if (id_hebraY != 0) {
			// Recibimos los volúmenes de comunicación inferiores del cluster superior
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterInf_1, 1, tipoFilasDouble2, hebra_sup, 33, comm_cartesiano, reqRecv_sup);
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterInf_2, 1, tipoFilasDouble3, hebra_sup, 34, comm_cartesiano, reqRecv_sup+1);
			// Enviamos los volúmenes de comunicación superiores al cluster superior
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterSup_1, 1, tipoFilasDouble2, hebra_sup, 31, comm_cartesiano, reqSend_sup);
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterSup_2, 1, tipoFilasDouble3, hebra_sup, 32, comm_cartesiano, reqSend_sup+1);
		}

		calcularCoeficientesReconstruccionNoCom<<<blockGridEst, threadBlockEst>>>(d_datosVolumenesNivel0Sig_1, d_datosVolumenesNivel0Sig_2,
			d_datosNivel0->vcos, d_datosNivel0->vtan, d_datosNivel0->vccos, d_aristaReconstruido, d_acumulador_2, nvx, nvy, numVolyTotalNivel0,
			d_datosNivel0->dx, d_datosNivel0->dy, radio_tierra, vmax, delta_T, iniySubmallaCluster, ultima_hebraY);

		// Esperamos a que lleguen los volúmenes de comunicación de todos los clusters adyacentes
		// y copiamos los volúmenes de comunicación recibidos a memoria GPU
		if (id_hebraX != 0) {
			MPI_Waitall(2, reqRecv_izq, MPI_STATUS_IGNORE);
			cudaMemcpy2D(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterDer_1, tam_pitchDouble2, datosClusterCPUNivel0->datosVolumenesComOtroClusterDer_1,
				tam_pitchDouble2_2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
			cudaMemcpy2D(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterDer_2, tam_pitchDouble3, datosClusterCPUNivel0->datosVolumenesComOtroClusterDer_2,
				tam_pitchDouble3_2, 2*sizeof(double3), nvy, cudaMemcpyHostToDevice);
		}
		if (! ultima_hebraX) {
			MPI_Waitall(2, reqRecv_der, MPI_STATUS_IGNORE);
			cudaMemcpy2D(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterIzq_1, tam_pitchDouble2, datosClusterCPUNivel0->datosVolumenesComOtroClusterIzq_1,
				tam_pitchDouble2_2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
			cudaMemcpy2D(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterIzq_2, tam_pitchDouble3, datosClusterCPUNivel0->datosVolumenesComOtroClusterIzq_2,
				tam_pitchDouble3_2, 2*sizeof(double3), nvy, cudaMemcpyHostToDevice);
		}
		if (id_hebraY != 0) {
			MPI_Waitall(2, reqRecv_sup, MPI_STATUS_IGNORE);
			cudaMemcpy(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterInf_1, datosClusterCPUNivel0->datosVolumenesComOtroClusterInf_1,
				2*tam_pitchDouble2, cudaMemcpyHostToDevice);
			cudaMemcpy(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterInf_2, datosClusterCPUNivel0->datosVolumenesComOtroClusterInf_2,
				2*tam_pitchDouble3, cudaMemcpyHostToDevice);
		}
		if (! ultima_hebraY) {
			MPI_Waitall(2, reqRecv_inf, MPI_STATUS_IGNORE);
			cudaMemcpy(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterSup_1, datosClusterCPUNivel0->datosVolumenesComOtroClusterSup_1,
				2*tam_pitchDouble2, cudaMemcpyHostToDevice);
			cudaMemcpy(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterSup_2, datosClusterCPUNivel0->datosVolumenesComOtroClusterSup_2,
				2*tam_pitchDouble3, cudaMemcpyHostToDevice);
		}

		calcularCoeficientesReconstruccionCom<<<blockGridFan, threadBlockEst>>>(d_datosVolumenesNivel0Sig_1, d_datosVolumenesNivel0Sig_2,
			d_datosNivel0->vcos, d_datosNivel0->vtan, d_datosNivel0->vccos, d_aristaReconstruido, d_acumulador_2, nvx, nvy, numVolyTotalNivel0,
			d_datosNivel0->dx, d_datosNivel0->dy, radio_tierra, vmax, delta_T, inixSubmallaCluster, iniySubmallaCluster, ultima_hebraX, ultima_hebraY);

		// Procesamos las aristas verticales
		procesarFlujoLonGPU<<<blockGridVer1, threadBlockAri>>>(d_datosNivel0->vcos, d_aristaReconstruido, nvx, nvy, delta_T, d_acumulador_1,
			d_acumulador_2, epsilon_h, dxR, 1, vmax, borde_sup, borde_inf, borde_izq, borde_der, iniySubmallaCluster, id_hebraX, ultima_hebraX);
		procesarFlujoLonGPU<<<blockGridVer2, threadBlockAri>>>(d_datosNivel0->vcos, d_aristaReconstruido, nvx, nvy, delta_T, d_acumulador_1,
			d_acumulador_2, epsilon_h, dxR, 2, vmax, borde_sup, borde_inf, borde_izq, borde_der, iniySubmallaCluster, id_hebraX, ultima_hebraX);
		// Procesamos las aristas horizontales
		procesarFlujoLatGPU<<<blockGridHor1, threadBlockAri>>>(d_datosNivel0->vcos, d_aristaReconstruido, nvx, nvy, delta_T, d_acumulador_1,
			d_acumulador_2, epsilon_h, dyR, 1, vmax, borde_sup, borde_inf, borde_izq, borde_der, iniySubmallaCluster, id_hebraY, ultima_hebraY);
		procesarFlujoLatGPU<<<blockGridHor2, threadBlockAri>>>(d_datosNivel0->vcos, d_aristaReconstruido, nvx, nvy, delta_T, d_acumulador_1,
			d_acumulador_2, epsilon_h, dyR, 2, vmax, borde_sup, borde_inf, borde_izq, borde_der, iniySubmallaCluster, id_hebraY, ultima_hebraY);

		switch(k) {
			case 1:
				coef1 = 0.0;
				coef2 = coef3 = 1.0;
				break;
			case 2:
				if (OrdenTiempo == 2) {
					coef1 = coef2 = coef3 = 0.5;
				}
				else {
					coef1 = 0.75;
					coef2 = coef3 = 0.25;
				}
				break;
			case 3:
				coef1 = 1.0/3.0;
				coef2 = coef3 = 2.0/3.0;
				break;
		}

		obtenerEstadoYDeltaTVolumenesGPU_RKTVD<<<blockGridEst, threadBlockEst>>>(d_datosVolumenesNivel0Sig_1, d_datosVolumenesNivel0Sig_2,
			d_datosNivel0->vccos, d_acumulador_1, d_acumulador_2, d_deltaTVolumenes, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
			k, nvx, nvy, CFL, delta_T, d_friccionesNivel0, vmax, epsilon_h, coef1, coef2, coef3, iniGlobalSubmallaNivel0, 1, inixSubmallaCluster,
			iniySubmallaCluster, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level, numVolxTotalNivel0, numVolyTotalNivel0);

		// Esperamos a que se hayan enviado los volúmenes de comunicación a todos los clusters adyacentes
		if (id_hebraX != 0) {
			MPI_Waitall(2, reqSend_izq, MPI_STATUS_IGNORE);
		}
		if (! ultima_hebraX) {
			MPI_Waitall(2, reqSend_der, MPI_STATUS_IGNORE);
		}
		if (id_hebraY != 0) {
			MPI_Waitall(2, reqSend_sup, MPI_STATUS_IGNORE);
		}
		if (! ultima_hebraY) {
			MPI_Waitall(2, reqSend_inf, MPI_STATUS_IGNORE);
		}

		if (numNiveles == 1) {
			// Copiamos los volúmenes de comunicación del nivel 0 a memoria CPU de forma asíncrona
			cudaStreamSynchronize(0);
			copiarVolumenesComAsincACPU(datosClusterCPUNivel0, datosClusterGPUNivel0, streamMemcpy, id_hebraX, id_hebraY,
				ultima_hebraX, ultima_hebraY);
		}
		else {
			// Copiamos los volúmenes de comunicación del nivel 1 a memoria CPU de forma asíncrona
			for (j=0; j<numSubmallasNivel[1]; j++) {
				if (datosClusterCPU[1][j].iniy != -1) {
					copiarVolumenesComAsincACPU(&(datosClusterCPU[1][j]), &(datosClusterGPU[1][j]), streamMemcpy,
						datosClusterCPU[1][j].inix, datosClusterCPU[1][j].iniy, ultimaHebraXSubmalla[1][j], ultimaHebraYSubmalla[1][j]);
				}
			}
		}
	} // for (k=0; k<OrdenTiempo; k++)

	// Actualizamos el tiempo actual
	*tiempo_act += delta_T;

	// Obtenemos el mínimo delta T del cluster aplicando un algoritmo de reducción
	dT_min = obtenerMinimoReduccion<double>(d_deltaTVolumenes, nvx*nvy);

	// Obtenemos el mínimo delta T de todos los clusters por reducción
	MPI_Allreduce(&dT_min, &delta_T, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	return delta_T;
}

/**************************/
/* Siguiente paso nivel 1 */
/**************************/

// Devuelve true si se ha alcanzado el tiempo de corte en todas las submallas, false en otro caso
/*bool siguientePasoNivel(int l, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		TDatosClusterGPU datosClusterGPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 **d_datosVolumenesNivel_1, double2 **d_datosVolumenesNivel_2,
		double2 *d_datosVolumenesNivelSupSig_1, double2 *d_datosVolumenesNivelSupSig_2, double2 *d_escrituraDatosVolumenes_1,
		double2 *d_escrituraDatosVolumenes_2, tipoDatosSubmalla *d_datosNivel1, bool **d_refinarNivel, double Hmin, double *d_deltaTVolumenesNivel1,
		double **d_correccionEtaNivel, double2 **d_correccionNivel_2, double2 *d_acumuladorNivel1_1, double2 *d_acumuladorNivel1_2,
		double borde_sup, double borde_inf, double borde_izq, double borde_der, int tam_spongeSup, int tam_spongeInf, int tam_spongeIzq,
		int tam_spongeDer, double sea_level, double tiempoAntSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], double tiempoActSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL],
		double *deltaTNivel1, double *deltaTNivelSinTruncar, double **d_friccionesNivel, double vmax, double CFL, double epsilon_h,
		double hpos, double cvis, int ratioRefNivel1, int ratioRefAcumNivel1, double L, double H, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int *numSubmallasNivel, int2 iniGlobalSubmallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *submallaNivelSuperior, int *posSubmallaNivelSuperior,
		int *d_posCopiaNivel1, bool inicializar_fantasma, bool haySubmallasAdyacentesNivel1, int64_t tam_datosVolDouble2Nivel1,
		dim3 *blockGridVer1Nivel1, dim3 *blockGridVer2Nivel1, dim3 *blockGridHor1Nivel1, dim3 *blockGridHor2Nivel1, dim3 threadBlockAri,
		dim3 *blockGridEstNivel1, dim3 *blockGridFanNivel1, dim3 threadBlockEst, cudaStream_t *streams, int nstreams,
		cudaStream_t streamMemcpy, double T, MPI_Datatype *tipo_filas, MPI_Datatype *tipo_columnas, int id_hebra,
		bool ultimaHebraXSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], bool ultimaHebraYSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL],
		MPI_Comm comm_cartesiano, int hebra_izq, int hebra_der, int hebra_sup, int hebra_inf)
{
	int j, k;
	int pos, pos_sup;
	int nvx, nvy;
	double peso1;
	double min_deltaT;
	double dT_min[MAX_GRIDS_LEVEL];
	double tiempo_corte;
	bool fin;
	int tam_pitch;
	int tam_pitch2 = 8*sizeof(double2);
	int nvxTotalNivelSup, nvyTotalNivelSup;
	int4 *submallasNivelAct = submallasNivel[l];
	int numSubmallasNivelAct = numSubmallasNivel[l];
	int2 *iniGlobalSubmallasNivelAct = iniGlobalSubmallasNivel[l];
	bool *d_refinarNivelSup = d_refinarNivel[l-1];
	bool *d_refinarNivel1 = d_refinarNivel[l];
	double *d_friccionesNivel1 = d_friccionesNivel[l];
	double *d_correccionEtaNivelSup = d_correccionEtaNivel[l-1];
	double *d_correccionEtaNivel1 = d_correccionEtaNivel[l];
	double2 *d_correccionNivelSup_2 = d_correccionNivel_2[l-1];
	double2 *d_correccionNivel1_2 = d_correccionNivel_2[l];
	double2 *d_datosVolumenesNivelSup_1 = d_datosVolumenesNivel_1[l-1];
	double2 *d_datosVolumenesNivelSup_2 = d_datosVolumenesNivel_2[l-1];
	double2 *d_datosVolumenesNivel1_1 = d_datosVolumenesNivel_1[l];
	double2 *d_datosVolumenesNivel1_2 = d_datosVolumenesNivel_2[l];
	TDatosClusterCPU *datosClusterCPUNivelSup = datosClusterCPU[l-1];
	TDatosClusterCPU *datosClusterCPUNivelAct = datosClusterCPU[l];
	TDatosClusterCPU *datosClusterCPUNivelInf;
	TDatosClusterGPU *datosClusterGPUNivelAct = datosClusterGPU[l];
	bool *ultimaHebraXSubmallaNivelAct = ultimaHebraXSubmalla[l];
	bool *ultimaHebraYSubmallaNivelAct = ultimaHebraYSubmalla[l];
	int numVolxTotalNivel0 = submallasNivel[0][0].z;
	int numVolyTotalNivel0 = submallasNivel[0][0].w;
	// Variables para MPI
	MPI_Request reqSend_sup[MAX_GRIDS_LEVEL], reqSend_inf[MAX_GRIDS_LEVEL];
	MPI_Request reqSend_izq[MAX_GRIDS_LEVEL], reqSend_der[MAX_GRIDS_LEVEL];
	MPI_Request reqRecv_sup[MAX_GRIDS_LEVEL], reqRecv_inf[MAX_GRIDS_LEVEL];
	MPI_Request reqRecv_izq[MAX_GRIDS_LEVEL], reqRecv_der[MAX_GRIDS_LEVEL];
	int num_req_izq, num_req_der;
	int num_req_sup, num_req_inf;

	// ENVIOS MPI DEL PASO 1
	num_req_izq = num_req_der = 0;
	num_req_sup = num_req_inf = 0;
	cudaStreamSynchronize(streamMemcpy);
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			if (datosClusterCPUNivelAct[j].inix != 0) {
				// Recibimos los volúmenes de comunicación derechos del cluster izquierdo
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterDer_2, 1, tipo_columnas[j], hebra_izq, 21,
					comm_cartesiano, reqRecv_izq+num_req_izq);
				// Enviamos los volúmenes de comunicación izquierdos al cluster izquierdo
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterIzq_2, 1, tipo_columnas[j], hebra_izq, 21,
					comm_cartesiano, reqSend_izq+num_req_izq);
				num_req_izq++;
			}
			if (! ultimaHebraXSubmallaNivelAct[j]) {
				// Recibimos los volúmenes de comunicación izquierdos del cluster derecho
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterIzq_2, 1, tipo_columnas[j], hebra_der, 21,
					comm_cartesiano, reqRecv_der+num_req_der);
				// Enviamos los volúmenes de comunicación derechos al cluster derecho
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterDer_2, 1, tipo_columnas[j], hebra_der, 21,
					comm_cartesiano, reqSend_der+num_req_der);
				num_req_der++;
			}
			if (datosClusterCPUNivelAct[j].iniy != 0) {
				// Recibimos los volúmenes de comunicación inferiores del cluster superior
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterInf_2, 1, tipo_filas[j], hebra_sup, 22,
					comm_cartesiano, reqRecv_sup+num_req_sup);
				// Enviamos los volúmenes de comunicación superiores al cluster superior
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterSup_2, 1, tipo_filas[j], hebra_sup, 22,
					comm_cartesiano, reqSend_sup+num_req_sup);
				num_req_sup++;
			}
			if (! ultimaHebraYSubmallaNivelAct[j]) {
				// Recibimos los volúmenes de comunicación superiores del cluster inferior
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterSup_2, 1, tipo_filas[j], hebra_inf, 22,
					comm_cartesiano, reqRecv_inf+num_req_inf);
				// Enviamos los volúmenes de comunicación inferiores al cluster inferior
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterInf_2, 1, tipo_filas[j], hebra_inf, 22,
					comm_cartesiano, reqSend_inf+num_req_inf);
				num_req_inf++;
			}
		}
	}

	// Inicializamos los acumuladores
	cudaMemset(d_acumuladorNivel1_1, 0, tam_datosVolDouble2Nivel1);
	cudaMemset(d_acumuladorNivel1_2, 0, tam_datosVolDouble2Nivel1);

	if (inicializar_fantasma) {
		pos = 0;
		for (j=0; j<numSubmallasNivelAct; j++) {
			if (datosClusterCPUNivelAct[j].iniy != -1) {
				nvx = datosClusterCPUNivelAct[j].numVolx;
				nvy = datosClusterCPUNivelAct[j].numVoly;
				k = submallaNivelSuperior[j];
				tiempo_corte = tiempoActSubmalla[l-1][k];
				if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
					// Inicializamos los caudales de los volúmenes fantasma de la submalla
					nvxTotalNivelSup = submallasNivel[l-1][k].z;
					nvyTotalNivelSup = submallasNivel[l-1][k].w;
					pos_sup = posSubmallaNivelSuperior[j];
#if (NESTED_ALGORITHM == 1)
					peso1 = (tiempoActSubmalla[l][j] - tiempoAntSubmalla[l][j]) / (tiempo_corte - tiempoAntSubmalla[l-1][k]);
					obtenerSiguientesCaudalesVolumenesFantasmaDiferenciasNivel1GPU<<<blockGridFanNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
						d_datosVolumenesNivelSup_2+pos_sup, d_datosVolumenesNivelSupSig_2+pos_sup, d_datosVolumenesNivel1_1+pos,
						d_datosVolumenesNivel1_2+pos, nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y,
						nvx, nvy, d_posCopiaNivel1+pos, d_datosVolumenesNivel1_2, peso1, ratioRefNivel1, borde_sup, borde_inf, borde_izq,
						borde_der, datosClusterCPUNivelSup[k].inix, datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelSup[k].numVolx,
						datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, ultimaHebraXSubmallaNivelAct[j], ultimaHebraYSubmallaNivelAct[j]);
#else
					peso1 = 1.0 - (tiempoActSubmalla[l][j] - tiempoAntSubmalla[l-1][k]) / (tiempo_corte - tiempoAntSubmalla[l-1][k]);
					obtenerSiguientesCaudalesVolumenesFantasmaValoresNivel1GPU<<<blockGridFanNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
						d_datosVolumenesNivelSup_2+pos_sup, d_datosVolumenesNivelSupSig_2+pos_sup, d_datosVolumenesNivel1_1+pos,
						d_datosVolumenesNivel1_2+pos, nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y,
						nvx, nvy, d_posCopiaNivel1+pos, d_datosVolumenesNivel1_2, peso1, ratioRefNivel1, borde_sup, borde_inf, borde_izq,
						borde_der, datosClusterCPUNivelSup[k].inix, datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelSup[k].numVolx,
						datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, ultimaHebraXSubmallaNivelAct[j], ultimaHebraYSubmallaNivelAct[j]);
#endif
				}
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}

	// PASO 1

	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas verticales impares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasVerNivel1Paso1GPU<<<blockGridVer2Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_datosVolumenesNivel1_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi, d_datosNivel1[j].altoVolumenes,
					datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y, nvx, nvy,
					submallasNivelAct[j].z, deltaTNivel1[j], d_acumuladorNivel1_1+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos,
					d_correccionEtaNivelSup+pos_sup, d_correccionEtaNivel1+pos, CFL, epsilon_h, hpos, ratioRefNivel1, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, 2, ultimaHebraXSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas horizontales impares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				nvyTotalNivelSup = submallasNivel[l-1][k].w;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasHorNivel1Paso1GPU<<<blockGridHor2Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_datosVolumenesNivel1_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi, d_datosNivel1[j].anchoVolumenes,
					datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y,
					nvx, nvy, submallasNivelAct[j].z, deltaTNivel1[j], d_acumuladorNivel1_1+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos,
					d_correccionEtaNivelSup+pos_sup, d_correccionEtaNivel1+pos, CFL, epsilon_h, hpos, ratioRefNivel1, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, 2, ultimaHebraYSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	// Esperamos a que lleguen los volúmenes de comunicación de todos los clusters adyacentes
	MPI_Waitall(num_req_izq, reqRecv_izq, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_der, reqRecv_der, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_sup, reqRecv_sup, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_inf, reqRecv_inf, MPI_STATUS_IGNORE);

	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			tam_pitch = (nvx+4)*sizeof(double2);
			// Copiamos los volúmenes de comunicación recibidos a memoria GPU
			if (datosClusterCPUNivelAct[j].inix != 0) {
				cudaMemcpy2D(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterDer_2, tam_pitch, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterDer_2,
					tam_pitch2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
			}
			if (! ultimaHebraXSubmallaNivelAct[j]) {
				cudaMemcpy2D(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterIzq_2, tam_pitch, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterIzq_2,
					tam_pitch2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
			}
			if (datosClusterCPUNivelAct[j].iniy != 0) {
				cudaMemcpy(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterInf_2, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterInf_2,
					2*tam_pitch, cudaMemcpyHostToDevice);
			}
			if (! ultimaHebraYSubmallaNivelAct[j]) {
				cudaMemcpy(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterSup_2, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterSup_2,
					2*tam_pitch, cudaMemcpyHostToDevice);
			}
		}
	}
	if (l < numNiveles-1) {
		cudaMemcpy(d_escrituraDatosVolumenes_1, d_datosVolumenesNivel1_1, tam_datosVolDouble2Nivel1, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_escrituraDatosVolumenes_2, d_datosVolumenesNivel1_2, tam_datosVolDouble2Nivel1, cudaMemcpyDeviceToDevice);
	}

	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas verticales pares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasVerNivel1Paso1GPU<<<blockGridVer1Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_datosVolumenesNivel1_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi, d_datosNivel1[j].altoVolumenes,
					datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y, nvx, nvy,
					submallasNivelAct[j].z, deltaTNivel1[j], d_acumuladorNivel1_1+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos,
					d_correccionEtaNivelSup+pos_sup, d_correccionEtaNivel1+pos, CFL, epsilon_h, hpos, ratioRefNivel1, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, 1, ultimaHebraXSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas horizontales pares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				nvyTotalNivelSup = submallasNivel[l-1][k].w;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasHorNivel1Paso1GPU<<<blockGridHor1Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_datosVolumenesNivel1_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi, d_datosNivel1[j].anchoVolumenes,
					datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y,
					nvx, nvy, submallasNivelAct[j].z, deltaTNivel1[j], d_acumuladorNivel1_1+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos,
					d_correccionEtaNivelSup+pos_sup, d_correccionEtaNivel1+pos, CFL, epsilon_h, hpos, ratioRefNivel1, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, 1, ultimaHebraYSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Actualizamos h en cada volumen
				obtenerEstadosPaso1NivelGPU<<<blockGridEstNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
					d_datosVolumenesNivel1_1+pos, d_datosNivel1[j].areaYCosPhi, d_datosNivel1[j].anchoVolumenes,
					d_datosNivel1[j].altoVolumenes, d_acumuladorNivel1_1+pos, d_escrituraDatosVolumenes_1+pos, nvx, nvy,
					deltaTNivel1[j], Hmin, iniGlobalSubmallasNivelAct[j], ratioRefAcumNivel1, datosClusterCPUNivelAct[j].inix,
					datosClusterCPUNivelAct[j].iniy, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer,
					sea_level, numVolxTotalNivel0, numVolyTotalNivel0);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	// Reinicializamos el acumulador1
	cudaMemset(d_acumuladorNivel1_1, 0, tam_datosVolDouble2Nivel1);

	// Esperamos a que se hayan enviado los volúmenes de comunicación a todos los clusters adyacentes
	MPI_Waitall(num_req_izq, reqSend_izq, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_der, reqSend_der, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_sup, reqSend_sup, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_inf, reqSend_inf, MPI_STATUS_IGNORE);

	// ENVIOS MPI DEL PASO 2
	num_req_izq = num_req_der = 0;
	num_req_sup = num_req_inf = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			tam_pitch = (nvx+4)*sizeof(double2);
			if (datosClusterCPUNivelAct[j].inix != 0) {
				// Recibimos los volúmenes de comunicación derechos del cluster izquierdo
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterDer_1, 1, tipo_columnas[j], hebra_izq, 23,
					comm_cartesiano, reqRecv_izq+num_req_izq);
				// Enviamos los volúmenes de comunicación izquierdos al cluster izquierdo
				cudaMemcpy2D(datosClusterCPUNivelAct[j].datosVolumenesComClusterIzq_1, tam_pitch2, datosClusterGPUNivelAct[j].d_datosVolumenesComClusterIzq_1,
					tam_pitch, 2*sizeof(double2), nvy, cudaMemcpyDeviceToHost);
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterIzq_1, 1, tipo_columnas[j], hebra_izq, 23,
					comm_cartesiano, reqSend_izq+num_req_izq);
				num_req_izq++;
			}
			if (! ultimaHebraXSubmallaNivelAct[j]) {
				// Recibimos los volúmenes de comunicación izquierdos del cluster derecho
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterIzq_1, 1, tipo_columnas[j], hebra_der, 23,
					comm_cartesiano, reqRecv_der+num_req_der);
				// Enviamos los volúmenes de comunicación derechos al cluster derecho
				cudaMemcpy2D(datosClusterCPUNivelAct[j].datosVolumenesComClusterDer_1, tam_pitch2, datosClusterGPUNivelAct[j].d_datosVolumenesComClusterDer_1,
					tam_pitch, 2*sizeof(double2), nvy, cudaMemcpyDeviceToHost);
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterDer_1, 1, tipo_columnas[j], hebra_der, 23,
					comm_cartesiano, reqSend_der+num_req_der);
				num_req_der++;
			}
			if (datosClusterCPUNivelAct[j].iniy != 0) {
				// Recibimos los volúmenes de comunicación inferiores del cluster superior
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterInf_1, 1, tipo_filas[j], hebra_sup, 24,
					comm_cartesiano, reqRecv_sup+num_req_sup);
				// Enviamos los volúmenes de comunicación superiores al cluster superior
				cudaMemcpy(datosClusterCPUNivelAct[j].datosVolumenesComClusterSup_1, datosClusterGPUNivelAct[j].d_datosVolumenesComClusterSup_1,
					2*tam_pitch, cudaMemcpyDeviceToHost);
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterSup_1, 1, tipo_filas[j], hebra_sup, 24,
					comm_cartesiano, reqSend_sup+num_req_sup);
				num_req_sup++;
			}
			if (! ultimaHebraYSubmallaNivelAct[j]) {
				// Recibimos los volúmenes de comunicación superiores del cluster inferior
				MPI_Irecv(datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterSup_1, 1, tipo_filas[j], hebra_inf, 24,
					comm_cartesiano, reqRecv_inf+num_req_inf);
				// Enviamos los volúmenes de comunicación inferiores al cluster inferior
				cudaMemcpy(datosClusterCPUNivelAct[j].datosVolumenesComClusterInf_1, datosClusterGPUNivelAct[j].d_datosVolumenesComClusterInf_1,
					2*tam_pitch, cudaMemcpyDeviceToHost);
				MPI_Isend(datosClusterCPUNivelAct[j].datosVolumenesComClusterInf_1, 1, tipo_filas[j], hebra_inf, 24,
					comm_cartesiano, reqSend_inf+num_req_inf);
				num_req_inf++;
			}
		}
	}

	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Inicializamos los espesores de los volúmenes fantasma de la submalla
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				nvyTotalNivelSup = submallasNivel[l-1][k].w;
				pos_sup = posSubmallaNivelSuperior[j];
#if (NESTED_ALGORITHM == 1)
				peso1 = deltaTNivel1[j] / (tiempo_corte - tiempoAntSubmalla[l-1][k]);
				obtenerSiguientesEspesoresVolumenesFantasmaDiferenciasNivel1GPU<<<blockGridFanNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
					d_datosVolumenesNivelSup_1+pos_sup, d_datosVolumenesNivelSupSig_1+pos_sup, d_escrituraDatosVolumenes_1+pos,
					nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y, nvx, nvy, d_posCopiaNivel1+pos,
					d_escrituraDatosVolumenes_1, peso1, ratioRefNivel1, borde_sup, borde_inf, borde_izq, borde_der, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelSup[k].numVolx, datosClusterCPUNivelAct[j].inix,
					datosClusterCPUNivelAct[j].iniy, ultimaHebraXSubmallaNivelAct[j], ultimaHebraYSubmallaNivelAct[j]);
#else
				peso1 = 1.0 - (tiempoActSubmalla[l][j] - tiempoAntSubmalla[l-1][k]) / (tiempo_corte - tiempoAntSubmalla[l-1][k]);
				obtenerSiguientesEspesoresVolumenesFantasmaValoresNivel1GPU<<<blockGridFanNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
					d_datosVolumenesNivelSup_1+pos_sup, d_datosVolumenesNivelSupSig_1+pos_sup, d_escrituraDatosVolumenes_1+pos,
					nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y, nvx, nvy, d_posCopiaNivel1+pos,
					d_escrituraDatosVolumenes_1, peso1, ratioRefNivel1, borde_sup, borde_inf, borde_izq, borde_der, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelSup[k].numVolx, datosClusterCPUNivelAct[j].inix,
					datosClusterCPUNivelAct[j].iniy, ultimaHebraXSubmallaNivelAct[j], ultimaHebraYSubmallaNivelAct[j]);
#endif
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	// PASO 2

	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas verticales impares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasVerNivel1Paso2GPU<<<blockGridVer2Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_escrituraDatosVolumenes_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi,
					d_datosNivel1[j].altoVolumenes, datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, submallasNivelAct[j].x,
					submallasNivelAct[j].y, nvx, nvy, submallasNivelAct[j].z, deltaTNivel1[j], d_acumuladorNivel1_1+pos,
					d_acumuladorNivel1_2+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos, d_correccionNivelSup_2+pos_sup,
					d_correccionNivel1_2+pos, CFL, epsilon_h, hpos, cvis, ratioRefNivel1, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, 2, ultimaHebraXSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas horizontales impares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				nvyTotalNivelSup = submallasNivel[l-1][k].w;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasHorNivel1Paso2GPU<<<blockGridHor2Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_escrituraDatosVolumenes_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi,
					d_datosNivel1[j].anchoVolumenes, datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, nvyTotalNivelSup,
					submallasNivelAct[j].x, submallasNivelAct[j].y, nvx, nvy, submallasNivelAct[j].z, deltaTNivel1[j],
					d_acumuladorNivel1_1+pos, d_acumuladorNivel1_2+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos,
					d_correccionNivelSup_2+pos_sup, d_correccionNivel1_2+pos, CFL, epsilon_h, hpos, cvis, ratioRefNivel1,
					datosClusterCPUNivelSup[k].inix, datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix,
					datosClusterCPUNivelAct[j].iniy, 2, ultimaHebraYSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	// Esperamos a que lleguen los volúmenes de comunicación de todos los clusters adyacentes
	MPI_Waitall(num_req_izq, reqRecv_izq, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_der, reqRecv_der, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_sup, reqRecv_sup, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_inf, reqRecv_inf, MPI_STATUS_IGNORE);

	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			tam_pitch = (nvx+4)*sizeof(double2);
			// Copiamos los volúmenes de comunicación recibidos a memoria GPU
			if (datosClusterCPUNivelAct[j].inix != 0) {
				cudaMemcpy2D(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterDer_1, tam_pitch, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterDer_1,
					tam_pitch2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
			}
			if (! ultimaHebraXSubmallaNivelAct[j]) {
				cudaMemcpy2D(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterIzq_1, tam_pitch, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterIzq_1,
					tam_pitch2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
			}
			if (datosClusterCPUNivelAct[j].iniy != 0) {
				cudaMemcpy(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterInf_1, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterInf_1,
					2*tam_pitch, cudaMemcpyHostToDevice);
			}
			if (! ultimaHebraYSubmallaNivelAct[j]) {
				cudaMemcpy(datosClusterGPUNivelAct[j].d_datosVolumenesComOtroClusterSup_1, datosClusterCPUNivelAct[j].datosVolumenesComOtroClusterSup_1,
					2*tam_pitch, cudaMemcpyHostToDevice);
			}
		}
	}

	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas verticales pares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasVerNivel1Paso2GPU<<<blockGridVer1Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_escrituraDatosVolumenes_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi,
					d_datosNivel1[j].altoVolumenes, datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, submallasNivelAct[j].x,
					submallasNivelAct[j].y, nvx, nvy, submallasNivelAct[j].z, deltaTNivel1[j], d_acumuladorNivel1_1+pos,
					d_acumuladorNivel1_2+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos, d_correccionNivelSup_2+pos_sup,
					d_correccionNivel1_2+pos, CFL, epsilon_h, hpos, cvis, ratioRefNivel1, datosClusterCPUNivelSup[k].inix,
					datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, 1, ultimaHebraXSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Procesamos las aristas horizontales pares
				nvxTotalNivelSup = submallasNivel[l-1][k].z;
				nvyTotalNivelSup = submallasNivel[l-1][k].w;
				pos_sup = posSubmallaNivelSuperior[j];
				procesarAristasHorNivel1Paso2GPU<<<blockGridHor1Nivel1[j], threadBlockAri, 0, streams[j&nstreams]>>>(l, numNiveles,
					d_escrituraDatosVolumenes_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi,
					d_datosNivel1[j].anchoVolumenes, datosClusterCPUNivelSup[k].numVolx, nvxTotalNivelSup, nvyTotalNivelSup,
					submallasNivelAct[j].x, submallasNivelAct[j].y, nvx, nvy, submallasNivelAct[j].z, deltaTNivel1[j],
					d_acumuladorNivel1_1+pos, d_acumuladorNivel1_2+pos, d_refinarNivelSup+pos_sup, d_refinarNivel1+pos,
					d_correccionNivelSup_2+pos_sup, d_correccionNivel1_2+pos, CFL, epsilon_h, hpos, cvis, ratioRefNivel1,
					datosClusterCPUNivelSup[k].inix, datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelAct[j].inix,
					datosClusterCPUNivelAct[j].iniy, 1, ultimaHebraYSubmallaNivelAct[j]);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}
	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			k = submallaNivelSuperior[j];
			tiempo_corte = tiempoActSubmalla[l-1][k];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				// Actualizamos los caudales y obtenemos el delta T local de cada volumen
				obtenerEstadoYDeltaTVolumenesNivelGPU<<<blockGridEstNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
					d_escrituraDatosVolumenes_1+pos, d_datosVolumenesNivel1_2+pos, d_datosNivel1[j].areaYCosPhi,
					d_datosNivel1[j].anchoVolumenes, d_datosNivel1[j].altoVolumenes, d_acumuladorNivel1_1+pos,
					d_acumuladorNivel1_2+pos, d_deltaTVolumenesNivel1+pos, d_escrituraDatosVolumenes_2+pos,
					nvx, nvy, CFL, deltaTNivel1[j], d_friccionesNivel1+pos, vmax, hpos, epsilon_h, Hmin,
					iniGlobalSubmallasNivelAct[j], ratioRefAcumNivel1, datosClusterCPUNivelAct[j].inix,
					datosClusterCPUNivelAct[j].iniy, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer,
					sea_level, numVolxTotalNivel0, numVolyTotalNivel0);
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	// Esperamos a que se hayan enviado los volúmenes de comunicación a todos los clusters adyacentes
	MPI_Waitall(num_req_izq, reqSend_izq, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_der, reqSend_der, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_sup, reqSend_sup, MPI_STATUS_IGNORE);
	MPI_Waitall(num_req_inf, reqSend_inf, MPI_STATUS_IGNORE);

	memcpy(tiempoAntSubmalla[l], tiempoActSubmalla[l], numSubmallasNivelAct*sizeof(double));
	memcpy(dT_min, deltaTNivel1, numSubmallasNivelAct*sizeof(double));

	fin = true;
	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		k = submallaNivelSuperior[j];
		tiempo_corte = tiempoActSubmalla[l-1][k];
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				tiempoActSubmalla[l][j] += deltaTNivel1[j];
				// Obtenemos el mínimo delta T aplicando un algoritmo de reducción
				dT_min[j] = obtenerMinimoReduccion<double>(d_deltaTVolumenesNivel1+pos, nvx*nvy);
				deltaTNivelSinTruncar[j] = dT_min[j];
				if (tiempoActSubmalla[l][j] + dT_min[j] > tiempo_corte)
					dT_min[j] = tiempo_corte - tiempoActSubmalla[l][j];
			}
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON) {
				fin = false;
			}
			pos += (nvx + 4)*(nvy + 4);
		}
		else {
			dT_min[j] = DBL_MAX;
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON)
				tiempoActSubmalla[l][j] += deltaTNivel1[j];
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) > EPSILON)
				fin = false;
		}
	}

	if ((l == numNiveles-1) && (! fin)) {
		// Copiamos los volúmenes de comunicación del nivel l a memoria CPU de forma asíncrona
		for (j=0; j<numSubmallasNivelAct; j++) {
			if (datosClusterCPUNivelAct[j].iniy != -1) {
				copiarVolumenesComAsincACPU(&(datosClusterCPUNivelAct[j]), &(datosClusterGPUNivelAct[j]), streamMemcpy,
					datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, ultimaHebraXSubmallaNivelAct[j], ultimaHebraYSubmallaNivelAct[j]);
			}
		}
	}
	else if (l < numNiveles-1) {
		// Copiamos los volúmenes de comunicación del nivel l+1 a memoria CPU de forma asíncrona
		datosClusterCPUNivelInf = datosClusterCPU[l+1];
		for (j=0; j<numSubmallasNivel[l+1]; j++) {
			if (datosClusterCPUNivelInf[j].iniy != -1) {
				copiarVolumenesComAsincACPU(&(datosClusterCPUNivelInf[j]), &(datosClusterGPU[l+1][j]), streamMemcpy,
					datosClusterCPUNivelInf[j].inix, datosClusterCPUNivelInf[j].iniy, ultimaHebraXSubmalla[l+1][j], ultimaHebraYSubmalla[l+1][j]);
			}
		}
	}

	// Obtenemos el mínimo delta T de cada submalla por reducción
	MPI_Allreduce (dT_min, deltaTNivel1, numSubmallasNivelAct, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	if (haySubmallasAdyacentesNivel1) {
		// Si hay submallas adyacentes, asignamos a todas las submallas el mínimo deltaT de las submallas
		min_deltaT = deltaTNivel1[0];
		for (j=1; j<numSubmallasNivelAct; j++)
			min_deltaT = min(min_deltaT, deltaTNivel1[j]);
		for (j=0; j<numSubmallasNivelAct; j++)
			deltaTNivel1[j] = min_deltaT;
	}
	if (id_hebra == 0) {
		for (j=0; j<numSubmallasNivelAct; j++) {
			for (k=0; k<l; k++)
				fprintf(stdout, "  ");
			fprintf(stdout, "Level %d, Submesh %d, deltaT = %e sec\n", l, j+1, deltaTNivel1[j]*T);
		}
	}

	pos = 0;
	for (j=0; j<numSubmallasNivelAct; j++) {
		k = submallaNivelSuperior[j];
		tiempo_corte = tiempoActSubmalla[l-1][k];
		if (datosClusterCPUNivelAct[j].iniy != -1) {
			nvx = datosClusterCPUNivelAct[j].numVolx;
			nvy = datosClusterCPUNivelAct[j].numVoly;
			if (fabs(tiempo_corte - tiempoActSubmalla[l][j]) <= EPSILON) {
				// Inicializamos los caudales de los volúmenes fantasma de la submalla para la siguiente iteración
				if (fabs(tiempoActSubmalla[l][j] - tiempoAntSubmalla[l][j]) > EPSILON) {
					nvxTotalNivelSup = submallasNivel[l-1][k].z;
					nvyTotalNivelSup = submallasNivel[l-1][k].w;
					pos_sup = posSubmallaNivelSuperior[j];
#if (NESTED_ALGORITHM == 1)
					peso1 = (tiempoActSubmalla[l][j] - tiempoAntSubmalla[l][j]) / (tiempo_corte - tiempoAntSubmalla[l-1][k]);
					obtenerSiguientesCaudalesVolumenesFantasmaDiferenciasNivel1GPU<<<blockGridFanNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
						d_datosVolumenesNivelSup_2+pos_sup, d_datosVolumenesNivelSupSig_2+pos_sup, d_escrituraDatosVolumenes_1+pos,
						d_escrituraDatosVolumenes_2+pos, nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y,
						nvx, nvy, d_posCopiaNivel1+pos, d_escrituraDatosVolumenes_2, peso1, ratioRefNivel1, borde_sup, borde_inf, borde_izq,
						borde_der, datosClusterCPUNivelSup[k].inix, datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelSup[k].numVolx,
						datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, ultimaHebraXSubmallaNivelAct[j], ultimaHebraYSubmallaNivelAct[j]);
#else
					peso1 = 1.0 - (tiempoActSubmalla[l][j] - tiempoAntSubmalla[l-1][k]) / (tiempo_corte - tiempoAntSubmalla[l-1][k]);
					obtenerSiguientesCaudalesVolumenesFantasmaValoresNivel1GPU<<<blockGridFanNivel1[j], threadBlockEst, 0, streams[j&nstreams]>>>(
						d_datosVolumenesNivelSup_2+pos_sup, d_datosVolumenesNivelSupSig_2+pos_sup, d_escrituraDatosVolumenes_1+pos,
						d_escrituraDatosVolumenes_2+pos, nvxTotalNivelSup, nvyTotalNivelSup, submallasNivelAct[j].x, submallasNivelAct[j].y,
						nvx, nvy, d_posCopiaNivel1+pos, d_escrituraDatosVolumenes_2, peso1, ratioRefNivel1, borde_sup, borde_inf, borde_izq,
						borde_der, datosClusterCPUNivelSup[k].inix, datosClusterCPUNivelSup[k].iniy, datosClusterCPUNivelSup[k].numVolx,
						datosClusterCPUNivelAct[j].inix, datosClusterCPUNivelAct[j].iniy, ultimaHebraXSubmallaNivelAct[j], ultimaHebraYSubmallaNivelAct[j]);
#endif
				}
			}
			pos += (nvx + 4)*(nvy + 4);
		}
	}

	return fin;
}*/

#endif
