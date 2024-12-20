#ifndef _TIMESTEP_NOHIDRO_H_
#define _TIMESTEP_NOHIDRO_H_

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
#include "NoHidro_kernel.cu"

/**************************/
/* Siguiente paso nivel 0 */
/**************************/

// Usamos d_aristaReconstruido para guardar los errores locales del sistema de ecuaciones (d_errorNH)
void pasoNoHidros(int numPesosJacobi, double *omega, int *ord, TDatosClusterCPU *datosClusterCPU, TDatosClusterGPU *datosClusterGPU,
		double2 *d0_datosVolumenesh, double3 *d0_datosVolumenesQ, tipoDatosSubmalla *d_datosNivel0, double2 *d_datosVolumenesh,
		double3 *d_datosVolumenesQ, int numVolyTotalNivel0, double *d_errorNH, double delta_T, int it, double coef2, double radio_tierra,
		double *d_RHS_dt, double *d_dtHs, double *d_CoefPE, double *d_CoefPW, double *d_CoefPN, double *d_CoefPS, double *d_Pnh0,
		double *d_Pnh1, int *iter_sistema, double *error_sistema, dim3 blockGridEst, dim3 blockGridVert, dim3 threadBlockEst,
		cudaStream_t streamMemcpy, MPI_Datatype tipoFilasDouble2, MPI_Datatype tipoFilasDouble3, MPI_Datatype tipoColumnasDouble2,
		MPI_Datatype tipoColumnasDouble3, MPI_Datatype tipoColumnasPresion, int64_t tam_datosVertDoubleNivel, double L, double H,
		int inixSubmallaCluster, int iniySubmallaCluster, int id_hebraX, int id_hebraY, bool ultima_hebraX, bool ultima_hebraY,
		MPI_Comm comm_cartesiano, int hebra_izq, int hebra_der, int hebra_sup, int hebra_inf)
{
	// Volumenes
	int nvx = datosClusterCPU->numVolx;
	int nvy = datosClusterCPU->numVoly;
	int tam_pitchDouble2 = (datosClusterCPU->numVolx+4)*sizeof(double2);
	int tam_pitchDouble2_2 = 8*sizeof(double2);
	int tam_pitchDouble3 = (datosClusterCPU->numVolx+4)*sizeof(double3);
	int tam_pitchDouble3_2 = 8*sizeof(double3);
	// Vértices (presiones)
	int numVertices = (nvx+1)*(nvy+1);
	int tam_pitchDoubleP_1 = (nvx+3)*sizeof(double);
	int tam_pitchDoubleP_2 = 4*sizeof(double);
	MPI_Request reqSend_izq[2], reqSend_der[2];
	MPI_Request reqSend_sup[2], reqSend_inf[2];
	MPI_Request reqRecv_izq[2], reqRecv_der[2];
	MPI_Request reqRecv_sup[2], reqRecv_inf[2];
	double error_max;
	int iomega;

	// COPIA A CPU DE h Y q Y ENVIOS MPI
	if (! ultima_hebraX) {
		// Recibimos los volúmenes de comunicación izquierdos del cluster derecho
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterIzq_1, 1, tipoColumnasDouble2, hebra_der, 41, comm_cartesiano, reqRecv_der);
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterIzq_2, 1, tipoColumnasDouble3, hebra_der, 42, comm_cartesiano, reqRecv_der+1);
		// Volúmenes de comunicación derechos
		cudaMemcpy2D(datosClusterCPU->datosVolumenesComClusterDer_1, tam_pitchDouble2_2, datosClusterGPU->d_datosVolumenesComClusterDer_1,
			tam_pitchDouble2, 2*sizeof(double2), nvy, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(datosClusterCPU->datosVolumenesComClusterDer_2, tam_pitchDouble3_2, datosClusterGPU->d_datosVolumenesComClusterDer_2,
			tam_pitchDouble3, 2*sizeof(double3), nvy, cudaMemcpyDeviceToHost);
		// Enviamos los volúmenes de comunicación derechos al cluster derecho
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterDer_1, 1, tipoColumnasDouble2, hebra_der, 43, comm_cartesiano, reqSend_der);
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterDer_2, 1, tipoColumnasDouble3, hebra_der, 44, comm_cartesiano, reqSend_der+1);
	}
	if (id_hebraX != 0) {
		// Recibimos los volúmenes de comunicación derechos del cluster izquierdo
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterDer_1, 1, tipoColumnasDouble2, hebra_izq, 43, comm_cartesiano, reqRecv_izq);
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterDer_2, 1, tipoColumnasDouble3, hebra_izq, 44, comm_cartesiano, reqRecv_izq+1);
		// Volúmenes de comunicación izquierdos
		cudaMemcpy2D(datosClusterCPU->datosVolumenesComClusterIzq_1, tam_pitchDouble2_2, datosClusterGPU->d_datosVolumenesComClusterIzq_1,
			tam_pitchDouble2, 2*sizeof(double2), nvy, cudaMemcpyDeviceToHost);
		cudaMemcpy2D(datosClusterCPU->datosVolumenesComClusterIzq_2, tam_pitchDouble3_2, datosClusterGPU->d_datosVolumenesComClusterIzq_2,
			tam_pitchDouble3, 2*sizeof(double3), nvy, cudaMemcpyDeviceToHost);
		// Enviamos los volúmenes de comunicación izquierdos al cluster izquierdo
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterIzq_1, 1, tipoColumnasDouble2, hebra_izq, 41, comm_cartesiano, reqSend_izq);
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterIzq_2, 1, tipoColumnasDouble3, hebra_izq, 42, comm_cartesiano, reqSend_izq+1);
	}
	if (! ultima_hebraY) {
		// Recibimos los volúmenes de comunicación superiores del cluster inferior
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterSup_1, 1, tipoFilasDouble2, hebra_inf, 51, comm_cartesiano, reqRecv_inf);
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterSup_2, 1, tipoFilasDouble3, hebra_inf, 52, comm_cartesiano, reqRecv_inf+1);
		// Volúmenes de comunicación inferiores
		cudaMemcpy(datosClusterCPU->datosVolumenesComClusterInf_1, datosClusterGPU->d_datosVolumenesComClusterInf_1, 2*tam_pitchDouble2, cudaMemcpyDeviceToHost);
		cudaMemcpy(datosClusterCPU->datosVolumenesComClusterInf_2, datosClusterGPU->d_datosVolumenesComClusterInf_2, 2*tam_pitchDouble3, cudaMemcpyDeviceToHost);
		// Enviamos los volúmenes de comunicación inferiores al cluster inferior
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterInf_1, 1, tipoFilasDouble2, hebra_inf, 53, comm_cartesiano, reqSend_inf);
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterInf_2, 1, tipoFilasDouble3, hebra_inf, 54, comm_cartesiano, reqSend_inf+1);
	}
	if (id_hebraY != 0) {
		// Recibimos los volúmenes de comunicación inferiores del cluster superior
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterInf_1, 1, tipoFilasDouble2, hebra_sup, 53, comm_cartesiano, reqRecv_sup);
		MPI_Irecv(datosClusterCPU->datosVolumenesComOtroClusterInf_2, 1, tipoFilasDouble3, hebra_sup, 54, comm_cartesiano, reqRecv_sup+1);
		// Volúmenes de comunicación superiores
		cudaMemcpy(datosClusterCPU->datosVolumenesComClusterSup_1, datosClusterGPU->d_datosVolumenesComClusterSup_1, 2*tam_pitchDouble2, cudaMemcpyDeviceToHost);
		cudaMemcpy(datosClusterCPU->datosVolumenesComClusterSup_2, datosClusterGPU->d_datosVolumenesComClusterSup_2, 2*tam_pitchDouble3, cudaMemcpyDeviceToHost);
		// Enviamos los volúmenes de comunicación superiores al cluster superior
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterSup_1, 1, tipoFilasDouble2, hebra_sup, 51, comm_cartesiano, reqSend_sup);
		MPI_Isend(datosClusterCPU->datosVolumenesComClusterSup_2, 1, tipoFilasDouble3, hebra_sup, 52, comm_cartesiano, reqSend_sup+1);
	}

	compute_NH_coefficientsNoCom<<<blockGridVert, threadBlockEst>>>(d_datosVolumenesh, d_datosVolumenesQ, d_dtHs, d_datosNivel0->vccos,
		d_datosNivel0->vtan, nvx, nvy, numVolyTotalNivel0, coef2, delta_T, d_datosNivel0->dx, d_datosNivel0->dy, radio_tierra, L, H,
		d_RHS_dt, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, inixSubmallaCluster, iniySubmallaCluster, ultima_hebraX, ultima_hebraY);

	// Esperamos a que lleguen los volúmenes de comunicación de todos los clusters adyacentes
	// y copiamos los volúmenes de comunicación recibidos a memoria GPU
	if (id_hebraX != 0) {
		MPI_Waitall(2, reqRecv_izq, MPI_STATUS_IGNORE);
		cudaMemcpy2D(datosClusterGPU->d_datosVolumenesComOtroClusterDer_1, tam_pitchDouble2, datosClusterCPU->datosVolumenesComOtroClusterDer_1,
			tam_pitchDouble2_2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
		cudaMemcpy2D(datosClusterGPU->d_datosVolumenesComOtroClusterDer_2, tam_pitchDouble3, datosClusterCPU->datosVolumenesComOtroClusterDer_2,
			tam_pitchDouble3_2, 2*sizeof(double3), nvy, cudaMemcpyHostToDevice);
	}
	if (! ultima_hebraX) {
		MPI_Waitall(2, reqRecv_der, MPI_STATUS_IGNORE);
		cudaMemcpy2D(datosClusterGPU->d_datosVolumenesComOtroClusterIzq_1, tam_pitchDouble2, datosClusterCPU->datosVolumenesComOtroClusterIzq_1,
			tam_pitchDouble2_2, 2*sizeof(double2), nvy, cudaMemcpyHostToDevice);
		cudaMemcpy2D(datosClusterGPU->d_datosVolumenesComOtroClusterIzq_2, tam_pitchDouble3, datosClusterCPU->datosVolumenesComOtroClusterIzq_2,
			tam_pitchDouble3_2, 2*sizeof(double3), nvy, cudaMemcpyHostToDevice);
	}
	if (id_hebraY != 0) {
		MPI_Waitall(2, reqRecv_sup, MPI_STATUS_IGNORE);
		cudaMemcpy(datosClusterGPU->d_datosVolumenesComOtroClusterInf_1, datosClusterCPU->datosVolumenesComOtroClusterInf_1,
			2*tam_pitchDouble2, cudaMemcpyHostToDevice);
		cudaMemcpy(datosClusterGPU->d_datosVolumenesComOtroClusterInf_2, datosClusterCPU->datosVolumenesComOtroClusterInf_2,
			2*tam_pitchDouble3, cudaMemcpyHostToDevice);
	}
	if (! ultima_hebraY) {
		MPI_Waitall(2, reqRecv_inf, MPI_STATUS_IGNORE);
		cudaMemcpy(datosClusterGPU->d_datosVolumenesComOtroClusterSup_1, datosClusterCPU->datosVolumenesComOtroClusterSup_1,
			2*tam_pitchDouble2, cudaMemcpyHostToDevice);
		cudaMemcpy(datosClusterGPU->d_datosVolumenesComOtroClusterSup_2, datosClusterCPU->datosVolumenesComOtroClusterSup_2,
			2*tam_pitchDouble3, cudaMemcpyHostToDevice);
	}

	compute_NH_coefficientsCom<<<blockGridVert, threadBlockEst>>>(d_datosVolumenesh, d_datosVolumenesQ, d_dtHs, d_datosNivel0->vccos,
		d_datosNivel0->vtan, nvx, nvy, numVolyTotalNivel0, coef2, delta_T, d_datosNivel0->dx, d_datosNivel0->dy, radio_tierra, L, H,
		d_RHS_dt, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, inixSubmallaCluster, iniySubmallaCluster, ultima_hebraX, ultima_hebraY);

	// Copiamos las presiones de comunicación a memoria CPU de forma asíncrona
	cudaStreamSynchronize(0);
	copiarPresionesComAsincACPU(datosClusterCPU, datosClusterGPU, streamMemcpy, id_hebraX, id_hebraY, ultima_hebraX, ultima_hebraY);

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

	*iter_sistema = 0;
	*error_sistema = 1e20;
	while ((*iter_sistema < ITER_MAX_NH) && ((*iter_sistema < 3) || (*error_sistema > ERROR_NH))) {
		iomega = (*iter_sistema)%numPesosJacobi;

		// ENVIOS MPI
		// Enviamos las presiones de comunicación al cluster que corresponda
		cudaStreamSynchronize(streamMemcpy);
		if (id_hebraX != 0) {
			// Recibimos las presiones de comunicación derechas del cluster izquierdo
			MPI_Irecv(datosClusterCPU->datosVerticesComOtroClusterDer_P, 1, tipoColumnasPresion, hebra_izq, 51, comm_cartesiano, reqRecv_izq);
			// Enviamos las presiones de comunicación izquierdas al cluster izquierdo
			MPI_Isend(datosClusterCPU->datosVerticesComClusterIzq_P, 1, tipoColumnasPresion, hebra_izq, 52, comm_cartesiano, reqSend_izq);
		}
		if (! ultima_hebraX) {
			// Recibimos las presiones de comunicación izquierdas del cluster derecho
			MPI_Irecv(datosClusterCPU->datosVerticesComOtroClusterIzq_P, 1, tipoColumnasPresion, hebra_der, 52, comm_cartesiano, reqRecv_der);
			// Enviamos las presiones de comunicación derechas al cluster derecho
			MPI_Isend(datosClusterCPU->datosVerticesComClusterDer_P, 1, tipoColumnasPresion, hebra_der, 51, comm_cartesiano, reqSend_der);
		}
		if (id_hebraY != 0) {
			// Recibimos las presiones de comunicación inferiores del cluster superior
			MPI_Irecv(datosClusterCPU->datosVerticesComOtroClusterInf_P, nvx+3, MPI_DOUBLE, hebra_sup, 55, comm_cartesiano, reqRecv_sup);
			// Enviamos las presiones de comunicación superiores al cluster superior
			MPI_Isend(datosClusterCPU->datosVerticesComClusterSup_P, nvx+3, MPI_DOUBLE, hebra_sup, 56, comm_cartesiano, reqSend_sup);
		}
		if (! ultima_hebraY) {
			// Recibimos las presiones de comunicación superiores del cluster inferior
			MPI_Irecv(datosClusterCPU->datosVerticesComOtroClusterSup_P, nvx+3, MPI_DOUBLE, hebra_inf, 56, comm_cartesiano, reqRecv_inf);
			// Enviamos las presiones de comunicación inferiores al cluster inferior
			MPI_Isend(datosClusterCPU->datosVerticesComClusterInf_P, nvx+3, MPI_DOUBLE, hebra_inf, 55, comm_cartesiano, reqSend_inf);
		}

		compute_NH_pressureNoCom<<<blockGridVert, threadBlockEst>>>(d_Pnh0, d_Pnh1, d_errorNH, nvx, nvy, d_RHS_dt, d_CoefPE, d_CoefPW,
			d_CoefPN, d_CoefPS, omega[ord[iomega]]);

		// Esperamos a que lleguen las presiones de comunicación de todos los clusters adyacentes
		if (id_hebraX != 0) {
			MPI_Wait(reqRecv_izq, MPI_STATUS_IGNORE);
			cudaMemcpy2D(datosClusterGPU->d_datosVerticesComOtroClusterDer_P, tam_pitchDoubleP_1, datosClusterCPU->datosVerticesComOtroClusterDer_P,
				tam_pitchDoubleP_2, 1*sizeof(double), nvy+1, cudaMemcpyHostToDevice);
		}
		if (! ultima_hebraX) {
			MPI_Wait(reqRecv_der, MPI_STATUS_IGNORE);
			cudaMemcpy2D(datosClusterGPU->d_datosVerticesComOtroClusterIzq_P, tam_pitchDoubleP_1, datosClusterCPU->datosVerticesComOtroClusterIzq_P,
				tam_pitchDoubleP_2, 1*sizeof(double), nvy+1, cudaMemcpyHostToDevice);
		}
		if (id_hebraY != 0) {
			MPI_Wait(reqRecv_sup, MPI_STATUS_IGNORE);
			cudaMemcpy(datosClusterGPU->d_datosVerticesComOtroClusterInf_P, datosClusterCPU->datosVerticesComOtroClusterInf_P,
				tam_pitchDoubleP_1, cudaMemcpyHostToDevice);
		}
		if (! ultima_hebraY) {
			MPI_Wait(reqRecv_inf, MPI_STATUS_IGNORE);
			cudaMemcpy(datosClusterGPU->d_datosVerticesComOtroClusterSup_P, datosClusterCPU->datosVerticesComOtroClusterSup_P,
				tam_pitchDoubleP_1, cudaMemcpyHostToDevice);
		}

		compute_NH_pressureCom<<<blockGridVert, threadBlockEst>>>(d_Pnh0, d_Pnh1, d_errorNH, nvx, nvy, d_RHS_dt, d_CoefPE, d_CoefPW,
			d_CoefPN, d_CoefPS, omega[ord[iomega]], inixSubmallaCluster, iniySubmallaCluster, ultima_hebraX, ultima_hebraY);

		// Esperamos a que se hayan enviado las presiones de comunicación a todos los clusters adyacentes
		if (id_hebraX != 0) {
			MPI_Wait(reqSend_izq, MPI_STATUS_IGNORE);
		}
		if (! ultima_hebraX) {
			MPI_Wait(reqSend_der, MPI_STATUS_IGNORE);
		}
		if (id_hebraY != 0) {
			MPI_Wait(reqSend_sup, MPI_STATUS_IGNORE);
		}
		if (! ultima_hebraY) {
			MPI_Wait(reqSend_inf, MPI_STATUS_IGNORE);
		}

		cudaMemcpy(d_Pnh0, d_Pnh1, tam_datosVertDoubleNivel, cudaMemcpyDeviceToDevice);

		// Copiamos las presiones de comunicación a memoria CPU de forma asíncrona
		cudaStreamSynchronize(0);
		copiarPresionesComAsincACPU(datosClusterCPU, datosClusterGPU, streamMemcpy, id_hebraX, id_hebraY, ultima_hebraX, ultima_hebraY);

		(*iter_sistema)++;

		error_max = -obtenerMinimoReduccion<double>(d_errorNH, numVertices);

		// Obtenemos el error máximo de todos los clusters por reducción
		MPI_Allreduce(&error_max, error_sistema, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
	if ((id_hebraX == 0) && (id_hebraY == 0)) {
		fprintf(stdout, "  Iterations NH system = %d, Error NH system = %e\n", *iter_sistema, *error_sistema);
	}

	NH_correction<<<blockGridEst, threadBlockEst>>>(d0_datosVolumenesQ, d_datosVolumenesh, d_datosVolumenesQ, d_datosNivel0->vccos,
		d_datosNivel0->vtan, d_Pnh1, nvx, nvy, numVolyTotalNivel0, coef2, delta_T, d_datosNivel0->dx, d_datosNivel0->dy,
		radio_tierra, L, H, it, inixSubmallaCluster, iniySubmallaCluster, ultima_hebraX, ultima_hebraY);
}

// El nuevo estado se escribe en d_datosVolumenesNivel0Sig_1 y d_datosVolumenesNivel0Sig_2.
double siguientePasoNivel0NoHidros(int numPesosJacobi, double *omega, int *ord, int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		TDatosClusterGPU datosClusterGPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 *d_datosVolumenesNivel0_1, double3 *d_datosVolumenesNivel0_2,
		tipoDatosSubmalla *d_datosNivel0, double2 *d_datosVolumenesNivel0Sig_1, double3 *d_datosVolumenesNivel0Sig_2, double *d_aristaReconstruido,
		bool *d_refinarNivel0, int numVolxTotalNivel0, int numVolyTotalNivel0, double Hmin, double *d_deltaTVolumenes, double *d_correccionEtaNivel0,
		double2 *d_correccionNivel0_2, double2 *d_acumulador_1, double3 *d_acumulador_2, double borde_sup, double borde_inf, double borde_izq,
		double borde_der, int tam_spongeSup, int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer, double sea_level, double *tiempo_act, double CFL,
		double delta_T, double *d_friccionesNivel0, double vmax, double epsilon_h, int *numSubmallasNivel, int2 iniGlobalSubmallaNivel0, double *d_RHS_dt,
		double *d_dtHs, double *d_CoefPE, double *d_CoefPW, double *d_CoefPN, double *d_CoefPS, double *d_Pnh0, double *d_Pnh1, int *iter_sistema,
		double *error_sistema, dim3 blockGridVer1, dim3 blockGridVer2, dim3 blockGridHor1, dim3 blockGridHor2, dim3 threadBlockAri, dim3 blockGridEst,
		dim3 blockGridFan, dim3 blockGridVert, dim3 threadBlockEst, cudaStream_t streamMemcpy, MPI_Datatype tipoFilasDouble2, MPI_Datatype tipoFilasDouble3,
		MPI_Datatype tipoColumnasDouble2, MPI_Datatype tipoColumnasDouble3, MPI_Datatype tipoColumnasPresion, int64_t tam_datosVertDoubleNivel0, double L,
		double H, int id_hebraX, int id_hebraY, bool ultimaHebraXSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL], bool ultimaHebraYSubmalla[MAX_LEVELS][MAX_GRIDS_LEVEL],
		MPI_Comm comm_cartesiano, int hebra_izq, int hebra_der, int hebra_sup, int hebra_inf)
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
	int tam_pitchDouble3 = (nvx+4)*sizeof(double3);
	int tam_pitchDouble3_2 = 8*sizeof(double3);
	MPI_Request reqSend_izq, reqSend_der;
	MPI_Request reqSend_sup, reqSend_inf;
	MPI_Request reqRecv_izq, reqRecv_der;
	MPI_Request reqRecv_sup, reqRecv_inf;
	double coef1, coef2, coef3;
	double dT_min;
	int j, k;

	for (k=1; k<=OrdenTiempo; k++) {
		// ENVIOS MPI
		// Esperamos a que lleguen los volúmenes de comunicación a CPU y los enviamos al cluster que corresponda
		cudaStreamSynchronize(streamMemcpy);
		if (! ultima_hebraX) {
			// Recibimos los volúmenes de comunicación izquierdos del cluster derecho
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterIzq_2, 1, tipoColumnasDouble3, hebra_der, 22, comm_cartesiano, &reqRecv_der);
			// Enviamos los volúmenes de comunicación derechos al cluster derecho
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterDer_2, 1, tipoColumnasDouble3, hebra_der, 24, comm_cartesiano, &reqSend_der);
		}
		if (id_hebraX != 0) {
			// Recibimos los volúmenes de comunicación derechos del cluster izquierdo
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterDer_2, 1, tipoColumnasDouble3, hebra_izq, 24, comm_cartesiano, &reqRecv_izq);
			// Enviamos los volúmenes de comunicación izquierdos al cluster izquierdo
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterIzq_2, 1, tipoColumnasDouble3, hebra_izq, 22, comm_cartesiano, &reqSend_izq);
		}
		if (! ultima_hebraY) {
			// Recibimos los volúmenes de comunicación superiores del cluster inferior
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterSup_2, 1, tipoFilasDouble3, hebra_inf, 32, comm_cartesiano, &reqRecv_inf);
			// Enviamos los volúmenes de comunicación inferiores al cluster inferior
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterInf_2, 1, tipoFilasDouble3, hebra_inf, 34, comm_cartesiano, &reqSend_inf);
		}
		if (id_hebraY != 0) {
			// Recibimos los volúmenes de comunicación inferiores del cluster superior
			MPI_Irecv(datosClusterCPUNivel0->datosVolumenesComOtroClusterInf_2, 1, tipoFilasDouble3, hebra_sup, 34, comm_cartesiano, &reqRecv_sup);
			// Enviamos los volúmenes de comunicación superiores al cluster superior
			MPI_Isend(datosClusterCPUNivel0->datosVolumenesComClusterSup_2, 1, tipoFilasDouble3, hebra_sup, 32, comm_cartesiano, &reqSend_sup);
		}

		calcularCoeficientesReconstruccionNoCom<<<blockGridEst, threadBlockEst>>>(d_datosVolumenesNivel0Sig_1, d_datosVolumenesNivel0Sig_2,
			d_datosNivel0->vcos, d_datosNivel0->vtan, d_datosNivel0->vccos, d_aristaReconstruido, d_acumulador_2, nvx, nvy, numVolyTotalNivel0,
			d_datosNivel0->dx, d_datosNivel0->dy, radio_tierra, vmax, delta_T, iniySubmallaCluster, ultima_hebraY);

		// Esperamos a que lleguen los volúmenes de comunicación de todos los clusters adyacentes
		// y copiamos los volúmenes de comunicación recibidos a memoria GPU
		if (id_hebraX != 0) {
			MPI_Wait(&reqRecv_izq, MPI_STATUS_IGNORE);
			cudaMemcpy2D(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterDer_2, tam_pitchDouble3, datosClusterCPUNivel0->datosVolumenesComOtroClusterDer_2,
				tam_pitchDouble3_2, 2*sizeof(double3), nvy, cudaMemcpyHostToDevice);
		}
		if (! ultima_hebraX) {
			MPI_Wait(&reqRecv_der, MPI_STATUS_IGNORE);
			cudaMemcpy2D(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterIzq_2, tam_pitchDouble3, datosClusterCPUNivel0->datosVolumenesComOtroClusterIzq_2,
				tam_pitchDouble3_2, 2*sizeof(double3), nvy, cudaMemcpyHostToDevice);
		}
		if (id_hebraY != 0) {
			MPI_Wait(&reqRecv_sup, MPI_STATUS_IGNORE);
			cudaMemcpy(datosClusterGPUNivel0->d_datosVolumenesComOtroClusterInf_2, datosClusterCPUNivel0->datosVolumenesComOtroClusterInf_2,
				2*tam_pitchDouble3, cudaMemcpyHostToDevice);
		}
		if (! ultima_hebraY) {
			MPI_Wait(&reqRecv_inf, MPI_STATUS_IGNORE);
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

		pasoNoHidros(numPesosJacobi, omega, ord, datosClusterCPUNivel0, datosClusterGPUNivel0, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
			d_datosNivel0, d_datosVolumenesNivel0Sig_1, d_datosVolumenesNivel0Sig_2, numVolyTotalNivel0, d_aristaReconstruido, delta_T,
			k, coef2, radio_tierra, d_RHS_dt, d_dtHs, d_CoefPE, d_CoefPW, d_CoefPN, d_CoefPS, d_Pnh0, d_Pnh1, iter_sistema, error_sistema,
			blockGridEst, blockGridVert, threadBlockEst, streamMemcpy, tipoFilasDouble2, tipoFilasDouble3, tipoColumnasDouble2, tipoColumnasDouble3,
			tipoColumnasPresion, tam_datosVertDoubleNivel0, L, H, inixSubmallaCluster, iniySubmallaCluster, id_hebraX, id_hebraY, ultima_hebraX,
			ultima_hebraY, comm_cartesiano, hebra_izq, hebra_der, hebra_sup, hebra_inf);

		// Esperamos a que se hayan enviado los volúmenes de comunicación a todos los clusters adyacentes
		if (id_hebraX != 0) {
			MPI_Wait(&reqSend_izq, MPI_STATUS_IGNORE);
		}
		if (! ultima_hebraX) {
			MPI_Wait(&reqSend_der, MPI_STATUS_IGNORE);
		}
		if (id_hebraY != 0) {
			MPI_Wait(&reqSend_sup, MPI_STATUS_IGNORE);
		}
		if (! ultima_hebraY) {
			MPI_Wait(&reqSend_inf, MPI_STATUS_IGNORE);
		}

		if (numNiveles == 1) {
			// Copiamos los volúmenes de comunicación del nivel 0 a memoria CPU de forma asíncrona
			cudaStreamSynchronize(0);
			copiarVolumenesComAsincACPUNoHidros(datosClusterCPUNivel0, datosClusterGPUNivel0, streamMemcpy, id_hebraX, id_hebraY,
				ultima_hebraX, ultima_hebraY);
		}
		else {
			// Copiamos los volúmenes de comunicación del nivel 1 a memoria CPU de forma asíncrona
			for (j=0; j<numSubmallasNivel[1]; j++) {
				if (datosClusterCPU[1][j].iniy != -1) {
					copiarVolumenesComAsincACPUNoHidros(&(datosClusterCPU[1][j]), &(datosClusterGPU[1][j]), streamMemcpy,
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
