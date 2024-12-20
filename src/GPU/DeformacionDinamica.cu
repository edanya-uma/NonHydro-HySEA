#ifndef _DEFORMACION_DINAMICA_H_
#define _DEFORMACION_DINAMICA_H_

#include <stdio.h>
#include "Constantes.hxx"
#include "Deformacion.cu"
#include <cufft.h>

/*********************************/
/* Interpolación para el nivel 0 */
/*********************************/

__global__ void interpolarDefEnTiempoNivel0GPU(double *d_defSigNivel0, double defTime_ant, double defTime_sig,
			double tiempoAnt, double tiempoAct, double *d_deformacionInterpoladaNivel0, int nvxDef, int nvyDef)
{
	int pos_x_hebra, pos_y_hebra;
	int pos_def;
	double peso, U_Z;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < nvxDef) && (pos_y_hebra < nvyDef)) {
		pos_def = pos_y_hebra*nvxDef + pos_x_hebra;
		U_Z = d_defSigNivel0[pos_def];

		if (fabs(defTime_sig-defTime_ant) > EPSILON) {
			// Estamos en un estado de la deformación dinámica distinto del inicial (para el estado inicial,
			// indiceFallaAnt=indiceFallaSig=0, con lo que defTime_ant=defTime_sig)
			// peso a aplicar al siguiente estado (de 0 a 1)
			peso = (tiempoAct - tiempoAnt) / (defTime_sig - defTime_ant);
			U_Z = peso*U_Z;
		}
		d_deformacionInterpoladaNivel0[pos_def] = U_Z;
	}
}


/*********************************************/
/* Interpolación para los niveles inferiores */
/*********************************************/

__global__ void interpolarDefDinamicaGPU(int l, int numNiveles, double2 *d_datosVolumenesNivel_1, double *d_eta1Inicial,
				double *d_deformacionInterpoladaNivel0, double *d_deformacionAcumuladaNivel, int inixSubmalla,
				int iniySubmalla, int nvxSubmalla, int nvySubmalla, int inixDef, int iniyDef, int nvxDef,
				int nvyDef, int inixSubmallaCluster, int iniySubmallaCluster, int ratio_ref, double *vccos)
{
	int pos_x_hebra, pos_y_hebra;
	int pos2, pos_datos;
	int posxNivel0, posyNivel0;
	int restox, restoy;
	int posx_izq, posx_der;
	int posy_sup, posy_inf;
	int j_global;
	double distx, disty;
	double factor, U_Z;
	// inixSubmalla e iniySubmalla se indican con respecto a la malla del nivel 0
	// y están en la resolución de la malla fina donde se aplica la deformación

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < nvxSubmalla+4) && (pos_y_hebra < nvySubmalla+4)) {
		// Aplicamos la deformación también en las celdas fantasma (si la deformación está definida para ellas)
		pos_datos = pos_y_hebra*(nvxSubmalla + 4) + pos_x_hebra;
		// Restamos 2 por la indexación debido a la inclusión de las celdas fantasma
		j_global = iniySubmallaCluster + pos_y_hebra-2;
		posxNivel0 = (inixSubmalla+inixSubmallaCluster+pos_x_hebra-2)/ratio_ref;
		posyNivel0 = (iniySubmalla+iniySubmallaCluster+pos_y_hebra-2)/ratio_ref;

		if ((posxNivel0 >= inixDef) && (posxNivel0 < inixDef+nvxDef) && (posyNivel0 >= iniyDef) && (posyNivel0 < iniyDef+nvyDef)) {
			// El punto está dentro de la ventana de computación de Okada
			restox = ((inixSubmalla+inixSubmallaCluster+pos_x_hebra-2)&(ratio_ref-1));
			restoy = ((iniySubmalla+iniySubmallaCluster+pos_y_hebra-2)&(ratio_ref-1));
			factor = 1.0/ratio_ref;
			distx = (restox + 0.5)*factor;
			disty = (restoy + 0.5)*factor;
			if (restox+0.5 < 0.5*ratio_ref) {
				posx_der = posxNivel0-inixDef;
				posx_izq = max(posx_der-1,0);
				distx += 0.5;
			}
			else {
				posx_izq = posxNivel0-inixDef;
				posx_der = min(posx_izq+1,nvxDef-1);
				distx -= 0.5;
			}
			if (restoy+0.5 < 0.5*ratio_ref) {
				posy_sup = posyNivel0-iniyDef;
				posy_inf = max(posy_sup-1,0);
				disty += 0.5;
			}
			else {
				posy_inf = posyNivel0-iniyDef;
				posy_sup = min(posy_inf+1,nvyDef-1);
				disty -= 0.5;
			}

			// interpolacionBilineal está definida en Deformacion.cu
			U_Z = interpolacionBilineal(d_deformacionInterpoladaNivel0, nvxDef, posx_izq,
					posx_der, posy_inf, posy_sup, distx, disty);
			U_Z *= vccos[j_global];
			d_datosVolumenesNivel_1[pos_datos].y -= U_Z;
			if ((pos_x_hebra > 1) && (pos_x_hebra < nvxSubmalla+2) && (pos_y_hebra > 1) && (pos_y_hebra < nvySubmalla+2)) {
				// Solo aplicamos la deformación en d_deformacionAcumuladaNivel y en d_eta1Inicial
				// si es una celda interna de la submalla (no fantasma)
				// Aplicamos la deformación también a d_eta1Inicial (para los tiempos de llegada
				// y la activación de las mallas anidadas)
				pos2 = (pos_y_hebra-2)*nvxSubmalla + pos_x_hebra-2;
				d_eta1Inicial[pos2] += U_Z;
				if (l < numNiveles-1)
					d_deformacionAcumuladaNivel[pos2] -= U_Z;
			}
		}
	}
}


/***********************/
/* Aplicar deformación */
/***********************/

// Aplica un estado de la deformación dinámica. Si kajiura_flag=1, aplica el filtro de Kajiura al resultado obtenido
void aplicarDefDinamica(int numNiveles, int indiceFallaSig, int kajiura_flag, double depth_kajiura,
		TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, int4 submallaDeformacion,
		double2 **d_datosVolumenesNivel_1, double **d_eta1Inicial, double **d_deformacionNivel0,
		double **d_deformacionAcumuladaNivel, double *d_deformacionInterpoladaNivel0, double *defTime,
		double tiempoAnt, double tiempoAct, cuDoubleComplex *d_datosKajiura, double *d_F2Sx, double *d_F2Sy,
		int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel, dim3 blockGridOkada,
		dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 blockGridFanNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		dim3 threadBlockEst, cudaStream_t *streams, int nstreams)
{
	int i, j, k, l;
	int nvx, nvy;
	int nvxDef, nvyDef;
	int inix, iniy;
	int submallaSup;
	int pos, ratio;
	double *d_defSigNivel0;
	double defTime_ant, defTime_sig;
	int indiceFallaAnt = max(0, indiceFallaSig-1);

	nvxDef = submallaDeformacion.z;
	nvyDef = submallaDeformacion.w;
	
	// La deformación a ponderar en tiempo está en d_deformacionNivel0[indiceFallaSig].
	// Ponemos el resultado en d_deformacionInterpoladaNivel0 (usamos d_deltaTVolumenesNivel)
	d_defSigNivel0 = d_deformacionNivel0[indiceFallaSig];
	defTime_ant = defTime[indiceFallaAnt];
	defTime_sig = defTime[indiceFallaSig];
	interpolarDefEnTiempoNivel0GPU<<<blockGridOkada, threadBlockEst>>>(d_defSigNivel0, defTime_ant,
		defTime_sig, tiempoAnt, tiempoAct, d_deformacionInterpoladaNivel0, nvxDef, nvyDef);

	if (kajiura_flag == 1) {
		aplicarKajiura(d_deformacionInterpoladaNivel0, d_datosKajiura, d_F2Sx, d_F2Sy, nvxDef, nvyDef,
			depth_kajiura, blockGridOkada, threadBlockEst);
	}

	// sumarDeformacionADatosGPU está definida en Deformacion.cu
	sumarDeformacionADatosGPU<<<blockGridEstNivel[0][0], threadBlockEst>>>(numNiveles, d_datosVolumenesNivel_1[0],
		d_eta1Inicial[0], d_deformacionInterpoladaNivel0, d_deformacionAcumuladaNivel[0], datosClusterCPU[0][0].numVolx,
		datosClusterCPU[0][0].numVoly, submallaDeformacion.x, submallaDeformacion.y, nvxDef, nvyDef,
		datosClusterCPU[0][0].inix, datosClusterCPU[0][0].iniy, d_datosNivel[0][0].vccos);
	// Interpolamos d_deformacionInterpoladaNivel0 en el resto de niveles
	for (l=1; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			j = datosClusterCPU[l][i].iniy;
			if (j != -1) {
				// Ponemos en ratio el ratio acumulado, y en inix e iniy las coordenadas de inicio de
				// la submalla con respecto a la malla del nivel 0 y en la resolución de la malla fina
				inix = submallasNivel[l][i].x;
				iniy = submallasNivel[l][i].y;
				ratio = ratioRefNivel[l];
				submallaSup = submallaNivelSuperior[l][i];
				for (k=l-1; k>=0; k--) {
					inix += ratio*submallasNivel[k][submallaSup].x;
					iniy += ratio*submallasNivel[k][submallaSup].y;
					ratio *= ratioRefNivel[k];
					submallaSup = submallaNivelSuperior[k][submallaSup];
				}
				// Interpolamos
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				interpolarDefDinamicaGPU<<<blockGridFanNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					l, numNiveles, d_datosVolumenesNivel_1[l]+pos, d_eta1Inicial[l]+pos, d_deformacionInterpoladaNivel0,
					d_deformacionAcumuladaNivel[l]+pos, inix, iniy, nvx, nvy, submallaDeformacion.x, submallaDeformacion.y,
					nvxDef, nvyDef, datosClusterCPU[l][i].inix, j, ratio, d_datosNivel[l][i].vccos);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

#endif
