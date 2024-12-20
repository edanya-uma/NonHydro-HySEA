#ifndef _FRICCION_VARIABLE_H_
#define _FRICCION_VARIABLE_H_

#include <stdio.h>
#include "Constantes.hxx"

/*****************/
/* Normalización */
/*****************/

__global__ void normalizarFriccionesGPU(double *d_friccionesNivel, int nvx, int nvy, double factor)
{
	int pos_x_hebra, pos_y_hebra, pos;
	double val;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < nvx+4) && (pos_y_hebra < nvy+4)) {
		// También lo aplicamos a las celdas fantasma
		pos = pos_y_hebra*(nvx + 4) + pos_x_hebra;
		val = d_friccionesNivel[pos];
		val *= factor;
		d_friccionesNivel[pos] = val;
	}
}

/*****************/
/* Interpolación */
/*****************/

__device__ double interpolacionBilineal2(double *d_friccionesNivel0, int nvxNivel0, int posx_izq,
					int posx_der, int posy_inf, int posy_sup, double distx, double disty)
{
	// distx es la distancia en x del punto a interpolar con respecto a la posición izquierda
	// disty es la distancia en y del punto a interpolar con respecto a la posición inferior
	// si: sup izq; sd: sup der; ii: inf izq; id: inf der
	int possi, possd, posii, posid;
	double defsi, defsd, defii, defid;
	double val;

	nvxNivel0 += 4;
	possi = (posy_sup+2)*nvxNivel0 + posx_izq+2;
	possd = (posy_sup+2)*nvxNivel0 + posx_der+2;
	posii = (posy_inf+2)*nvxNivel0 + posx_izq+2;
	posid = (posy_inf+2)*nvxNivel0 + posx_der+2;

	defsi = d_friccionesNivel0[possi];
	defsd = d_friccionesNivel0[possd];
	defii = d_friccionesNivel0[posii];
	defid = d_friccionesNivel0[posid];

	val = defii*(1.0-distx)*(1.0-disty) + defid*distx*(1.0-disty) + defsi*(1.0-distx)*disty + defsd*distx*disty;

	return val;
}

__global__ void interpolarFriccionesGPU(double *d_friccionesNivel1, double *d_friccionesNivel0, int inixNivel0,
				int iniyNivel0, int nvxNivel0, int nvyNivel0, int inixSubmalla, int iniySubmalla, int nvxSubmalla,
				int nvySubmalla, int inixSubmallaCluster, int iniySubmallaCluster, int ratio_ref, int id_hebraX,
				int id_hebraY, bool ultimaHebraXNivel0, bool ultimaHebraYNivel0)
{
	int pos_x_hebra, pos_y_hebra;
	int pos_datos;
	int posxNivel0, posyNivel0;
	int restox, restoy;
	int posx_izq, posx_der;
	int posy_sup, posy_inf;
	double distx, disty;
	double factor, val;
	// inixSubmalla e iniySubmalla se indican con respecto a la malla del nivel 0
	// y están en la resolución de la malla fina donde se aplica la deformación

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < nvxSubmalla) && (pos_y_hebra < nvySubmalla)) {
		pos_datos = (pos_y_hebra+2)*(nvxSubmalla + 4) + pos_x_hebra+2;
		posxNivel0 = (inixSubmalla+inixSubmallaCluster+pos_x_hebra)/ratio_ref - inixNivel0;
		posyNivel0 = (iniySubmalla+iniySubmallaCluster+pos_y_hebra)/ratio_ref - iniyNivel0;

		restox = ((inixSubmalla+inixSubmallaCluster+pos_x_hebra)&(ratio_ref-1));
		restoy = ((iniySubmalla+iniySubmallaCluster+pos_y_hebra)&(ratio_ref-1));
		factor = 1.0/ratio_ref;
		distx = (restox + 0.5)*factor;
		disty = (restoy + 0.5)*factor;
		if (restox+0.5 < 0.5*ratio_ref) {
			posx_der = posxNivel0;
			posx_izq = (((posx_der == 0) && (id_hebraX == 0)) ? 0 : posx_der-1);
			distx += 0.5;
		}
		else {
			posx_izq = posxNivel0;
			posx_der = (((posx_izq == nvxNivel0-1) && ultimaHebraXNivel0) ? nvxNivel0-1 : posx_izq+1);
			distx -= 0.5;
		}
		if (restoy+0.5 < 0.5*ratio_ref) {
			posy_sup = posyNivel0;
			posy_inf = (((posy_sup == 0) && (id_hebraY == 0)) ? 0 : posy_sup-1);
			disty += 0.5;
		}
		else {
			posy_inf = posyNivel0;
			posy_sup = (((posy_inf == nvyNivel0-1) && ultimaHebraYNivel0) ? nvyNivel0-1 : posy_inf+1);
			disty -= 0.5;
		}

		val = interpolacionBilineal2(d_friccionesNivel0, nvxNivel0, posx_izq, posx_der,
				posy_inf, posy_sup, distx, disty);
		d_friccionesNivel1[pos_datos] = val;
	}
}

/*************************/
/* Interpolar fricciones */
/*************************/

void normalizarEInterpolarFricciones(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, int tipo_friccion,
		double **d_friccionesNivel, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel,
		dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 blockGridFanNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		dim3 threadBlockEst, double L, double H, cudaStream_t *streams, int nstreams, int id_hebraX, int id_hebraY,
		bool ultimaHebraXNivel0, bool ultimaHebraYNivel0)
{
	int i, j, k, l;
	int nvx, nvy;
	int inixNivel0, iniyNivel0;
	int nvxNivel0, nvyNivel0;
	int inix, iniy;
	int submallaSup;
	int pos, ratio;
	double factor = L/pow(H,4.0/3.0);

	inixNivel0 = datosClusterCPU[0][0].inix;
	iniyNivel0 = datosClusterCPU[0][0].iniy;
	nvxNivel0 = datosClusterCPU[0][0].numVolx;
	nvyNivel0 = datosClusterCPU[0][0].numVoly;

	// Normalizamos las fricciones del nivel 0 (incluyendo los volúmenes fantasma para que la interpolación
	// funcione en los niveles inferiores)
	normalizarFriccionesGPU<<<blockGridFanNivel[0][0], threadBlockEst>>>(d_friccionesNivel[0], nvxNivel0, nvyNivel0, factor);

	for (l=1; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			j = datosClusterCPU[l][i].iniy;
			if (j != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				if (tipo_friccion == VARIABLE_FRICTION_ALL) {
					// Normalizamos las fricciones de los niveles inferiores (se copiaron en ShallowWater.cu)
					normalizarFriccionesGPU<<<blockGridFanNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(d_friccionesNivel[l]+pos,
						nvx, nvy, factor);
				}
				else {
					// Interpolamos las fricciones en los niveles inferiores sin incluir los volúmenes fantasma
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
					interpolarFriccionesGPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
						d_friccionesNivel[l]+pos, d_friccionesNivel[0], inixNivel0, iniyNivel0, nvxNivel0,
						nvyNivel0, inix, iniy, nvx, nvy, datosClusterCPU[l][i].inix, j, ratio, id_hebraX,
						id_hebraY, ultimaHebraXNivel0, ultimaHebraYNivel0);
				}
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}

#endif
