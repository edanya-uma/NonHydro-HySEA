#ifndef _NESTED_MESHES_DIF_H_
#define _NESTED_MESHES_DIF_H_

#include "NestedMeshesCommon.cu"

/*********************************/
/* Interpolación con diferencias */
/*********************************/

__device__ void obtenerDatosEInterpolarEspesoresDif(double2 *datosVolumenesNivel0_1, double2 *datosVolumenesNivel0Sig_1,
				double2 datosNivel1_1, int numVolxNivel0, int xi, int yi, double peso1, double2 *res1)
{
	double2 datos1, datos_sig1;
	int pos;

	pos = (yi + 2)*(numVolxNivel0 + 4) + (xi + 2);
	datos1 = datosVolumenesNivel0_1[pos];
	datos_sig1 = datosVolumenesNivel0Sig_1[pos];
	*res1 = datosNivel1_1;
	res1->x -= res1->y;
	// eta_{fina}^n = eta_{fina}^{n-1} + peso1*(eta_{gruesa}^n - eta_{gruesa}^{n-1})
	res1->x += peso1*(datos_sig1.x - datos1.x);
	res1->x = max(res1->x + res1->y, 0.0);  // h = eta+H

	if (res1->x < EPSILON) {
		// No hay agua
		res1->x = 0.0;
	}
}

__device__ void obtenerDatosEInterpolarCaudalesDif(double2 *datosVolumenesNivel0_2, double2 *datosVolumenesNivel0Sig_2,
				double2 datosNivel1_1, double2 datosNivel1_2, int numVolxNivel0, int xi, int yi, double peso1,
				double2 *res2)
{
	double2 datos2, datos_sig2;
	int pos;

	pos = (yi + 2)*(numVolxNivel0 + 4) + (xi + 2);
	datos2 = datosVolumenesNivel0_2[pos];
	datos_sig2 = datosVolumenesNivel0Sig_2[pos];

	if (datosNivel1_1.x == 0.0) {
		// No hay agua
		res2->x = res2->y = 0.0;
	}
	else {
		// Hay agua
		res2->x = datosNivel1_2.x + peso1*(datos_sig2.x - datos2.x);
		res2->y = datosNivel1_2.y + peso1*(datos_sig2.y - datos2.y);
	}
}

__global__ void obtenerSiguientesEspesoresVolumenesFantasmaDiferenciasNivel1GPU(double2 *datosVolumenesNivel0_1,
				double2 *datosVolumenesNivel0Sig_1, double2 *datosVolumenesNivel1_1, int numVolxTotalNivel0,
				int numVolyTotalNivel0, int inix, int iniy, int numVolxSubmallaNivel1, int numVolySubmallaNivel1,
				int *posCopiaNivel1, double2 *datosCopiaNivel1_1, double peso1, int ratio_ref, double borde_sup,
				double borde_inf, double borde_izq, double borde_der, int inixSubmallaSupCluster,
				int iniySubmallaSupCluster, int numVolxSubmallaSupCluster, int inixSubmallaCluster,
				int iniySubmallaCluster, bool ultima_hebraX_submalla, bool ultima_hebraY_submalla)
{
	int posNivel1;
	int pos_ghost, pos2;
	int xi, yi;
	int finx, finy;
	double2 datosNivel1_1;
	double2 res1;
	int pos_x_hebra, pos_y_hebra;
	bool procesada = false;
	// peso1 es el peso que se aplica a la diferencia entre el siguiente estado y el actual (entre 0 y 1)

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// datosVolumenesNivel0 contiene los datos del estado actual del nivel 0, mientras que
	// datosVolumenesNivel0Sig contiene los datos del siguiente estado del nivel 0 (para
	// poder interpolar en tiempo). datosVolumenesNivel1 sólo contiene los datos de la
	// submalla actual. datosCopiaNivel1 contiene los datos de todas las submallas del
	// nivel 1, y posCopiaNivel1 indica la posición de datosCopiaNivel1 desde donde hay que
	// copiar los datos del volumen fastasma que trate la hebra (-1 si hay que interpolar).

	if ((pos_y_hebra > 1) && (pos_y_hebra < numVolySubmallaNivel1+2)) {
		posNivel1 = pos_y_hebra*(numVolxSubmallaNivel1+4) + pos_x_hebra;
		finx = inix + inixSubmallaCluster + numVolxSubmallaNivel1 - 1;
		if ((pos_x_hebra < 2) && (inixSubmallaCluster == 0)) {
			// Celda fantasma izquierda (de las dos primeras columnas)
			procesada = true;
			pos_ghost = inix - 1;
			if (pos_ghost > -1) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res1 = datosCopiaNivel1_1[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;  // (se le resta 2 por las 2 primeras celdas fantasma en y)
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarEspesoresDif(datosVolumenesNivel0_1, datosVolumenesNivel0Sig_1,
						datosNivel1_1, numVolxSubmallaSupCluster, xi, yi, peso1, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_x_hebra == 0) ? 2 : 1);
				res1 = datosVolumenesNivel1_1[posNivel1 + pos2];
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
		}
		else if (((pos_x_hebra == numVolxSubmallaNivel1+2) || (pos_x_hebra == numVolxSubmallaNivel1+3)) && ultima_hebraX_submalla) {
			// Celda fantasma derecha (de las dos últimas columnas)
			procesada = true;
			pos_ghost = finx + 1;
			if (pos_ghost < numVolxTotalNivel0*ratio_ref) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res1 = datosCopiaNivel1_1[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarEspesoresDif(datosVolumenesNivel0_1, datosVolumenesNivel0Sig_1,
						datosNivel1_1, numVolxSubmallaSupCluster, xi, yi, peso1, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_x_hebra - numVolxSubmallaNivel1 - 1;
				res1 = datosVolumenesNivel1_1[posNivel1 - pos2];
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
		}
	}
	if ((! procesada) && (pos_x_hebra > 1) && (pos_x_hebra < numVolxSubmallaNivel1+2)) {
		posNivel1 = pos_y_hebra*(numVolxSubmallaNivel1+4) + pos_x_hebra;
		finy = iniy + iniySubmallaCluster + numVolySubmallaNivel1 - 1;
		if ((pos_y_hebra < 2) && (iniySubmallaCluster == 0)) {
			// Celda fantasma superior (de las dos primeras filas)
			pos_ghost = iniy - 1;
			if (pos_ghost > -1) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res1 = datosCopiaNivel1_1[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarEspesoresDif(datosVolumenesNivel0_1, datosVolumenesNivel0Sig_1,
						datosNivel1_1, numVolxSubmallaSupCluster, xi, yi, peso1, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_y_hebra == 0) ? 2 : 1);
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res1 = datosVolumenesNivel1_1[posNivel1 + pos2];
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
		}
		else if (((pos_y_hebra == numVolySubmallaNivel1+2) || (pos_y_hebra == numVolySubmallaNivel1+3)) && ultima_hebraY_submalla) {
			// Celda fantasma inferior (de las dos últimas filas)
			pos_ghost = finy + 1;
			if (pos_ghost < numVolyTotalNivel0*ratio_ref) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res1 = datosCopiaNivel1_1[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarEspesoresDif(datosVolumenesNivel0_1, datosVolumenesNivel0Sig_1,
						datosNivel1_1, numVolxSubmallaSupCluster, xi, yi, peso1, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_y_hebra - numVolySubmallaNivel1 - 1;
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res1 = datosVolumenesNivel1_1[posNivel1 - pos2];
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
		}
	}
}

__global__ void obtenerSiguientesCaudalesVolumenesFantasmaDiferenciasNivel1GPU(double2 *datosVolumenesNivel0_2,
				double2 *datosVolumenesNivel0Sig_2, double2 *datosVolumenesNivel1_1, double2 *datosVolumenesNivel1_2,
				int numVolxTotalNivel0, int numVolyTotalNivel0, int inix, int iniy, int numVolxSubmallaNivel1,
				int numVolySubmallaNivel1, int *posCopiaNivel1, double2 *datosCopiaNivel1_2, double peso1,
				int ratio_ref, double borde_sup, double borde_inf, double borde_izq, double borde_der,
				int inixSubmallaSupCluster, int iniySubmallaSupCluster, int numVolxSubmallaSupCluster,
				int inixSubmallaCluster, int iniySubmallaCluster, bool ultima_hebraX_submalla, bool ultima_hebraY_submalla)
{
	int posNivel1;
	int pos_ghost, pos2;
	int xi, yi;
	int finx, finy;
	double2 datosNivel1_1, datosNivel1_2;
	double2 res2;
	int pos_x_hebra, pos_y_hebra;
	bool procesada = false;
	// peso1 es el peso que se aplica a la diferencia entre el siguiente estado y el actual (entre 0 y 1)

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// datosVolumenesNivel0 contiene los datos del estado actual del nivel 0, mientras que
	// datosVolumenesNivel0Sig contiene los datos del siguiente estado del nivel 0 (para
	// poder interpolar en tiempo). datosVolumenesNivel1 sólo contiene los datos de la
	// submalla actual. datosCopiaNivel1 contiene los datos de todas las submallas del
	// nivel 1, y posCopiaNivel1 indica la posición de datosCopiaNivel1 desde donde hay que
	// copiar los datos del volumen fastasma que trate la hebra (-1 si hay que interpolar).

	if ((pos_y_hebra > 1) && (pos_y_hebra < numVolySubmallaNivel1+2)) {
		posNivel1 = pos_y_hebra*(numVolxSubmallaNivel1+4) + pos_x_hebra;
		finx = inix + inixSubmallaCluster + numVolxSubmallaNivel1 - 1;
		if ((pos_x_hebra < 2) && (inixSubmallaCluster == 0)) {
			// Celda fantasma izquierda (de las dos primeras columnas)
			procesada = true;
			pos_ghost = inix - 1;
			if (pos_ghost > -1) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					datosNivel1_2 = datosVolumenesNivel1_2[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;  // (se le resta 2 por las 2 primeras celdas fantasma en y)
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarCaudalesDif(datosVolumenesNivel0_2, datosVolumenesNivel0Sig_2,
						datosNivel1_1, datosNivel1_2, numVolxSubmallaSupCluster, xi, yi, peso1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_x_hebra == 0) ? 2 : 1);
				res2 = datosVolumenesNivel1_2[posNivel1 + pos2];
				res2.x *= borde_izq;
			}
			datosVolumenesNivel1_2[posNivel1] = res2;
		}
		else if (((pos_x_hebra == numVolxSubmallaNivel1+2) || (pos_x_hebra == numVolxSubmallaNivel1+3)) && ultima_hebraX_submalla) {
			// Celda fantasma derecha (de las dos últimas columnas)
			procesada = true;
			pos_ghost = finx + 1;
			if (pos_ghost < numVolxTotalNivel0*ratio_ref) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					datosNivel1_2 = datosVolumenesNivel1_2[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarCaudalesDif(datosVolumenesNivel0_2, datosVolumenesNivel0Sig_2,
						datosNivel1_1, datosNivel1_2, numVolxSubmallaSupCluster, xi, yi, peso1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_x_hebra - numVolxSubmallaNivel1 - 1;
				res2 = datosVolumenesNivel1_2[posNivel1 - pos2];
				res2.x *= borde_der;
			}
			datosVolumenesNivel1_2[posNivel1] = res2;
		}
	}
	if ((! procesada) && (pos_x_hebra > 1) && (pos_x_hebra < numVolxSubmallaNivel1+2)) {
		posNivel1 = pos_y_hebra*(numVolxSubmallaNivel1+4) + pos_x_hebra;
		finy = iniy + iniySubmallaCluster + numVolySubmallaNivel1 - 1;
		if ((pos_y_hebra < 2) && (iniySubmallaCluster == 0)) {
			// Celda fantasma superior (de las dos primeras filas)
			pos_ghost = iniy - 1;
			if (pos_ghost > -1) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					datosNivel1_2 = datosVolumenesNivel1_2[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarCaudalesDif(datosVolumenesNivel0_2, datosVolumenesNivel0Sig_2,
						datosNivel1_1, datosNivel1_2, numVolxSubmallaSupCluster, xi, yi, peso1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_y_hebra == 0) ? 2 : 1);
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res2 = datosVolumenesNivel1_2[posNivel1 + pos2];
				res2.y *= borde_sup;
			}
			datosVolumenesNivel1_2[posNivel1] = res2;
		}
		else if (((pos_y_hebra == numVolySubmallaNivel1+2) || (pos_y_hebra == numVolySubmallaNivel1+3)) && ultima_hebraY_submalla) {
			// Celda fantasma inferior (de las dos últimas filas)
			pos_ghost = finy + 1;
			if (pos_ghost < numVolyTotalNivel0*ratio_ref) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					datosNivel1_1 = datosVolumenesNivel1_1[posNivel1];
					datosNivel1_2 = datosVolumenesNivel1_2[posNivel1];
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					obtenerDatosEInterpolarCaudalesDif(datosVolumenesNivel0_2, datosVolumenesNivel0Sig_2,
						datosNivel1_1, datosNivel1_2, numVolxSubmallaSupCluster, xi, yi, peso1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_y_hebra - numVolySubmallaNivel1 - 1;
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res2 = datosVolumenesNivel1_2[posNivel1 - pos2];
				res2.y *= borde_inf;
			}
			datosVolumenesNivel1_2[posNivel1] = res2;
		}
	}
}

/**************************************************************/
/* Corrección de flujos de celdas fantasma del nivel inferior */
/**************************************************************/

__global__ void corregirFlujosFantasmaNivel1GPU(double2 *d_datosVolumenesNivel1_1, double2 *d_datosVolumenesNivel1_2,
				double *d_correccionEtaNivel0, double2 *d_correccionNivel0_2, int numVolxNivel0, int numVolxTotalNivel0,
				int numVolyTotalNivel0, int inixNivel1, int iniyNivel1, int numVolxSubmallaNivel1, int numVolySubmallaNivel1,
				int ratio_ref, int inixSubmallaSupCluster, int iniySubmallaSupCluster, int inixSubmallaCluster, 
				int iniySubmallaCluster, bool ultima_hebraX_submalla, bool ultima_hebraY_submalla)
{
	int pos_corr, pos_ghost;
	int posNivel1;
	int finx, finy;
	double2 W, corr;
	double corr_eta;
	double alfa = 1.0;
	int pos_x_hebra, pos_y_hebra;
	bool aplicarCorreccion = false;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_y_hebra > 1) && (pos_y_hebra < numVolySubmallaNivel1+2)) {
		finx = inixNivel1 + inixSubmallaCluster + numVolxSubmallaNivel1 - 1;
		if ((pos_x_hebra < 2) && (inixSubmallaCluster == 0)) {
			// Celda fantasma izquierda (de las dos primeras columnas)
			pos_ghost = inixNivel1 - 1;
			if (pos_ghost > -1)
				aplicarCorreccion = true;
		}
		else if (((pos_x_hebra == numVolxSubmallaNivel1+2) || (pos_x_hebra == numVolxSubmallaNivel1+3)) && ultima_hebraX_submalla) {
			// Celda fantasma derecha (de las dos últimas columnas)
			pos_ghost = finx + 1;
			if (pos_ghost < numVolxTotalNivel0*ratio_ref)
				aplicarCorreccion = true;
		}
	}
	else if ((pos_x_hebra > 1) && (pos_x_hebra < numVolxSubmallaNivel1+2)) {
		finy = iniyNivel1 + iniySubmallaCluster + numVolySubmallaNivel1 - 1;
		if ((pos_y_hebra < 2) && (iniySubmallaCluster == 0)) {
			// Celda fantasma superior (de las dos primeras filas)
			pos_ghost = iniyNivel1 - 1;
			if (pos_ghost > -1)
				aplicarCorreccion = true;
		}
		else if (((pos_y_hebra == numVolySubmallaNivel1+2) || (pos_y_hebra == numVolySubmallaNivel1+3)) && ultima_hebraY_submalla) {
			// Celda fantasma inferior (de las dos últimas filas)
			pos_ghost = finy + 1;
			if (pos_ghost < numVolyTotalNivel0*ratio_ref)
				aplicarCorreccion = true;
		}
	}

	if (aplicarCorreccion) {
		// Aplicamos la corrección que se aplicó a la celda gruesa asociada del nivel superior
		posNivel1 = pos_y_hebra*(numVolxSubmallaNivel1+4) + pos_x_hebra;
		pos_corr = (iniyNivel1 + iniySubmallaCluster + pos_y_hebra - 2)/ratio_ref;
		pos_corr = (pos_corr - iniySubmallaSupCluster)*numVolxNivel0 + ((inixNivel1 + inixSubmallaCluster + pos_x_hebra-2)/ratio_ref - inixSubmallaSupCluster);
		W = d_datosVolumenesNivel1_1[posNivel1];
		corr_eta = d_correccionEtaNivel0[pos_corr];

		if (corr_eta > W.x) {
			alfa = W.x / (corr_eta + EPSILON);
		}
		W.x += alfa*corr_eta;
		if (W.x < 0.0)  W.x = 0.0;
		d_datosVolumenesNivel1_1[posNivel1] = W;

		W = d_datosVolumenesNivel1_2[posNivel1];
		corr = d_correccionNivel0_2[pos_corr];
		W.x += alfa*corr.x;
		W.y += alfa*corr.y;
		d_datosVolumenesNivel1_2[posNivel1] = W;
	}
}

/***************************************/
/* Proyección con diferencia de medias */
/***************************************/

__global__ void obtenerMediaSubmallaNivel1GPU(double *d_mediaEtaNivel1, double2 *d_mediaNivel1_2,
				double2 *d_datosVolumenesNivel1_1, double2 *d_datosVolumenesNivel1_2, int numVolxNivel1,
				int numVolyNivel1, int ratio_ref, double factor_correccion)
{
	int i, j, k, l;
	int posMedia, posNivel1;
	double h, datos_eta;
	double2 datos2;
	int pos_x_hebra, pos_y_hebra;
	int finsx, finsy;
	bool hay_celda_seca = false;

	__shared__ double2 s_datosNivel1_1[NUM_HEBRAS_ANCHO_EST][NUM_HEBRAS_ALTO_EST];
	__shared__ double2 s_datosNivel1_2[NUM_HEBRAS_ANCHO_EST][NUM_HEBRAS_ALTO_EST];

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// Comprobamos si la hebra está dentro de los límites de la malla
	if ((pos_x_hebra < numVolxNivel1) && (pos_y_hebra < numVolyNivel1)) {
		posNivel1 = (pos_y_hebra+2)*(numVolxNivel1+4) + pos_x_hebra+2;
		i = pos_x_hebra&(NUM_HEBRAS_ANCHO_EST-1);  // pos_x_hebra%NUM_HEBRAS_ANCHO_EST
		j = pos_y_hebra&(NUM_HEBRAS_ALTO_EST-1);   // pos_y_hebra%NUM_HEBRAS_ALTO_EST
		s_datosNivel1_1[i][j] = d_datosVolumenesNivel1_1[posNivel1];
		s_datosNivel1_2[i][j] = d_datosVolumenesNivel1_2[posNivel1];
	}
	__syncthreads();

	if ((pos_x_hebra < numVolxNivel1) && (pos_y_hebra < numVolyNivel1)) {
		if (((i&(ratio_ref-1)) == 0) && ((j&(ratio_ref-1)) == 0)) {
			// La hebra correspondiente a la esquina superior izquierda de la celda gruesa
			// asociada obtiene la media de las celdas finas
			posMedia = (pos_y_hebra/ratio_ref)*(numVolxNivel1/ratio_ref) + pos_x_hebra/ratio_ref;
			datos_eta = 0.0;
			datos2.x = datos2.y = 0.0;
			finsx = i + ratio_ref;
			finsy = j + ratio_ref;
			for (l=i; l<finsx; l++) {
				for (k=j; k<finsy; k++) {
					h = s_datosNivel1_1[l][k].x;
					if (h < EPSILON)
						hay_celda_seca = true;
					datos_eta += h - s_datosNivel1_1[l][k].y;
					datos2.x += s_datosNivel1_2[l][k].x;
					datos2.y += s_datosNivel1_2[l][k].y;
				}
			}
			datos_eta *= factor_correccion;
			datos2.x *= factor_correccion;
			datos2.y *= factor_correccion;
			// Escribimos la media si todas las celdas finas están mojadas
			if (hay_celda_seca)
				datos_eta = DBL_MAX;
			d_mediaEtaNivel1[posMedia] = datos_eta;
			d_mediaNivel1_2[posMedia] = datos2;
		}
	}
}

__global__ void obtenerMediaEtaSubmallaNivel1GPU(double *d_mediaEtaNivel1, double2 *d_datosVolumenesNivel1_1,
				int numVolxNivel1, int numVolyNivel1, int ratio_ref, double factor_correccion)
{
	int i, j, k, l;
	int posMedia, posNivel1;
	double h, datos_eta;
	int pos_x_hebra, pos_y_hebra;
	int finsx, finsy;
	bool hay_celda_seca = false;

	__shared__ double2 s_datosNivel1_1[NUM_HEBRAS_ANCHO_EST][NUM_HEBRAS_ALTO_EST];

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// Comprobamos si la hebra está dentro de los límites de la malla
	if ((pos_x_hebra < numVolxNivel1) && (pos_y_hebra < numVolyNivel1)) {
		posNivel1 = (pos_y_hebra+2)*(numVolxNivel1+4) + pos_x_hebra+2;
		i = pos_x_hebra&(NUM_HEBRAS_ANCHO_EST-1);  // pos_x_hebra%NUM_HEBRAS_ANCHO_EST
		j = pos_y_hebra&(NUM_HEBRAS_ALTO_EST-1);   // pos_y_hebra%NUM_HEBRAS_ALTO_EST
		s_datosNivel1_1[i][j] = d_datosVolumenesNivel1_1[posNivel1];
	}
	__syncthreads();

	if ((pos_x_hebra < numVolxNivel1) && (pos_y_hebra < numVolyNivel1)) {
		if (((i&(ratio_ref-1)) == 0) && ((j&(ratio_ref-1)) == 0)) {
			// La hebra correspondiente a la esquina superior izquierda de la celda gruesa
			// asociada obtiene la media de las celdas finas
			posMedia = (pos_y_hebra/ratio_ref)*(numVolxNivel1/ratio_ref) + pos_x_hebra/ratio_ref;
			datos_eta = 0.0;
			finsx = i + ratio_ref;
			finsy = j + ratio_ref;
			for (l=i; l<finsx; l++) {
				for (k=j; k<finsy; k++) {
					h = s_datosNivel1_1[l][k].x;
					if (h < EPSILON)
						hay_celda_seca = true;
					datos_eta += h - s_datosNivel1_1[l][k].y;
				}
			}
			datos_eta *= factor_correccion;
			// Escribimos la media si todas las celdas finas están mojadas
			if (hay_celda_seca)
				datos_eta = DBL_MAX;
			d_mediaEtaNivel1[posMedia] = datos_eta;
		}
	}
}

__global__ void corregirSolucionSubmallaDiferenciasNivel1GPU(double2 *d_datosVolumenesNivel0_1, double2 *d_datosVolumenesNivel0_2,
				double2 *d_datosVolumenesNivel0Sig_1, double2 *d_datosVolumenesNivel0Sig_2, double *d_mediaEtaNivel1,
				double2 *d_mediaNivel1_2, double2 *d_datosVolumenesNivel1_1, double2 *d_datosVolumenesNivel1_2,
				int inix, int iniy, int numVolxNivel1, int numVolyNivel1, int numVolxNivel0, int ratio_ref,
				double factor_correccion, int inixSubmallaSupCluster, int iniySubmallaSupCluster,
				int inixSubmallaCluster, int iniySubmallaCluster)
{
	int i, j, k, l;
	int xi, yi;
	int posMedia, posNivel0, posNivel1;
	double2 datosNivel0_1, datosNivel0_2;
	double h, datosMedia_eta, datosMediaAnt_eta;
	double2 datosMedia_2, datosMediaAnt_2;
	int pos_x_hebra, pos_y_hebra;
	int finsx, finsy;
	bool hay_celda_seca = false;

	__shared__ double2 s_datosNivel1_1[NUM_HEBRAS_ANCHO_EST][NUM_HEBRAS_ALTO_EST];
	__shared__ double2 s_datosNivel1_2[NUM_HEBRAS_ANCHO_EST][NUM_HEBRAS_ALTO_EST];

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// Comprobamos si la hebra está dentro de los límites de la malla
	if ((pos_x_hebra < numVolxNivel1) && (pos_y_hebra < numVolyNivel1)) {
		posNivel1 = (pos_y_hebra+2)*(numVolxNivel1+4) + pos_x_hebra+2;
		i = pos_x_hebra&(NUM_HEBRAS_ANCHO_EST-1);  // pos_x_hebra%NUM_HEBRAS_ANCHO_EST
		j = pos_y_hebra&(NUM_HEBRAS_ALTO_EST-1);   // pos_y_hebra%NUM_HEBRAS_ALTO_EST
		s_datosNivel1_1[i][j] = d_datosVolumenesNivel1_1[posNivel1];
		s_datosNivel1_2[i][j] = d_datosVolumenesNivel1_2[posNivel1];
	}
	__syncthreads();

	if ((pos_x_hebra < numVolxNivel1) && (pos_y_hebra < numVolyNivel1)) {
		if (((i&(ratio_ref-1)) == 0) && ((j&(ratio_ref-1)) == 0)) {
			// La hebra correspondiente a la esquina superior izquierda de la celda gruesa
			// asociada obtiene la media de las celdas finas
			datosMedia_eta = 0.0;
			datosMedia_2.x = datosMedia_2.y = 0.0;
			finsx = i + ratio_ref;
			finsy = j + ratio_ref;
			for (l=i; l<finsx; l++) {
				for (k=j; k<finsy; k++) {
					h = s_datosNivel1_1[l][k].x;
					if (h < EPSILON)
						hay_celda_seca = true;
					datosMedia_eta += h - s_datosNivel1_1[l][k].y;
					datosMedia_2.x += s_datosNivel1_2[l][k].x;
					datosMedia_2.y += s_datosNivel1_2[l][k].y;
				}
			}
			datosMedia_eta *= factor_correccion;
			datosMedia_2.x *= factor_correccion;
			datosMedia_2.y *= factor_correccion;

			// Actualizamos el estado de la celda gruesa del nivel 0 si todas las celdas finas
			// del estado actual y el anterior estan mojadas
			posMedia = (pos_y_hebra/ratio_ref)*(numVolxNivel1/ratio_ref) + pos_x_hebra/ratio_ref;
			datosMediaAnt_eta = d_mediaEtaNivel1[posMedia];
			datosMediaAnt_2 = d_mediaNivel1_2[posMedia];
			if (hay_celda_seca) {
				// Hay alguna celda seca en el estado actual
				datosMedia_eta = DBL_MAX;
			}
			else if (datosMediaAnt_eta < 1e30) {
				// Todas las celdas finas del estado actual y el anterior están mojadas
				xi = (inix + inixSubmallaCluster + pos_x_hebra)/ratio_ref - inixSubmallaSupCluster;
				yi = (iniy + iniySubmallaCluster + pos_y_hebra)/ratio_ref - iniySubmallaSupCluster;
				posNivel0 = (yi + 2)*(numVolxNivel0 + 4) + (xi + 2);
				datosNivel0_1 = d_datosVolumenesNivel0_1[posNivel0];
				// eta_{gruesa}^n = eta_{gruesa}^{n-1} + (mediaEta_{fina}^n - mediaEta_{fina}^{n-1})
				datosNivel0_1.x = (datosNivel0_1.x - datosNivel0_1.y) + (datosMedia_eta - datosMediaAnt_eta);
				datosNivel0_1.x = max(datosNivel0_1.x + datosNivel0_1.y, 0.0);
				if (datosNivel0_1.x < EPSILON) {
					datosNivel0_1.x = datosNivel0_2.x = datosNivel0_2.y = 0.0;
				}
				else {
					datosNivel0_2 = d_datosVolumenesNivel0_2[posNivel0];
					datosNivel0_2.x += datosMedia_2.x - datosMediaAnt_2.x;
					datosNivel0_2.y += datosMedia_2.y - datosMediaAnt_2.y;
				}
				d_datosVolumenesNivel0Sig_1[posNivel0] = datosNivel0_1;
				d_datosVolumenesNivel0Sig_2[posNivel0] = datosNivel0_2;
			}

			// Actualizamos la media con la media del estado actual
			d_mediaEtaNivel1[posMedia] = datosMedia_eta;
			d_mediaNivel1_2[posMedia] = datosMedia_2;
		}
	}
}

void corregirSolucionDiferenciasNivelGPU(int nivel, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
				TDatosClusterGPU datosClusterGPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 *d_datosVolumenesNivel0_1, double2 *d_datosVolumenesNivel0_2,
				double2 *d_datosVolumenesNivel0Sig_1, double2 *d_datosVolumenesNivel0Sig_2, double *d_mediaEtaNivel1, double2 *d_mediaNivel1_2,
				double *d_correccionEtaNivel0, double2 *d_correccionNivel0_2, double2 *d_datosVolumenesNivel1_1, double2 *d_datosVolumenesNivel1_2,
				int ratioRefNivelInf, double factorCorreccionNivelInf, double epsilon_h, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
				int *numSubmallasNivel, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
				dim3 blockGridFanNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, cudaStream_t *streams, int nstreams,
				bool *ultimaHebraXSubmalla, bool *ultimaHebraYSubmalla)
{
	int i, j, k;
	int posNivel, posNivelInf;
	int nvxNivel, nvyNivel;
	int nvxNivelInf, nvyNivelInf;
	int nvxTotalNivel, nvyTotalNivel;
	int nivelInf = nivel+1;

	// nivel es el nivel en el que se aplica la corrección
	posNivel = 0;
	for (i=0; i<numSubmallasNivel[nivel]; i++) {
		if (datosClusterCPU[nivel][i].iniy != -1) {
			nvxNivel = datosClusterCPU[nivel][i].numVolx;
			nvyNivel = datosClusterCPU[nivel][i].numVoly;
			nvxTotalNivel = submallasNivel[nivel][i].z;
			nvyTotalNivel = submallasNivel[nivel][i].w;
			// Diferencias de las medias para cada celda del nivel
			posNivelInf = k = 0;
			for (j=0; j<numSubmallasNivel[nivelInf]; j++) {
				if (datosClusterCPU[nivelInf][j].iniy != -1) {
					nvxNivelInf = datosClusterCPU[nivelInf][j].numVolx;
					nvyNivelInf = datosClusterCPU[nivelInf][j].numVoly;
					if (submallaNivelSuperior[nivelInf][j] == i) {
						// La submalla j de NivelInf está contenida en la submalla i del nivel
						corregirSolucionSubmallaDiferenciasNivel1GPU<<<blockGridEstNivel[nivelInf][j], threadBlockEst, 0, streams[j&nstreams]>>>(
							d_datosVolumenesNivel0_1+posNivel, d_datosVolumenesNivel0_2+posNivel, d_datosVolumenesNivel0Sig_1+posNivel,
							d_datosVolumenesNivel0Sig_2+posNivel, d_mediaEtaNivel1+k, d_mediaNivel1_2+k, d_datosVolumenesNivel1_1+posNivelInf,
							d_datosVolumenesNivel1_2+posNivelInf, submallasNivel[nivelInf][j].x, submallasNivel[nivelInf][j].y, nvxNivelInf,
							nvyNivelInf, nvxNivel, ratioRefNivelInf, factorCorreccionNivelInf, datosClusterCPU[nivel][i].inix,
							datosClusterCPU[nivel][i].iniy, datosClusterCPU[nivelInf][j].inix, datosClusterCPU[nivelInf][j].iniy);
					}
					posNivelInf += (nvxNivelInf + 4)*(nvyNivelInf + 4);
					k += nvxNivelInf*nvyNivelInf/(ratioRefNivelInf*ratioRefNivelInf);
				}
			}

			// Sincronización necesaria para evitar que corrija flujos sin haber terminado la proyección de la solución fina
			cudaDeviceSynchronize();
			// Corrección debido al ajuste de flujos en las fronteras entre niveles
			corregirFlujosNivelGPU<<<blockGridEstNivel[nivel][i], threadBlockEst>>>(d_datosVolumenesNivel0Sig_1+posNivel,
				d_datosVolumenesNivel0Sig_2+posNivel, d_correccionEtaNivel0+posNivel, d_correccionNivel0_2+posNivel,
				nvxNivel, nvyNivel);

			// Aplicamos la misma corrección a las celdas finas del nivel inferior
			// Aquí no se necesita cudaDeviceSynchronize porque los kernels anterior y actual operan sobre celdas diferentes
			// (interiores y fantasma), no hay conflictos.
			posNivelInf = 0;
			for (j=0; j<numSubmallasNivel[nivelInf]; j++) {
				if (datosClusterCPU[nivelInf][j].iniy != -1) {
					nvxNivelInf = datosClusterCPU[nivelInf][j].numVolx;
					nvyNivelInf = datosClusterCPU[nivelInf][j].numVoly;
					if (submallaNivelSuperior[nivelInf][j] == i) {
						// La submalla j de NivelInf está contenida en la submalla i del nivel
						corregirFlujosFantasmaNivel1GPU<<<blockGridFanNivel[nivelInf][j], threadBlockEst, 0, streams[j&nstreams]>>>(
							d_datosVolumenesNivel1_1+posNivelInf, d_datosVolumenesNivel1_2+posNivelInf, d_correccionEtaNivel0+posNivel,
							d_correccionNivel0_2+posNivel, datosClusterCPU[nivel][i].numVolx, nvxTotalNivel, nvyTotalNivel,
							submallasNivel[nivelInf][j].x, submallasNivel[nivelInf][j].y, nvxNivelInf, nvyNivelInf, ratioRefNivelInf, 
							datosClusterCPU[nivel][i].inix, datosClusterCPU[nivel][i].iniy, datosClusterCPU[nivelInf][j].inix,
							datosClusterCPU[nivelInf][j].iniy, ultimaHebraXSubmalla[j], ultimaHebraYSubmalla[j]);
					}
					posNivelInf += (nvxNivelInf + 4)*(nvyNivelInf + 4);
				}
			}

			posNivel += (nvxNivel + 4)*(nvyNivel + 4);
		}
	}
}

#endif

