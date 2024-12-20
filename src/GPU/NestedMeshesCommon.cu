#ifndef _NESTED_MESHES_COMMON_H_
#define _NESTED_MESHES_COMMON_H_

#include "Constantes.hxx"
#include <stdlib.h>
#include <stdio.h>

#define EPSILON_AMR 1e-12

/*************************************************/
/* Interpolación de celdas fantasma al principio */
/*************************************************/

__device__ double minmod(double a, double b, double c)
{
	double d = 0.0;

	if ((a > 0) && (b > 0) && (c > 0))
		d = min(a, min(b,c));
	else if ((a < 0) && (b < 0) && (c < 0))
		d = max(a, max(b,c));

	return d;
}

__device__ void interpolacionMUSCL(double2 datos1, double2 datos2, double2 datosx1, double2 datosx2, double2 datosy1,
						double2 datosy2, double distx, double disty, double2 *res1, double2 *res2)
{
	res1->x = datos1.x + distx*datosx1.x + disty*datosy1.x;
	res2->x = datos2.x + distx*datosx2.x + disty*datosy2.x;
	res2->y = datos2.y + distx*datosx2.y + disty*datosy2.y;
	// El fondo H (en res1->y) no se interpola porque ya se interpoló al principio.
}

__device__ void obtenerDatosParaInterpolacion(double2 datos1, double2 datos2, double2 datos_izq1, double2 datos_izq2,
						double2 datos_der1, double2 datos_der2, double2 *datos_int1, double2 *datos_int2)
{
	double Hl, Hr, Hlr;
	double eta0l, eta1l, eta0r, eta1r;
	double etal, etar;
	double eta0lr, eta1lr, etalr;

	// eta
	Hl = min(datos_izq1.y, datos1.y);
	Hr = min(datos1.y, datos_der1.y);
	eta0l = max(datos_izq1.x - datos_izq1.y + Hl, 0.0);
	eta1l = max(datos1.x - datos1.y + Hl, 0.0);
	etal = eta1l - eta0l;
	eta0r = max(datos1.x - datos1.y + Hr, 0.0);
	eta1r = max(datos_der1.x - datos_der1.y + Hr, 0.0);
	etar = eta1r - eta0r;
	if ((datos1.x < EPSILON_AMR) && (datos_der1.x < EPSILON_AMR))
		datos_int1->x = etal;
	else if ((datos1.x < EPSILON_AMR) && (datos_izq1.x < EPSILON_AMR))
		datos_int1->x = etar;
	else {
		Hlr = min(datos_izq1.y, datos_der1.y);
		eta0lr = max(datos_izq1.x - datos_izq1.y + Hlr, 0.0);
		eta1lr = max(datos_der1.x - datos_der1.y + Hlr, 0.0);
		etalr = 0.5*(eta1lr - eta0lr);
		datos_int1->x = minmod(etal, etalr, etar);
	}
	// ux y uy
	if (datos1.x >= EPSILON_AMR) {
		datos2.x /= datos1.x;
		datos2.y /= datos1.x;
	}
	else {
		datos2.x = datos2.y = 0.0;
	}
	if (datos_izq1.x >= EPSILON_AMR) {
		datos_izq2.x /= datos_izq1.x;
		datos_izq2.y /= datos_izq1.x;
	}
	else {
		datos_izq2.x = datos_izq2.y = 0.0;
	}
	if (datos_der1.x >= EPSILON_AMR) {
		datos_der2.x /= datos_der1.x;
		datos_der2.y /= datos_der1.x;
	}
	else {
		datos_der2.x = datos_der2.y = 0.0;
	}
	datos_int2->x = minmod(datos_der2.x-datos2.x, 0.5*(datos_der2.x-datos_izq2.x), datos2.x-datos_izq2.x);
	datos_int2->y = minmod(datos_der2.y-datos2.y, 0.5*(datos_der2.y-datos_izq2.y), datos2.y-datos_izq2.y);
}

__device__ void obtenerDatosCentroParaInterpolacion(double2 datos1, double2 datos2, double2 datos_izq1, double2 datos_izq2,
					double2 datos_der1, double2 datos_der2, double2 datos_sup1, double2 datos_sup2,
					double2 datos_inf1, double2 datos_inf2, double2 *datos_int1, double2 *datos_int2)
{
	double suma_eta;
	int nvecinos_humedos;
	// Si el volumen central está seco y emerge por encima de todos
	// sus vecinos húmedos
	bool emerge;

	if (datos1.x >= EPSILON_AMR) {
		// eta
		datos_int1->x = datos1.x - datos1.y;
		// ux y uy
		datos_int2->x = datos2.x/datos1.x;
		datos_int2->y = datos2.y/datos1.x;
	}
	else {
		// eta
		emerge = true;
		nvecinos_humedos = 0;
		suma_eta = 0.0;
		if (datos_izq1.x >= EPSILON_AMR) {
			if (datos_izq1.x - datos_izq1.y > -datos1.y)
				emerge = false;
			else {
				suma_eta += datos_izq1.x - datos_izq1.y;
				nvecinos_humedos++;
			}
		}
		if (datos_der1.x >= EPSILON_AMR) {
			if (datos_der1.x - datos_der1.y > -datos1.y)
				emerge = false;
			else {
				suma_eta += datos_der1.x - datos_der1.y;
				nvecinos_humedos++;
			}
		}
		if (datos_sup1.x >= EPSILON_AMR) {
			if (datos_sup1.x - datos_sup1.y > -datos1.y)
				emerge = false;
			else {
				suma_eta += datos_sup1.x - datos_sup1.y;
				nvecinos_humedos++;
			}
		}
		if (datos_inf1.x >= EPSILON_AMR) {
			if (datos_inf1.x - datos_inf1.y > -datos1.y)
				emerge = false;
			else {
				suma_eta += datos_inf1.x - datos_inf1.y;
				nvecinos_humedos++;
			}
		}
		datos_int1->x = (emerge ? suma_eta/(float)nvecinos_humedos : datos1.x - datos1.y);
		// ux y uy
		datos_int2->x = datos_int2->y = 0.0;
	}
}

__device__ void obtenerDatosEInterpolarEnTiempoYEspacio(double2 *datosVolumenesNivel0_1, double2 *datosVolumenesNivel0_2,
				double2 *datosVolumenesNivel0Sig_1, double2 *datosVolumenesNivel0Sig_2, int xi, int yi, double distx,
				double disty, double peso1, double prof, int inixSubmallaSupCluster, int iniySubmallaSupCluster,
				int numVolxSubmallaSupCluster, int numVolySubmallaSupCluster, bool ultima_hebraX_nivel0,
				bool ultima_hebraY_nivel0, int nivelSup, double2 *res1, double2 *res2)
{
	double2 datos1, datos2, datos_sig1, datos_sig2;
	double2 datosx1, datosx2, datosy1, datosy2;
	double2 datos_sup1, datos_inf1, datos_izq1, datos_der1;
	double2 datos_sup2, datos_inf2, datos_izq2, datos_der2;
	int xizq, xder, ysup, yinf;
	int pos, pos_izq, pos_der, pos_sup, pos_inf;
	// peso1: peso del estado actual, peso2: peso del siguiente estado
	double peso2 = 1.0-peso1;

	if (nivelSup == 0) {
		// Nivel 0
		if (inixSubmallaSupCluster == 0)
			xizq = max(xi-1, 0) + 2;
		else
			xizq = xi-1 + 2;
		if (ultima_hebraX_nivel0)
			xder = min(xi+1, numVolxSubmallaSupCluster-1) + 2;
		else
			xder = xi+1 + 2;

		if (iniySubmallaSupCluster == 0)
			ysup = max(yi-1, 0) + 2;
		else
			ysup = yi-1 + 2;
		if (ultima_hebraY_nivel0)
			yinf = min(yi+1, numVolySubmallaSupCluster-1) + 2;
		else
			yinf = yi+1 + 2;
	}
	else {
		xizq = xi-1 + 2;
		xder = xi+1 + 2;
		ysup = yi-1 + 2;
		yinf = yi+1 + 2;
	}
	xi += 2;  // offsetX
	yi += 2;  // offsetY
	numVolxSubmallaSupCluster += 4;  // 2*offsetX

	pos = yi*numVolxSubmallaSupCluster + xi;
	pos_izq = yi*numVolxSubmallaSupCluster + xizq;
	pos_der = yi*numVolxSubmallaSupCluster + xder;
	pos_sup = ysup*numVolxSubmallaSupCluster + xi;
	pos_inf = yinf*numVolxSubmallaSupCluster + xi;

	// El fondo H no es necesario interpolarlo en tiempo, ya que no varía
	if (fabs(peso2) < EPSILON) {
		datos1  = datosVolumenesNivel0_1[pos];
		datos2  = datosVolumenesNivel0_2[pos];
		datos1.x = peso1*datos1.x;
		datos2.x = peso1*datos2.x;
		datos2.y = peso1*datos2.y;

		datos_izq1  = datosVolumenesNivel0_1[pos_izq];
		datos_izq2  = datosVolumenesNivel0_2[pos_izq];
		datos_izq1.x = peso1*datos_izq1.x;
		datos_izq2.x = peso1*datos_izq2.x;
		datos_izq2.y = peso1*datos_izq2.y;

		datos_der1  = datosVolumenesNivel0_1[pos_der];
		datos_der2  = datosVolumenesNivel0_2[pos_der];
		datos_der1.x = peso1*datos_der1.x;
		datos_der2.x = peso1*datos_der2.x;
		datos_der2.y = peso1*datos_der2.y;

		datos_sup1 = datosVolumenesNivel0_1[pos_sup];
		datos_sup2 = datosVolumenesNivel0_2[pos_sup];
		datos_sup1.x = peso1*datos_sup1.x;
		datos_sup2.x = peso1*datos_sup2.x;
		datos_sup2.y = peso1*datos_sup2.y;

		datos_inf1 = datosVolumenesNivel0_1[pos_inf];
		datos_inf2 = datosVolumenesNivel0_2[pos_inf];
		datos_inf1.x = peso1*datos_inf1.x;
		datos_inf2.x = peso1*datos_inf2.x;
		datos_inf2.y = peso1*datos_inf2.y;
	}
	else {
		datos1  = datosVolumenesNivel0_1[pos];
		datos2  = datosVolumenesNivel0_2[pos];
		datos_sig1 = datosVolumenesNivel0Sig_1[pos];
		datos_sig2 = datosVolumenesNivel0Sig_2[pos];
		datos1.x = peso1*datos1.x + peso2*datos_sig1.x;
		datos2.x = peso1*datos2.x + peso2*datos_sig2.x;
		datos2.y = peso1*datos2.y + peso2*datos_sig2.y;

		datos_izq1  = datosVolumenesNivel0_1[pos_izq];
		datos_izq2  = datosVolumenesNivel0_2[pos_izq];
		datos_sig1 = datosVolumenesNivel0Sig_1[pos_izq];
		datos_sig2 = datosVolumenesNivel0Sig_2[pos_izq];
		datos_izq1.x = peso1*datos_izq1.x + peso2*datos_sig1.x;
		datos_izq2.x = peso1*datos_izq2.x + peso2*datos_sig2.x;
		datos_izq2.y = peso1*datos_izq2.y + peso2*datos_sig2.y;

		datos_der1  = datosVolumenesNivel0_1[pos_der];
		datos_der2  = datosVolumenesNivel0_2[pos_der];
		datos_sig1 = datosVolumenesNivel0Sig_1[pos_der];
		datos_sig2 = datosVolumenesNivel0Sig_2[pos_der];
		datos_der1.x = peso1*datos_der1.x + peso2*datos_sig1.x;
		datos_der2.x = peso1*datos_der2.x + peso2*datos_sig2.x;
		datos_der2.y = peso1*datos_der2.y + peso2*datos_sig2.y;

		datos_sup1 = datosVolumenesNivel0_1[pos_sup];
		datos_sup2 = datosVolumenesNivel0_2[pos_sup];
		datos_sig1 = datosVolumenesNivel0Sig_1[pos_sup];
		datos_sig2 = datosVolumenesNivel0Sig_2[pos_sup];
		datos_sup1.x = peso1*datos_sup1.x + peso2*datos_sig1.x;
		datos_sup2.x = peso1*datos_sup2.x + peso2*datos_sig2.x;
		datos_sup2.y = peso1*datos_sup2.y + peso2*datos_sig2.y;

		datos_inf1 = datosVolumenesNivel0_1[pos_inf];
		datos_inf2 = datosVolumenesNivel0_2[pos_inf];
		datos_sig1 = datosVolumenesNivel0Sig_1[pos_inf];
		datos_sig2 = datosVolumenesNivel0Sig_2[pos_inf];
		datos_inf1.x = peso1*datos_inf1.x + peso2*datos_sig1.x;
		datos_inf2.x = peso1*datos_inf2.x + peso2*datos_sig2.x;
		datos_inf2.y = peso1*datos_inf2.y + peso2*datos_sig2.y;
	}

	if ((datos1.x < EPSILON_AMR) && (datos_izq1.x < EPSILON_AMR) && (datos_der1.x < EPSILON_AMR) && 
		(datos_sup1.x < EPSILON_AMR) && (datos_inf1.x < EPSILON_AMR)) {
		// No hay agua en ningún volumen
		res1->x = res2->x = res2->y = 0.0;
	}
	else {
		// Hay agua en algún volumen
		obtenerDatosParaInterpolacion(datos1, datos2, datos_izq1, datos_izq2, datos_der1, datos_der2, &datosx1, &datosx2);
		obtenerDatosParaInterpolacion(datos1, datos2, datos_sup1, datos_sup2, datos_inf1, datos_inf2, &datosy1, &datosy2);
		obtenerDatosCentroParaInterpolacion(datos1, datos2, datos_izq1, datos_izq2, datos_der1, datos_der2,
			datos_sup1, datos_sup2, datos_inf1, datos_inf2, &datos_sig1, &datos_sig2);
		interpolacionMUSCL(datos_sig1, datos_sig2, datosx1, datosx2, datosy1, datosy2, distx, disty, res1, res2);
		res1->x = max(res1->x + prof, 0.0);  // h = eta+H
		res2->x *= res1->x;
		res2->y *= res1->x;
	}
	res1->y = prof;
}

__global__ void obtenerSiguientesDatosVolumenesFantasmaSecoMojadoNivel1GPU(double2 *datosVolumenesNivel0_1, double2 *datosVolumenesNivel0_2,
				double2 *datosVolumenesNivel0Sig_1, double2 *datosVolumenesNivel0Sig_2, double2 *datosVolumenesNivel1_1,
				double2 *datosVolumenesNivel1_2, int numVolxTotalNivel0, int numVolyTotalNivel0, int inix, int iniy,
				int numVolxSubmallaNivel1, int numVolySubmallaNivel1, int *posCopiaNivel1, double2 *datosCopiaNivel1_1,
				double2 *datosCopiaNivel1_2, double peso1, int ratio_ref, double borde_sup, double borde_inf, double borde_izq,
				double borde_der, int inixSubmallaSupCluster, int iniySubmallaSupCluster, int numVolxSubmallaSupCluster,
				int numVolySubmallaSupCluster, int inixSubmallaCluster, int iniySubmallaCluster, bool ultima_hebraX_nivel0,
				bool ultima_hebraY_nivel0, bool ultima_hebraX_submalla, bool ultima_hebraY_submalla, int nivelSup)
{
	int posNivel1;
	int pos_ghost, pos2;
	int xi, yi;
	double distx, disty;
	int finx, finy;
	double prof;
	double2 res1, res2;
	int pos_x_hebra, pos_y_hebra;
	bool procesada = false;
	// peso1 es el peso del estado actual (no del siguiente estado)

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// datosVolumenesNivel0 contiene los datos del estado actual del nivel 0, mientras que
	// datosVolumenesNivel0Sig contiene los datos del siguiente estado del nivel 0 (para
	// poder interpolar en tiempo). datosVolumenesNivel1 sólo contiene los datos de la
	// submalla actual. datosCopiaNivel1 contiene los datos de todas las submallas del
	// nivel 1, y posCopiaNivel1 indica la posición de datosCopiaNivel1 desde donde hay que
	// copiar los datos del volumen fastasma que trate la hebra (-1 si hay que interpolar).

	// Interpolamos las celdas internas fantasma que no se salgan del dominio a partir
	// de los datos del nivel 0 aplicando interpolación MUSCL en tiempo y espacio.
	// Si la celda interna fantasma es una celda de otra submalla, copiamos los datos
	// y no interpolamos. También asignamos las celdas fantasma que se salen
	// del dominio. El fondo H no hace falta asignarlo en ningún caso porque
	// ya se asignó al inicializar los datos. Ponemos los datos en datosVolumenesNivel1.

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
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					prof = datosVolumenesNivel1_1[posNivel1].y;
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					obtenerDatosEInterpolarEnTiempoYEspacio(datosVolumenesNivel0_1, datosVolumenesNivel0_2,
						datosVolumenesNivel0Sig_1, datosVolumenesNivel0Sig_2, xi, yi, distx, disty, peso1,
						prof, inixSubmallaSupCluster, iniySubmallaSupCluster, numVolxSubmallaSupCluster,
						numVolySubmallaSupCluster, ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_x_hebra == 0) ? 2 : 1);
				res1 = datosVolumenesNivel1_1[posNivel1 + pos2];
				res2 = datosVolumenesNivel1_2[posNivel1 + pos2];
				res2.x *= borde_izq;
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
			datosVolumenesNivel1_2[posNivel1] = res2;
		}
		else if (((pos_x_hebra == numVolxSubmallaNivel1+2) || (pos_x_hebra == numVolxSubmallaNivel1+3)) && ultima_hebraX_submalla) {
			// Celda fantasma derecha (de las dos últimas columnas)
			procesada = true;
			pos_ghost = finx + 1;
			if (pos_ghost < numVolxTotalNivel0*ratio_ref) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res1 = datosCopiaNivel1_1[pos2];
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					prof = datosVolumenesNivel1_1[posNivel1].y;
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					obtenerDatosEInterpolarEnTiempoYEspacio(datosVolumenesNivel0_1, datosVolumenesNivel0_2,
						datosVolumenesNivel0Sig_1, datosVolumenesNivel0Sig_2, xi, yi, distx, disty, peso1,
						prof, inixSubmallaSupCluster, iniySubmallaSupCluster, numVolxSubmallaSupCluster,
						numVolySubmallaSupCluster, ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_x_hebra - numVolxSubmallaNivel1 - 1;
				res1 = datosVolumenesNivel1_1[posNivel1 - pos2];
				res2 = datosVolumenesNivel1_2[posNivel1 - pos2];
				res2.x *= borde_der;
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
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
					res1 = datosCopiaNivel1_1[pos2];
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					prof = datosVolumenesNivel1_1[posNivel1].y;
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					obtenerDatosEInterpolarEnTiempoYEspacio(datosVolumenesNivel0_1, datosVolumenesNivel0_2,
						datosVolumenesNivel0Sig_1, datosVolumenesNivel0Sig_2, xi, yi, distx, disty, peso1,
						prof, inixSubmallaSupCluster, iniySubmallaSupCluster, numVolxSubmallaSupCluster,
						numVolySubmallaSupCluster, ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_y_hebra == 0) ? 2 : 1);
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res1 = datosVolumenesNivel1_1[posNivel1 + pos2];
				res2 = datosVolumenesNivel1_2[posNivel1 + pos2];
				res2.y *= borde_sup;
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
			datosVolumenesNivel1_2[posNivel1] = res2;
		}
		else if ((pos_y_hebra == numVolySubmallaNivel1+2) || (pos_y_hebra == numVolySubmallaNivel1+3) && ultima_hebraY_submalla) {
			// Celda fantasma inferior (de las dos últimas filas)
			pos_ghost = finy + 1;
			if (pos_ghost < numVolyTotalNivel0*ratio_ref) {
				pos2 = posCopiaNivel1[posNivel1];
				if (pos2 > -1) {
					res1 = datosCopiaNivel1_1[pos2];
					res2 = datosCopiaNivel1_2[pos2];
				}
				else {
					prof = datosVolumenesNivel1_1[posNivel1].y;
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					obtenerDatosEInterpolarEnTiempoYEspacio(datosVolumenesNivel0_1, datosVolumenesNivel0_2,
						datosVolumenesNivel0Sig_1, datosVolumenesNivel0Sig_2, xi, yi, distx, disty, peso1,
						prof, inixSubmallaSupCluster, iniySubmallaSupCluster, numVolxSubmallaSupCluster,
						numVolySubmallaSupCluster, ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1, &res2);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_y_hebra - numVolySubmallaNivel1 - 1;
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res1 = datosVolumenesNivel1_1[posNivel1 - pos2];
				res2 = datosVolumenesNivel1_2[posNivel1 - pos2];
				res2.y *= borde_inf;
			}
			datosVolumenesNivel1_1[posNivel1] = res1;
			datosVolumenesNivel1_2[posNivel1] = res2;
		}
	}
}

/****************************************/
/* Interpolación del fondo al principio */
/****************************************/

__device__ void interpolarFondoEnEspacio(double2 *datosVolumenesNivel0_1, int xi, int yi, double distx,
				double disty, int inixSubmallaSupCluster, int iniySubmallaSupCluster, int numVolxSubmallaSupCluster,
				int numVolySubmallaSupCluster, bool ultima_hebraX_nivel0, bool ultima_hebraY_nivel0, int nivelSup, double2 *res1)
{
	double2 datos1, datosx1, datosy1;
	double2 datos_sup1, datos_inf1, datos_izq1, datos_der1;
	int xizq, xder, ysup, yinf;
	int pos, pos_izq, pos_der, pos_sup, pos_inf;

	if (nivelSup == 0) {
		// Nivel 0
		if (inixSubmallaSupCluster == 0)
			xizq = max(xi-1, 0) + 2;
		else
			xizq = xi-1 + 2;
		if (ultima_hebraX_nivel0)
			xder = min(xi+1, numVolxSubmallaSupCluster-1) + 2;
		else
			xder = xi+1 + 2;

		if (iniySubmallaSupCluster == 0)
			ysup = max(yi-1, 0) + 2;
		else
			ysup = yi-1 + 2;
		if (ultima_hebraY_nivel0)
			yinf = min(yi+1, numVolySubmallaSupCluster-1) + 2;
		else
			yinf = yi+1 + 2;
	}
	else {
		xizq = xi-1 + 2;
		xder = xi+1 + 2;
		ysup = yi-1 + 2;
		yinf = yi+1 + 2;
	}
	xi += 2;  // offsetX
	yi += 2;  // offsetY
	numVolxSubmallaSupCluster += 4;  // 2*offsetX

	pos = yi*numVolxSubmallaSupCluster + xi;
	pos_izq = yi*numVolxSubmallaSupCluster + xizq;
	pos_der = yi*numVolxSubmallaSupCluster + xder;
	pos_sup = ysup*numVolxSubmallaSupCluster + xi;
	pos_inf = yinf*numVolxSubmallaSupCluster + xi;

	datos1  = datosVolumenesNivel0_1[pos];
	datos_izq1 = datosVolumenesNivel0_1[pos_izq];
	datos_der1 = datosVolumenesNivel0_1[pos_der];
	datos_sup1 = datosVolumenesNivel0_1[pos_sup];
	datos_inf1 = datosVolumenesNivel0_1[pos_inf];

	datosx1.y = minmod(datos_der1.y-datos1.y, 0.5*(datos_der1.y-datos_izq1.y), datos1.y-datos_izq1.y);
	datosy1.y = minmod(datos_inf1.y-datos1.y, 0.5*(datos_inf1.y-datos_sup1.y), datos1.y-datos_sup1.y);
	res1->y = datos1.y + distx*datosx1.y + disty*datosy1.y;
}

__global__ void obtenerFondoVolumenesFantasmaNivel1GPU(double2 *datosVolumenesNivel0_1, double2 *datosVolumenesNivel1_1,
				int numVolxTotalNivel0, int numVolyTotalNivel0, int inix, int iniy, int numVolxSubmallaNivel1, int numVolySubmallaNivel1,
				int *posCopiaNivel1, double2 *datosCopiaNivel1_1, int ratio_ref, int inixSubmallaSupCluster, int iniySubmallaSupCluster,
				int numVolxSubmallaSupCluster, int numVolySubmallaSupCluster, int inixSubmallaCluster, int iniySubmallaCluster,
				bool ultima_hebraX_nivel0, bool ultima_hebraY_nivel0, bool ultima_hebraX_submalla, bool ultima_hebraY_submalla, int nivelSup)
{
	int posNivel1;
	int pos_ghost, pos2;
	int xi, yi;
	double distx, disty;
	int finx, finy;
	double2 res1;
	int pos_x_hebra, pos_y_hebra;
	bool procesada = false;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// datosVolumenesNivel0 contiene los datos del estado actual del nivel 0.
	// datosVolumenesNivel1 sólo contiene los datos de la submalla actual.
	// datosCopiaNivel1 contiene los datos de todas las submallas del nivel 1,
	// y posCopiaNivel1 indica la posición de datosCopiaNivel1 desde donde hay que
	// copiar los datos del volumen fastasma que trate la hebra (-1 si hay que interpolar).

	// Interpolamos las celdas internas fantasma que no se salgan del dominio a partir
	// de los datos del nivel 0 aplicando interpolación MUSCL en espacio.
	// Si la celda interna fantasma es una celda de otra submalla, copiamos los datos
	// y no interpolamos. También asignamos las celdas fantasma que se salen
	// del dominio. Ponemos los datos en datosVolumenesNivel1.

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
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					interpolarFondoEnEspacio(datosVolumenesNivel0_1, xi, yi, distx, disty, inixSubmallaSupCluster,
						iniySubmallaSupCluster, numVolxSubmallaSupCluster, numVolySubmallaSupCluster,
						ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_x_hebra == 0) ? 2 : 1);
				res1 = datosVolumenesNivel1_1[posNivel1 + pos2];
			}
			datosVolumenesNivel1_1[posNivel1].y = res1.y;
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
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = iniy + iniySubmallaCluster + pos_y_hebra-2;
					xi = pos_ghost/ratio_ref - inixSubmallaSupCluster;
					yi = pos2/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					interpolarFondoEnEspacio(datosVolumenesNivel0_1, xi, yi, distx, disty, inixSubmallaSupCluster,
						iniySubmallaSupCluster, numVolxSubmallaSupCluster, numVolySubmallaSupCluster,
						ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_x_hebra - numVolxSubmallaNivel1 - 1;
				res1 = datosVolumenesNivel1_1[posNivel1 - pos2];
			}
			datosVolumenesNivel1_1[posNivel1].y = res1.y;
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
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					interpolarFondoEnEspacio(datosVolumenesNivel0_1, xi, yi, distx, disty, inixSubmallaSupCluster,
						iniySubmallaSupCluster, numVolxSubmallaSupCluster, numVolySubmallaSupCluster,
						ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = ((pos_y_hebra == 0) ? 2 : 1);
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res1 = datosVolumenesNivel1_1[posNivel1 + pos2];
			}
			datosVolumenesNivel1_1[posNivel1].y = res1.y;
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
					// (xi, yi): Posición de la celda asociada del nivel 0
					pos2 = inix + inixSubmallaCluster + pos_x_hebra-2;
					xi = pos2/ratio_ref - inixSubmallaSupCluster;
					yi = pos_ghost/ratio_ref - iniySubmallaSupCluster;
					distx = (-ratio_ref + 1.0 + 2.0*(pos2&(ratio_ref-1))) / (2.0*ratio_ref);
					disty = (-ratio_ref + 1.0 + 2.0*(pos_ghost&(ratio_ref-1))) / (2.0*ratio_ref);
					interpolarFondoEnEspacio(datosVolumenesNivel0_1, xi, yi, distx, disty, inixSubmallaSupCluster,
						iniySubmallaSupCluster, numVolxSubmallaSupCluster, numVolySubmallaSupCluster,
						ultima_hebraX_nivel0, ultima_hebraY_nivel0, nivelSup, &res1);
				}
			}
			else {
				// Celda que se sale del dominio
				pos2 = pos_y_hebra - numVolySubmallaNivel1 - 1;
				pos2 = pos2*(numVolxSubmallaNivel1+4);
				res1 = datosVolumenesNivel1_1[posNivel1 - pos2];
			}
			datosVolumenesNivel1_1[posNivel1].y = res1.y;
		}
	}
}

/*******************************************/
/* Corrección de flujos de celdas internas */
/*******************************************/

__global__ void corregirFlujosNivelGPU(double2 *d_datosVolumenesNivel0_1, double2 *d_datosVolumenesNivel0_2,
				double *d_correccionEtaNivel0, double2 *d_correccionNivel0_2, int numVolxNivel0, int numVolyNivel0)
{
	double2 W, corr;
	double alfa, corr_eta;
	int pos_x_hebra, pos_y_hebra;
	int pos_datos, pos_corr;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < numVolxNivel0) && (pos_y_hebra < numVolyNivel0)) {
		pos_datos = (pos_y_hebra+2)*(numVolxNivel0+4) + (pos_x_hebra + 2);
		pos_corr = pos_y_hebra*numVolxNivel0 + pos_x_hebra;
		alfa = 1.0;
		W = d_datosVolumenesNivel0_1[pos_datos];
		corr_eta = d_correccionEtaNivel0[pos_corr];

		if (corr_eta > W.x) {
			alfa = W.x / (corr_eta + EPSILON);
		}
		W.x += alfa*corr_eta;
		if (W.x < 0.0)  W.x = 0.0;
		d_datosVolumenesNivel0_1[pos_datos] = W;

		W = d_datosVolumenesNivel0_2[pos_datos];
		corr = d_correccionNivel0_2[pos_corr];
		W.x += alfa*corr.x;
		W.y += alfa*corr.y;
		d_datosVolumenesNivel0_2[pos_datos] = W;
	}
}

/*********************************/
/* Activación de mallas anidadas */
/*********************************/

// 1 si hay que procesar las mallas anidadas, 0 en otro caso
__device__ int d_mallasAnidadasActivadas;

__global__ void inicializarActivacionMallasAnidadasGPU(bool *d_activacionMallasAnidadas, bool *d_refinarNivel0,
				int numVolxNivel0, int numVolyNivel0)
{
	int pos_x_hebra, pos_y_hebra;
	int pos, pos2;
	int i, j;
	int inix, finx;
	int iniy, finy;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// Comprobamos si la hebra está dentro de los límites de la malla
	if ((pos_x_hebra < numVolxNivel0) && (pos_y_hebra < numVolyNivel0)) {
		pos = pos_y_hebra*numVolxNivel0 + pos_x_hebra;
		if (d_refinarNivel0[pos]) {
			inix = max(pos_x_hebra-1,0);
			finx = min(pos_x_hebra+1,numVolxNivel0-1);
			iniy = max(pos_y_hebra-1,0);
			finy = min(pos_y_hebra+1,numVolyNivel0-1);
			for (j=iniy; j<=finy; j++) {
				for (i=inix; i<=finx; i++) {
					pos2 = j*numVolxNivel0 + i;
					d_activacionMallasAnidadas[pos2] = true;
				}
			}
		}
		if (pos == 0)
			d_mallasAnidadasActivadas = 0;
	}
}

__global__ void obtenerActivacionMallasAnidadasGPU(double2 *d_datosVolumenesNivel0Sig_1, double *d_eta1InicialNivel0,
				bool *d_activacionMallasAnidadas, int numVolxNivel0, int numVolyNivel0, double epsilon_h)
{
	int pos_x_hebra, pos_y_hebra;
	int pos_datos, pos;
	double2 W;
	double eta_ini, dif_eta;
	bool act;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	// Comprobamos si la hebra está dentro de los límites de la malla
	if ((pos_x_hebra < numVolxNivel0) && (pos_y_hebra < numVolyNivel0)) {
		pos_datos = (pos_y_hebra+2)*(numVolxNivel0+4) + pos_x_hebra+2;
		W = d_datosVolumenesNivel0Sig_1[pos_datos];

		pos = pos_y_hebra*numVolxNivel0 + pos_x_hebra;
		eta_ini = d_eta1InicialNivel0[pos];
		act = d_activacionMallasAnidadas[pos];

		dif_eta = fabs(W.x - W.y - eta_ini);
		if ((dif_eta > 0.5*epsilon_h) && (W.x > epsilon_h) && act)
			d_mallasAnidadasActivadas = 1;
	}
}

#endif

