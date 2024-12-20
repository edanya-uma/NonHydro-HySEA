#ifndef _CARGAY_
#define _CARGAY_

#include "CargaCommon.cxx"

// FUNCIONES PARA EL EQUILIBRADO DE CARGA EN LA DIMENSIÓN Y (POR FILAS)

/******/
/* 1D */
/******/

double obtenerPesoAristasHorNivel0(double2 *datosVolumenesNivel0_1, int num_volx, int num_voly, int fila_sup, int fila_inf)
{
	double peso;
	double h0, h1;
	bool waf;
	int y0, y1, i;

	y0 = fila_sup;
	y1 = min(fila_inf,num_voly-1);
	peso = 0.0;
	for (i=0; i<num_volx; i++) {
		h0  = datosVolumenesNivel0_1[y0*num_volx + i].x;
		h1  = datosVolumenesNivel0_1[y1*num_volx + i].x;
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}

	return peso;
}

double obtenerPesoAristasHorNivel1(double2 *datosVolumenesNivel1_1, int num_volx, int num_voly, int fila_sup, int fila_inf)
{
	double peso;
	double h0, h1;
	bool waf;
	int y0, y1, i;

	// fila_sup y fila_inf empiezan por 0 (sumamos 2 para obtener la coordenada y en el vector de datos)
	y0 = fila_sup+2;
	y1 = min(fila_inf+2,num_voly+1);
	num_volx += 4;
	peso = 0.0;
	for (i=0; i<num_volx; i++) {
		h0  = datosVolumenesNivel1_1[y0*num_volx + i+2].x;
		h1  = datosVolumenesNivel1_1[y1*num_volx + i+2].x;
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}

	return peso;
}

double obtenerPesoVolumenesYAristasVerNivel0(double2 *datosVolumenesNivel0_1, int num_volx, int fila)
{
	double peso;
	double h0, h1;
	bool waf;
	int x0, x1, i;
	int indfila = fila*num_volx;

	peso = 0.0;
	for (i=0; i<num_volx; i++) {
		x0 = max(i-1,0);
		x1 = i;
		h0  = datosVolumenesNivel0_1[indfila + x0].x;
		h1  = datosVolumenesNivel0_1[indfila + x1].x;
		// Peso del volumen
		peso += ((h1 > EPSILON) ? PESO_VOL_MOJADO : PESO_VOL_SECO);
		// Peso de la arista
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}
	// Arista frontera derecha (waf)
	h0  = datosVolumenesNivel0_1[indfila + num_volx-1].x;
	h1  = h0;
	peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);

	return peso;
}

double obtenerPesoVolumenesYAristasVerNivel1(double2 *datosVolumenesNivel1_1, int num_volx, int fila)
{
	double peso;
	double h0, h1;
	bool waf;
	int x0, x1, i;
	int indfila = (fila+2)*(num_volx+4);

	// fila empieza por 0 (sumamos 2 para obtener la coordenada y en el vector de datos)
	peso = 0.0;
	for (i=0; i<num_volx; i++) {
		x0 = max(i+1,2);
		x1 = i+2;
		h0  = datosVolumenesNivel1_1[indfila + x0].x;
		h1  = datosVolumenesNivel1_1[indfila + x1].x;
		// Peso del volumen
		peso += ((h1 > EPSILON) ? PESO_VOL_MOJADO : PESO_VOL_SECO);
		// Peso de la arista
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}
	// Arista frontera derecha (waf)
	h0  = datosVolumenesNivel1_1[indfila + num_volx+1].x;
	h1  = h0;
	peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);

	return peso;
}

void obtenerPesoFilas(double2 **datosVolumenesNivel_1, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
				int *numSubmallasNivel, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL],
				int *ratioRefNivel, int num_niveles, double *pesoFila)
{
	int i, l, s, pos_ini;
	int numVolxNivel0, numVolyNivel0;
	int numVolx, numVoly, filaNivel0;
	int ratio_acumulado = 1;
	double peso;

	numVolxNivel0 = submallasNivel[0][0].z;
	numVolyNivel0 = submallasNivel[0][0].w;

	// Inicializamos el vector pesoFila
	memset(pesoFila, 0, numVolyNivel0*sizeof(double));

	// Nivel 0
	for (i=0; i<numVolyNivel0; i++) {
		pesoFila[i] += obtenerPesoVolumenesYAristasVerNivel0(datosVolumenesNivel_1[0], numVolxNivel0, i);
		pesoFila[i] += obtenerPesoAristasHorNivel0(datosVolumenesNivel_1[0], numVolxNivel0, numVolyNivel0, i, i+1);
	}

	// Resto de niveles
	for (l=1; l<num_niveles; l++) {
#if (GAMMA == 0)
		ratio_acumulado *= ratioRefNivel[l];
#else
		ratio_acumulado *= 2;
#endif
		pos_ini = 0;
		for (s=0; s<numSubmallasNivel[l]; s++) {
			numVolx = submallasNivel[l][s].z;
			numVoly = submallasNivel[l][s].w;
			for (i=0; i<numVoly; i++) {
				peso  = obtenerPesoVolumenesYAristasVerNivel1(datosVolumenesNivel_1[l]+pos_ini, numVolx, i);
				peso += obtenerPesoAristasHorNivel1(datosVolumenesNivel_1[l]+pos_ini, numVolx, numVoly, i, i+1);
				filaNivel0 = obtenerFilaNivel0(i, s, l, submallasNivel, submallaNivelSuperior, ratioRefNivel);
				pesoFila[filaNivel0] += peso*ratio_acumulado;
			}
			pos_ini += (numVolx+4)*(numVoly+4);
		}
	}
}

// Devuelve 0 si el equilibrado ha ido bien, 1 si algun proceso no tiene ningún volumen
int obtenerEquilibradoY(double *pesoFila, int numVoly, int numProcsX, int numProcsY, int *yIniCluster,
					int numNiveles, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel,
					int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel,
					double *pesosGPU, double *pesoIdealAcum)
{
	double peso_total, peso_ideal, peso_actual;
	double val;
	int i, j, indice, num_voly_cluster;
	bool es_frontera;

	peso_total = 0.0;
	for (i=0; i<numVoly; i++)
		peso_total += pesoFila[i];
	// Peso ideal que debería tener cada subproblema si todas las GPUs tuvieran el mismo peso
	peso_ideal = peso_total/numProcsY;
	// Calculamos los pesos ideales acumulados para cada fila de procesos ponderados por los pesos de las GPUs
	for (i=0; i<numProcsY; i++) {
		// val: media de los pesos de las GPUs de la fila i-ésima de la matriz de procesos
		val = 0.0;
		for (j=0; j<numProcsX; j++)
			val += pesosGPU[i*numProcsX+j];
		val /= numProcsX;

		if (i == 0)
			pesoIdealAcum[i] = peso_ideal*val;
		else
			pesoIdealAcum[i] = pesoIdealAcum[i-1] + peso_ideal*val;
	}

	// indice: indice para recorrer el vector pesoFila
	indice = 0;
	peso_actual = 0.0;
	for (i=0; i<numProcsY; i++) {
		yIniCluster[i] = indice;
		num_voly_cluster = 0;
		while ((peso_actual < pesoIdealAcum[i]) && (indice < numVoly)) {
			num_voly_cluster++;
			peso_actual += pesoFila[indice];
			indice++;
		}
		if (i != numProcsY-1) {
			// Si no estamos procesando la última hebra, forzamos a que el número de filas
			// del subproblema sea un número par
			if (num_voly_cluster % 2 != 0) {
				num_voly_cluster++;
				peso_actual += pesoFila[indice];
				indice++;
			}

			// Comprobamos si la frontera del subdominio coincide con la frontera de alguna submalla y,
			// si es así, añadimos dos filas más
			es_frontera = esFronteraSubmallaY(numNiveles, submallasNivel, numSubmallasNivel, submallaNivelSuperior,
							ratioRefNivel, indice);
			while (es_frontera) {
				num_voly_cluster += 2;
				peso_actual += pesoFila[indice] + pesoFila[indice+1];
				indice += 2;
				es_frontera = esFronteraSubmallaY(numNiveles, submallasNivel, numSubmallasNivel, submallaNivelSuperior,
								ratioRefNivel, indice);
			}
		}
		if (num_voly_cluster == 0) {
			cerr << "Error: Process " << i << " in Y has 0 volumes. Please use less processes in the Y dimension" << endl;
			return 1;
		}
	}

	return 0;
}

#endif

