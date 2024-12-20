#ifndef _CARGAX_
#define _CARGAX_

#include "CargaCommon.cxx"

// FUNCIONES PARA EL EQUILIBRADO DE CARGA EN LA DIMENSIÓN X (POR COLUMNAS)

/******/
/* 1D */
/******/

double obtenerPesoAristasVerNivel0(double2 *datosVolumenesNivel0_1, int num_volx, int num_voly, int col_izda, int col_dcha)
{
	double peso;
	double h0, h1;
	bool waf;
	int x0, x1, i;
	int indfila;

	x0 = col_izda;
	x1 = min(col_dcha,num_volx-1);
	peso = 0.0;
	for (i=0; i<num_voly; i++) {
		indfila = i*num_volx;
		h0  = datosVolumenesNivel0_1[indfila + x0].x;
		h1  = datosVolumenesNivel0_1[indfila + x1].x;
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}

	return peso;
}

double obtenerPesoAristasVerNivel1(double2 *datosVolumenesNivel1_1, int num_volx, int num_voly, int col_izda, int col_dcha)
{
	double peso;
	double h0, h1;
	bool waf;
	int x0, x1, i;
	int indfila;

	// col_izda y col_dcha empiezan por 0 (sumamos 2 para obtener la coordenada x en el vector de datos)
	x0 = col_izda+2;
	x1 = min(col_dcha+2,num_volx+1);
	num_volx += 4;
	peso = 0.0;
	for (i=0; i<num_voly; i++) {
		indfila = (i+2)*num_volx;
		h0  = datosVolumenesNivel1_1[indfila + x0].x;
		h1  = datosVolumenesNivel1_1[indfila + x1].x;
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}

	return peso;
}

double obtenerPesoVolumenesYAristasHorNivel0(double2 *datosVolumenesNivel0_1, int num_volx, int num_voly, int col)
{
	double peso;
	double h0, h1;
	bool waf;
	int y0, y1, i;

	peso = 0.0;
	for (i=0; i<num_voly; i++) {
		y0 = max(i-1,0);
		y1 = i;
		h0  = datosVolumenesNivel0_1[y0*num_volx + col].x;
		h1  = datosVolumenesNivel0_1[y1*num_volx + col].x;
		// Peso del volumen
		peso += ((h1 > EPSILON) ? PESO_VOL_MOJADO : PESO_VOL_SECO);
		// Peso de la arista
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}
	// Arista frontera inferior (waf)
	h0  = datosVolumenesNivel0_1[(num_voly-1)*num_volx + col].x;
	h1  = h0;
	peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);

	return peso;
}

double obtenerPesoVolumenesYAristasHorNivel1(double2 *datosVolumenesNivel1_1, int num_volx, int num_voly, int col)
{
	double peso;
	double h0, h1;
	bool waf;
	int y0, y1, i;

	// col empieza por 0 (sumamos 2 para obtener la coordenada x en el vector de datos)
	num_volx += 4;
	col += 2;
	peso = 0.0;
	for (i=0; i<num_voly; i++) {
		y0 = max(i+1,2);
		y1 = i+2;
		h0  = datosVolumenesNivel1_1[y0*num_volx + col].x;
		h1  = datosVolumenesNivel1_1[y1*num_volx + col].x;
		// Peso del volumen
		peso += ((h1 > EPSILON) ? PESO_VOL_MOJADO : PESO_VOL_SECO);
		// Peso de la arista
		waf = ( ((h0 > EPSILON) && (h1 > EPSILON)) ? false : true);
		if (waf)
			peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);
		else
			peso += PESO_ARI_2S;
	}
	// Arista frontera inferior (waf)
	h0  = datosVolumenesNivel1_1[(num_voly+1)*num_volx + col].x;
	h1  = h0;
	peso += ( ((h0 > EPSILON) || (h1 > EPSILON)) ? PESO_ARI_WAF_MOJADA : PESO_ARI_WAF_SECA);

	return peso;
}

void obtenerPesoColumnas(double2 **datosVolumenesNivel_1, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
				int *numSubmallasNivel, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL],
				int *ratioRefNivel, int num_niveles, double *pesoColumna)
{
	int i, l, s, pos_ini;
	int numVolxNivel0, numVolyNivel0;
	int numVolx, numVoly, colNivel0;
	int ratio_acumulado = 1;
	double peso;

	numVolxNivel0 = submallasNivel[0][0].z;
	numVolyNivel0 = submallasNivel[0][0].w;

	// Inicializamos el vector pesoColumna
	memset(pesoColumna, 0, numVolxNivel0*sizeof(double));

	// Nivel 0
	for (i=0; i<numVolxNivel0; i++) {
		pesoColumna[i] += obtenerPesoVolumenesYAristasHorNivel0(datosVolumenesNivel_1[0], numVolxNivel0, numVolyNivel0, i);
		pesoColumna[i] += obtenerPesoAristasVerNivel0(datosVolumenesNivel_1[0], numVolxNivel0, numVolyNivel0, i, i+1);
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
			for (i=0; i<numVolx; i++) {
				peso  = obtenerPesoVolumenesYAristasHorNivel1(datosVolumenesNivel_1[l]+pos_ini, numVolx, numVoly, i);
				peso += obtenerPesoAristasVerNivel1(datosVolumenesNivel_1[l]+pos_ini, numVolx, numVoly, i, i+1);
				colNivel0 = obtenerColumnaNivel0(i, s, l, submallasNivel, submallaNivelSuperior, ratioRefNivel);
				pesoColumna[colNivel0] += peso*ratio_acumulado;
			}
			pos_ini += (numVolx+4)*(numVoly+4);
		}
	}
}

// Devuelve 0 si el equilibrado ha ido bien, 1 si algun proceso no tiene ningún volumen
int obtenerEquilibradoX(double *pesoColumna, int numVolx, int numProcsX, int numProcsY, int *xIniCluster,
					int numNiveles, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel,
					int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel,
					double *pesosGPU, double *pesoIdealAcum)
{
	double peso_total, peso_ideal, peso_actual;
	double val;
	int i, j, indice, num_volx_cluster;
	bool es_frontera;

	peso_total = 0.0;
	for (i=0; i<numVolx; i++)
		peso_total += pesoColumna[i];
	// Peso ideal que debería tener cada subproblema si todas las GPUs tuvieran el mismo peso
	peso_ideal = peso_total/numProcsX;
	// Calculamos los pesos ideales acumulados para cada columna de procesos ponderados por los pesos de las GPUs
	for (i=0; i<numProcsX; i++) {
		// val: media de los pesos de las GPUs de la columna i-ésima de la matriz de procesos
		val = 0.0;
		for (j=0; j<numProcsY; j++)
			val += pesosGPU[j*numProcsX+i];
		val /= numProcsY;

		if (i == 0)
			pesoIdealAcum[i] = peso_ideal*val;
		else
			pesoIdealAcum[i] = pesoIdealAcum[i-1] + peso_ideal*val;
	}

	// indice: indice para recorrer el vector pesoColumna
	indice = 0;
	peso_actual = 0.0;
	for (i=0; i<numProcsX; i++) {
		xIniCluster[i] = indice;
		num_volx_cluster = 0;
		while ((peso_actual < pesoIdealAcum[i]) && (indice < numVolx)) {
			num_volx_cluster++;
			peso_actual += pesoColumna[indice];
			indice++;
		}
		if (i != numProcsX-1) {
			// Si no estamos procesando la última hebra, forzamos a que el número de filas
			// del subproblema sea un número par
			if (num_volx_cluster % 2 != 0) {
				num_volx_cluster++;
				peso_actual += pesoColumna[indice];
				indice++;
			}

			// Comprobamos si la frontera del subdominio coincide con la frontera de alguna submalla y,
			// si es así, añadimos dos filas más
			es_frontera = esFronteraSubmallaX(numNiveles, submallasNivel, numSubmallasNivel, submallaNivelSuperior,
							ratioRefNivel, indice);
			while (es_frontera) {
				num_volx_cluster += 2;
				peso_actual += pesoColumna[indice] + pesoColumna[indice+1];
				indice += 2;
				es_frontera = esFronteraSubmallaX(numNiveles, submallasNivel, numSubmallasNivel, submallaNivelSuperior,
								ratioRefNivel, indice);
			}
		}
		if (num_volx_cluster == 0) {
			cerr << "Error: Process " << i << " in X has 0 volumes. Please use less processes in the X dimension" << endl;
			return 1;
		}
	}

	return 0;
}

#endif

