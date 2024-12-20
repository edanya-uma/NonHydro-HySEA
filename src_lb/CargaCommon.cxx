#ifndef _CARGA_COMMON_
#define _CARGA_COMMON_

#include "Constantes.hxx"
#include <fstream>
#include <cmath>
#include <string.h>

// Devuelve la columna del nivel 0 donde se encuentra el volumen con coordenada x de la submalla y el nivel indicados
int obtenerColumnaNivel0(int x, int submalla, int nivel, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
					int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel)
{
	int col;
	int l, sact, ssup;

	sact = submalla;
	col = submallasNivel[nivel][sact].x + x;
	for (l=nivel; l>0; l--) {
		ssup = submallaNivelSuperior[l][sact];
		col = submallasNivel[l-1][ssup].x + col/ratioRefNivel[l];
		sact = ssup;
	}

	return col;
}

// Devuelve la fila del nivel 0 donde se encuentra el volumen con coordenada y de la submalla y el nivel indicados
int obtenerFilaNivel0(int y, int submalla, int nivel, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
					int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel)
{
	int fila;
	int l, sact, ssup;

	sact = submalla;
	fila = submallasNivel[nivel][sact].y + y;
	for (l=nivel; l>0; l--) {
		ssup = submallaNivelSuperior[l][sact];
		fila = submallasNivel[l-1][ssup].y + fila/ratioRefNivel[l];
		sact = ssup;
	}

	return fila;
}

// Devuelve true si inixCluster coincide con la frontera de alguna submalla, false en otro caso
bool esFronteraSubmallaX(int numNiveles, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel,
					int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel, int inixCluster)
{
	int l, l2, i, k;
	int inix_submalla, posx;
	bool es_frontera = false;

	for (l=1; (l < numNiveles) && (! es_frontera); l++) {
		for (i=0; (i < numSubmallasNivel[l]) && (! es_frontera); i++) {
			// Comprobamos la frontera izquierda de la submalla i
			posx = submallasNivel[l][i].x/ratioRefNivel[l];
			k = submallaNivelSuperior[l][i];
			for (l2=l-1; l2>=0; l2--) {
				inix_submalla = submallasNivel[l2][k].x;
				posx = (inix_submalla + posx)/ratioRefNivel[l2];
				k = submallaNivelSuperior[l2][k];
			}
			if (posx == inixCluster) {
				es_frontera = true;
			}
			else {
				// Comprobamos la frontera derecha de la submalla i
				posx = (submallasNivel[l][i].x + submallasNivel[l][i].z)/ratioRefNivel[l];
				k = submallaNivelSuperior[l][i];
				for (l2=l-1; l2>=0; l2--) {
					inix_submalla = submallasNivel[l2][k].x;
					posx = (inix_submalla + posx)/ratioRefNivel[l2];
					k = submallaNivelSuperior[l2][k];
				}
				if (posx == inixCluster) {
					es_frontera = true;
				}
			}
		}
	}

	return es_frontera;
}

// Devuelve true si iniyCluster coincide con la frontera de alguna submalla, false en otro caso
bool esFronteraSubmallaY(int numNiveles, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel,
					int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel, int iniyCluster)
{
	int l, l2, i, k;
	int iniy_submalla, posy;
	bool es_frontera = false;

	for (l=1; (l < numNiveles) && (! es_frontera); l++) {
		for (i=0; (i < numSubmallasNivel[l]) && (! es_frontera); i++) {
			// Comprobamos la frontera superior de la submalla i
			posy = submallasNivel[l][i].y/ratioRefNivel[l];
			k = submallaNivelSuperior[l][i];
			for (l2=l-1; l2>=0; l2--) {
				iniy_submalla = submallasNivel[l2][k].y;
				posy = (iniy_submalla + posy)/ratioRefNivel[l2];
				k = submallaNivelSuperior[l2][k];
			}
			if (posy == iniyCluster) {
				es_frontera = true;
			}
			else {
				// Comprobamos la frontera inferior de la submalla i
				posy = (submallasNivel[l][i].y + submallasNivel[l][i].w)/ratioRefNivel[l];
				k = submallaNivelSuperior[l][i];
				for (l2=l-1; l2>=0; l2--) {
					iniy_submalla = submallasNivel[l2][k].y;
					posy = (iniy_submalla + posy)/ratioRefNivel[l2];
					k = submallaNivelSuperior[l2][k];
				}
				if (posy == iniyCluster) {
					es_frontera = true;
				}
			}
		}
	}

	return es_frontera;
}

#endif

