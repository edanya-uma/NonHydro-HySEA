#ifndef _PROBLEMA_H_
#define _PROBLEMA_H_

#include "Constantes.hxx"
#include "OkadaTriangular.cxx"
#include <sys/stat.h> 
#include <fstream>
#include <sstream>
#include <cstring>
#include <limits.h>
#include <mpi.h>

/********************/
/* Funciones NetCDF */
/********************/

extern void leerTamanoSubmallaGRD(const char *nombre_fich, int nivel, int submalla, int *nvx, int *nvy);
extern int  abrirGRD(const char *nombre_fich, int nivel, int submalla, MPI_Comm comm);
extern void leerLongitudGRD(int nivel, int submalla, double *lon);
extern void leerLatitudGRD(int nivel, int submalla, double *lat);
extern void leerBatimetriaGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *bati);
extern void leerEtaGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *eta);
extern void leerUxGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *ux);
extern void leerUyGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *uy);
extern void leerUzGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *uz);
extern void cerrarGRD(int nivel, int submalla);
extern void leerDeformacionGRD(const char *nombre_fich, int *nvx, int *nvy, double *lon, double *lat, float *def);

extern int  abrirGRDFricciones(const char *nombre_fich, int nivel, int submalla, MPI_Comm comm);
extern void leerTamanoSubmallaGRDFricciones(int nivel, int submalla, int *nvx, int *nvy);
extern void leerFriccionesGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *fric);
extern void cerrarGRDFricciones(int nivel, int submalla);

extern void leerTamanoYNumEstadosDefDinamicaGRD(const char *nombre_fich, int *nvx, int *nvy, int *num_estados);
extern int abrirGRDDefDinamica(const char *nombre_fich);
extern void leerLongitudLatitudYTiemposDefDinamicaGRD(double *lon, double *lat, float *time);
extern void leerEstadoDefDinamicaGRD(int nvx, int nvy, int num, float *def);
extern void cerrarGRDDefDinamica();

extern bool existeVariableNetCDF(int nivel, int submalla, const char *nombre_var);
extern void abrirTimeSeriesOldNC(const char *nombre_fich);
extern int obtenerNumPuntosTimeSeriesOldNC();
extern void leerLongitudesYLatitudesTimeSeriesOldNC(double *lonPuntos, double *latPuntos);
extern void cerrarTimeSeriesOldNC();

/**************/
/* Ordenación */
/**************/

void ordenarFallasOkadaStandardPorTiempo(int numFaults, double *LON_C, double *LAT_C, double *DEPTH_C, double *FAULT_L,
		double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *defTime)
{
	int i, j;
	int pos_min;
	double val;

	for (i=0; i<numFaults-1; i++) {
		pos_min = i;
		val = defTime[i];
		for (j=i+1; j<numFaults; j++) {
			if (defTime[j] < defTime[pos_min]) {
				pos_min = j;
				val = defTime[j];
			}
		}
		// Intercambiar elementos i y pos_min
		if (i != pos_min) {
			val = LON_C[i];    LON_C[i] = LON_C[pos_min];      LON_C[pos_min] = val;
			val = LAT_C[i];    LAT_C[i] = LAT_C[pos_min];      LAT_C[pos_min] = val;
			val = DEPTH_C[i];  DEPTH_C[i] = DEPTH_C[pos_min];  DEPTH_C[pos_min] = val;
			val = FAULT_L[i];  FAULT_L[i] = FAULT_L[pos_min];  FAULT_L[pos_min] = val;
			val = FAULT_W[i];  FAULT_W[i] = FAULT_W[pos_min];  FAULT_W[pos_min] = val;
			val = STRIKE[i];   STRIKE[i] = STRIKE[pos_min];    STRIKE[pos_min] = val;
			val = DIP[i];      DIP[i] = DIP[pos_min];          DIP[pos_min] = val;
			val = RAKE[i];     RAKE[i] = RAKE[pos_min];        RAKE[pos_min] = val;
			val = SLIP[i];     SLIP[i] = SLIP[pos_min];        SLIP[pos_min] = val;
			val = defTime[i];  defTime[i] = defTime[pos_min];  defTime[pos_min] = val;
		}
	}
}

void ordenarFallasOkadaTriangularPorTiempo(int numFaults, double2 LON_LAT_v[MAX_FAULTS][3], double DEPTH_v[MAX_FAULTS][4],
		double2 vc[MAX_FAULTS][4], double *LONCTRI, double *LATCTRI, double SLIPVEC[MAX_FAULTS][3], double *RAKE,
		double *SLIP, double *defTime)
{
	int i, j;
	int pos_min;
	double val;
	double2 val2;

	for (i=0; i<numFaults-1; i++) {
		pos_min = i;
		val = defTime[i];
		for (j=i+1; j<numFaults; j++) {
			if (defTime[j] < defTime[pos_min]) {
				pos_min = j;
				val = defTime[j];
			}
		}
		// Intercambiar elementos i y pos_min
		val2 = LON_LAT_v[i][0];  LON_LAT_v[i][0] = LON_LAT_v[pos_min][0];  LON_LAT_v[pos_min][0] = val2;
		val2 = LON_LAT_v[i][1];  LON_LAT_v[i][1] = LON_LAT_v[pos_min][1];  LON_LAT_v[pos_min][1] = val2;
		val2 = LON_LAT_v[i][2];  LON_LAT_v[i][2] = LON_LAT_v[pos_min][2];  LON_LAT_v[pos_min][2] = val2;
		val  = DEPTH_v[i][0];    DEPTH_v[i][0] = DEPTH_v[pos_min][0];      DEPTH_v[pos_min][0] = val;
		val  = DEPTH_v[i][1];    DEPTH_v[i][1] = DEPTH_v[pos_min][1];      DEPTH_v[pos_min][1] = val;
		val  = DEPTH_v[i][2];    DEPTH_v[i][2] = DEPTH_v[pos_min][2];      DEPTH_v[pos_min][2] = val;
		val  = DEPTH_v[i][3];    DEPTH_v[i][3] = DEPTH_v[pos_min][3];      DEPTH_v[pos_min][3] = val;
		val2 = vc[i][0];         vc[i][0] = vc[pos_min][0];                vc[pos_min][0] = val2;
		val2 = vc[i][1];         vc[i][1] = vc[pos_min][1];                vc[pos_min][1] = val2;
		val2 = vc[i][2];         vc[i][2] = vc[pos_min][2];                vc[pos_min][2] = val2;
		val2 = vc[i][3];         vc[i][3] = vc[pos_min][3];                vc[pos_min][3] = val2;
		val  = LONCTRI[i];       LONCTRI[i] = LONCTRI[pos_min];            LONCTRI[pos_min] = val;
		val  = LATCTRI[i];       LATCTRI[i] = LATCTRI[pos_min];            LATCTRI[pos_min] = val;
		val  = SLIPVEC[i][0];    SLIPVEC[i][0] = SLIPVEC[pos_min][0];      SLIPVEC[pos_min][0] = val;
		val  = SLIPVEC[i][1];    SLIPVEC[i][1] = SLIPVEC[pos_min][1];      SLIPVEC[pos_min][1] = val;
		val  = SLIPVEC[i][2];    SLIPVEC[i][2] = SLIPVEC[pos_min][2];      SLIPVEC[pos_min][2] = val;
		val  = RAKE[i];          RAKE[i] = RAKE[pos_min];                  RAKE[pos_min] = val;
		val  = SLIP[i];          SLIP[i] = SLIP[pos_min];                  SLIP[pos_min] = val;
		val  = defTime[i];       defTime[i] = defTime[pos_min];            defTime[pos_min] = val;
	}
}

void ordenarFallasFicherosPorTiempo(int numFaults, double **deformacionNivel0, int4 *submallasDef,
		double *defTime, double *temp, int64_t tam_datos)
{
	int i, j;
	int pos_min;
	double vald;
	int4 vali;

	for (i=0; i<numFaults-1; i++) {
		pos_min = i;
		vald = defTime[i];
		for (j=i+1; j<numFaults; j++) {
			if (defTime[j] < defTime[pos_min]) {
				pos_min = j;
				vald = defTime[j];
			}
		}
		// Intercambiar elementos i y pos_min
		if (i != pos_min) {
			memcpy(temp, deformacionNivel0[i], tam_datos);
			memcpy(deformacionNivel0[i], deformacionNivel0[pos_min], tam_datos);
			memcpy(deformacionNivel0[pos_min], temp, tam_datos);
			vali = submallasDef[i];  submallasDef[i] = submallasDef[pos_min];  submallasDef[pos_min] = vali;
			vald = defTime[i];       defTime[i] = defTime[pos_min];            defTime[pos_min] = vald;
		}
	}
}

// Devuelve true si existe el fichero, false en otro caso
bool existeFichero(string fichero)
{
	struct stat stFichInfo;
	bool existe;
	int intStat;

	// Obtenemos los atributos del fichero
	intStat = stat(fichero.c_str(), &stFichInfo);
	if (intStat == 0) {
		// Hemos obtenido los atributos del fichero. Por tanto, el fichero existe.
		existe = true;
	}
	else {
		// No hemos obtenido los atributos del fichero. Notar que esto puede
		// significar que no tenemos permiso para acceder al fichero. Para
		// hacer esta comprobación, comprobar los valores de intStat.
		existe = false;
	}

	return existe;
}

void liberarMemoria(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], double2 **datosVolumenesNivel_1,
		double3 **datosVolumenesNivel_2, double **datosPnh, float *batiOriginal[MAX_LEVELS], tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int *numSubmallasNivel, int **posCopiaNivel, bool **refinarNivel, double2 *bufferEnviosMPISupInf_1, double2 *bufferEnviosMPIIzqDer_1,
		double3 *bufferEnviosMPISupInf_2, double3 *bufferEnviosMPIIzqDer_2, double *bufferEnviosMPISupInf_P, double *bufferEnviosMPIIzqDer_P,
		int tipo_friccion, double **friccionesNivel, int4 *posicionesVolumenesGuardado, double *lonPuntos, double *latPuntos, int okada_flag,
		int numFaults, int numEstadosDefDinamica, double **deformacionNivel0, double *deformacionAcumNivel0, double *datosGRD, int id_hebra)
{
	int l, j;

	if (bufferEnviosMPISupInf_1 != NULL)		cudaFreeHost(bufferEnviosMPISupInf_1);
	if (bufferEnviosMPIIzqDer_1 != NULL)		cudaFreeHost(bufferEnviosMPIIzqDer_1);
	if (bufferEnviosMPISupInf_2 != NULL)		cudaFreeHost(bufferEnviosMPISupInf_2);
	if (bufferEnviosMPIIzqDer_2 != NULL)		cudaFreeHost(bufferEnviosMPIIzqDer_2);
	if (bufferEnviosMPISupInf_P != NULL)		cudaFreeHost(bufferEnviosMPISupInf_P);
	if (bufferEnviosMPIIzqDer_P != NULL)		cudaFreeHost(bufferEnviosMPIIzqDer_P);
	if (posicionesVolumenesGuardado != NULL)	free(posicionesVolumenesGuardado);
	if (lonPuntos != NULL)		free(lonPuntos);
	if (latPuntos != NULL)		free(latPuntos);
	if (datosGRD != NULL)		free(datosGRD);
	if (okada_flag == DEFORMATION_FROM_FILE) {
		if (deformacionAcumNivel0 != NULL)		free(deformacionAcumNivel0);
		for (j=0; j<numFaults; j++)
			if (deformacionNivel0[j] != NULL)	free(deformacionNivel0[j]);
	}
	else if (okada_flag == DYNAMIC_DEFORMATION) {
		if (deformacionAcumNivel0 != NULL)		free(deformacionAcumNivel0);
		for (j=0; j<numEstadosDefDinamica; j++)
			if (deformacionNivel0[j] != NULL)	free(deformacionNivel0[j]);
	}
	for (l=0; l<numNiveles; l++) {
		for (j=0; j<numSubmallasNivel[l]; j++) {
			if (datosNivel[l][j].vcos != NULL)		free(datosNivel[l][j].vcos);
			if (datosNivel[l][j].vccos != NULL)		free(datosNivel[l][j].vccos);
			if (datosNivel[l][j].vtan != NULL)		free(datosNivel[l][j].vtan);
			if (datosNivel[l][j].longitud != NULL)	free(datosNivel[l][j].longitud);
			if (datosNivel[l][j].latitud != NULL)	free(datosNivel[l][j].latitud);
		}
		if (batiOriginal[l] != NULL)			free(batiOriginal[l]);
		if (datosVolumenesNivel_1[l] != NULL)	free(datosVolumenesNivel_1[l]);
		if (datosVolumenesNivel_2[l] != NULL)	free(datosVolumenesNivel_2[l]);
		if (datosPnh[l] != NULL)				free(datosPnh[l]);
		if ((l == 0) || (tipo_friccion == VARIABLE_FRICTION_ALL)) {
			if (friccionesNivel[l] != NULL)			free(friccionesNivel[l]);
		}
		if (l > 0) {
			if (posCopiaNivel[l] != NULL)		free(posCopiaNivel[l]);
		}
		if ((numNiveles > 1) && (l < numNiveles-1)) {
			if (refinarNivel[l] != NULL)		free(refinarNivel[l]);
		}
	}
}

void liberarMemoriaDeformacion(double *longitudDeformacion, double *latitudDeformacion)
{
	if (longitudDeformacion != NULL) free(longitudDeformacion);
	if (latitudDeformacion != NULL) free(latitudDeformacion);
}

/*******************************/
/* Lectura de datos de fichero */
/*******************************/

template <class T>
void obtenerSiguienteDato(ifstream &fich, T &dato)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> dato;
}

void obtenerDatosFallaOkadaStandard(ifstream &fich, double &time, double &LON_C, double &LAT_C, double &DEPTH_C,
			double &FAULT_L, double &FAULT_W, double &STRIKE, double &DIP, double &RAKE, double &SLIP)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> time >> LON_C >> LAT_C >> DEPTH_C >> FAULT_L >> FAULT_W >> STRIKE >> DIP >> RAKE >> SLIP;
}

void obtenerDatosVentanaComputacion(ifstream &fich, double &lon_centro, double &lat_centro, double &radioComp)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");  
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> lon_centro >> lat_centro >> radioComp;
}

void obtenerDatosFallaOkadaTriangular(ifstream &fich, double &time, double &LON_v1, double &LAT_v1, double &DEPTH_v1,
			double &LON_v2, double &LAT_v2, double &DEPTH_v2, double &LON_v3, double &LAT_v3, double &DEPTH_v3,
			double &RAKE, double &SLIP)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");  
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> time >> LON_v1 >> LAT_v1 >> DEPTH_v1 >> LON_v2 >> LAT_v2 >> DEPTH_v2 >> LON_v3 >> LAT_v3 >> DEPTH_v3 >> RAKE >> SLIP;   
}

void obtenerDatosGaussiana(ifstream &fich, double &lon, double &lat, double &height, double &sigma)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> lon >> lat >> height >> sigma;
}

void obtenerDatosFallaFichero(ifstream &fich, double &time, string &fich_def)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> time >> fich_def;
}

void obtenerDatosGuardadoSubmalla(ifstream &fich, int &eta, int &eta_max, int &velocidades, int &velocidades_max,
		int &modulo_velocidades, int &modulo_velocidades_max, int &modulo_caudales_max, int &presion_no_hidrostatica,
		int &flujo_momento, int &flujo_momento_max, int &tiempos_llegada)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> eta >> eta_max >> velocidades >> velocidades_max >> modulo_velocidades >> modulo_velocidades_max >> modulo_caudales_max >> presion_no_hidrostatica >> flujo_momento >> flujo_momento_max >> tiempos_llegada;
}

void obtenerTiemposGuardadoNetCDFNiveles(ifstream &fich, int numNiveles, double *tiempoGuardarNetCDF)
{
	string linea;
	istringstream iss;
	size_t pos;
	bool repetir = true;
	int i;

	while (repetir) {
		if (! getline(fich,linea))
			repetir = false;
		// Eliminamos blancos al principio de linea
		pos = linea.find_first_not_of(" \t\r\n");  
		if (pos != string::npos) {
			linea.erase(0,pos);
			if (linea[0] != '#')
				repetir = false;
		}
	}
	iss.str(linea);
	iss >> tiempoGuardarNetCDF[0];
	for (i=1; i<numNiveles; i++) {
		// Si hay menos tiempos que niveles, asignamos el último tiempo al resto de niveles
		if (! (iss >> tiempoGuardarNetCDF[i]))
			tiempoGuardarNetCDF[i] = tiempoGuardarNetCDF[i-1];
	}
}

template void obtenerSiguienteDato<int>(ifstream &fich, int &dato);
template void obtenerSiguienteDato<double>(ifstream &fich, double &dato);
template void obtenerSiguienteDato<string>(ifstream &fich, string &dato);

/****************/
/* Cargar datos */
/****************/

bool hayErrorEnTiemposGuardarNetCDF(int numNiveles, double *tiempoGuardarNetCDF)
{
	double tiempo_nivel0 = tiempoGuardarNetCDF[0];
	double t, part_dec, part_int;
	bool hay_error = false;
	int l;

	if ((numNiveles > 1) && (tiempo_nivel0 >= 0.0)) {
		for (l=1; l<numNiveles; l++) {
			t = tiempoGuardarNetCDF[l];
			part_dec = modf(tiempo_nivel0/t, &part_int);
			if (fabs(part_dec) > 1e-6)
				hay_error = true;
		}
	}

	return hay_error;
}

// Devuelve la posición correspondiente del punto (x,y) en los vectores de datos
// de las submallas indicadas, -1 si el punto no está en las submallas. Sólo se
// comprueban las submallas que tengan como padre submallaSup
int posicionEnSubmallas(TDatosClusterCPU *datosClusterCPU, int x, int y, int4 *submallas, int numSubmallas,
						int submallaSup, int *submallaNivelSuperior)
{
	int inix, finx, iniy, finy;
	int i, pos;
	bool encontrado;

	pos = i = 0;
	encontrado = false;
	while ((i < numSubmallas) && (! encontrado)) {
		if (datosClusterCPU[i].iniy != -1) {
			if (submallaNivelSuperior[i] == submallaSup) {
				inix = submallas[i].x;
				iniy = submallas[i].y;
				finx = inix + submallas[i].z;
				finy = iniy + submallas[i].w;
				if ((x >= inix) && (x < finx) && (y >= iniy) && (y < finy)) {
					encontrado = true;
					pos += (y-iniy+2-datosClusterCPU[i].iniy)*(datosClusterCPU[i].numVolx+4) + (x-inix+2-datosClusterCPU[i].inix);
				}
				else {
					pos += (datosClusterCPU[i].numVolx + 4)*(datosClusterCPU[i].numVoly + 4);
				}
			}
			else {
				pos += (datosClusterCPU[i].numVolx + 4)*(datosClusterCPU[i].numVoly + 4);
			}
		}
		i++;
	}

	if (! encontrado) pos = -1;

	return pos;
}

int4 obtenerIndicePunto(int num_niveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double lon, double lat)
{
	int i, j, l, s;
	int pos_ini, inix, iniy;
	int4 pos;
	double *longitud, *latitud;
	int nvx, nvy;
	int nvxTotal, nvyTotal;
	bool encontrado;
	// En submallasNivel[l][s].w está num_voly_total
	// En datosClusterCPU[l][s].numVoly está num_voly del cluster

	pos.x = pos.y = pos.z = pos.w = -1;
	encontrado = false;
	l = num_niveles-1;
	while ((l >= 0) && (! encontrado)) {
		s = 0;
		pos_ini = 0;
		while ((s < numSubmallasNivel[l]) && (! encontrado)) {
			if (datosClusterCPU[l][s].iniy != -1) {
				longitud = datosNivel[l][s].longitud;
				latitud = datosNivel[l][s].latitud;
				nvxTotal = submallasNivel[l][s].z;
				nvyTotal = submallasNivel[l][s].w;
				nvx = datosClusterCPU[l][s].numVolx;
				nvy = datosClusterCPU[l][s].numVoly;
				// Buscamos el índice en longitud
				if (lon >= longitud[0]) {
					i = 0;
					while ((i < nvxTotal) && (! encontrado))  {
						if (longitud[i] >= lon) {
							encontrado = true;
							if (fabs(lon - longitud[max(i-1,0)]) < fabs(lon - longitud[i]))
								i = max(i-1,0);
						}
						else i++;
					}
					if (encontrado) {
						// Si la longitud está dentro del dominio, buscamos el índice en latitud
						encontrado = false;
						if (lat >= latitud[0]) {
							j = 0;
							while ((j < nvyTotal) && (! encontrado))  {
								if (latitud[j] >= lat) {
									encontrado = true;
									if (fabs(lat - latitud[max(j-1,0)]) < fabs(lat - latitud[j]))
										j = max(j-1,0);
								}
								else j++;
							}
							inix = datosClusterCPU[l][s].inix;
							iniy = datosClusterCPU[l][s].iniy;
							if (encontrado && (j >= iniy) && (j < iniy+nvy) && (i >= inix) && (i < inix+nvx)) {
								// x: nivel, y: posición en datosVolumenes[nivel], z: submalla, w: coordenada y de la submalla
								pos.x = l;
								pos.y = pos_ini + (j-iniy+2)*(nvx+4) + (i-inix+2);
								pos.z = s;
								pos.w = j;
							}
						}
					}
				}
				pos_ini += (nvx + 4)*(nvy + 4);
			}
			s++;
		}
		l--;
	}

	return pos;
}

// Se utiliza al obtener las coordenadas del nivel 0 de la esquina superior izquierda de una submalla
// que tiene la deformación de Okada. También se usa al asignar submallasDeformacion, para obtener
// la posición del centro de la ventana de computación
void obtenerIndicePuntoNivel0(tipoDatosSubmalla *datosNivel0, int4 submallaNivel0, double lon,
		double lat, int *inix, int *iniy)
{
	int i, j;
	double *longitud, *latitud;
	int nvx, nvy;
	bool encontrado;

	encontrado = false;
	longitud = datosNivel0->longitud;
	latitud = datosNivel0->latitud;
	nvx = submallaNivel0.z;
	nvy = submallaNivel0.w;
	// Buscamos el índice en longitud
	if (lon >= longitud[0]) {
		i = 0;
		while ((i < nvx) && (! encontrado))  {
			if (longitud[i] >= lon) {
				encontrado = true;
				if (fabs(lon - longitud[max(i-1,0)]) < fabs(lon - longitud[i]))
					i = max(i-1,0);
			}
			else i++;
		}
		if (encontrado) {
			// Si la longitud está dentro del dominio, buscamos el índice en latitud
			encontrado = false;
			if (lat >= latitud[0]) {
				j = 0;
				while ((j < nvy) && (! encontrado))  {
					if (latitud[j] >= lat) {
						encontrado = true;
						if (fabs(lat - latitud[max(j-1,0)]) < fabs(lat - latitud[j]))
							j = max(j-1,0);
					}
					else j++;
				}
			}
		}
		if (encontrado) {
			*inix = i;
			*iniy = j;
		}
	}
}

// Se usa al procesar los puntos de la serie de tiempos, para saber si el punto (lon,lat) ya estaba
// en la lista de puntos. Devuelve su posición o -1 si no está
int obtenerPosEnPuntosGuardado(double *lonPuntos, double *latPuntos, int num_puntos, double lon, double lat)
{
#if (PREPROCESS_POINTS)
	int i;
	bool encontrado = false;

	i = 0;
	while ((i < num_puntos) && (! encontrado)) {
		if ((fabs(lonPuntos[i] - lon) < 1e-6) && (fabs(latPuntos[i] - lat) < 1e-6))
			encontrado = true;
		else i++;
	}

	return (encontrado ? i : -1);
#else
	return -1;
#endif
}

void copiarDeformacionEnNivel0(double **deformacionNivel0, double *deformacionAcumNivel0, float *datosDef, int num_falla,
		int4 submallaDef, double H)
{
	int i, j, pos;
	int nvx, nvy;
	double def;

	nvx = submallaDef.z;
	nvy = submallaDef.w;
	// La deformación en datosDef es una deformación acumulada. Para cada deformación almacenamos su resta de la deformación
	// anterior, menos para la primera deformación, que se almacena tal cual
	if (num_falla == 0) {
		for (j=0; j<nvy; j++) {
			for (i=0; i<nvx; i++) {
				pos = j*nvx + i;
				def = ((double) datosDef[pos])/H;
				deformacionNivel0[num_falla][pos] = def;
				deformacionAcumNivel0[pos] = def;
			}
		}
	}
	else {
		// Restamos la deformación anterior por ser deformaciones acumuladas (guardamos la deformación actual)
		for (j=0; j<nvy; j++) {
			for (i=0; i<nvx; i++) {
				pos = j*nvx + i;
				def = ((double) datosDef[pos])/H;
				deformacionNivel0[num_falla][pos] = def - deformacionAcumNivel0[pos];
				deformacionAcumNivel0[pos] = def;
			}
		}
	}
}

// Asigna inix, numVolx, iniy, numVoly, numVolumenes, iniLeer y numVolsLeer en datosClusterCPU[nivel][submalla]
void obtenerDatosClusterSubmalla(TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int nivel, int submalla,
			int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel,
			int2 *iniGlobalSubmalla, int *ratioRefAcumNivel)
{
	int i, j;
	int inix_n0, finx_n0, numVolx_n0;
	int inix_sub, finx_sub, numVolx_sub;
	int iniy_n0, finy_n0, numVoly_n0;
	int iniy_sub, finy_sub, numVoly_sub;
	int ratio;

	// PROCESAMIENTO EN LA DIMENSIÓN X
	// Calculamos inix_sub, finx_sub, numVolx_sub, inix_n0, finx_n0 y numVolx_n0 en coordenadas
	// de la resolución de la submalla
	inix_sub = submallasNivel[nivel][submalla].x;
	ratio = ratioRefNivel[nivel];
	j = submallaNivelSuperior[nivel][submalla];
	for (i=nivel-1; i>=0; i--) {
		inix_sub += ratio*submallasNivel[i][j].x;
		ratio *= ratioRefNivel[i];
		j = submallaNivelSuperior[i][j];
	}
	numVolx_sub = submallasNivel[nivel][submalla].z;
	finx_sub = inix_sub + numVolx_sub - 1;

	inix_n0 = ratio*datosClusterCPU[0][0].inix;
	numVolx_n0 = ratio*datosClusterCPU[0][0].numVolx;
	finx_n0 = inix_n0 + numVolx_n0 - 1;

	if ((inix_sub <= finx_n0) && (finx_sub >= inix_n0)) {
		// La submalla está total o parcialmente dentro del subdominio
		if (inix_sub >= inix_n0) {
			// La submalla empieza dentro del subdominio
			datosClusterCPU[nivel][submalla].inix = 0;
			datosClusterCPU[nivel][submalla].iniLeerX = 0;
			if (finx_sub <= finx_n0) {
				datosClusterCPU[nivel][submalla].numVolx = numVolx_sub;
				datosClusterCPU[nivel][submalla].numVolsLeerX = numVolx_sub;
			}
			else {
				datosClusterCPU[nivel][submalla].numVolx = finx_n0 - inix_sub + 1;
				datosClusterCPU[nivel][submalla].numVolsLeerX = finx_n0 - inix_sub + 3;
			}
		}
		else {
			// La submalla empieza antes del subdominio
			datosClusterCPU[nivel][submalla].inix = inix_n0 - inix_sub;
			datosClusterCPU[nivel][submalla].iniLeerX = inix_n0 - inix_sub - 2;
			if (finx_sub <= finx_n0) {
				datosClusterCPU[nivel][submalla].numVolx = finx_sub - inix_n0 + 1;
				datosClusterCPU[nivel][submalla].numVolsLeerX = finx_sub - inix_n0 + 3;
			}
			else {
				datosClusterCPU[nivel][submalla].numVolx = finx_n0 - inix_n0 + 1;
				datosClusterCPU[nivel][submalla].numVolsLeerX = finx_n0 - inix_n0 + 5;
			}
		}
	}
	else {
		// La submalla está fuera del subdominio en la dimensión X
		datosClusterCPU[nivel][submalla].inix = -1;
	}

	// PROCESAMIENTO EN LA DIMENSIÓN Y
	// Calculamos iniy_sub, finy_sub, numVoly_sub, iniy_n0, finy_n0 y numVoly_n0 en coordenadas
	// de la resolución de la submalla
	iniy_sub = submallasNivel[nivel][submalla].y;
	ratio = ratioRefNivel[nivel];
	j = submallaNivelSuperior[nivel][submalla];
	for (i=nivel-1; i>=0; i--) {
		iniy_sub += ratio*submallasNivel[i][j].y;
		ratio *= ratioRefNivel[i];
		j = submallaNivelSuperior[i][j];
	}
	numVoly_sub = submallasNivel[nivel][submalla].w;
	finy_sub = iniy_sub + numVoly_sub - 1;

	iniy_n0 = ratio*datosClusterCPU[0][0].iniy;
	numVoly_n0 = ratio*datosClusterCPU[0][0].numVoly;
	finy_n0 = iniy_n0 + numVoly_n0 - 1;

	if ((iniy_sub <= finy_n0) && (finy_sub >= iniy_n0)) {
		// La submalla está total o parcialmente dentro del subdominio
		if (iniy_sub >= iniy_n0) {
			// La submalla empieza dentro del subdominio
			datosClusterCPU[nivel][submalla].iniy = 0;
			datosClusterCPU[nivel][submalla].iniLeerY = 0;
			if (finy_sub <= finy_n0) {
				datosClusterCPU[nivel][submalla].numVoly = numVoly_sub;
				datosClusterCPU[nivel][submalla].numVolsLeerY = numVoly_sub;
			}
			else {
				datosClusterCPU[nivel][submalla].numVoly = finy_n0 - iniy_sub + 1;
				datosClusterCPU[nivel][submalla].numVolsLeerY = finy_n0 - iniy_sub + 3;
			}
		}
		else {
			// La submalla empieza antes del subdominio
			datosClusterCPU[nivel][submalla].iniy = iniy_n0 - iniy_sub;
			datosClusterCPU[nivel][submalla].iniLeerY = iniy_n0 - iniy_sub - 2;
			if (finy_sub <= finy_n0) {
				datosClusterCPU[nivel][submalla].numVoly = finy_sub - iniy_n0 + 1;
				datosClusterCPU[nivel][submalla].numVolsLeerY = finy_sub - iniy_n0 + 3;
			}
			else {
				datosClusterCPU[nivel][submalla].numVoly = finy_n0 - iniy_n0 + 1;
				datosClusterCPU[nivel][submalla].numVolsLeerY = finy_n0 - iniy_n0 + 5;
			}
		}
		datosClusterCPU[nivel][submalla].numVolumenes = (datosClusterCPU[nivel][submalla].numVolx)*(datosClusterCPU[nivel][submalla].numVoly);
	}
	else {
		// La submalla está fuera del subdominio en la dimensión Y
		datosClusterCPU[nivel][submalla].iniy = -1;
	}

	if ((datosClusterCPU[nivel][submalla].inix == -1) || (datosClusterCPU[nivel][submalla].iniy == -1)) {
		// La submalla está fuera del subdominio en la dimensión X o en la Y, o sea,
		// está totalmente fuera del subdominio
		datosClusterCPU[nivel][submalla].inix = -1;
		datosClusterCPU[nivel][submalla].iniy = -1;
		datosClusterCPU[nivel][submalla].numVolx = -1;
		datosClusterCPU[nivel][submalla].numVoly = -1;
		datosClusterCPU[nivel][submalla].numVolumenes = -1;
		datosClusterCPU[nivel][submalla].iniLeerX = -1;
		datosClusterCPU[nivel][submalla].numVolsLeerX = -1;
		datosClusterCPU[nivel][submalla].iniLeerY = -1;
		datosClusterCPU[nivel][submalla].numVolsLeerY = -1;
	}

	// Asignamos iniGlobalSubmalla y el ratio acumulado
	iniGlobalSubmalla->x = inix_sub;
	iniGlobalSubmalla->y = iniy_sub;
	ratioRefAcumNivel[nivel] = ratio;
}

// Devuelve 0 si todo ha ido bien, 1 si ha habido algún error
int cargarDatosProblema(string fich_ent, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], string &nombre_bati,
		string prefijo[MAX_LEVELS][MAX_GRIDS_LEVEL], int *no_hidros, int *numPesosJacobi, int *numNiveles, int *okada_flag, int *kajiura_flag,
		double *depth_kajiura, string &fich_okada, int *numFaults, int *numEstadosDefDinamica, double *LON_C, double *LAT_C, double *DEPTH_C,
		double *FAULT_L, double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *defTime, double2 LON_LAT_v[MAX_FAULTS][3],
		double DEPTH_v[MAX_FAULTS][4], double2 vc[MAX_FAULTS][4], double *LONCTRI, double *LATCTRI, double SLIPVEC[MAX_FAULTS][3],
		double **deformacionNivel0, double **deformacionAcumNivel0, string *fich_def, double *lonGauss, double *latGauss, double *heightGauss,
		double *sigmaGauss, float *batiOriginal[MAX_LEVELS], double2 **datosVolumenesNivel_1, double3 **datosVolumenesNivel_2, double **datosPnh,
		double **datosGRD, tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel,
		int2 iniGlobalSubmallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int4 *submallasDeformacion, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int64_t *numVolumenesNivel, int64_t *numVerticesNivel, int posSubmallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int **posCopiaNivel,
		bool **refinarNivel, int *leer_fichero_puntos, double2 **bufferEnviosMPISupInf_1, double2 **bufferEnviosMPIIzqDer_1,
		double3 **bufferEnviosMPISupInf_2, double3 **bufferEnviosMPIIzqDer_2, double **bufferEnviosMPISupInf_P, double **bufferEnviosMPIIzqDer_P,
		int4 **posicionesVolumenesGuardado, int *numPuntosGuardarAnt, int *numPuntosGuardarTotal, double **lonPuntos, double **latPuntos,
		int *numVolxTotalNivel0, int *numVolyTotalNivel0, double *Hmin_global, double *borde_sup, double *borde_inf, double *borde_izq,
		double *borde_der, int *tam_spongeSup, int *tam_spongeInf, int *tam_spongeIzq, int *tam_spongeDer, double *tiempo_tot,
		double *tiempoGuardarNetCDF, VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL], double *tiempoGuardarSeries,
		double *CFL, int *tipo_friccion, string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL], double **friccionesNivel, double *vmax,
		double *epsilon_h, int *continuar_simulacion, double *tiempo_continuar, int *numEstadoNetCDF, int *ratioRefNivel,
		int *ratioRefAcumNivel, bool *haySubmallasAdyacentesNivel, double *difh_at, double *L, double *H, double *U, double *T,
		int num_procs, int *num_procsX, int *num_procsY, MPI_Comm *comm_cartesiano, int id_hebra)
{
	int i, j, k, l, m;
	int pos, pos_ini;
	int4 posPunto;
	int nvx, nvy;
	int nvxTotal, nvyTotal;
	int inix, iniy;
	int inixP, iniyP;
	int ratio;
	int ini_leerx, num_vols_leerx;
	int ini_leery, num_vols_leery;
	int numVolumenesTotalNivel0;
	int sizei = sizeof(int);
	int sizef = sizeof(float);
	int error_formato;
	tipoDatosSubmalla *tds;
	double val;
	double grad2rad = M_PI/180.0;
	double Hmin;
	double cfb, factor;
	// Stream para leer los ficheros de fricciones, puntos y fallas de Okada
	ifstream fich_fric, fich_ptos, fich_fallas;
	// Directorio donde se encuentran los ficheros de datos
	string directorio;
	string fich_topo[MAX_LEVELS][MAX_GRIDS_LEVEL];
	string fich_est[MAX_LEVELS][MAX_GRIDS_LEVEL];
	string fich_puntos, fich_lb;
	char nombre_fich[256];
	double lon, lat;
	bool restar_fila;
	// datosGRD se usará para almacenar tanto el contenido de los ficheros GRD (float)
	// como el contenido de una simulación guardada (double). datosGRDFloat es datosGRD
	// pero leyendo el vector como floats
	float *datosGRD_float;
	int64_t tam_datosGRD;
	int64_t tam_datosMPIDouble, tamf_Double, tamc_Double;
	int64_t tam_datosMPIDouble2, tamf_Double2, tamc_Double2;
	int64_t tam_datosMPIDouble3, tamf_Double3, tamc_Double3;
	double *longitudDeformacion = NULL;
	double *latitudDeformacion = NULL;
	bool hayQueCrearAlgunFicheroNetCDF = false;
	bool leer_difh_at = false;
	// Variables para la ventana de computación de Okada
	int usar_ventana;
	double lon_centro, lat_centro, radioComp;
	double incx, incy;
	int centrox, centroy;
	int radiox, radioy;
	// Variables para MPI
	MPI_Comm comunicadores[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int id_hebra_cart;
	int id_hebraX, id_hebraY;
	int coord_proc[2];
	int dims[2];
	int periods[2];

	// Ponemos en directorio el directorio donde están los ficheros de datos
	i = fich_ent.find_last_of("/");
	if (i > -1) {
		// Directorios indicados con '/' (S.O. distinto de windows)
		directorio = fich_ent.substr(0,i)+"/";
	}
	else {
		i = fich_ent.find_last_of("\\");
		if (i > -1) {
			// Directorios indicados con '\' (S.O. Windows)
			directorio = fich_ent.substr(0,i)+"\\";
		}
		else {
			// No se ha especificado ningún directorio para los ficheros de datos
			directorio = "";
		}
	}

	// Leemos los datos del problema del fichero de entrada
	*kajiura_flag = 0;
	*depth_kajiura = 0.0;
	*numFaults = 0;
	*numEstadosDefDinamica = 0;
	*difh_at = -1.0;
	usar_ventana = 0;
	ifstream fich(fich_ent.c_str());
	obtenerSiguienteDato<string>(fich, nombre_bati);
	obtenerSiguienteDato<int>(fich, *no_hidros);
	if (*no_hidros == 1) {
		obtenerSiguienteDato<int>(fich, *numPesosJacobi);
		if ((*numPesosJacobi != 1) && (*numPesosJacobi != 16) && (*numPesosJacobi != 27)) {
			if (id_hebra == 0)
				cerr << "Error: The number of Jacobi weights should be 1, 16 or 27" << endl;
			fich.close();
			return 1;
		}
	}
	obtenerSiguienteDato<string>(fich, fich_topo[0][0]);
	obtenerSiguienteDato<int>(fich, *okada_flag);
	if ((*okada_flag != SEA_SURFACE_FROM_FILE) && (*okada_flag != OKADA_STANDARD) && (*okada_flag != OKADA_STANDARD_FROM_FILE) &&
		(*okada_flag != OKADA_TRIANGULAR) && (*okada_flag != OKADA_TRIANGULAR_FROM_FILE) && (*okada_flag != DEFORMATION_FROM_FILE) &&
		(*okada_flag != DYNAMIC_DEFORMATION) && (*okada_flag != GAUSSIAN))
	{
		if (id_hebra == 0) {
			cerr << "Error: The initialization flag should be " << SEA_SURFACE_FROM_FILE << ", " << OKADA_STANDARD << ", ";
			cerr << OKADA_STANDARD_FROM_FILE << ", " << OKADA_TRIANGULAR << ", " << OKADA_TRIANGULAR_FROM_FILE << ", ";
			cerr << DEFORMATION_FROM_FILE << ", " << DYNAMIC_DEFORMATION << " or " << GAUSSIAN << endl;
		}
		fich.close();
		return 1;
	}
	if (*okada_flag == SEA_SURFACE_FROM_FILE) {
		// Leer el estado inicial de fichero
		obtenerSiguienteDato<string>(fich, fich_est[0][0]);
	}
	else if (*okada_flag == OKADA_STANDARD) {
		// Aplicar Okada estándar
		obtenerSiguienteDato<int>(fich, *kajiura_flag);
		if (*kajiura_flag == 1)
			obtenerSiguienteDato<double>(fich, *depth_kajiura);
		obtenerSiguienteDato<int>(fich, *numFaults);
		if (*numFaults > MAX_FAULTS) {
			if (id_hebra == 0)
				cerr << "Error: The maximum number of faults is " << MAX_FAULTS << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerDatosFallaOkadaStandard(fich, defTime[i], LON_C[i], LAT_C[i], DEPTH_C[i], FAULT_L[i],
				FAULT_W[i], STRIKE[i], DIP[i], RAKE[i], SLIP[i]);
		}
		obtenerSiguienteDato<int>(fich, usar_ventana);
		if (usar_ventana == 1) {
			obtenerDatosVentanaComputacion(fich, lon_centro, lat_centro, radioComp);
		}
	}
	else if (*okada_flag == OKADA_STANDARD_FROM_FILE) {
		// Aplicar Okada estándar leyendo de fichero
		obtenerSiguienteDato<int>(fich, *kajiura_flag);
		if (*kajiura_flag == 1)
			obtenerSiguienteDato<double>(fich, *depth_kajiura);
		obtenerSiguienteDato<string>(fich, fich_okada);
		if (! existeFichero(directorio+fich_okada)) {
			if (id_hebra == 0)
				cerr << "Error: File '" << directorio+fich_okada << "' not found" << endl;
			fich.close();
			return 1;
		}
		obtenerSiguienteDato<int>(fich, usar_ventana);
		if (usar_ventana == 1) {
			obtenerDatosVentanaComputacion(fich, lon_centro, lat_centro, radioComp);
		}
	}
	else if (*okada_flag == OKADA_TRIANGULAR) {
		// Aplicar Okada triangular
		obtenerSiguienteDato<int>(fich, *kajiura_flag);
		if (*kajiura_flag == 1)
			obtenerSiguienteDato<double>(fich, *depth_kajiura);
		obtenerSiguienteDato<int>(fich, *numFaults);
		if (*numFaults > MAX_FAULTS) {
			if (id_hebra == 0)
				cerr << "Error: The maximum number of faults is " << MAX_FAULTS << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerDatosFallaOkadaTriangular(fich, defTime[i], LON_LAT_v[i][0].x, LON_LAT_v[i][0].y, DEPTH_v[i][0], LON_LAT_v[i][1].x,
				LON_LAT_v[i][1].y, DEPTH_v[i][1], LON_LAT_v[i][2].x, LON_LAT_v[i][2].y, DEPTH_v[i][2], RAKE[i], SLIP[i]);
			if ((DEPTH_v[i][0] < 0.0) || (DEPTH_v[i][1] < 0.0) || (DEPTH_v[i][2] < 0.0)) {
				if (id_hebra == 0)
					cerr << "Error: The Okada depths should be greater or equal to 0" << endl;
				fich.close();
				return 1;
			}
			obtenerDatosOkadaTriangular(LON_LAT_v[i], RAKE[i], SLIP[i], vc[i], DEPTH_v[i], LATCTRI+i, LONCTRI+i, SLIPVEC[i]);
		}
		obtenerSiguienteDato<int>(fich, usar_ventana);
		if (usar_ventana == 1) {
			obtenerDatosVentanaComputacion(fich, lon_centro, lat_centro, radioComp);
		}
	}
	else if (*okada_flag == OKADA_TRIANGULAR_FROM_FILE) {
		// Aplicar Okada triangular leyendo de fichero
		obtenerSiguienteDato<int>(fich, *kajiura_flag);
		if (*kajiura_flag == 1)
			obtenerSiguienteDato<double>(fich, *depth_kajiura);
		obtenerSiguienteDato<string>(fich, fich_okada);
		if (! existeFichero(directorio+fich_okada)) {
			if (id_hebra == 0)
				cerr << "Error: File '" << directorio+fich_okada << "' not found" << endl;
			fich.close();
			return 1;
		}
		obtenerSiguienteDato<int>(fich, usar_ventana);
		if (usar_ventana == 1) {
			obtenerDatosVentanaComputacion(fich, lon_centro, lat_centro, radioComp);
		}
	}
	else if (*okada_flag == DEFORMATION_FROM_FILE) {
		// Leer las deformaciones de Okada de fichero
		obtenerSiguienteDato<int>(fich, *kajiura_flag);
		if (*kajiura_flag == 1)
			obtenerSiguienteDato<double>(fich, *depth_kajiura);
		obtenerSiguienteDato<int>(fich, *numFaults);
		if (*numFaults > MAX_FAULTS) {
			if (id_hebra == 0)
				cerr << "Error: The maximum number of faults is " << MAX_FAULTS << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerDatosFallaFichero(fich, defTime[i], fich_def[i]);
			if (! existeFichero(directorio+fich_def[i])) {
				if (id_hebra == 0)
					cerr << "Error: File '" << directorio+fich_def[i] << "' not found" << endl;
				fich.close();
				return 1;
			}
		}
	}
	else if (*okada_flag == DYNAMIC_DEFORMATION) {
		// Leer una deformación dinámica de fichero
		obtenerSiguienteDato<int>(fich, *kajiura_flag);
		if (*kajiura_flag == 1)
			obtenerSiguienteDato<double>(fich, *depth_kajiura);
		obtenerSiguienteDato<int>(fich, *numFaults);
		if (*numFaults != 1) {
			if (id_hebra == 0)
				cerr << "Error: Only one dynamic deformation is supported" << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerSiguienteDato<string>(fich, fich_def[i]);
			if (! existeFichero(directorio+fich_def[i])) {
				if (id_hebra == 0)
					cerr << "Error: File '" << directorio+fich_def[i] << "' not found" << endl;
				fich.close();
				return 1;
			}
			// submallaDeformacion.x y submallaDeformacion.y se asignarán después, al leer los estados
			// de la deformación dinámica
			leerTamanoYNumEstadosDefDinamicaGRD((directorio+fich_def[i]).c_str(), &(submallasDeformacion[i].z),
				&(submallasDeformacion[i].w), numEstadosDefDinamica);
			if (*numEstadosDefDinamica > MAX_FAULTS) {
				if (id_hebra == 0)
					cerr << "Error: The maximum number of states in a dynamic deformation is " << MAX_FAULTS << endl;
				fich.close();
				return 1;
			}
		}
	}
	else if (*okada_flag == GAUSSIAN) {
		// Construir gaussiana
		obtenerDatosGaussiana(fich, *lonGauss, *latGauss, *heightGauss, *sigmaGauss);
	}
	obtenerSiguienteDato<string>(fich, prefijo[0][0]);
	obtenerDatosGuardadoSubmalla(fich, guardarVariables[0][0].eta, guardarVariables[0][0].eta_max, guardarVariables[0][0].velocidades,
		guardarVariables[0][0].velocidades_max, guardarVariables[0][0].modulo_velocidades, guardarVariables[0][0].modulo_velocidades_max,
		guardarVariables[0][0].modulo_caudales_max, guardarVariables[0][0].presion_no_hidrostatica, guardarVariables[0][0].flujo_momento,
		guardarVariables[0][0].flujo_momento_max, guardarVariables[0][0].tiempos_llegada);
	if (! no_hidros)
		guardarVariables[0][0].presion_no_hidrostatica = 0;
	if (guardarVariables[0][0].tiempos_llegada != 0)
		leer_difh_at = true;
	if (! existeFichero(directorio+fich_topo[0][0])) {
		if (id_hebra == 0)
			cerr << "Error: File '" << directorio+fich_topo[0][0] << "' not found" << endl;
		fich.close();
		return 1;
	}
	if (*okada_flag == SEA_SURFACE_FROM_FILE) {
		if (! existeFichero(directorio+fich_est[0][0])) {
			if (id_hebra == 0)
				cerr << "Error: File '" << directorio+fich_est[0][0] << "' not found" << endl;
			fich.close();
			return 1;
		}
	}
	leerTamanoSubmallaGRD((directorio+fich_topo[0][0]).c_str(), 0, 0, numVolxTotalNivel0, numVolyTotalNivel0);
	datosClusterCPU[0][0].numVoly = *numVolyTotalNivel0;
	// Los datos topográficos se leerán después

	if ((*numVolxTotalNivel0 < 2) || (*numVolyTotalNivel0 < 2)) {
		if (id_hebra == 0)
			cerr << "Error: Mesh size too small. The number of rows and columns should be >= 2" << endl;
		fich.close();
		return 1;
	}

	numSubmallasNivel[0] = 1;
	ratioRefNivel[0] = 1;
	*tiempoGuardarSeries = -1.0;
	obtenerSiguienteDato<int>(fich, *numNiveles);
	if (*numNiveles > MAX_LEVELS) {
		if (id_hebra == 0)
			cerr << "Error: The maximum number of levels is " << MAX_LEVELS << endl;
		fich.close();
		return 1;
	}
	for (l=1; l<(*numNiveles); l++) {
		obtenerSiguienteDato<int>(fich, ratioRefNivel[l]);
		if ((ratioRefNivel[l] != 2) && (ratioRefNivel[l] != 4) && (ratioRefNivel[l] != 8) && (ratioRefNivel[l] != 16)) {
			if (id_hebra == 0)
				cerr << "Error: The refinement ratio should be 2, 4, 8 or 16" << endl;
			fich.close();
			return 1;
		}
		// Leemos el tamaño de las submallas
		obtenerSiguienteDato<int>(fich, numSubmallasNivel[l]);
		if (numSubmallasNivel[l] > MAX_GRIDS_LEVEL) {
			if (id_hebra == 0)
				cerr << "Error: The maximum number of grids per level is " << MAX_GRIDS_LEVEL << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<numSubmallasNivel[l]; i++) {
			obtenerSiguienteDato<string>(fich, fich_topo[l][i]);
			if (*okada_flag == SEA_SURFACE_FROM_FILE) {
				obtenerSiguienteDato<string>(fich, fich_est[l][i]);
				if (! existeFichero(directorio+fich_est[l][i])) {
					if (id_hebra == 0)
						cerr << "Error: File '" << directorio+fich_est[l][i] << "' not found" << endl;
					fich.close();
					return 1;
				}
			}
			if (! existeFichero(directorio+fich_topo[l][i])) {
				if (id_hebra == 0)
					cerr << "Error: File '" << directorio+fich_topo[l][i] << "' not found" << endl;
				fich.close();
				return 1;
			}
			obtenerSiguienteDato<string>(fich, prefijo[l][i]);
			obtenerDatosGuardadoSubmalla(fich, guardarVariables[l][i].eta, guardarVariables[l][i].eta_max, guardarVariables[l][i].velocidades,
				guardarVariables[l][i].velocidades_max, guardarVariables[l][i].modulo_velocidades, guardarVariables[l][i].modulo_velocidades_max,
				guardarVariables[l][i].modulo_caudales_max, guardarVariables[l][i].presion_no_hidrostatica, guardarVariables[l][i].flujo_momento,
				guardarVariables[l][i].flujo_momento_max, guardarVariables[l][i].tiempos_llegada);
			if (! no_hidros)
				guardarVariables[l][i].presion_no_hidrostatica = 0;
			if (guardarVariables[l][i].tiempos_llegada != 0)
				leer_difh_at = true;
			leerTamanoSubmallaGRD((directorio+fich_topo[l][i]).c_str(), l, i, &(submallasNivel[l][i].z), &(submallasNivel[l][i].w));
		}
	}
	obtenerSiguienteDato<double>(fich, *borde_sup);
	obtenerSiguienteDato<double>(fich, *borde_inf);
	obtenerSiguienteDato<double>(fich, *borde_izq);
	obtenerSiguienteDato<double>(fich, *borde_der);
	if (((fabs(*borde_sup-1.0) > EPSILON) && (fabs(*borde_sup+1.0) > EPSILON)) ||
		((fabs(*borde_inf-1.0) > EPSILON) && (fabs(*borde_inf+1.0) > EPSILON)) ||
		((fabs(*borde_izq-1.0) > EPSILON) && (fabs(*borde_izq+1.0) > EPSILON)) ||
		((fabs(*borde_der-1.0) > EPSILON) && (fabs(*borde_der+1.0) > EPSILON))) {
		if (id_hebra == 0)
			cerr << "Error: The border conditions should be 1 or -1" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<double>(fich, *tiempo_tot);
	obtenerTiemposGuardadoNetCDFNiveles(fich, *numNiveles, tiempoGuardarNetCDF);
	if (hayErrorEnTiemposGuardarNetCDF(*numNiveles, tiempoGuardarNetCDF)) {
		if (id_hebra == 0)
			cerr << "Error: The NetCDF saving time of level 0 should be multiple of the NetCDF saving times of the other levels" << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *leer_fichero_puntos);
	if (*leer_fichero_puntos == 1) {
		obtenerSiguienteDato<string>(fich, fich_puntos);
		obtenerSiguienteDato<double>(fich, *tiempoGuardarSeries);
		if (! existeFichero(directorio+fich_puntos)) {
			if (id_hebra == 0)
				cerr << "Error: File '" << directorio+fich_puntos << "' not found" << endl;
			fich.close();
			return 1;
		}
		if (*tiempoGuardarSeries < 0.0) {
			if (id_hebra == 0)
				cerr << "Error: The saving time of the time series should be >= 0" << endl;
			fich.close();
			return 1;
		}
	}
	obtenerSiguienteDato<double>(fich, *CFL);
	obtenerSiguienteDato<double>(fich, *epsilon_h);
	obtenerSiguienteDato<int>(fich, *tipo_friccion);
	if (*tipo_friccion == FIXED_FRICTION) {
		obtenerSiguienteDato<double>(fich, cfb);
	}
	else if (*tipo_friccion == VARIABLE_FRICTION_0) {
		obtenerSiguienteDato<string>(fich, fich_friccion[0][0]);
		if (! existeFichero(directorio+fich_friccion[0][0])) {
			if (id_hebra == 0)
				cerr << "Error: File '" << directorio+fich_friccion[0][0] << "' not found" << endl;
			fich.close();
			return 1;
		}
	}
	else if (*tipo_friccion == VARIABLE_FRICTION_ALL) {
		for (l=0; l<(*numNiveles); l++) {
			for (i=0; i<numSubmallasNivel[l]; i++) {
				obtenerSiguienteDato<string>(fich, fich_friccion[l][i]);
				if (! existeFichero(directorio+fich_friccion[l][i])) {
					if (id_hebra == 0)
						cerr << "Error: File '" << directorio+fich_friccion[l][i] << "' not found" << endl;
					fich.close();
					return 1;
				}
			}
		}
	}
	else {
		if (id_hebra == 0)
			cerr << "Error: The friction type flag should be " << FIXED_FRICTION << ", " << VARIABLE_FRICTION_0 << " or " << VARIABLE_FRICTION_ALL << endl;
		fich.close();
		return 1;
	}
	obtenerSiguienteDato<double>(fich, *vmax);
	obtenerSiguienteDato<double>(fich, *L);
	obtenerSiguienteDato<double>(fich, *H);
	if (leer_difh_at) {
		*difh_at = *epsilon_h;
		obtenerSiguienteDato<double>(fich, *difh_at);
		*difh_at /= *H;
	}
	*U = sqrt(9.81*(*H));
	*T = (*L)/(*U);
	*tiempo_tot /= *T;
	*tiempoGuardarSeries /= *T;
	*vmax /= (*U);
	*epsilon_h /= *H;
	*depth_kajiura /= *H;
	for (l=0; l<(*numNiveles); l++) {
		if (tiempoGuardarNetCDF[l] >= 0.0)
			hayQueCrearAlgunFicheroNetCDF = true;
		tiempoGuardarNetCDF[l] /= *T;
		numEstadoNetCDF[l] = 0;
	}
	fich.close();

	// Tamaños del sponge layer en los cuatro bordes de la malla
	*tam_spongeIzq = ((fabs(*borde_izq-1.0) < EPSILON) ? SPONGE_SIZE : 0);
	*tam_spongeDer = ((fabs(*borde_der-1.0) < EPSILON) ? SPONGE_SIZE : 0);
	*tam_spongeSup = ((fabs(*borde_sup-1.0) < EPSILON) ? SPONGE_SIZE : 0);
	*tam_spongeInf = ((fabs(*borde_inf-1.0) < EPSILON) ? SPONGE_SIZE : 0);

	// Leemos las fallas de Okada si se cargan de fichero
	if ((*okada_flag == OKADA_STANDARD_FROM_FILE) || (*okada_flag == OKADA_TRIANGULAR_FROM_FILE)) {
		fich_fallas.open((directorio+fich_okada).c_str());
		fich_fallas >> (*numFaults);
		if (*numFaults > MAX_FAULTS) {
			if (id_hebra == 0)
				cerr << "Error: The maximum number of faults is " << MAX_FAULTS << endl;
			fich_fallas.close();
			return 1;
		}
		if (*okada_flag == OKADA_STANDARD_FROM_FILE) {
			for (i=0; i<(*numFaults); i++) {
				obtenerDatosFallaOkadaStandard(fich_fallas, defTime[i], LON_C[i], LAT_C[i], DEPTH_C[i], FAULT_L[i],
					FAULT_W[i], STRIKE[i], DIP[i], RAKE[i], SLIP[i]);
			}
		}
		else if (*okada_flag == OKADA_TRIANGULAR_FROM_FILE) {
			for (i=0; i<(*numFaults); i++) {
				obtenerDatosFallaOkadaTriangular(fich_fallas, defTime[i], LON_LAT_v[i][0].x, LON_LAT_v[i][0].y, DEPTH_v[i][0], LON_LAT_v[i][1].x,
					LON_LAT_v[i][1].y, DEPTH_v[i][1], LON_LAT_v[i][2].x, LON_LAT_v[i][2].y, DEPTH_v[i][2], RAKE[i], SLIP[i]);
				if ((DEPTH_v[i][0] < 0.0) || (DEPTH_v[i][1] < 0.0) || (DEPTH_v[i][2] < 0.0)) {
					if (id_hebra == 0)
						cerr << "Error: The Okada depths should be greater or equal to 0" << endl;
					fich_fallas.close();
					return 1;
				}
				obtenerDatosOkadaTriangular(LON_LAT_v[i], RAKE[i], SLIP[i], vc[i], DEPTH_v[i], LATCTRI+i, LONCTRI+i, SLIPVEC[i]);
			}
		}
		fich_fallas.close();
	}

	// Si tiempoGuardarNetCDF < 0 para todos los niveles, ponemos que no se guarde ninguna variable
	if (! hayQueCrearAlgunFicheroNetCDF) {
		for (l=0; l<(*numNiveles); l++) {
			for (i=0; i<numSubmallasNivel[l]; i++) {
				guardarVariables[l][i].eta = 0;
				guardarVariables[l][i].eta_max = 0;
				guardarVariables[l][i].velocidades = 0;
				guardarVariables[l][i].velocidades_max = 0;
				guardarVariables[l][i].modulo_velocidades = 0;
				guardarVariables[l][i].modulo_velocidades_max = 0;
				guardarVariables[l][i].modulo_caudales_max = 0;
				guardarVariables[l][i].presion_no_hidrostatica = 0;
				guardarVariables[l][i].flujo_momento = 0;
				guardarVariables[l][i].flujo_momento_max = 0;
				guardarVariables[l][i].tiempos_llegada = 0;
			}
		}
	}

	// Leemos el fichero de equilibrado de carga y asignamos los datos leídos
	if ((*numNiveles == 1) && (num_procs == 1))
		sprintf(nombre_fich, "%s_lb_01level_01proc.bin", nombre_bati.c_str());
	else if ((*numNiveles > 1) && (num_procs == 1))
		sprintf(nombre_fich, "%s_lb_%02dlevels_01proc.bin", nombre_bati.c_str(), *numNiveles);
	else if ((*numNiveles == 1) && (num_procs > 1))
		sprintf(nombre_fich, "%s_lb_01level_%02dprocs.bin", nombre_bati.c_str(), num_procs);
	else
		sprintf(nombre_fich, "%s_lb_%02dlevels_%02dprocs.bin", nombre_bati.c_str(), *numNiveles, num_procs);
	fich_lb = string(nombre_fich);
	if (! existeFichero(directorio+fich_lb)) {
		if (id_hebra == 0)
			cerr << "Error: File '" << directorio+fich_lb << "' not found" << endl;
		return 1;
	}
	fich.open((directorio+fich_lb).c_str(), ios::in | ios::binary);
	// Número de niveles
	fich.read((char *)&k, sizei);
	if (*numNiveles != k) {
		if (id_hebra == 0)
			cerr << "Error: The number of levels in file '" << directorio+fich_lb << "' do not match (" << k << " != " << (*numNiveles) << ")" << endl;
		fich.close();
		return 1;
	}
	for (l=0; l<(*numNiveles); l++) {
		// haySubmallasAdyacentesNivel
		fich.read((char *)&k, sizei);
		haySubmallasAdyacentesNivel[l] = ((k == 1) ? true : false);
		// Número de submallas
		fich.read((char *)&pos, sizei);
		if (numSubmallasNivel[l] != pos) {
			if (id_hebra == 0)
				cerr << "Error: The number of submeshes of level " << l << " in file '" << directorio+fich_lb << "' do not match (" << pos << " != " << numSubmallasNivel[l] << ")" << endl;
			fich.close();
			return 1;
		}
		for (j=0; j<pos; j++) {
			// Inicio x
			fich.read((char *)&k, sizei);
			submallasNivel[l][j].x = k;
			// Inicio y
			fich.read((char *)&k, sizei);
			submallasNivel[l][j].y = k;
			// Submalla superior
			fich.read((char *)&k, sizei);
			submallaNivelSuperior[l][j] = k;
		}
	}
	// Número de procesos en X e Y
	fich.read((char *)&k, sizei);
	*num_procsX = k;
	fich.read((char *)&k, sizei);
	*num_procsY = k;
	k = (*num_procsX)*(*num_procsY);
	if (num_procs != k) {
		if (id_hebra == 0)
			cerr << "Error: The number of processes in file '" << directorio+fich_lb << "' do not match (" << k << " != " << num_procs << ")" << endl;
		fich.close();
		return 1;
	}
	// Creamos el comunicador cartesiano y obtenemos el nuevo id_hebra_cart
	dims[0] = *num_procsX;
	dims[1] = *num_procsY;
	periods[0] = periods[1] = 0;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, comm_cartesiano);
	MPI_Comm_rank(*comm_cartesiano, &id_hebra_cart);
	MPI_Cart_coords(*comm_cartesiano, id_hebra_cart, 2, coord_proc);
	id_hebraX = coord_proc[0];
	id_hebraY = coord_proc[1];

	// Inicio x
	for (i=0; i<id_hebraX; i++) {  // Saltamos el inicio_x de los procesos anteriores a id_hebraX
		fich.read((char *)&k, sizei);
	}
	fich.read((char *)&k, sizei);  // Inicio_x del proceso id_hebra_cart
	datosClusterCPU[0][0].inix = k;
	if (id_hebraX == (*num_procsX)-1) {
		datosClusterCPU[0][0].numVolx = (*numVolxTotalNivel0) - datosClusterCPU[0][0].inix;
	}
	else {
		fich.read((char *)&k, sizei);  // Inicio_x del siguiente proceso en la dimensión X
		datosClusterCPU[0][0].numVolx = k - datosClusterCPU[0][0].inix;
	}
	for (i=id_hebraX+2; i<(*num_procsX); i++) {  // Saltamos el inicio_x del resto de procesos
		fich.read((char *)&k, sizei);
	}

	// Inicio y
	for (i=0; i<id_hebraY; i++) {  // Saltamos el inicio_y de los procesos anteriores a id_hebraY
		fich.read((char *)&k, sizei);
	}
	fich.read((char *)&k, sizei);  // Inicio_y del proceso id_hebra_cart
	datosClusterCPU[0][0].iniy = k;
	if (id_hebraY == (*num_procsY)-1) {
		datosClusterCPU[0][0].numVoly = (*numVolyTotalNivel0) - datosClusterCPU[0][0].iniy;
	}
	else {
		fich.read((char *)&k, sizei);  // Inicio_y del siguiente proceso en la dimensión Y
		datosClusterCPU[0][0].numVoly = k - datosClusterCPU[0][0].iniy;
	}
	datosClusterCPU[0][0].numVolumenes = (datosClusterCPU[0][0].numVolx) * (datosClusterCPU[0][0].numVoly);
	fich.close();

	// Inicio sin equilibrado
/*	datosClusterCPU[0][0].numVolx = (int) ceil((*numVolxTotalNivel0)/(*num_procsX));
	if (datosClusterCPU[0][0].numVolx % 2 != 0) {
		(datosClusterCPU[0][0].numVolx)++;
	}
	int numVolxOtrosNivel0 = datosClusterCPU[0][0].numVolx;
	if (id_hebraX == (*num_procsX)-1) {
		datosClusterCPU[0][0].numVolx = (*numVolxTotalNivel0) - ((*num_procsX)-1)*numVolxOtrosNivel0;
	}
	datosClusterCPU[0][0].inix = id_hebraX*numVolxOtrosNivel0;

	datosClusterCPU[0][0].numVoly = (int) ceil((*numVolyTotalNivel0)/(*num_procsY));
	if (datosClusterCPU[0][0].numVoly % 2 != 0) {
		(datosClusterCPU[0][0].numVoly)++;
	}
	int numVolyOtrosNivel0 = datosClusterCPU[0][0].numVoly;
	if (id_hebraY == (*num_procsY)-1) {
		datosClusterCPU[0][0].numVoly = (*numVolyTotalNivel0) - ((*num_procsY)-1)*numVolyOtrosNivel0;
	}
	datosClusterCPU[0][0].iniy = id_hebraY*numVolyOtrosNivel0;
	datosClusterCPU[0][0].numVolumenes = (datosClusterCPU[0][0].numVolx) * (datosClusterCPU[0][0].numVoly);*/
	// Fin sin equilibrado

	if (datosClusterCPU[0][0].numVolx < 2) {
		cerr << "Error in process " << id_hebra_cart << ": Submesh size too small. Please use less processes in the X dimension" << endl;
		return 1;
	}
	if (datosClusterCPU[0][0].numVoly < 2) {
		cerr << "Error in process " << id_hebra_cart << ": Submesh size too small. Please use less processes in the Y dimension" << endl;
		return 1;
	}

	*continuar_simulacion = 0;

	// Obtenemos inix, iniy, numVolx, numVoly, numVolumenes, iniLeer y numVolsLeer de todas las submallas.
	// También inicializamos iniGlobalSubmallasNivel y ratioRefAcumNivel
	iniGlobalSubmallasNivel[0][0].x = 0;
	iniGlobalSubmallasNivel[0][0].y = 0;
	ratioRefAcumNivel[0] = 1;
	for (l=1; l<(*numNiveles); l++) {
		for (i=0; i<numSubmallasNivel[l]; i++)
			obtenerDatosClusterSubmalla(datosClusterCPU, submallasNivel, l, i, submallaNivelSuperior, ratioRefNivel,
				&(iniGlobalSubmallasNivel[l][i]), ratioRefAcumNivel);
	}

	// Obtenemos posSubmallaNivelSuperior
	posSubmallaNivelSuperior[0][0] = -1;
	for (l=1; l<(*numNiveles); l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				pos = k = 0;
				while (k < submallaNivelSuperior[l][i]) {
					if (datosClusterCPU[l-1][k].iniy != -1) {
						nvx = datosClusterCPU[l-1][k].numVolx;
						nvy = datosClusterCPU[l-1][k].numVoly;
						pos += (nvx+4)*(nvy+4);
					}
					k++;
				}
				posSubmallaNivelSuperior[l][i] = pos;
			}
			else {
				posSubmallaNivelSuperior[l][i] = -1;
			}
		}
	}

	// Obtenemos numVolumenesNivel y numVerticesNivel
	numVolumenesNivel[0] = (datosClusterCPU[0][0].numVolx+4)*(datosClusterCPU[0][0].numVoly+4);
	numVerticesNivel[0] = (datosClusterCPU[0][0].numVolx+3)*(datosClusterCPU[0][0].numVoly+3);
	for (l=1; l<(*numNiveles); l++) {
		numVolumenesNivel[l] = 0;
		numVerticesNivel[l] = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				numVolumenesNivel[l] += (nvx+4)*(nvy+4);
				numVerticesNivel[l] += (nvx+3)*(nvy+3);
			}
		}
	}

	// Leemos el número de puntos de guardado (actual y de la simulación anterior, si existe). Es necesario
	// para la reserva de memoria de datosGRD, ya que los datos de los puntos a guardar se copian
	// en este vector en ShallowWater.cu
	// El resto de los ficheros de puntos se leerá después.
	m = 0;
	*numPuntosGuardarAnt = 0;
	if (*leer_fichero_puntos == 1) {
		fich_ptos.open((directorio+fich_puntos).c_str());
		fich_ptos >> m;
		fich_ptos.close();

		if (*continuar_simulacion == 1) {
			abrirTimeSeriesOldNC(((prefijo[0][0])+"_ts.nc").c_str());
			*numPuntosGuardarAnt = obtenerNumPuntosTimeSeriesOldNC();
		}
	}

	// Reservamos memoria
	cudaSetDevice(id_hebra%GPUS_PER_NODE);
	nvxTotal = *numVolxTotalNivel0;
	nvyTotal = *numVolyTotalNivel0;
	submallasNivel[0][0].z = nvxTotal;
	submallasNivel[0][0].w = nvyTotal;
	numVolumenesTotalNivel0 = nvxTotal*nvyTotal;
	tam_datosGRD = ((int64_t) numVolumenesTotalNivel0)*sizeof(float);  // (Si se lee la deformación inicial de fichero GRD)
	tam_datosGRD = max(tam_datosGRD, MAX_FAULTS*((int64_t) sizeof(float)));  // (Si se lee una deformación dinámica, para guardar los tiempos)
	tam_datosMPIDouble = tam_datosMPIDouble2 = tam_datosMPIDouble3 = 0;
	for (l=0; l<(*numNiveles); l++) {
		tam_datosGRD = max(tam_datosGRD, numVolumenesNivel[l]*((int64_t) sizeof(double)));  // (Si se continúa una simulación)
		tamf_Double = tamf_Double2 = tamf_Double3 = 0;
		tamc_Double = tamc_Double2 = tamc_Double3 = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			tds = &(datosNivel[l][i]);
			nvxTotal = submallasNivel[l][i].z;
			nvyTotal = submallasNivel[l][i].w;
			tds->vcos = (double *) malloc((2*nvyTotal+1)*sizeof(double));
			tds->vccos  = (double *) malloc(nvyTotal*sizeof(double));
			tds->vtan  = (double *) malloc(nvyTotal*sizeof(double));
			tds->longitud = (double *) malloc(nvxTotal*sizeof(double));
			tds->latitud  = (double *) malloc(nvyTotal*sizeof(double));
			tam_datosGRD = max(tam_datosGRD, nvxTotal*nvyTotal*((int64_t) sizeof(float)));  // (Si se lee una submalla de fichero GRD)
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				// El 8 significa: 2 filas de volComOtroClusterInf + 2 filas de volComClusterSup +
				// 2 filas de volComClusterInf + 2 filas de volComOtroClusterSup (lo mismo para columnas)
				// El 4 de Double es porque solo se envía una fila o columna de las presiones, no dos
				tamf_Double  += (nvx+3)*4*sizeof(double);
				tamc_Double  += (nvy+1)*4*sizeof(double);
				tamf_Double2 += (nvx+4)*8*sizeof(double2);
				tamc_Double2 += nvy*8*sizeof(double2);
				tamf_Double3 += (nvx+4)*8*sizeof(double3);
				tamc_Double3 += nvy*8*sizeof(double3);
			}
		}
		tam_datosMPIDouble  = max(tam_datosMPIDouble, max(tamf_Double,tamc_Double));
		tam_datosMPIDouble2 = max(tam_datosMPIDouble2, max(tamf_Double2,tamc_Double2));
		tam_datosMPIDouble3 = max(tam_datosMPIDouble3, max(tamf_Double3,tamc_Double3));
	}
	k = (*numPuntosGuardarAnt) + m;
	tam_datosGRD = max(tam_datosGRD, 2*k*((int64_t) sizeof(double3)));
	*datosGRD = (double *) malloc(tam_datosGRD);
	datosGRD_float = (float *) (*datosGRD);
	longitudDeformacion = (double *) malloc((*numVolxTotalNivel0)*sizeof(double));
	latitudDeformacion = (double *) malloc((*numVolyTotalNivel0)*sizeof(double));
	cudaMallocHost((void **) bufferEnviosMPISupInf_1, tam_datosMPIDouble2);
	cudaMallocHost((void **) bufferEnviosMPIIzqDer_1, tam_datosMPIDouble2);
	cudaMallocHost((void **) bufferEnviosMPISupInf_2, tam_datosMPIDouble3);
	cudaMallocHost((void **) bufferEnviosMPIIzqDer_2, tam_datosMPIDouble3);
	cudaMallocHost((void **) bufferEnviosMPISupInf_P, tam_datosMPIDouble);
	cudaMallocHost((void **) bufferEnviosMPIIzqDer_P, tam_datosMPIDouble);
	if (*okada_flag == DEFORMATION_FROM_FILE) {
		*deformacionAcumNivel0 = (double *) malloc((int64_t) numVolumenesTotalNivel0*sizeof(double));
		for (i=0; i<(*numFaults); i++)
			deformacionNivel0[i] = (double *) malloc((int64_t) numVolumenesTotalNivel0*sizeof(double));
	}
	if (*okada_flag == DYNAMIC_DEFORMATION) {
		*deformacionAcumNivel0 = (double *) malloc((int64_t) numVolumenesTotalNivel0*sizeof(double));
		for (i=0; i<(*numEstadosDefDinamica); i++) {
			j = (submallasDeformacion[0].z)*(submallasDeformacion[0].w);
			deformacionNivel0[i] = (double *) malloc((int64_t) j*sizeof(double));
		}
	}
	for (l=0; l<(*numNiveles); l++) {
		if (l > 0)
			posCopiaNivel[l] = (int *) malloc(numVolumenesNivel[l]*sizeof(int));
		if (((*numNiveles) > 1) && (l < (*numNiveles)-1))
			refinarNivel[l] = (bool *) malloc(numVolumenesNivel[l]*sizeof(bool));
		batiOriginal[l] = (float *) malloc(numVolumenesNivel[l]*sizeof(float));
		datosVolumenesNivel_1[l] = (double2 *) malloc(numVolumenesNivel[l]*sizeof(double2));
		datosVolumenesNivel_2[l] = (double3 *) malloc(numVolumenesNivel[l]*sizeof(double3));
		datosPnh[l] = (double *) malloc(numVerticesNivel[l]*sizeof(double));
		if ((l == 0) || ((*tipo_friccion) == VARIABLE_FRICTION_ALL)) {
			friccionesNivel[l] = (double *) malloc(numVolumenesNivel[l]*sizeof(double));
		}
	}
	if (datosVolumenesNivel_2[(*numNiveles)-1] == NULL) {
		cerr << "Error in process " << id_hebra_cart << ": Not enough CPU memory" << endl;
		liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
			datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
			*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
			*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
			*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
		liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
		return 1;
	}

	// Asignamos los punteros CPU del cluster
	for (l=0; l<(*numNiveles); l++) {
		inix = iniy = 0;
		inixP = iniyP = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				// j = número de volúmenes en una fila contando los volúmenes fantasma
				j = nvx + 4;
				datosClusterCPU[l][i].datosVolumenesComClusterIzq_1 = (*bufferEnviosMPIIzqDer_1) + iniy + 2;
				datosClusterCPU[l][i].datosVolumenesComClusterIzq_2 = (*bufferEnviosMPIIzqDer_2) + iniy + 2;
				datosClusterCPU[l][i].datosVolumenesComClusterDer_1 = (*bufferEnviosMPIIzqDer_1) + iniy + 4;
				datosClusterCPU[l][i].datosVolumenesComClusterDer_2 = (*bufferEnviosMPIIzqDer_2) + iniy + 4;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterIzq_1 = (*bufferEnviosMPIIzqDer_1) + iniy + 6;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterIzq_2 = (*bufferEnviosMPIIzqDer_2) + iniy + 6;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterDer_1 = (*bufferEnviosMPIIzqDer_1) + iniy;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterDer_2 = (*bufferEnviosMPIIzqDer_2) + iniy;

				datosClusterCPU[l][i].datosVolumenesComClusterSup_1 = (*bufferEnviosMPISupInf_1) + inix + 2*j;
				datosClusterCPU[l][i].datosVolumenesComClusterSup_2 = (*bufferEnviosMPISupInf_2) + inix + 2*j;
				datosClusterCPU[l][i].datosVolumenesComClusterInf_1 = (*bufferEnviosMPISupInf_1) + inix + 4*j;
				datosClusterCPU[l][i].datosVolumenesComClusterInf_2 = (*bufferEnviosMPISupInf_2) + inix + 4*j;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterSup_1 = (*bufferEnviosMPISupInf_1) + inix + 6*j;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterSup_2 = (*bufferEnviosMPISupInf_2) + inix + 6*j;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterInf_1 = (*bufferEnviosMPISupInf_1) + inix;
				datosClusterCPU[l][i].datosVolumenesComOtroClusterInf_2 = (*bufferEnviosMPISupInf_2) + inix;

				datosClusterCPU[l][i].datosVerticesComClusterIzq_P = (*bufferEnviosMPIIzqDer_P) + iniyP + 1;
				datosClusterCPU[l][i].datosVerticesComClusterDer_P = (*bufferEnviosMPIIzqDer_P) + iniyP + 2;
				datosClusterCPU[l][i].datosVerticesComOtroClusterIzq_P = (*bufferEnviosMPIIzqDer_P) + iniyP + 3;
				datosClusterCPU[l][i].datosVerticesComOtroClusterDer_P = (*bufferEnviosMPIIzqDer_P) + iniyP;

				datosClusterCPU[l][i].datosVerticesComClusterSup_P = (*bufferEnviosMPISupInf_P) + inixP + 1*(nvx+3);
				datosClusterCPU[l][i].datosVerticesComClusterInf_P = (*bufferEnviosMPISupInf_P) + inixP + 2*(nvx+3);
				datosClusterCPU[l][i].datosVerticesComOtroClusterSup_P = (*bufferEnviosMPISupInf_P) + inixP + 3*(nvx+3);
				datosClusterCPU[l][i].datosVerticesComOtroClusterInf_P = (*bufferEnviosMPISupInf_P) + inixP;

				inix += (nvx + 4)*8;
				iniy += 8*nvy;
				inixP += (nvx + 3)*4;
				iniyP += 4*(nvy+1);
			}
		}
	}

	// Ponemos en iniLeerX e iniLeerY el índice inicial de los volúmenes que hay que leer en los ficheros de datos
	// (considerando los volúmenes de comunicación de los clusters adyacentes, que también se almacenan).
	// Ponemos en numVolsLeerX y numVolsLeerY el número de volúmenes que hay que leer en los ficheros de datos.
	// Dimensión X
	if (id_hebraX == 0) {
		datosClusterCPU[0][0].iniLeerX = 0;
	}
	else {
		datosClusterCPU[0][0].iniLeerX = datosClusterCPU[0][0].inix - 2;
	}
	if (id_hebraX == 0) {
		// Es la primera hebra en X
		if (*num_procsX == 1)
			datosClusterCPU[0][0].numVolsLeerX = datosClusterCPU[0][0].numVolx;
		else
			datosClusterCPU[0][0].numVolsLeerX = datosClusterCPU[0][0].numVolx + 2;
	}
	else if (id_hebraX == (*num_procsX)-1) {
		// Es la última hebra en X
		datosClusterCPU[0][0].numVolsLeerX = datosClusterCPU[0][0].numVolx + 2;
	}
	else {
		// Es una hebra intermedia en X
		datosClusterCPU[0][0].numVolsLeerX = datosClusterCPU[0][0].numVolx + 4;
	}
	// Dimensión Y
	if (id_hebraY == 0) {
		datosClusterCPU[0][0].iniLeerY = 0;
	}
	else {
		datosClusterCPU[0][0].iniLeerY = datosClusterCPU[0][0].iniy - 2;
	}
	if (id_hebraY == 0) {
		// Es la primera hebra en Y
		if (*num_procsY == 1)
			datosClusterCPU[0][0].numVolsLeerY = datosClusterCPU[0][0].numVoly;
		else
			datosClusterCPU[0][0].numVolsLeerY = datosClusterCPU[0][0].numVoly + 2;
	}
	else if (id_hebraY == (*num_procsY)-1) {
		// Es la última hebra en Y
		datosClusterCPU[0][0].numVolsLeerY = datosClusterCPU[0][0].numVoly + 2;
	}
	else {
		// Es una hebra intermedia en Y
		datosClusterCPU[0][0].numVolsLeerY = datosClusterCPU[0][0].numVoly + 4;
	}

	// Creamos los comunicadores de las submallas
	comunicadores[0][0] = *comm_cartesiano;
	for (l=1; l<(*numNiveles); l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			k = ((datosClusterCPU[l][i].iniy != -1) ? 1 : 0);
			MPI_Comm_split(*comm_cartesiano, k, id_hebra_cart, &(comunicadores[l][i]));
		}
	}

	// Cargamos fricciones
	if (*tipo_friccion == FIXED_FRICTION) {
		// Fricción fija
		nvx = datosClusterCPU[0][0].numVolx;
		nvy = datosClusterCPU[0][0].numVoly;
		for (j=0; j<nvy+4; j++) {
			pos_ini = j*(nvx+4);
			for (i=0; i<nvx+4; i++) {
				pos = pos_ini + i;
				friccionesNivel[0][pos] = cfb;
			}
		}
	}
	else {
		// Fricción variable
		// Para pasar de bin a grd: gmt xyz2grd fricciones.bin -Gfricciones.grd -Rbati.grd -bi3f
		for (l=0; l<(*numNiveles); l++) {
			pos_ini = 0;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				if ((l == 0) || ((*tipo_friccion) == VARIABLE_FRICTION_ALL)) {
					error_formato = abrirGRDFricciones((directorio+fich_friccion[l][i]).c_str(), l, i, comunicadores[l][i]);
					if (error_formato != 0) {
						if (id_hebra_cart == 0)
							cerr << "Error: The NetCDF friction files should have NetCDF4 format" << endl;
						liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
							datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
							*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
							*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
							*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
						liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
						cerrarGRDFricciones(l, i);
						return 1;
					}
					nvx = datosClusterCPU[l][i].numVolx;
					nvy = datosClusterCPU[l][i].numVoly;
					ini_leerx = datosClusterCPU[l][i].iniLeerX;
					ini_leery = datosClusterCPU[l][i].iniLeerY;
					num_vols_leerx = datosClusterCPU[l][i].numVolsLeerX;
					num_vols_leery = datosClusterCPU[l][i].numVolsLeerY;
					leerTamanoSubmallaGRDFricciones(l, i, &nvxTotal, &nvyTotal);
					if ((nvxTotal != submallasNivel[l][i].z) || (nvyTotal != submallasNivel[l][i].w)) {
						if (id_hebra_cart == 0)
							cerr << "Error: The size of the dimensions in a frictions file and bathymetry file of the level " << l << " is different" << endl;
						liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
							datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
							*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
							*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
							*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
						liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
						cerrarGRDFricciones(l, i);
						return 1;
					}

					if (datosClusterCPU[l][i].iniy != -1) {
						// Almacenamos las fricciones en friccionesNivel
						// inix, iniy: coordenadas x, y de friccionesNivel a partir de las cuales se guardan los datos
						leerFriccionesGRD(l, i, ini_leerx, ini_leery, num_vols_leerx, num_vols_leery, datosGRD_float);
						inix = ((datosClusterCPU[l][i].inix == 0) ? 2 : 0);
						iniy = ((datosClusterCPU[l][i].iniy == 0) ? 2 : 0);
						factor = (*L)/pow(*H,4.0/3.0);
						for (j=0; j<num_vols_leery; j++) {
							pos = (iniy+j)*(nvx+4);
							for (k=0; k<num_vols_leerx; k++) {
								val = (double) (datosGRD_float[j*num_vols_leerx + k]);
								friccionesNivel[l][pos_ini + pos + inix+k] = val*factor;
							}
						}
						pos_ini += (nvx + 4)*(nvy + 4);
					}
					cerrarGRDFricciones(l, i);
				}
			}
		}
	}

	Hmin = 1e30;
	if (*continuar_simulacion == 1) {
		// Continuación de una simulación anterior
		// Cargamos longitudes, latitudes y topografía del fichero NetCDF resultado
	}
	else {
		// Nueva simulación
		// Leemos longitudes y latitudes y topografía
		for (l=0; l<(*numNiveles); l++) {
			pos_ini = 0;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				nvxTotal = submallasNivel[l][i].z;
				nvyTotal = submallasNivel[l][i].w;
				nvx = datosClusterCPU[l][i].numVolx;
				// Las longitudes y latitudes siempre se leen (también para la hebra 0 para mostrar después los datos correctamente)
				// Leemos longitudes
				error_formato = abrirGRD((directorio+fich_topo[l][i]).c_str(), l, i, comunicadores[l][i]);
				if (error_formato != 0) {
					if (id_hebra_cart == 0)
						cerr << "Error: The grd files should have NetCDF4 format" << endl;
					liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
						datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
						*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
						*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
						*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
					liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
					cerrarGRD(l, i);
					return 1;
				}
				leerLongitudGRD(l, i, *datosGRD);
				for (k=0; k<nvxTotal; k++)
					datosNivel[l][i].longitud[k] = (*datosGRD)[k];
				// Leemos latitudes
				leerLatitudGRD(l, i, *datosGRD);
				for (k=0; k<nvyTotal; k++)
					datosNivel[l][i].latitud[k] = (*datosGRD)[k];

				// Calculamos vcos, vccos y vtan de la submalla i del nivel l
				tds = &(datosNivel[l][i]);
				nvyTotal = submallasNivel[l][i].w;
				tds->dx = fabs(tds->longitud[1] - tds->longitud[0]);
				tds->dy = fabs(tds->latitud[1] - tds->latitud[0]);
				for (k=0; k<nvyTotal; k++) {
					lat = tds->latitud[k];
					tds->vcos[2*k] = cos((lat - 0.5*tds->dy)*grad2rad);
					tds->vcos[2*k+1] = cos(lat*grad2rad);
					tds->vccos[k] = tds->vcos[2*k];
					tds->vtan[k] = tan(lat*grad2rad);
				}
				tds->vcos[2*nvyTotal] = cos((lat + 0.5*tds->dy)*grad2rad);
				tds->dx *= grad2rad;
				tds->dy *= grad2rad;

				// Leemos topografía
				if (datosClusterCPU[l][i].iniy != -1) {
					ini_leerx = datosClusterCPU[l][i].iniLeerX;
					ini_leery = datosClusterCPU[l][i].iniLeerY;
					num_vols_leerx = datosClusterCPU[l][i].numVolsLeerX;
					num_vols_leery = datosClusterCPU[l][i].numVolsLeerY;
					leerBatimetriaGRD(l, i, ini_leerx, ini_leery, num_vols_leerx, num_vols_leery, datosGRD_float);
					// inix, iniy: coordenadas x, y de datosVolumenesNivel[l] (empezando por la primera)
					// a partir de la cual se guardan los datos
					inix = ((ini_leerx == 0) ? 2 : 0);
					iniy = ((ini_leery == 0) ? 2 : 0);
					for (j=0; j<num_vols_leery; j++) {
						m = datosClusterCPU[l][i].iniy + (iniy-2);
						for (k=0; k<num_vols_leerx; k++) {
							pos = pos_ini + iniy*(nvx+4) + inix+k;
							val = -1.0*datosGRD_float[j*num_vols_leerx + k];
							val *= (tds->vccos[m])/(*H);
							datosVolumenesNivel_1[l][pos].y = val;
							// Si hay que aplicar Okada o gaussiana inicializamos el estado del volumen con agua plana
							// (el Okada o la gaussiana se aplicarán después en ShallowWater.cu). Si no, el estado
							// se inicializará después al leer el fichero
							datosVolumenesNivel_1[l][pos].x = ((val > 0.0) ? val : 0.0);
							datosVolumenesNivel_2[l][pos].x = 0.0;
							datosVolumenesNivel_2[l][pos].y = 0.0;
							datosVolumenesNivel_2[l][pos].z = 0.0;

							if (val < Hmin)
								Hmin = val;
						}
						iniy++;
					}
					pos_ini += (nvx + 4)*(datosClusterCPU[l][i].numVoly + 4);
				}
				cerrarGRD(l, i);
			}
		}
	}

	// Obtenemos el mí­nimo Hmin de todos los clusters por reducción
	MPI_Allreduce (&Hmin, Hmin_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

	// Almacenamos la batimetría original
	for (l=0; l<(*numNiveles); l++) {
		pos_ini = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				for (j=0; j<nvy; j++) {
					for (k=0; k<nvx; k++) {
						pos = pos_ini + (j+2)*(nvx+4) + (k+2);
						batiOriginal[l][pos_ini + j*nvx+k] = (float) ((datosVolumenesNivel_1[l][pos].y)*(*H));
					}
				}
				pos_ini += (nvx + 4)*(nvy + 4);
			}
		}
	}

	// Leemos las deformaciones de Okada si están en fichero, y las ordenamos por el tiempo en el que se producen
	nvxTotal = submallasNivel[0][0].z;
	nvyTotal = submallasNivel[0][0].w;
	if ((*okada_flag == OKADA_STANDARD) || (*okada_flag == OKADA_STANDARD_FROM_FILE)) {
		if (usar_ventana == 0) {
			// La deformación se aplica a todo el dominio
			for (i=0; i<(*numFaults); i++) {
				submallasDeformacion[i].x = 0;
				submallasDeformacion[i].y = 0;
				submallasDeformacion[i].z = nvxTotal;
				submallasDeformacion[i].w = nvyTotal;
			}
		}
		else {
			// La deformación se aplica a la ventana de computación de Okada
			// Convertimos el radio de computación: km->grados->volúmenes
			radioComp = 180.0*radioComp/(M_PI*EARTH_RADIUS*1e-3);
			incx = (datosNivel[0][0].longitud[nvxTotal-1] - datosNivel[0][0].longitud[0])/nvxTotal;
			incy = (datosNivel[0][0].latitud[nvyTotal-1] - datosNivel[0][0].latitud[0])/nvyTotal;
			radiox = (int) round(radioComp/incx);
			radioy = (int) round(radioComp/incy);

			obtenerIndicePuntoNivel0(&(datosNivel[0][0]), submallasNivel[0][0], lon_centro, lat_centro, &centrox, &centroy);
			if ((centrox < 0) || (centroy < 0)) {
				if (id_hebra_cart == 0)
					cerr << "Error: The center of the Okada computation window is outside of the spatial domain" << endl;
				liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
					datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
					*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
					*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
					*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
				liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
				return 1;
			}

			for (i=0; i<(*numFaults); i++) {
				submallasDeformacion[i].x = max(0, centrox-radiox);
				submallasDeformacion[i].y = max(0, centroy-radioy);
				submallasDeformacion[i].z = min(nvxTotal-1, centrox+radiox) - submallasDeformacion[i].x + 1;
				submallasDeformacion[i].w = min(nvyTotal-1, centroy+radioy) - submallasDeformacion[i].y + 1;
			}
		}
		ordenarFallasOkadaStandardPorTiempo(*numFaults, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, defTime);
	}
	else if ((*okada_flag == OKADA_TRIANGULAR) || (*okada_flag == OKADA_TRIANGULAR_FROM_FILE)) {
		if (usar_ventana == 0) {
			// La deformación se aplica a todo el dominio
			for (i=0; i<(*numFaults); i++) {
				submallasDeformacion[i].x = 0;
				submallasDeformacion[i].y = 0;
				submallasDeformacion[i].z = nvxTotal;
				submallasDeformacion[i].w = nvyTotal;
			}
		}
		else {
			// La deformación se aplica a la ventana de computación de Okada
			// Convertimos el radio de computación: km->grados->volúmenes
			radioComp = 180.0*radioComp/(M_PI*EARTH_RADIUS*1e-3);
			incx = (datosNivel[0][0].longitud[nvxTotal-1] - datosNivel[0][0].longitud[0])/nvxTotal;
			incy = (datosNivel[0][0].latitud[nvyTotal-1] - datosNivel[0][0].latitud[0])/nvyTotal;
			radiox = (int) round(radioComp/incx);
			radioy = (int) round(radioComp/incy);

			obtenerIndicePuntoNivel0(&(datosNivel[0][0]), submallasNivel[0][0], lon_centro, lat_centro, &centrox, &centroy);
			if ((centrox < 0) || (centroy < 0)) {
				if (id_hebra_cart == 0)
					cerr << "Error: The center of the Okada computation window is outside of the spatial domain" << endl;
				liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
					datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
					*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
					*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
					*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
				liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
				return 1;
			}

			for (i=0; i<(*numFaults); i++) {
				submallasDeformacion[i].x = max(0, centrox-radiox);
				submallasDeformacion[i].y = max(0, centroy-radioy);
				submallasDeformacion[i].z = min(nvxTotal-1, centrox+radiox) - submallasDeformacion[i].x + 1;
				submallasDeformacion[i].w = min(nvyTotal-1, centroy+radioy) - submallasDeformacion[i].y + 1;
			}
		}
		ordenarFallasOkadaTriangularPorTiempo(*numFaults, LON_LAT_v, DEPTH_v, vc, LONCTRI, LATCTRI, SLIPVEC, RAKE, SLIP, defTime);
	}
	else if (*okada_flag == DEFORMATION_FROM_FILE) {
		// La deformación se aplica a la ventana de computación que viene dada por los ficheros GRD que contienen las deformaciones
		tamf_Double2 = ((int64_t) numVolumenesTotalNivel0)*sizeof(double);
		j = submallasNivel[0][0].z;
		k = submallasNivel[0][0].w;
		for (i=0; i<(*numFaults); i++) {
			leerDeformacionGRD((directorio+fich_def[i]).c_str(), &nvx, &nvy, longitudDeformacion, latitudDeformacion, datosGRD_float);
			submallasDeformacion[i].z = nvx;
			submallasDeformacion[i].w = nvy;
			if ((longitudDeformacion[0]+EPSILON < datosNivel[0][0].longitud[0]) || (longitudDeformacion[nvx-1]-EPSILON > datosNivel[0][0].longitud[j-1]) ||
				(latitudDeformacion[0]+EPSILON < datosNivel[0][0].latitud[0]) || (latitudDeformacion[nvy-1]-EPSILON > datosNivel[0][0].latitud[k-1]) ) {
				if (id_hebra_cart == 0)
					cerr << "Error: Okada deformation " << i+1 << " is not a subset of level 0 mesh" << endl;
				liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
					datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
					*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
					*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
					*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
				liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
				return 1;
			}

			obtenerIndicePuntoNivel0(&(datosNivel[0][0]), submallasNivel[0][0], longitudDeformacion[0], latitudDeformacion[0],
				&(submallasDeformacion[i].x), &(submallasDeformacion[i].y));
			if ((submallasDeformacion[i].x != submallasDeformacion[0].x) || (submallasDeformacion[i].y != submallasDeformacion[0].y) ||
				(submallasDeformacion[i].z != submallasDeformacion[0].z) || (submallasDeformacion[i].w != submallasDeformacion[0].w)) {
				if (id_hebra_cart == 0)
					cerr << "Error: All the Okada deformations should have the same domain" << endl;
				liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
					datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
					*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
					*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
					*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
				liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
				return 1;
			}

			// Copiamos la deformación de datosGRD_float a deformacionNivel0[i]
			copiarDeformacionEnNivel0(deformacionNivel0, *deformacionAcumNivel0, datosGRD_float, i, submallasDeformacion[i], *H);
		}

		ordenarFallasFicherosPorTiempo(*numFaults, deformacionNivel0, submallasDeformacion, defTime, *datosGRD, tamf_Double2);
	}
	else if (*okada_flag == DYNAMIC_DEFORMATION) {
		// La deformación se aplica a la ventana de computación que viene dada por el fichero GRD que contiene la deformación
		j = submallasNivel[0][0].z;
		k = submallasNivel[0][0].w;
		for (i=0; i<(*numFaults); i++) {
			nvx = submallasDeformacion[i].z;
			nvy = submallasDeformacion[i].w;
			abrirGRDDefDinamica((directorio+fich_def[i]).c_str());
			leerLongitudLatitudYTiemposDefDinamicaGRD(longitudDeformacion, latitudDeformacion, datosGRD_float);
			for (m=0; m<(*numEstadosDefDinamica); m++) {
				defTime[m] = (double) datosGRD_float[m];
			}
			if ((longitudDeformacion[0]+EPSILON < datosNivel[0][0].longitud[0]) || (longitudDeformacion[nvx-1]-EPSILON > datosNivel[0][0].longitud[j-1]) ||
				(latitudDeformacion[0]+EPSILON < datosNivel[0][0].latitud[0]) || (latitudDeformacion[nvy-1]-EPSILON > datosNivel[0][0].latitud[k-1]) ) {
				if (id_hebra_cart == 0)
					cerr << "Error: Dynamic deformation " << i+1 << " is not a subset of level 0 mesh" << endl;
				liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
					datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
					*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
					*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
					*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
				liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
				cerrarGRDDefDinamica();
				return 1;
			}
			// Cargamos los estados de la deformación dinámica en deformacionNivel0
			for (m=0; m<(*numEstadosDefDinamica); m++) {
				leerEstadoDefDinamicaGRD(submallasDeformacion[i].z, submallasDeformacion[i].w, m, datosGRD_float);
				copiarDeformacionEnNivel0(deformacionNivel0, *deformacionAcumNivel0, datosGRD_float, m, submallasDeformacion[i], *H);
			}
			cerrarGRDDefDinamica();
		}
	}

	// Leemos los puntos de guardado
	*numPuntosGuardarTotal = 0;
	if (*leer_fichero_puntos == 1) {
		// k = Número de puntos actuales
		fich_ptos.open((directorio+fich_puntos).c_str());
		fich_ptos >> k;

		m = (*numPuntosGuardarAnt) + k;
		*lonPuntos = (double *) malloc(m*sizeof(double));
		*latPuntos = (double *) malloc(m*sizeof(double));
		*posicionesVolumenesGuardado = (int4 *) malloc(m*sizeof(int4));
		if (*posicionesVolumenesGuardado == NULL) {
			fprintf(stderr, "Error in process %d: Not enough CPU memory\n", id_hebra_cart);
			liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
				datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
				*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
				*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
				*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
			liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
			fich_ptos.close();
			if (*continuar_simulacion == 1)
				cerrarTimeSeriesOldNC();
			return 1;
		}
		if (*continuar_simulacion == 1) {
			// Cargamos las longitudes y latitudes de los puntos de la simulación anterior
			leerLongitudesYLatitudesTimeSeriesOldNC(*lonPuntos, *latPuntos);
			cerrarTimeSeriesOldNC();

			// Procesamos los puntos de la simulación anterior
			for (i=0; i<(*numPuntosGuardarAnt); i++) {
				posPunto = obtenerIndicePunto(*numNiveles, datosClusterCPU, datosNivel, submallasNivel,
								numSubmallasNivel, (*lonPuntos)[i], (*latPuntos)[i]);
				(*posicionesVolumenesGuardado)[i] = posPunto;
			}
		}
		// Procesamos los nuevos puntos
		*numPuntosGuardarTotal = *numPuntosGuardarAnt;
		for (i=0; i<k; i++) {
			fich_ptos >> lon;
			fich_ptos >> lat;
			pos = obtenerPosEnPuntosGuardado(*lonPuntos, *latPuntos, *numPuntosGuardarTotal, lon, lat);
			if (pos == -1) {
				// Añadimos el punto a la lista
				(*lonPuntos)[*numPuntosGuardarTotal] = lon;
				(*latPuntos)[*numPuntosGuardarTotal] = lat;
				posPunto = obtenerIndicePunto(*numNiveles, datosClusterCPU, datosNivel, submallasNivel,
								numSubmallasNivel, lon, lat);
				(*posicionesVolumenesGuardado)[*numPuntosGuardarTotal] = posPunto;
				(*numPuntosGuardarTotal)++;
			}
		}
		fich_ptos.close();
	}

	// Marcamos las celdas refinadas en refinarNivel
	if ((*numNiveles) > 1) {
		for (l=0; l < (*numNiveles)-1; l++) {
			memset(refinarNivel[l], 0, numVolumenesNivel[l]*sizeof(bool));
			ratio = ratioRefNivel[l+1];
			pos_ini = 0;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				if (datosClusterCPU[l][i].iniy != -1) {
					// Para cada submalla del nivel l+1, comprobamos si está dentro de la submalla
					// i del nivel l y dentro del cluster y, si es así, la procesamos
					for (j=0; j<numSubmallasNivel[l+1]; j++) {
						if ((datosClusterCPU[l+1][j].iniy != -1) && (submallaNivelSuperior[l+1][j] == i)) {
							// Marcamos las posiciones de la submalla j en el vector refinarNivel
							// asociado a la submalla i del nivel l
							inix = submallasNivel[l+1][j].x;
							iniy = submallasNivel[l+1][j].y;
							nvx = datosClusterCPU[l+1][j].numVolx;
							nvy = datosClusterCPU[l+1][j].numVoly;
							for (m=0; m<nvy; m+=ratio) {
								for (k=0; k<nvx; k+=ratio) {
									pos = (iniy + datosClusterCPU[l+1][j].iniy + m)/ratio - datosClusterCPU[l][i].iniy;
									pos = pos_ini + pos*datosClusterCPU[l][i].numVolx + ((inix + datosClusterCPU[l+1][j].inix + k)/ratio - datosClusterCPU[l][i].inix);
									refinarNivel[l][pos] = true;
								}
							}
						}
					}
					pos_ini += (datosClusterCPU[l][i].numVolx + 4)*(datosClusterCPU[l][i].numVoly + 4);
				}
			}
		}
	}

	// Asignamos las posiciones de copia de las celdas fantasma de las submallas
	for (l=1; l<(*numNiveles); l++) {
		for (i=0; i<numVolumenesNivel[l]; i++)
			posCopiaNivel[l][i] = -1;
		pos_ini = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if (datosClusterCPU[l][i].iniy != -1) {
				inix = submallasNivel[l][i].x;
				iniy = submallasNivel[l][i].y;
				nvxTotal = submallasNivel[l][i].z;
				nvyTotal = submallasNivel[l][i].w;
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				// xini, yini: = índices iniciales en x e y para procesar las celdas fantasma de la submalla i
				// xfin, yfin: = índices finales en x e y para procesar las celdas fantasma de la submalla i
				int xini = ((datosClusterCPU[l][i].inix == 0) ? -2 : datosClusterCPU[l][i].inix);
				int yini = ((datosClusterCPU[l][i].iniy == 0) ? -2 : datosClusterCPU[l][i].iniy);
				int xfin = ((datosClusterCPU[l][i].inix + nvx == nvxTotal) ? nvxTotal+2 : datosClusterCPU[l][i].inix + nvx);
				int yfin = ((datosClusterCPU[l][i].iniy + nvy == nvyTotal) ? nvyTotal+2 : datosClusterCPU[l][i].iniy + nvy);
				for (j=yini; j<yfin; j++) {
					for (k=xini; k<xfin; k++) {
						if ((j < 0) || (j >= nvyTotal) || (k < 0) || (k >= nvxTotal)) {
							pos = posicionEnSubmallas(datosClusterCPU[l], inix+k, iniy+j, submallasNivel[l], numSubmallasNivel[l],
									submallaNivelSuperior[l][i], submallaNivelSuperior[l]);
							if (pos >= 0)
								posCopiaNivel[l][pos_ini + (j+2-datosClusterCPU[l][i].iniy)*(nvx+4) + (k+2-datosClusterCPU[l][i].inix)] = pos;
						}
					}
				}
				pos_ini += (nvx + 4)*(nvy + 4);
			}
		}
	}

	// Quitamos el restado de Hmin_global en este código
	// Corregimos los valores de profundidad, si hay alguna negativa
/*	if (*Hmin_global < 0.0) {
		for (l=0; l<(*numNiveles); l++) {
			pos_ini = 0;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				if (datosClusterCPU[l][i].iniy != -1) {
					// inix e iniy: índices a partir del cuales se guardan los datos en datosVolumenes
					inix = ((datosClusterCPU[l][i].inix == 0) ? 2 : 0);
					iniy = ((datosClusterCPU[l][i].iniy == 0) ? 2 : 0);
					nvx = datosClusterCPU[l][i].numVolx;
					nvy = datosClusterCPU[l][i].numVoly;
					num_vols_leerx = datosClusterCPU[l][i].numVolsLeerX;
					num_vols_leery = datosClusterCPU[l][i].numVolsLeerY;
					for (j=iniy; j<iniy+num_vols_leery; j++) {
						for (k=inix; k<inix+num_vols_leerx; k++) {
							pos = pos_ini + j*(nvx+4) + k;
							datosVolumenesNivel_1[l][pos].y -= *Hmin_global;
						}
						m++;
					}
					pos_ini += (nvx + 4)*(nvy + 4);
				}
			}
		}
	}
	else {*/
		*Hmin_global = 0.0;
//	}

	if (*continuar_simulacion == 1) {
		// Continuación de una simulación anterior
		// Cargamos el estado inicial del fichero NetCDF resultado
	}
	else if (*okada_flag == SEA_SURFACE_FROM_FILE) {
		// Nueva simulación
		// Leemos estado inicial de fichero
		for (l=0; l<(*numNiveles); l++) {
			pos_ini = 0;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				nvxTotal = submallasNivel[l][i].z;
				nvyTotal = submallasNivel[l][i].w;
				nvx = datosClusterCPU[l][i].numVolx;
				if (datosClusterCPU[l][i].iniy != -1) {
					ini_leerx = datosClusterCPU[l][i].iniLeerX;
					ini_leery = datosClusterCPU[l][i].iniLeerY;
					num_vols_leerx = datosClusterCPU[l][i].numVolsLeerX;
					num_vols_leery = datosClusterCPU[l][i].numVolsLeerY;
					error_formato = abrirGRD((directorio+fich_est[l][i]).c_str(), l, i, comunicadores[l][i]);
					if (error_formato != 0) {
						if (id_hebra_cart == 0)
							cerr << "Error: The initial state files should have NetCDF4 format" << endl;
						liberarMemoria(*numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
							datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, *bufferEnviosMPISupInf_1, *bufferEnviosMPIIzqDer_1,
							*bufferEnviosMPISupInf_2, *bufferEnviosMPIIzqDer_2, *bufferEnviosMPISupInf_P, *bufferEnviosMPIIzqDer_P,
							*tipo_friccion, friccionesNivel, *posicionesVolumenesGuardado, *lonPuntos, *latPuntos, *okada_flag,
							*numFaults, *numEstadosDefDinamica, deformacionNivel0, *deformacionAcumNivel0, *datosGRD, id_hebra_cart);
						liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);
						cerrarGRD(l, i);
						return 1;
					}
					leerEtaGRD(l, i, ini_leerx, ini_leery, num_vols_leerx, num_vols_leery, datosGRD_float);
					// inix, iniy: coordenadas x, y de datosVolumenesNivel[l] (empezando por la primera)
					// a partir de la cual se guardan los datos
					tds = &(datosNivel[l][i]);
					inix = ((datosClusterCPU[l][i].inix == 0) ? 2 : 0);
					iniy = ((datosClusterCPU[l][i].iniy == 0) ? 2 : 0);
					for (j=0; j<num_vols_leery; j++) {
						m = datosClusterCPU[l][i].iniy + (iniy-2);
						for (k=0; k<num_vols_leerx; k++) {
							pos = pos_ini + iniy*(nvx+4) + inix+k;
							val = (double) datosGRD_float[j*num_vols_leerx + k];
							val *= (tds->vccos[m])/(*H);
							val = val + datosVolumenesNivel_1[l][pos].y + (*Hmin_global);  // h = eta+H
							val = max(val, 0.0);
							datosVolumenesNivel_1[l][pos].x = val;
							datosVolumenesNivel_2[l][pos].x = 0.0;
							datosVolumenesNivel_2[l][pos].y = 0.0;
							datosVolumenesNivel_2[l][pos].z = 0.0;
						}
						iniy++;
					}
					if (existeVariableNetCDF(l, i, "ux")) {
						leerUxGRD(l, i, ini_leerx, ini_leery, num_vols_leerx, num_vols_leery, datosGRD_float);
						iniy = ((datosClusterCPU[l][i].iniy == 0) ? 2 : 0);
						for (j=0; j<num_vols_leery; j++) {
							for (k=0; k<num_vols_leerx; k++) {
								pos = pos_ini + iniy*(nvx+4) + inix+k;
								val = datosVolumenesNivel_1[l][pos].x;  // h
								val *= datosGRD_float[j*num_vols_leerx + k]*(*H);  // qx = h*ux
								datosVolumenesNivel_2[l][pos].x = val/((*H)*(*U));
							}
							iniy++;
						}
					}
					if (existeVariableNetCDF(l, i, "uy")) {
						leerUyGRD(l, i, ini_leerx, ini_leery, num_vols_leerx, num_vols_leery, datosGRD_float);
						iniy = ((datosClusterCPU[l][i].iniy == 0) ? 2 : 0);
						for (j=0; j<num_vols_leery; j++) {
							for (k=0; k<num_vols_leerx; k++) {
								pos = pos_ini + iniy*(nvx+4) + inix+k;
								val = datosVolumenesNivel_1[l][pos].x;  // h
								val *= datosGRD_float[j*num_vols_leerx + k]*(*H);  // qy = h*uy
								datosVolumenesNivel_2[l][pos].y = val/((*H)*(*U));
							}
							iniy++;
						}
					}
					if ((*no_hidros == 1) && existeVariableNetCDF(l, i, "uz")) {
						leerUzGRD(l, i, ini_leerx, ini_leery, num_vols_leerx, num_vols_leery, datosGRD_float);
						iniy = ((datosClusterCPU[l][i].iniy == 0) ? 2 : 0);
						for (j=0; j<num_vols_leery; j++) {
							for (k=0; k<num_vols_leerx; k++) {
								pos = pos_ini + iniy*(nvx+4) + inix+k;
								val = datosVolumenesNivel_1[l][pos].x;  // h
								val *= datosGRD_float[j*num_vols_leerx + k]*(*H);  // qz = h*uz
								datosVolumenesNivel_2[l][pos].z = val/((*H)*(*H)/(*T));
							}
							iniy++;
						}
					}
					cerrarGRD(l, i);

					pos_ini += (nvx + 4)*(datosClusterCPU[l][i].numVoly + 4);
				}
			}
		}
	}

	liberarMemoriaDeformacion(longitudDeformacion, latitudDeformacion);

	return 0;
}

void mostrarDatosProblema(string version, int no_hidros, int numPesosJacobi, int numNiveles, tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int okada_flag, int kajiura_flag, double depth_kajiura, string fich_okada, int numFaults, int numVolxTotalNivel0, int numVolyTotalNivel0,
			int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double tiempo_tot, double *tiempoGuardarNetCDF,
			int leer_fichero_puntos, double tiempoGuardarSeries, double CFL, int tipo_friccion, string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double **friccionesNivel, double vmax, double epsilon_h, double difh_at, int *ratioRefNivel, double L, double H, double U, double T)
{
	int i, l;
	int nvx, nvy;
	int nsubmallas;
	double factor;

	cout << "/*************************************************************" << endl;
	cout << " NH-HySEA numerical model v" << version << endl;
	cout << " Copyright (C) 2010-2024                                      " << endl;
	cout << " EDANYA Research Group, University of Malaga (Spain).         " << endl;
	cout << "                                                              " << endl;
	cout << " NH-HySEA can not be copied, modified and/or distributed      " << endl;
	cout << " without the express permission of the EDANYA Research Group. " << endl;
	cout << " NH-HySEA is distributed under license. For more information, " << endl;
	cout << " visit:                                                       " << endl;
	cout << " https://edanya.uma.es/hysea/tsunami-hysea_license.html       " << endl;
	cout << "*************************************************************/" << endl;
	cout << endl;

	cout << "Problem data" << endl;
	if (no_hidros) {
		cout << "Non-hidrostatic simulation (Jacobi weights: " << numPesosJacobi << ")" << endl;
	}
	else {
		cout << "Hidrostatic simulation" << endl;
	}
	if (okada_flag == SEA_SURFACE_FROM_FILE) {
		cout << "Initialization: Sea surface displacement from file" << endl;
	}
	else if (okada_flag == OKADA_STANDARD) {
		if (kajiura_flag == 1)
			cout << "Initialization: Standard Okada with Kajiura filter (reference depth: " << depth_kajiura*H << " m)" << endl;
		else
			cout << "Initialization: Standard Okada without Kajiura filter" << endl;
		cout << "Number of faults: " << numFaults << endl;
	}
	else if (okada_flag == OKADA_STANDARD_FROM_FILE) {
		if (kajiura_flag == 1)
			cout << "Initialization: Standard Okada with Kajiura filter (reference depth: " << depth_kajiura*H << " m)" << endl;
		else
			cout << "Initialization: Standard Okada without Kajiura filter" << endl;
		cout << "File with faults: " << fich_okada << endl;
		cout << "Number of faults: " << numFaults << endl;
	}
	else if (okada_flag == OKADA_TRIANGULAR) {
		if (kajiura_flag == 1)
			cout << "Initialization: Triangular Okada with Kajiura filter (reference depth: " << depth_kajiura*H << " m)" << endl;
		else
			cout << "Initialization: Triangular Okada without Kajiura filter" << endl;
		cout << "Number of faults: " << numFaults << endl;
	}
	else if (okada_flag == OKADA_TRIANGULAR_FROM_FILE) {
		if (kajiura_flag == 1)
			cout << "Initialization: Triangular Okada with Kajiura filter (reference depth: " << depth_kajiura*H << " m)" << endl;
		else
			cout << "Initialization: Triangular Okada without Kajiura filter" << endl;
		cout << "File with faults: " << fich_okada << endl;
		cout << "Number of faults: " << numFaults << endl;
	}
	else if (okada_flag == DEFORMATION_FROM_FILE) {
		if (kajiura_flag == 1)
			cout << "Initialization: Sea floor deformation from file with Kajiura filter (reference depth: " << depth_kajiura*H << " m)" << endl;
		else
			cout << "Initialization: Sea floor deformation from file without Kajiura filter" << endl;
		cout << "Number of faults: " << numFaults << endl;
	}
	else if (okada_flag == DYNAMIC_DEFORMATION) {
		if (kajiura_flag == 1)
			cout << "Initialization: Sea floor dynamic deformation with Kajiura filter (reference depth: " << depth_kajiura*H << " m)" << endl;
		else
			cout << "Initialization: Sea floor dynamic deformation without Kajiura filter" << endl;
		cout << "Number of faults: " << numFaults << endl;
	}
	else if (okada_flag == GAUSSIAN) {
		cout << "Initialization: Gaussian" << endl;
	}
	cout << "CFL: " << CFL << endl;
	if (tipo_friccion == FIXED_FRICTION) {
		factor = L/pow(H,4.0/3.0);
		cout << "Friction type: fixed" << endl;
		cout << "Water-bottom friction: " << friccionesNivel[0][0]/factor << endl;
	}
	else {
		cout << "Friction type: variable" << endl;
		cout << "Files with water-bottom frictions" << endl;
		cout << "  Level 0: " << fich_friccion[0][0] << endl;
		if (tipo_friccion == VARIABLE_FRICTION_ALL) {
			for (l=1; l<numNiveles; l++) {
				nsubmallas = numSubmallasNivel[l];
				cout << "  Level " << l << ": ";
				for (i=0; i<nsubmallas-1; i++)
					cout << fich_friccion[l][i] << ", ";
				cout << fich_friccion[l][nsubmallas-1] << endl;
			}
		}
	}
	cout << "Maximum allowed velocity of water: " << vmax*U << endl;
	cout << "Epsilon h: " << epsilon_h*H << " m" << endl;
	if (difh_at >= 0.0) {
		cout << "Threshold for arrival times: " << difh_at*H << " m" << endl;
	}
	cout << "Simulation time: " << tiempo_tot*T << " sec" << endl;
	cout << "Saving time of NetCDF files" << endl;
	for (l=0; l<numNiveles; l++) {
		cout << "  Level " << l << ": " << tiempoGuardarNetCDF[l]*T << " sec" << endl;
	}
	if (leer_fichero_puntos) {
		cout << "Time series: yes (saving time: " << tiempoGuardarSeries*T << " sec)" << endl;
	}
	else {
		cout << "Time series: no" << endl;
	}
	cout << "Number of levels: " << numNiveles << endl;
	cout << "Level 0" << endl;
	cout << "  Volumes: " << numVolxTotalNivel0 << " x " << numVolyTotalNivel0 << " = " << numVolxTotalNivel0*numVolyTotalNivel0 << endl;
	cout << "  Longitude: [" << datosNivel[0][0].longitud[0] << ", " << datosNivel[0][0].longitud[numVolxTotalNivel0-1] << "]" << endl;
	cout << "  Latitude: [" << datosNivel[0][0].latitud[0] << ", " << datosNivel[0][0].latitud[numVolyTotalNivel0-1] << "]" << endl;
	for (l=1; l<numNiveles; l++) {
		if (numSubmallasNivel[l] > 0) {
			cout << "Level " << l << endl;
			cout << "  Refinement ratio: " << ratioRefNivel[l] << endl;
			for (i=0; i<numSubmallasNivel[l]; i++) {
				nvx = submallasNivel[l][i].z;
				nvy = submallasNivel[l][i].w;
				cout << "  Submesh " << i+1 << endl;
				cout << "    Volumes: " << nvx << " x " << nvy << " = " << nvx*nvy << endl;
				cout << "    Longitude: [" << datosNivel[l][i].longitud[0] << ", " << datosNivel[l][i].longitud[nvx-1] << "]" << endl;
				cout << "    Latitude: [" << datosNivel[l][i].latitud[0] << ", " << datosNivel[l][i].latitud[nvy-1] << "]" << endl;
			}
		}
	}
	cout << endl;
}

#endif

