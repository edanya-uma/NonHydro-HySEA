#ifndef _PROBLEMA_H_
#define _PROBLEMA_H_

#include "Constantes.hxx"
#include <sys/stat.h> 
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdlib.h>
#include <limits.h>
#include "CargaX.cxx"
#include "CargaY.cxx"
#include "netcdf.cxx"

/********************/
/* Funciones NetCDF */
/********************/

extern void abrirGRD(const char *nombre_fich, int nivel, int submalla, int *nvx, int *nvy);
extern void leerLongitudGRD(int nivel, int submalla, double *lon);
extern void leerLatitudGRD(int nivel, int submalla, double *lat);
extern void leerBatimetriaGRD(int nivel, int submalla, int num_volx, int num_voly, float *bati);
extern void cerrarGRD(int nivel, int submalla);


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

void liberarMemoria(int numNiveles, double2 **datosVolumenesNivel_1, double2 **datosVolumenesNivel_2,
		tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel,
		int *xIniCluster, int *yIniCluster)
{
	int i, j;

	if (xIniCluster != NULL)					free(xIniCluster);
	if (yIniCluster != NULL)					free(yIniCluster);

	for (i=0; i<numNiveles; i++) {
		for (j=0; j<numSubmallasNivel[i]; j++) {
			if (datosNivel[i][j].longitud != NULL)			free(datosNivel[i][j].longitud);
			if (datosNivel[i][j].latitud != NULL)			free(datosNivel[i][j].latitud);
		}
		if (datosVolumenesNivel_1[i] != NULL)	free(datosVolumenesNivel_1[i]);
		if (datosVolumenesNivel_2[i] != NULL)	free(datosVolumenesNivel_2[i]);
	}
}

void liberarMemoriaLocal(double *datosGRD, double *pesoFila, double *pesoColumna, double *pesosGPU, double *pesoIdealAcum)
{
	free(datosGRD);
	free(pesosGPU);
	free(pesoFila);
	free(pesoColumna);
	free(pesoIdealAcum);
}

bool sonSubmallasAdyacentes(int4 submalla1, int4 submalla2)
{
	int inix1 = submalla1.x;
	int iniy1 = submalla1.y;
	int inix2 = submalla2.x;
	int iniy2 = submalla2.y;
	int finx1 = inix1 + submalla1.z;
	int finy1 = iniy1 + submalla1.w;
	int finx2 = inix2 + submalla2.z;
	int finy2 = iniy2 + submalla2.w;
	bool adyacentes = false;

	if ( ((inix2 == finx1) && (iniy2 < finy1) && (finy2 >= iniy1)) ||
		 ((iniy2 == finy1) && (inix2 < finx1) && (finx2 >= inix1)) ||
		 ((inix1 == finx2) && (iniy1 < finy2) && (finy1 >= iniy2)) ||
		 ((iniy1 == finy2) && (inix1 < finx2) && (finx1 >= inix2)) ) {
		adyacentes = true;
	}

	return adyacentes;
}

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
		int &modulo_velocidades, int &modulo_velocidades_max, int &modulo_caudales_max, int &flujo_momento,
		int &flujo_momento_max, int &tiempos_llegada)
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
	iss >> eta >> eta_max >> velocidades >> velocidades_max >> modulo_velocidades >> modulo_velocidades_max >> modulo_caudales_max >> flujo_momento >> flujo_momento_max >> tiempos_llegada;
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

void obtenerCvisNiveles(ifstream &fich, int numNiveles, double *cvisNivel)
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
	iss >> cvisNivel[0];
	for (i=1; i<numNiveles; i++) {
		// Si hay menos cvis que niveles, asignamos el último cvis al resto de niveles
		if (! (iss >> cvisNivel[i]))
			cvisNivel[i] = cvisNivel[i-1];
	}
}

void obtenerPesoGPU(ifstream &fich, double &peso)
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
	iss >> peso;
}

template void obtenerSiguienteDato<int>(ifstream &fich, int &dato);
template void obtenerSiguienteDato<double>(ifstream &fich, double &dato);
template void obtenerSiguienteDato<string>(ifstream &fich, string &dato);

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

// Devuelve 0 si todo ha ido bien, 1 si ha habido algún error
int cargarDatosProblema(string fich_ent, string &nombre_bati, string prefijo[MAX_LEVELS][MAX_GRIDS_LEVEL], int *no_hidros, int *numPesosJacobi,
		int *numNiveles, int *okada_flag, int *kajiura_flag, double *depth_kajiura, string &fich_okada, int *numFaults, int *usar_ventana,
		double2 **datosVolumenesNivel_1, double2 **datosVolumenesNivel_2, tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int64_t *numVolumenesNivel, int posSubmallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *leer_fichero_puntos,
		int *num_puntos_guardar, double *borde_sup, double *borde_inf, double *borde_izq, double *borde_der, double *tiempo_tot,
		double *tiempoGuardarNetCDF, VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL], double *tiempoGuardarSeries,
		double *CFL, int *tipo_friccion, string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL], double *mf0, double *vmax, double *epsilon_h,
		int *ratioRefNivel, bool *haySubmallasAdyacentesNivel, double *difh_at, int **xIniCluster, int **yIniCluster, double *L,
		double *H, double *Q, double *T, int numProcsX, int numProcsY, string fich_pesos)
{
	int i, j, k, l, m;
	int pos, pos_ini;
	int2 posPunto;
	int nvx, nvy;
	int inix, iniy;
	int ratio;
	int numVolxNivel0, numVolyNivel0;
	tipoDatosSubmalla *tds;
	double val, delta_lon, delta_lat;
	double radio_tierra = EARTH_RADIUS;
	double grad2rad = M_PI/180.0;
	// Stream para leer los ficheros de datos topográficos y el estado inicial
	ifstream fich2;
	// Directorio donde se encuentran los ficheros de datos
	string directorio;
	string fich_topo, fich_est;
	string fich_puntos;
	int sized = sizeof(double);
	double lon, lat;
	bool encontrado;
	bool leer_difh_at = false;
	// Variables para Okada estándar
	double LON_C[MAX_FAULTS], LAT_C[MAX_FAULTS], DEPTH_C[MAX_FAULTS], FAULT_L[MAX_FAULTS], FAULT_W[MAX_FAULTS];
	double STRIKE[MAX_FAULTS], DIP[MAX_FAULTS], RAKE[MAX_FAULTS], SLIP[MAX_FAULTS], defTime[MAX_FAULTS];
	// Variables para la ventana de computación de Okada
	double lon_centro, lat_centro, radioComp;
	// Variables para Okada triangular
	double LON_v[MAX_FAULTS][3];
	double LAT_v[MAX_FAULTS][3];
	double DEPTH_v[MAX_FAULTS][3];
	// Variables para gaussiana
	double lonGauss, latGauss;
	double heightGauss, sigmaGauss;
	// Vectores para el equilibrado de carga
	double *pesoFila, *pesoColumna;
	double *pesosGPU, *pesoIdealAcum;
	// Vector para la lectura de los ficheros grd
	string fich_def[MAX_FAULTS];
	double *datosGRD;
	float *datosGRD_float;
	int64_t tam_datosGRD, tam;

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
	*difh_at = -1.0;
	ifstream fich(fich_ent.c_str());
	// Leemos el fichero con los datos de la topografía
	obtenerSiguienteDato<string>(fich, nombre_bati);
	obtenerSiguienteDato<int>(fich, *no_hidros);
	if (*no_hidros == 1) {
		obtenerSiguienteDato<int>(fich, *numPesosJacobi);
		if ((*numPesosJacobi != 1) && (*numPesosJacobi != 16) && (*numPesosJacobi != 27)) {
			cerr << "Error: The number of Jacobi weights should be 1, 16 or 27" << endl;
			fich.close();
			return 1;
		}
	}
	obtenerSiguienteDato<string>(fich, fich_topo);
	obtenerSiguienteDato<int>(fich, *okada_flag);
	if ((*okada_flag != SEA_SURFACE_FROM_FILE) && (*okada_flag != OKADA_STANDARD) && (*okada_flag != OKADA_STANDARD_FROM_FILE) &&
		(*okada_flag != OKADA_TRIANGULAR) && (*okada_flag != OKADA_TRIANGULAR_FROM_FILE) && (*okada_flag != DEFORMATION_FROM_FILE) &&
		(*okada_flag != DYNAMIC_DEFORMATION) && (*okada_flag != GAUSSIAN))
	{
		cerr << "Error: The initialization flag should be " << SEA_SURFACE_FROM_FILE << ", " << OKADA_STANDARD << ", ";
		cerr << OKADA_STANDARD_FROM_FILE << ", " << OKADA_TRIANGULAR << ", " << OKADA_TRIANGULAR_FROM_FILE << ", ";
		cerr << DEFORMATION_FROM_FILE << ", " << DYNAMIC_DEFORMATION << " or " << GAUSSIAN << endl;
		fich.close();
		return 1;
	}
	if (*okada_flag == SEA_SURFACE_FROM_FILE) {
		// Leer el estado inicial de fichero
		obtenerSiguienteDato<string>(fich, fich_est);
	}
	else if (*okada_flag == OKADA_STANDARD) {
		// Aplicar Okada estándar
		obtenerSiguienteDato<int>(fich, *kajiura_flag);
		if (*kajiura_flag == 1)
			obtenerSiguienteDato<double>(fich, *depth_kajiura);
		obtenerSiguienteDato<int>(fich, *numFaults);
		if (*numFaults > MAX_FAULTS) {
			cerr << "Error: The maximum number of faults is " << MAX_FAULTS << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerDatosFallaOkadaStandard(fich, defTime[i], LON_C[i], LAT_C[i], DEPTH_C[i], FAULT_L[i],
				FAULT_W[i], STRIKE[i], DIP[i], RAKE[i], SLIP[i]);
		}
		obtenerSiguienteDato<int>(fich, *usar_ventana);
		if (*usar_ventana == 1) {
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
			cerr << "Error: File '" << directorio+fich_okada << "' not found" << endl;
			fich.close();
			return 1;
		}
		obtenerSiguienteDato<int>(fich, *usar_ventana);
		if (*usar_ventana == 1) {
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
			cerr << "Error: The maximum number of faults is " << MAX_FAULTS << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerDatosFallaOkadaTriangular(fich, defTime[i], LON_v[i][0], LAT_v[i][0], DEPTH_v[i][0], LON_v[i][1],
				LAT_v[i][1], DEPTH_v[i][1], LON_v[i][2], LAT_v[i][2], DEPTH_v[i][2], RAKE[i], SLIP[i]);
			if ((DEPTH_v[i][0] < 0.0) || (DEPTH_v[i][1] < 0.0) || (DEPTH_v[i][2] < 0.0)) {
				cerr << "Error: The Okada depths should be greater or equal to 0" << endl;
				fich.close();
				return 1;
			}
		}
		obtenerSiguienteDato<int>(fich, *usar_ventana);
		if (*usar_ventana == 1) {
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
			cerr << "Error: File '" << directorio+fich_okada << "' not found" << endl;
			fich.close();
			return 1;
		}
		obtenerSiguienteDato<int>(fich, *usar_ventana);
		if (*usar_ventana == 1) {
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
			cerr << "Error: The maximum number of faults is " << MAX_FAULTS << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerDatosFallaFichero(fich, defTime[i], fich_def[i]);
			if (! existeFichero(directorio+fich_def[i])) {
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
			cerr << "Error: Only one dynamic deformation is supported" << endl;
			fich.close();
			return 1;
		}
		for (i=0; i<(*numFaults); i++) {
			obtenerSiguienteDato<string>(fich, fich_def[i]);
			if (! existeFichero(directorio+fich_def[i])) {
				cerr << "Error: File '" << directorio+fich_def[i] << "' not found" << endl;
				fich.close();
				return 1;
			}
		}
	}
	else if (*okada_flag == GAUSSIAN) {
		// Construir gaussiana
		obtenerDatosGaussiana(fich, lonGauss, latGauss, heightGauss, sigmaGauss);
	}
	obtenerSiguienteDato<string>(fich, prefijo[0][0]);
	obtenerDatosGuardadoSubmalla(fich, guardarVariables[0][0].eta, guardarVariables[0][0].eta_max, guardarVariables[0][0].velocidades,
		guardarVariables[0][0].velocidades_max, guardarVariables[0][0].modulo_velocidades, guardarVariables[0][0].modulo_velocidades_max,
		guardarVariables[0][0].modulo_caudales_max, guardarVariables[0][0].flujo_momento, guardarVariables[0][0].flujo_momento_max,
		guardarVariables[0][0].tiempos_llegada);
	if (guardarVariables[0][0].tiempos_llegada != 0)
		leer_difh_at = true;
	if (! existeFichero(directorio+fich_topo)) {
		cerr << "Error: File '" << directorio+fich_topo << "' not found" << endl;
		fich.close();
		return 1;
	}
	if (*okada_flag == SEA_SURFACE_FROM_FILE) {
		if (! existeFichero(directorio+fich_est)) {
			cerr << "Error: File '" << directorio+fich_est << "' not found" << endl;
			fich.close();
			return 1;
		}
	}
	abrirGRD((directorio+fich_topo).c_str(), 0, 0, &numVolxNivel0, &numVolyNivel0);
	// Los datos topográficos y el estado inicial se leerán después

	if ((numVolxNivel0 < 2) || (numVolyNivel0 < 2)) {
		cerr << "Error: Mesh size too small. The number of rows and columns should be >= 2" << endl;
		fich.close();
		cerrarGRD(0, 0);
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *numNiveles);
	// Borrar este if al meter mallas anidadas
	if (*numNiveles > 1) {
		cerr << "Error: The number of levels should be 1" << endl;
		fich.close();
		return 1;
	}
	if (*numNiveles > MAX_LEVELS) {
		cerr << "Error: The maximum number of levels is " << MAX_LEVELS << endl;
		fich.close();
		cerrarGRD(0, 0);
		return 1;
	}
	ratioRefNivel[0] = 1;
	numSubmallasNivel[0] = 1;
	for (l=1; l<(*numNiveles); l++) {
		obtenerSiguienteDato<int>(fich, ratioRefNivel[l]);
		if ((ratioRefNivel[l] != 2) && (ratioRefNivel[l] != 4) && (ratioRefNivel[l] != 8) && (ratioRefNivel[l] != 16)) {
			cerr << "Error: The refinement ratio should be 2, 4, 8 or 16" << endl;
			fich.close();
			cerrarGRD(0, 0);
			return 1;
		}
		// Leemos el tamaño de las submallas
		obtenerSiguienteDato<int>(fich, numSubmallasNivel[l]);
		if (numSubmallasNivel[l] > MAX_GRIDS_LEVEL) {
			cerr << "Error: The maximum number of grids per level is " << MAX_GRIDS_LEVEL << endl;
			fich.close();
			cerrarGRD(0, 0);
			return 1;
		}
		numVolumenesNivel[l] = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			obtenerSiguienteDato<string>(fich, fich_topo);
			if (*okada_flag == SEA_SURFACE_FROM_FILE) {
				obtenerSiguienteDato<string>(fich, fich_est);
				if (! existeFichero(directorio+fich_est)) {
					cerr << "Error: File '" << directorio+fich_est << "' not found" << endl;
					fich.close();
					cerrarGRD(0, 0);
					return 1;
				}
			}
			if (! existeFichero(directorio+fich_topo)) {
				cerr << "Error: File '" << directorio+fich_topo << "' not found" << endl;
				fich.close();
				cerrarGRD(0, 0);
				return 1;
			}
			obtenerSiguienteDato<string>(fich, prefijo[l][i]);
			obtenerDatosGuardadoSubmalla(fich, guardarVariables[l][i].eta, guardarVariables[l][i].eta_max, guardarVariables[l][i].velocidades,
				guardarVariables[l][i].velocidades_max, guardarVariables[l][i].modulo_velocidades, guardarVariables[l][i].modulo_velocidades_max,
				guardarVariables[l][i].modulo_caudales_max, guardarVariables[l][i].flujo_momento, guardarVariables[l][i].flujo_momento_max,
				guardarVariables[l][i].tiempos_llegada);
			if (guardarVariables[l][i].tiempos_llegada != 0)
				leer_difh_at = true;
			abrirGRD((directorio+fich_topo).c_str(), l, i, &(submallasNivel[l][i].z), &(submallasNivel[l][i].w));
			numVolumenesNivel[l] += (submallasNivel[l][i].z + 4)*(submallasNivel[l][i].w + 4);
		}
	}
	*tiempoGuardarSeries = -1.0;
	obtenerSiguienteDato<double>(fich, *borde_sup);
	obtenerSiguienteDato<double>(fich, *borde_inf);
	obtenerSiguienteDato<double>(fich, *borde_izq);
	obtenerSiguienteDato<double>(fich, *borde_der);
	obtenerSiguienteDato<double>(fich, *tiempo_tot);
	obtenerTiemposGuardadoNetCDFNiveles(fich, *numNiveles, tiempoGuardarNetCDF);
	if (hayErrorEnTiemposGuardarNetCDF(*numNiveles, tiempoGuardarNetCDF)) {
		cerr << "Error: The NetCDF saving time of level 0 should be multiple of the NetCDF saving times of the other levels" << endl;
		fich.close();
		cerrarGRD(0, 0);
		for (l=1; l<(*numNiveles); l++) {
			for (i=0; i<numSubmallasNivel[l]; i++)
				cerrarGRD(l, i);
		}
		return 1;
	}
	obtenerSiguienteDato<int>(fich, *leer_fichero_puntos);
	if (*leer_fichero_puntos == 1) {
		obtenerSiguienteDato<string>(fich, fich_puntos);
		obtenerSiguienteDato<double>(fich, *tiempoGuardarSeries);
		if (! existeFichero(directorio+fich_puntos)) {
			cerr << "Error: File '" << directorio+fich_puntos << "' not found" << endl;
			fich.close();
			cerrarGRD(0, 0);
			for (l=1; l<(*numNiveles); l++) {
				for (i=0; i<numSubmallasNivel[l]; i++)
					cerrarGRD(l, i);
			}
			return 1;
		}
		if (*tiempoGuardarSeries < 0.0) {
			cerr << "Error: The saving time of the time series should be >= 0" << endl;
			fich.close();
			cerrarGRD(0, 0);
			for (l=1; l<(*numNiveles); l++) {
				for (i=0; i<numSubmallasNivel[l]; i++)
					cerrarGRD(l, i);
			}
			return 1;
		}
	}
	obtenerSiguienteDato<double>(fich, *CFL);
	obtenerSiguienteDato<double>(fich, *epsilon_h);
	obtenerSiguienteDato<int>(fich, *tipo_friccion);
	if (*tipo_friccion == FIXED_FRICTION) {
		obtenerSiguienteDato<double>(fich, *mf0);
	}
	else if (*tipo_friccion == VARIABLE_FRICTION_0) {
		obtenerSiguienteDato<string>(fich, fich_friccion[0][0]);
		if (! existeFichero(directorio+fich_friccion[0][0])) {
			cerr << "Error: File '" << directorio+fich_friccion[0][0] << "' not found" << endl;
			fich.close();
			cerrarGRD(0, 0);
			for (l=1; l<(*numNiveles); l++) {
				for (i=0; i<numSubmallasNivel[l]; i++)
					cerrarGRD(l, i);
			}
			return 1;
		}
	}
	else if (*tipo_friccion == VARIABLE_FRICTION_ALL) {
		for (l=0; l<(*numNiveles); l++) {
			for (i=0; i<numSubmallasNivel[l]; i++) {
				obtenerSiguienteDato<string>(fich, fich_friccion[l][i]);
				if (! existeFichero(directorio+fich_friccion[l][i])) {
					cerr << "Error: File '" << directorio+fich_friccion[l][i] << "' not found" << endl;
					fich.close();
					cerrarGRD(0, 0);
					for (l=1; l<(*numNiveles); l++) {
						for (i=0; i<numSubmallasNivel[l]; i++)
							cerrarGRD(l, i);
					}
					return 1;
				}
			}
		}
	}
	else {
		cerr << "Error: The friction type flag should be " << FIXED_FRICTION << ", " << VARIABLE_FRICTION_0 << " or " << VARIABLE_FRICTION_ALL << endl;
		fich.close();
		cerrarGRD(0, 0);
		for (l=1; l<(*numNiveles); l++) {
			for (i=0; i<numSubmallasNivel[l]; i++)
				cerrarGRD(l, i);
		}
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
	*Q = sqrt(9.81*pow((*H),3.0));
	*T = (*L)*(*H)/(*Q);
	*tiempo_tot /= *T;
	*tiempoGuardarSeries /= *T;
	*mf0 *= 9.81*(*mf0)*(*L)/pow((*H),4.0/3.0);
	*vmax /= (*Q)/(*H);
	*epsilon_h /= *H;
	radio_tierra /= *L;
	*depth_kajiura /= *H;
	for (l=0; l<(*numNiveles); l++) {
		tiempoGuardarNetCDF[l] /= *T;
	}
	fich.close();

	// Reservamos memoria
	submallasNivel[0][0].x = 0;
	submallasNivel[0][0].y = 0;
	submallasNivel[0][0].z = numVolxNivel0;
	submallasNivel[0][0].w = numVolyNivel0;
	numVolumenesNivel[0] = numVolxNivel0*numVolyNivel0;
	tam_datosGRD = numVolumenesNivel[0]*sizeof(float);
	for (l=0; l<(*numNiveles); l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			nvx = submallasNivel[l][i].z;
			nvy = submallasNivel[l][i].w;
			datosNivel[l][i].longitud = (double *) malloc(nvx*sizeof(double));
			datosNivel[l][i].latitud  = (double *) malloc(nvy*sizeof(double));
			tam = ((int64_t) nvx)*nvy*sizeof(float);
			if (tam > tam_datosGRD)
				tam_datosGRD = tam;
		}
	}
	datosGRD = (double *) malloc(tam_datosGRD);
	datosGRD_float = (float *) datosGRD;
	for (l=0; l<(*numNiveles); l++) {
		datosVolumenesNivel_1[l] = (double2 *) malloc(numVolumenesNivel[l]*sizeof(double2));
		datosVolumenesNivel_2[l] = (double2 *) malloc(numVolumenesNivel[l]*sizeof(double2));
	}
	if (datosVolumenesNivel_2[(*numNiveles)-1] == NULL) {
		cerr << "Error: Not enough CPU memory" << endl;
		liberarMemoria(*numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel,
			*xIniCluster, *yIniCluster);
		if (datosGRD != NULL) free(datosGRD);
		cerrarGRD(0, 0);
		for (l=1; l<(*numNiveles); l++) {
			for (i=0; i<numSubmallasNivel[l]; i++)
				cerrarGRD(l, i);
		}
		return 1;
	}

	// Leemos longitudes, latitudes y topografía
	nvx = submallasNivel[0][0].z;
	nvy = submallasNivel[0][0].w;
	leerLongitudGRD(0, 0, datosGRD);
	for (k=0; k<nvx; k++)
		datosNivel[0][0].longitud[k] = datosGRD[k];
	leerLatitudGRD(0, 0, datosGRD);
	for (k=0; k<nvy; k++)
		datosNivel[0][0].latitud[k] = datosGRD[k];
	leerBatimetriaGRD(0, 0, numVolxNivel0, numVolyNivel0, datosGRD_float);
	for (j=0; j<numVolyNivel0; j++) {
		for (i=0; i<numVolxNivel0; i++) {
			pos = j*numVolxNivel0 + i;
			val = -1.0*datosGRD_float[pos];
			val /= *H;
			datosVolumenesNivel_1[0][pos].y = val;
			// Si hay que aplicar Okada o gaussiana inicializamos el estado del volumen con agua plana.
			// Si no, el estado se inicializará después al leer el fichero
			datosVolumenesNivel_1[0][pos].x = ((val > 0.0) ? val : 0.0);
			datosVolumenesNivel_2[0][pos].x = 0.0;
			datosVolumenesNivel_2[0][pos].y = 0.0;
		}
	}
	cerrarGRD(0, 0);
	for (l=1; l<(*numNiveles); l++) {
		pos_ini = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			nvx = submallasNivel[l][i].z;
			nvy = submallasNivel[l][i].w;
			leerLongitudGRD(l, i, datosGRD);
			for (k=0; k<nvx; k++)
				datosNivel[l][i].longitud[k] = datosGRD[k];
			leerLatitudGRD(l, i, datosGRD);
			for (k=0; k<nvy; k++)
				datosNivel[l][i].latitud[k] = datosGRD[k];
			leerBatimetriaGRD(l, i, nvx, nvy, datosGRD_float);
			for (j=0; j<nvy; j++) {
				for (k=0; k<nvx; k++) {
					pos = pos_ini + (j+2)*(nvx+4) + k+2;
					m = j*nvx + i;
					val = -1.0*datosGRD_float[m];
					val /= *H;
					datosVolumenesNivel_1[l][pos].y = val;
					// Si hay que aplicar Okada o gaussiana inicializamos el estado del volumen con agua plana.
					// Si no, el estado se inicializará después al leer el fichero
					datosVolumenesNivel_1[l][pos].x = ((val > 0.0) ? val : 0.0);
					datosVolumenesNivel_2[l][pos].x = 0.0;
					datosVolumenesNivel_2[l][pos].y = 0.0;
				}
			}
			pos_ini += (nvx+4)*(nvy+4);
			cerrarGRD(l, i);
		}
	}

	// Asignamos la esquina superior izquierda de las submallas en submallasNivel
	submallaNivelSuperior[0][0] = -1;
	posSubmallaNivelSuperior[0][0] = -1;
	for (l=1; l<(*numNiveles); l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			nvx = submallasNivel[l][i].z;
			nvy = submallasNivel[l][i].w;
			lon = datosNivel[l][i].longitud[0];
			lat = datosNivel[l][i].latitud[0];

			// Ponemos en k el índice de la submalla del nivel superior donde está la submalla i del nivel l,
			// y ponemos en pos la posición donde empiezan sus datos en los vectores datosVolumenesNivel
			encontrado = false;
			pos = k = 0;
			while ((k < numSubmallasNivel[l-1]) && (! encontrado)) {
				if ( (lon < datosNivel[l-1][k].longitud[submallasNivel[l-1][k].z-1]) &&
					 (datosNivel[l-1][k].longitud[0] < datosNivel[l][i].longitud[nvx-1]) ) {
					if ( (lat < datosNivel[l-1][k].latitud[submallasNivel[l-1][k].w-1]) &&
						 (datosNivel[l-1][k].latitud[0] < datosNivel[l][i].latitud[nvy-1]) ) {
						encontrado = true;
					}
					else {
						pos += (submallasNivel[l-1][k].z + 4)*(submallasNivel[l-1][k].w + 4);
						k++;
					}
				}
				else {
					pos += (submallasNivel[l-1][k].z + 4)*(submallasNivel[l-1][k].w + 4);
					k++;
				}
			}
			if (! encontrado) {
				cerr << "Error: Submesh " << i+1 << " of level "<< l << " is not contained in any submesh of level " << l-1 << endl;
				liberarMemoria(*numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel,
					*xIniCluster, *yIniCluster);
				if (datosGRD != NULL) free(datosGRD);
				return 1;
			}

			submallaNivelSuperior[l][i] = k;
			posSubmallaNivelSuperior[l][i] = pos;
			nvx = submallasNivel[l-1][k].z;
			nvy = submallasNivel[l-1][k].w;

			j = 0;
			while ((datosNivel[l-1][k].longitud[j] < lon) && (j < nvx))
				j++;
			submallasNivel[l][i].x = j*ratioRefNivel[l];

			j = 0;
			while ((datosNivel[l-1][k].latitud[j] < lat) && (j < nvy))
				j++;
			submallasNivel[l][i].y = j*ratioRefNivel[l];
		}
	}

	// Comprobamos que las submallas encajan correctamente y tienen el tamaño adecuado
	for (l=1; l<(*numNiveles); l++) {
		for (i=0; i<numSubmallasNivel[l]; i++) {
			if ( (submallasNivel[l][i].x % ratioRefNivel[l] != 0) || (submallasNivel[l][i].y % ratioRefNivel[l] != 0) ||
				 (submallasNivel[l][i].z % ratioRefNivel[l] != 0) || (submallasNivel[l][i].w % ratioRefNivel[l] != 0) ) {
				cerr << "Error: Submesh " << i+1 << " of level " << l << " does not fit properly into cells of level " << l-1 << endl;
				liberarMemoria(*numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel,
					*xIniCluster, *yIniCluster);
				if (datosGRD != NULL) free(datosGRD);
				return 1;
			}
		}
	}

	// Comprobamos si hay submallas adyacentes
	memset(haySubmallasAdyacentesNivel, 0, (*numNiveles)*sizeof(bool));
	l = (*numNiveles) - 1;
	while ((l >= 0) && (! haySubmallasAdyacentesNivel[l])) {
		i = 0;
		while ((i < numSubmallasNivel[l]) && (! haySubmallasAdyacentesNivel[l])) {
			m = submallaNivelSuperior[l][i];
			j = i+1;
			while ((j < numSubmallasNivel[l]) && (! haySubmallasAdyacentesNivel[l])) {
				if (submallaNivelSuperior[l][j] == m) {
					if (sonSubmallasAdyacentes(submallasNivel[l][i], submallasNivel[l][j])) {
						for (k=0; k<=l; k++)
							haySubmallasAdyacentesNivel[k] = true;
					}
				}
				j++;
			}
			i++;
		}
		l--;
	}

	// EQUILIBRADO DE CARGA
	// Reservamos memoria
	k = numProcsX*numProcsY;
	pesosGPU = (double *) malloc(k*sizeof(double));
	pesoFila = (double *) malloc(numVolyNivel0*sizeof(double));
	pesoColumna = (double *) malloc(numVolxNivel0*sizeof(double));
	pesoIdealAcum = (double *) malloc(max(numProcsX,numProcsY)*sizeof(double));
	*xIniCluster = (int *) malloc(numProcsX*sizeof(int));
	*yIniCluster = (int *) malloc(numProcsY*sizeof(int));
	if ((pesosGPU == NULL) || (pesoColumna == NULL) || (pesoFila == NULL) || (pesoIdealAcum == NULL) ||
		(*xIniCluster == NULL) || (*yIniCluster == NULL)) {
		if (datosGRD != NULL)	   free(datosGRD);
		if (pesosGPU != NULL)      free(pesosGPU);
		if (pesoFila != NULL)      free(pesoFila);
		if (pesoColumna != NULL)   free(pesoColumna);
		if (pesoIdealAcum != NULL) free(pesoIdealAcum);
		liberarMemoria(*numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel,
			*xIniCluster, *yIniCluster);
		return 1;
	}

	// Leemos los pesos de cada GPU
	if (fich_pesos != "") {
		// Hay fichero de pesos de las GPUs
		fich2.open(fich_pesos.c_str(), ios::in);
		i = 0;
		encontrado = false;
		val = 0.0;
		while ((i < k) && (! encontrado)) {
			obtenerPesoGPU(fich2, pesosGPU[i]);
			val += pesosGPU[i];
			if (fich2.eof()) {
				encontrado = true;
				if (k == 1) {
					cerr << "Error: There are less weights than processes in the file '" << fich_pesos << "'. ";
					cerr << "It should be 1 weight" << endl;
				}
				else {
					cerr << "Error: There are less weights than processes in the file '" << fich_pesos << "'. ";
					cerr << "There should be " << k << " weights" << endl;
				}
			}
			else if (pesosGPU[i] <= 0.0) {
				encontrado = true;
				cerr << "Error: The GPU weights should be greater than 0" << endl;
			}
			i++;
		}
		fich2.close();

		if (encontrado) {
			liberarMemoriaLocal(datosGRD, pesoFila, pesoColumna, pesosGPU, pesoIdealAcum);
			liberarMemoria(*numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel,
				*xIniCluster, *yIniCluster);
			return 1;
		}

		// Normalizamos los pesos para que tengan media igual a 1
		for (i=0; i<k; i++)
			pesosGPU[i] = (pesosGPU[i]/val)*k;
	}
	else {
		// No hay fichero de pesos de las GPUs
		for (i=0; i<numProcsX*numProcsY; i++)
			pesosGPU[i] = 1.0;
	}

	(*xIniCluster)[0] = 0;
	(*yIniCluster)[0] = 0;
	if (numProcsX > 1) {
		// Obtenemos el equilibrado en la dimensión X (por columnas)
		obtenerPesoColumnas(datosVolumenesNivel_1, submallasNivel, numSubmallasNivel, submallaNivelSuperior,
			ratioRefNivel, *numNiveles, pesoColumna);
		m = obtenerEquilibradoX(pesoColumna, numVolxNivel0, numProcsX, numProcsY, *xIniCluster, *numNiveles, submallasNivel,
				numSubmallasNivel, submallaNivelSuperior, ratioRefNivel, pesosGPU, pesoIdealAcum);
		if (m == 1) {
			liberarMemoriaLocal(datosGRD, pesoFila, pesoColumna, pesosGPU, pesoIdealAcum);
			liberarMemoria(*numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel,
				*xIniCluster, *yIniCluster);
			return 1;
		}
	}
	if (numProcsY > 1) {
		// Obtenemos el equilibrado en la dimensión Y (por filas)
		obtenerPesoFilas(datosVolumenesNivel_1, submallasNivel, numSubmallasNivel, submallaNivelSuperior,
			ratioRefNivel, *numNiveles, pesoFila);
		m = obtenerEquilibradoY(pesoFila, numVolyNivel0, numProcsX, numProcsY, *yIniCluster, *numNiveles, submallasNivel,
				numSubmallasNivel, submallaNivelSuperior, ratioRefNivel, pesosGPU, pesoIdealAcum);
		if (m == 1) {
			liberarMemoriaLocal(datosGRD, pesoFila, pesoColumna, pesosGPU, pesoIdealAcum);
			liberarMemoria(*numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel,
				*xIniCluster, *yIniCluster);
			return 1;
		}
	}

	liberarMemoriaLocal(datosGRD, pesoFila, pesoColumna, pesosGPU, pesoIdealAcum);

	return 0;
}

void mostrarDatosProblema(string version, int no_hidros, int numPesosJacobi, int numNiveles, int okada_flag, int kajiura_flag,
		double depth_kajiura, string fich_okada, int numFaults, tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double tiempo_tot, double *tiempoGuardarNetCDF,
		int leer_fichero_puntos, double tiempoGuardarSeries, double CFL, int tipo_friccion, string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL],
		double mf0, double vmax, double epsilon_h, double difh_at, int *ratioRefNivel, int *xIniCluster, int *yIniCluster, double L,
		double H, double Q, double T, int numProcsX, int numProcsY)
{
	int i, l;
	int nvx, nvy;
	int nsubmallas;

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
		cout << "Friction type: fixed" << endl;
		cout << "Water-bottom friction: " << sqrt(mf0*pow(H,4.0/3.0)/(9.81*L)) << endl;
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
	cout << "Maximum allowed velocity of water: " << vmax*(Q/H) << endl;
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
	nvx = submallasNivel[0][0].z;
	nvy = submallasNivel[0][0].w;
	cout << "Level 0" << endl;
	cout << "  Volumes: " << nvx << " x " << nvy << " = " << nvx*nvy << endl;
	cout << "  Longitude: [" << datosNivel[0][0].longitud[0] << ", " << datosNivel[0][0].longitud[nvx-1] << "]" << endl;
	cout << "  Latitude: [" << datosNivel[0][0].latitud[0] << ", " << datosNivel[0][0].latitud[nvy-1] << "]" << endl;
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
	cout << "Load balancing data" << endl;
	for (i=0; i<numProcsX-1; i++)
		cout << "Process " << i << " in X: columns " << xIniCluster[i] << "-" << xIniCluster[i+1]-1 << endl;
	cout << "Process " << numProcsX-1 << " in X: columns " << xIniCluster[numProcsX-1] << "-" << submallasNivel[0][0].z-1 << endl;
	for (i=0; i<numProcsY-1; i++)
		cout << "Process " << i << " in Y: rows " << yIniCluster[i] << "-" << yIniCluster[i+1]-1 << endl;
	cout << "Process " << numProcsY-1 << " in Y: rows " << yIniCluster[numProcsY-1] << "-" << submallasNivel[0][0].w-1 << endl;
}

// Formato del fichero de salida (todos los datos son int):
// <num_niveles>
// Para cada nivel:
//   <hay_submallas_adyacentes>
//   <num_submallas>
//   Para cada submalla
//     <inicio_x>
//     <inicio_y>
//     <submalla_superior>
// <num_procesos_en_x>
// <num_procesos_en_y>
// Para cada proceso en x:
//   <inicio_x_nivel0>
// Para cada proceso en y:
//   <inicio_y_nivel0>
void escribirFicheroSalida(string &nombre_bati, int *numNiveles, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
				int *numSubmallasNivel, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL],
				bool *haySubmallasAdyacentesNivel, int *xIniCluster, int *yIniCluster, int *numProcsX, int *numProcsY)
{
	int l, k, val;
	int hay_submallas_adyacentes;
	int sizei = sizeof(int);
	int numProcs = (*numProcsX)*(*numProcsY);
	char nombre_fich[256];
	FILE *fp;

	if ((*numNiveles == 1) && (numProcs == 1))
		sprintf(nombre_fich, "%s_lb_01level_01proc.bin", nombre_bati.c_str());
	else if ((*numNiveles > 1) && (numProcs == 1))
		sprintf(nombre_fich, "%s_lb_%02dlevels_01proc.bin", nombre_bati.c_str(), *numNiveles);
	else if ((*numNiveles == 1) && (numProcs > 1))
		sprintf(nombre_fich, "%s_lb_01level_%02dprocs.bin", nombre_bati.c_str(), numProcs);
	else
		sprintf(nombre_fich, "%s_lb_%02dlevels_%02dprocs.bin", nombre_bati.c_str(), *numNiveles, numProcs);
	cout << endl << "Writing '" << nombre_fich << "'" << endl;
	fp = fopen(nombre_fich, "wb");
	fwrite(numNiveles, sizei, 1, fp);
	for (l=0; l<(*numNiveles); l++) {
		hay_submallas_adyacentes = (haySubmallasAdyacentesNivel[l] ? 1 : 0);
		fwrite(&hay_submallas_adyacentes, sizei, 1, fp);
		fwrite(numSubmallasNivel+l, sizei, 1, fp);
		for (k=0; k<numSubmallasNivel[l]; k++) {
			// Coordenadas de inicio de la submalla
			val = submallasNivel[l][k].x;
			fwrite(&val, sizei, 1, fp);
			val = submallasNivel[l][k].y;
			fwrite(&val, sizei, 1, fp);
			// Submalla del nivel superior
			val = submallaNivelSuperior[l][k];
			fwrite(&val, sizei, 1, fp);
		}
	}

	fwrite(numProcsX, sizei, 1, fp);
	fwrite(numProcsY, sizei, 1, fp);
	for (k=0; k<(*numProcsX); k++)
		fwrite(xIniCluster+k, sizei, 1, fp);
	for (k=0; k<(*numProcsY); k++)
		fwrite(yIniCluster+k, sizei, 1, fp);
	fclose(fp);
}

#endif

