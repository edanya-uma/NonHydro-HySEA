/*************************************************************
 NH-HySEA numerical model v1.0.1
 Copyright (C) 2010-2024
 EDANYA Research Group, University of Malaga (Spain).

 NH-HySEA can not be copied, modified and/or distributed
 without the express permission of the EDANYA Research Group.
 NH-HySEA is distributed under license. For more information,
 visit:
 https://edanya.uma.es/hysea/tsunami-hysea_license.html
*************************************************************/

#include "Constantes.hxx"
#include "Problema_grd.cxx"

/*****************/
/* Funciones GPU */
/*****************/

extern "C" int comprobarSoporteCUDA();
extern "C" int shallowWater(TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], int no_hidros, int numPesosJacobi, int numNiveles,
			int okada_flag, int kajiura_flag, double depth_kajiura, char *fich_okada, int numFaults, int numEstadosDefDinamica, double *LON_C,
			double *LAT_C, double *DEPTH_C, double *FAULT_L, double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *defTime,
			double DEPTH_v[MAX_FAULTS][4], double2 vc[MAX_FAULTS][4], double *LONCTRI, double *LATCTRI, double SLIPVEC[MAX_FAULTS][3],
			double **deformacionNivel0, string *fich_def, double lonGauss, double latGauss, double heightGauss, double sigmaGauss,
			float *batiOriginal[MAX_LEVELS], double2 **datosVolumenesNivel_1, double3 **datosVolumenesNivel_2, double **datosPnh, double *datosGRD,
			tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int **posCopiaNivel, int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
			int *numSubmallasNivel, int2 iniGlobalSubmallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int4 *submallasDeformacion, bool **refinarNivel,
			int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int posSubmallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int leer_fichero_puntos,
			int4 *posicionesVolumenesGuardado, int numPuntosGuardarAnt, int numPuntosGuardarTotal, double *lonPuntos, double *latPuntos,
			int numVolxTotalNivel0, int numVolyTotalNivel0, int64_t *numVolumenesNivel, int64_t *numVerticesNivel, double Hmin, char *nombre_bati,
			string prefijo[MAX_LEVELS][MAX_GRIDS_LEVEL], double borde_sup, double borde_inf, double borde_izq, double borde_der,
			int tam_spongeSup, int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer, double tiempo_tot, double *tiempoGuardarNetCDF,
			VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL], double tiempoGuardarSeries, double CFL, int tipo_friccion,
			string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL], double **friccionesNivel, double vmax, double epsilon_h, int continuar_simulacion,
			double tiempo_continuar, int *numEstadoNetCDF, int *ratioRefNivel, int *ratioRefAcumNivel, bool *haySubmallasAdyacentesNivel,
			double difh_at, double L, double H, double U, double T, int num_procs, int num_procsX, int num_procsY, int id_hebra,
			MPI_Comm comm_cartesiano, char *version, double *tiempo);

/*********************/
/* Fin funciones GPU */
/*********************/

void mostrarFormatoProgramaGRD(char *argv[], string version)
{
	cerr << "/*************************************************************" << endl;
	cerr << " NH-HySEA numerical model v" << version << endl;
	cerr << " Copyright (C) 2010-2024                                      " << endl;
	cerr << " EDANYA Research Group, University of Malaga (Spain).         " << endl;
	cerr << "                                                              " << endl;
	cerr << " NH-HySEA can not be copied, modified and/or distributed      " << endl;
	cerr << " without the express permission of the EDANYA Research Group. " << endl;
	cerr << " NH-HySEA is distributed under license. For more information, " << endl;
	cerr << " visit:                                                       " << endl;
	cerr << " https://edanya.uma.es/hysea/tsunami-hysea_license.html       " << endl;
	cerr << "*************************************************************/" << endl;
	cerr << endl;
	cerr << "Use:" << endl;
	cerr << argv[0] << " dataFile" << endl << endl; 
	cerr << "dataFile format:" << endl;
	cerr << "  Problem name" << endl;
	cerr << "  Non hidrostatic simulation (0: no, 1: yes)" << endl;
	cerr << "  If 1:" << endl;
	cerr << "    Number of Jacobi weights (1, 16 or 27)" << endl;
	cerr << "  Bathymetry file" << endl;
	cerr << "  Initialization of states (" << SEA_SURFACE_FROM_FILE << ": Sea surface displacement from file," << endl;
	cerr << "                            " << OKADA_STANDARD << ": Standard Okada," << endl;
	cerr << "                            " << OKADA_STANDARD_FROM_FILE << ": Standard Okada from file," << endl;
	cerr << "                            " << OKADA_TRIANGULAR << ": Triangular Okada," << endl;
	cerr << "                            " << OKADA_TRIANGULAR_FROM_FILE << ": Triangular Okada from file," << endl;
	cerr << "                            " << DEFORMATION_FROM_FILE << ": Sea floor deformation from file," << endl;
	cerr << "                            " << DYNAMIC_DEFORMATION << ": Sea floor dynamic deformation from file," << endl;
	cerr << "                            " << GAUSSIAN << ": Gaussian)" << endl;
	cerr << "  If " << SEA_SURFACE_FROM_FILE << ":" << endl;
	cerr << "    Initial state file" << endl;
	cerr << "  Else if " << OKADA_STANDARD << ":" << endl;
	cerr << "    Apply Kajiura filter to the Okada deformation (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Reference depth for Kajiura filter (m)" << endl;
	cerr << "    Number of faults (>= 1)" << endl;
	cerr << "    For every fault, a line containing:" << endl;
	cerr << "      Time(sec) Lon_epicenter Lat_epicenter Depth_hypocenter(km) Fault_length(km) Fault_width(km) Strike Dip Rake Slip(m)" << endl;
	cerr << "    Use Okada computation window (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Lon_center Lat_center Radius(km)" << endl;
	cerr << "  Else if " << OKADA_STANDARD_FROM_FILE << ":" << endl;
	cerr << "    Apply Kajiura filter to the Okada deformation (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Reference depth for Kajiura filter (m)" << endl;
	cerr << "    File with faults" << endl;
	cerr << "    Use Okada computation window (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Lon_center Lat_center Radius(km)" << endl;
	cerr << "  Else if " << OKADA_TRIANGULAR << ":" << endl;
	cerr << "    Apply Kajiura filter to the Okada deformation (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Reference depth for Kajiura filter (m)" << endl;
	cerr << "    Number of faults (>= 1)" << endl;
	cerr << "    For every fault, a line containing:"<< endl;
	cerr << "      Time(sec) Lon_v1 Lat_v1 Depth_v1(km) Lon_v2 Lat_v2 Depth_v2(km) Lon_v3 Lat_v3 Depth_v3(km) Rake Slip(m)" << endl;
	cerr << "    Use Okada computation window (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Lon_center Lat_center Radius(km)" << endl;
	cerr << "  Else if " << OKADA_TRIANGULAR_FROM_FILE << ":" << endl;
	cerr << "    Apply Kajiura filter to the Okada deformation (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Reference depth for Kajiura filter (m)" << endl;
	cerr << "    File with faults" << endl;
	cerr << "    Use Okada computation window (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Lon_center Lat_center Radius(km)" << endl;
	cerr << "  Else if " << DEFORMATION_FROM_FILE << ":" << endl;
	cerr << "    Apply Kajiura filter to the deformation (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Reference depth for Kajiura filter (m)" << endl;
	cerr << "    Number of faults (>= 1)" << endl;
	cerr << "    For every fault, a line containing:"<< endl;
	cerr << "      Time(sec) File with the accumulated deformation" << endl;
	cerr << "  Else if " << DYNAMIC_DEFORMATION << ":" << endl;
	cerr << "    Apply Kajiura filter to the deformation (0: no, 1: yes)" << endl;
	cerr << "    If 1:" << endl;
	cerr << "      Reference depth for Kajiura filter (m)" << endl;
	cerr << "    Number of faults (it should be 1)" << endl;
	cerr << "    For every fault, a line containing:"<< endl;
	cerr << "      File with the dynamic deformation" << endl;
	cerr << "  Else if " << GAUSSIAN << ":" << endl;
	cerr << "    A line containing:" << endl;
	cerr << "      Lon_center Lat_center Height(m) Sigma(km)" << endl;
	cerr << "  NetCDF file prefix" << endl;
	cerr << "  A line specifying if the following variables are saved (1: save, 0: do not save):" << endl;
	cerr << "    eta maximum_eta velocities maximum_velocities modulus_of_velocity maximum_modulus_of_velocity maximum_modulus_of_mass_flow non_hydrostatic_pressure momentum_flux maximum_momentum_flux arrival_times" << endl;
	cerr << "  Number of levels (should be 1)" << endl;
/*	cerr << "  If > 1:" << endl;
	cerr << "    For each level below 0:" << endl;
	cerr << "      Refinement ratio" << endl;
	cerr << "      Number of submeshes" << endl;
	cerr << "      For each submesh:" << endl;
	cerr << "        Bathymetry file" << endl;
	cerr << "        If (initialization of states == " << SEA_SURFACE_FROM_FILE << "):" << endl;
	cerr << "          Initial state file" << endl;
	cerr << "        NetCDF file prefix" << endl;
	cerr << "        A line specifying if the following variables are saved (1: save, 0: do not save):" << endl;
	cerr << "          eta maximum_eta velocities maximum_velocities modulus_of_velocity maximum_modulus_of_velocity maximum_modulus_of_mass_flow momentum_flux maximum_momentum_flux arrival_times" << endl;*/
	cerr << "  Upper border condition (1: open, -1: wall)" << endl;
	cerr << "  Lower border condition" << endl;
	cerr << "  Left border condition" << endl;
	cerr << "  Right border condition" << endl;
	cerr << "  Simulation time (sec)" << endl;
	cerr << "  Saving time of NetCDF files (sec) (-1: do not save)" << endl;
	cerr << "  Read points from file (0: no, 1: yes)" << endl;
	cerr << "  If 1:" << endl;
	cerr << "    File with points" << endl;
	cerr << "    Saving time of time series (sec)" << endl;
	cerr << "  CFL" << endl;
	cerr << "  Epsilon h (m)" << endl;
	cerr << "  Friction type (" << FIXED_FRICTION << ": fixed," << endl;
	cerr << "                 " << VARIABLE_FRICTION_0 << ": Variable friction specifying the frictions of the level 0 mesh," << endl;
	cerr << "                 " << VARIABLE_FRICTION_ALL << ": Variable friction specifying the frictions of all the submeshes)" << endl;
	cerr << "  If " << FIXED_FRICTION << ":" << endl;
	cerr << "    Water-bottom friction (Manning coefficient)" << endl;
	cerr << "  Else if " << VARIABLE_FRICTION_0 << ":" << endl;
	cerr << "    File with water-bottom frictions" << endl;
	cerr << "  Else if " << VARIABLE_FRICTION_ALL << ":" << endl;
	cerr << "    For each level:" << endl;
	cerr << "      For each submesh:" << endl;
	cerr << "        File with water-bottom frictions" << endl;
	cerr << "  Maximum allowed velocity of water" << endl;
	cerr << "  L (typical length)" << endl;
	cerr << "  H (typical depth)" << endl;
	cerr << "  if (arrival times are stored):" << endl;
	cerr << "    Threshold for arrival times (m)" << endl;
	cerr << endl;
}

int main(int argc, char *argv[])
{
	string version = "1.0.1";
	TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL];
	double2 *datosVolumenesNivel_1[MAX_LEVELS];  // h, H
	double3 *datosVolumenesNivel_2[MAX_LEVELS];  // qx, qy, qz
	double *datosPnh[MAX_LEVELS];
	tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int *posCopiaNivel[MAX_LEVELS];
	float *batiOriginal[MAX_LEVELS];
	bool *refinarNivel[MAX_LEVELS];
	int4 *posicionesVolumenesGuardado = NULL;
	double2 *bufferEnviosMPISupInf_1 = NULL;
	double2 *bufferEnviosMPIIzqDer_1 = NULL;
	double3 *bufferEnviosMPISupInf_2 = NULL;
	double3 *bufferEnviosMPIIzqDer_2 = NULL;
	double *bufferEnviosMPISupInf_P = NULL;
	double *bufferEnviosMPIIzqDer_P = NULL;
	int leer_fichero_puntos;
	// numPuntosGuardarAnt es el número de puntos del fichero NetCDF de series de tiempos
	// de la simulación anterior (0 si no se continúa la simulación). numPuntosGuardarTotal
	// es la suma de numPuntosGuardarAnt más el número de puntos actual
	int numPuntosGuardarAnt, numPuntosGuardarTotal;
	int soporteCUDA, err, err2;
	int numNiveles, numFaults;
	int numVolxNivel0, numVolyNivel0;
	int64_t numVolumenesNivel[MAX_LEVELS];
	int64_t numVerticesNivel[MAX_LEVELS];
	int no_hidros, numPesosJacobi;
	double borde_sup, borde_inf, borde_izq, borde_der;
	int tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer;
	double tiempo_tot;
	double tiempoGuardarNetCDF[MAX_LEVELS];
	double tiempoGuardarSeries;
	VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL];
	double Hmin, CFL, vmax;
	double epsilon_h, difh_at;
	double L, H, U, T;
	int tipo_friccion;
	string fich_okada;
	string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL];
	// Ficheros con las deformaciones de Okada si la inicialización es DEFORMATION_FROM_FILE o DYNAMIC_DEFORMATION
	string fich_def[MAX_FAULTS];
	double *friccionesNivel[MAX_LEVELS];
	int ratioRefNivel[MAX_LEVELS];
	bool haySubmallasAdyacentesNivel[MAX_LEVELS];
	char fich_ent_char[256];
	string fich_ent;
	string nombre_bati;
	string prefijo[MAX_LEVELS][MAX_GRIDS_LEVEL];  // Prefijos de los ficheros NetCDF
	int numSubmallasNivel[MAX_LEVELS+1];
	int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	// iniGlobalSubmallasNivel: x->inicio global de la submalla en x; y->inicio global de la submalla en y
	// (iniGlobalSubmallasNivel y ratioRefAcumNivel se usan en Volumen_kernel al aplicar el sponge layer)
	int2 iniGlobalSubmallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int ratioRefAcumNivel[MAX_LEVELS];
	int continuar_simulacion;
	double tiempo_continuar;
	// Indice del estado que se va guardando en cada nivel en los ficheros NetCDF
	int numEstadoNetCDF[MAX_LEVELS];
	double *datosGRD = NULL;
	// Longitudes y latitudes de los puntos a guardar
	double *lonPuntos = NULL;
	double *latPuntos = NULL;
	// Variables para Okada estándar
	int okada_flag, kajiura_flag;
	double depth_kajiura;
	double LON_C[MAX_FAULTS], LAT_C[MAX_FAULTS], DEPTH_C[MAX_FAULTS], FAULT_L[MAX_FAULTS], FAULT_W[MAX_FAULTS];
	double STRIKE[MAX_FAULTS], DIP[MAX_FAULTS], RAKE[MAX_FAULTS], SLIP[MAX_FAULTS], defTime[MAX_FAULTS];
	int4 submallasDeformacion[MAX_FAULTS];
	double *deformacionNivel0[MAX_FAULTS];
	double *deformacionAcumNivel0;
	// Variables para Okada triangular
	double2 LON_LAT_v[MAX_FAULTS][3];
	double DEPTH_v[MAX_FAULTS][4];
	double2 vc[MAX_FAULTS][4];
	double LONCTRI[MAX_FAULTS], LATCTRI[MAX_FAULTS];
	double SLIPVEC[MAX_FAULTS][3];
	// Variables para deformación dinámica
	int numEstadosDefDinamica;
	// Variables para gaussiana
	double lonGauss, latGauss;
	double heightGauss, sigmaGauss;
	// submallaNivelSuperior[l][i] indica la submalla del nivel l-1
	// (empieza por 0) que contiene la submalla i del nivel l.
	// posSubmallaNivelSuperior indica la posición en el vector datosVolumenesNivel
	// donde empiezan los datos de la malla del nivel superior
	int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int posSubmallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL];
	double tiempo_gpu, tiempo_multigpu;
	double tiempo_lectura0, tiempo_lectura1;
	double tiempo_lectura, tiempo_lectura_total;
	// Variables para MPI
	MPI_Comm comm_cartesiano;
	MPI_Status status;
	int num_procsX, num_procsY, num_procs;  // num_procs = num_procsX*num_procsY
	int id_hebra;
	int modo_hebras;

	err = err2 = 0;
#if (FILE_WRITING_MODE == 1)
	// Escritura síncrona de ficheros NetCDF
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &modo_hebras);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_hebra);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#else
	// Escritura asíncrona de ficheros NetCDF
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &modo_hebras);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_hebra);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	if (modo_hebras != MPI_THREAD_MULTIPLE) {
		if (id_hebra == 0)
			cerr << "Error: MPI_THREAD_MULTIPLE not supported, needed for asynchronous file writing" << endl;
		err = 1;
	}
#endif

	if (err == 0) {
		if (id_hebra == 0) {
			// El proceso 0 lee los datos de entrada
			if (argc < 2) {
				mostrarFormatoProgramaGRD(argv, version);
				err = 1;
			}
			else {
				// Fichero de datos
				fich_ent = argv[1];
				strcpy(fich_ent_char, argv[1]);
				if (! existeFichero(fich_ent)) {
					cerr << "Error in process " << id_hebra << ": File '" << fich_ent << "' not found" << endl;
					err = 1;
				}
			}
		}
	}

	// El proceso 0 envía err y fich_prob al resto de procesos
	MPI_Bcast (&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (fich_ent_char, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
	if (err == 0) {
		// No ha habido error
		// Todos los procesos ejecutan esto

		// Comprobamos si la tarjeta gráfica soporta CUDA
		soporteCUDA = comprobarSoporteCUDA();
		if (soporteCUDA == 1) {
			fprintf(stderr, "Error in process %d: There is no graphics card\n", id_hebra);
			err = 1;
		}
		else if (soporteCUDA == 2) {
			fprintf(stderr, "Error in process %d: There is no graphics card supporting CUDA\n", id_hebra);
			err = 1;
		}

		if (err == 0) {
			fich_ent = fich_ent_char;
			fprintf(stdout, "Process %d loading data\n", id_hebra);
			tiempo_lectura0 = MPI_Wtime();
			err = cargarDatosProblema(fich_ent, datosClusterCPU, nombre_bati, prefijo, &no_hidros, &numPesosJacobi, &numNiveles, &okada_flag,
					&kajiura_flag, &depth_kajiura, fich_okada, &numFaults, &numEstadosDefDinamica, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W,
					STRIKE, DIP, RAKE, SLIP, defTime, LON_LAT_v, DEPTH_v, vc, LONCTRI, LATCTRI, SLIPVEC, deformacionNivel0, &deformacionAcumNivel0,
					fich_def, &lonGauss, &latGauss, &heightGauss, &sigmaGauss, batiOriginal, datosVolumenesNivel_1, datosVolumenesNivel_2,
					datosPnh, &datosGRD, datosNivel, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel, submallasDeformacion,
					submallaNivelSuperior, numVolumenesNivel, numVerticesNivel, posSubmallaNivelSuperior, posCopiaNivel, refinarNivel,
					&leer_fichero_puntos, &bufferEnviosMPISupInf_1, &bufferEnviosMPIIzqDer_1, &bufferEnviosMPISupInf_2, &bufferEnviosMPIIzqDer_2,
					&bufferEnviosMPISupInf_P, &bufferEnviosMPIIzqDer_P, &posicionesVolumenesGuardado, &numPuntosGuardarAnt, &numPuntosGuardarTotal,
					&lonPuntos, &latPuntos, &numVolxNivel0, &numVolyNivel0, &Hmin, &borde_sup, &borde_inf, &borde_izq, &borde_der, &tam_spongeSup,
					&tam_spongeInf, &tam_spongeIzq, &tam_spongeDer, &tiempo_tot, tiempoGuardarNetCDF, guardarVariables, &tiempoGuardarSeries,
					&CFL, &tipo_friccion, fich_friccion, friccionesNivel, &vmax, &epsilon_h, &continuar_simulacion, &tiempo_continuar,
					numEstadoNetCDF, ratioRefNivel, ratioRefAcumNivel, haySubmallasAdyacentesNivel, &difh_at, &L, &H, &U, &T, num_procs,
					&num_procsX, &num_procsY, &comm_cartesiano, id_hebra);
			tiempo_lectura1 = MPI_Wtime();
			tiempo_lectura = tiempo_lectura1 - tiempo_lectura0;
		}

		// Comprobamos si ha habido error en algún proceso
		MPI_Allreduce(&err, &err2, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		MPI_Reduce(&tiempo_lectura, &tiempo_lectura_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		if (err2 == 0) {
			// Obtenemos el nuevo id_hebra del comunicador cartesiano
			MPI_Comm_rank(comm_cartesiano, &id_hebra);
			if (id_hebra == 0) {
				mostrarDatosProblema(version, no_hidros, numPesosJacobi, numNiveles, datosNivel, okada_flag, kajiura_flag, depth_kajiura,
					fich_okada, numFaults, numVolxNivel0, numVolyNivel0, submallasNivel, numSubmallasNivel, tiempo_tot, tiempoGuardarNetCDF,
					leer_fichero_puntos, tiempoGuardarSeries, CFL, tipo_friccion, fich_friccion, friccionesNivel, vmax, epsilon_h,
					difh_at, ratioRefNivel, L, H, U, T);
			}
		}
	}

	cout << scientific;
	if ((err == 0) && (err2 == 0)) {
		if (id_hebra == 0) {
			cout << "Reading time: " << tiempo_lectura_total << " sec" << endl;
			cout << "Running " << num_procsX << "x" << num_procsY << " processes" << endl;
		}
		err = shallowWater(datosClusterCPU, no_hidros, numPesosJacobi, numNiveles, okada_flag, kajiura_flag, depth_kajiura, (char *) fich_okada.c_str(),
				numFaults, numEstadosDefDinamica, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, defTime, DEPTH_v, vc, LONCTRI,
				LATCTRI, SLIPVEC, deformacionNivel0, fich_def, lonGauss, latGauss, heightGauss, sigmaGauss, batiOriginal, datosVolumenesNivel_1,
				datosVolumenesNivel_2, datosPnh, datosGRD, datosNivel, posCopiaNivel, submallasNivel, numSubmallasNivel, iniGlobalSubmallasNivel,
				submallasDeformacion, refinarNivel, submallaNivelSuperior, posSubmallaNivelSuperior, leer_fichero_puntos, posicionesVolumenesGuardado,
				numPuntosGuardarAnt, numPuntosGuardarTotal, lonPuntos, latPuntos, numVolxNivel0, numVolyNivel0, numVolumenesNivel, numVerticesNivel,
				Hmin, (char *) nombre_bati.c_str(), prefijo, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup, tam_spongeInf, tam_spongeIzq,
				tam_spongeDer, tiempo_tot, tiempoGuardarNetCDF, guardarVariables, tiempoGuardarSeries, CFL, tipo_friccion, fich_friccion,
				friccionesNivel, vmax, epsilon_h, continuar_simulacion, tiempo_continuar, numEstadoNetCDF, ratioRefNivel, ratioRefAcumNivel,
				haySubmallasAdyacentesNivel, difh_at, L, H, U, T, num_procs, num_procsX, num_procsY, id_hebra, comm_cartesiano,
				(char *) version.c_str(), &tiempo_gpu);
		MPI_Allreduce(&err, &err2, 1, MPI_INT, MPI_MAX, comm_cartesiano);
		if (err2 > 0) {
			if (err == 1)
				fprintf(stderr, "Error in process %d: Not enough GPU memory\n", id_hebra);
			else if (err == 2)
				fprintf(stderr, "Error in process %d: Not enough CPU memory\n", id_hebra);
			liberarMemoria(numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
				datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, bufferEnviosMPISupInf_1, bufferEnviosMPIIzqDer_1,
				bufferEnviosMPISupInf_2, bufferEnviosMPIIzqDer_2, bufferEnviosMPISupInf_P, bufferEnviosMPIIzqDer_P,
				tipo_friccion, friccionesNivel, posicionesVolumenesGuardado, lonPuntos, latPuntos, okada_flag,
				numFaults, numEstadosDefDinamica, deformacionNivel0, deformacionAcumNivel0, datosGRD, id_hebra);
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		// El tiempo total es el máximo de los tiempos locales
		MPI_Reduce (&tiempo_gpu, &tiempo_multigpu, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cartesiano);
		if (id_hebra == 0)
			cout << endl << "Runtime: " << tiempo_multigpu << " sec" << endl;
		liberarMemoria(numNiveles, datosClusterCPU, datosVolumenesNivel_1, datosVolumenesNivel_2, datosPnh, batiOriginal,
			datosNivel, numSubmallasNivel, posCopiaNivel, refinarNivel, bufferEnviosMPISupInf_1, bufferEnviosMPIIzqDer_1,
			bufferEnviosMPISupInf_2, bufferEnviosMPIIzqDer_2, bufferEnviosMPISupInf_P, bufferEnviosMPIIzqDer_P,
			tipo_friccion, friccionesNivel, posicionesVolumenesGuardado, lonPuntos, latPuntos, okada_flag,
			numFaults, numEstadosDefDinamica, deformacionNivel0, deformacionAcumNivel0, datosGRD, id_hebra);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
