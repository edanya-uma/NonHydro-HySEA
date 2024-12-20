/*************************************************************
 NH-HySEA numerical model v1.0.0
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
	cerr << argv[0] << " dataFile numberOfDivisionsInX numberOfDivisionsInY [weightsFile]" << endl << endl; 
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
	cerr << "    Apply Kajiura filter to the Okada deformation (0: no, 1: yes)" << endl;
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
	cerr << "    eta maximum_eta velocities maximum_velocities modulus_of_velocity maximum_modulus_of_velocity maximum_modulus_of_mass_flow momentum_flux maximum_momentum_flux arrival_times" << endl;
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
	cerr << "weightsFile format:" << endl;
	cerr << "  Weights of the GPUs (higher if faster), in row-order in the matrix of processes and one per line" << endl;
}

int main(int argc, char *argv[])
{
	string version = "1.0.0";
	double2 *datosVolumenesNivel_1[MAX_LEVELS];  // h, H
	double2 *datosVolumenesNivel_2[MAX_LEVELS];  // qx, qy
	tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int *xIniCluster = NULL;
	int *yIniCluster = NULL;
	int leer_fichero_puntos, num_puntos_guardar;
	int numVolxNivel0, numVolyNivel0;
	int64_t numVolumenesNivel[MAX_LEVELS];
	int no_hidros, numPesosJacobi;
	double borde_sup, borde_inf, borde_izq, borde_der;
	double tiempo_tot;
	double tiempoGuardarNetCDF[MAX_LEVELS];
	double tiempoGuardarSeries;
	VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL];
	double epsilon_h, difh_at;
	double mf0, vmax;
	double CFL;
	double L, H, Q, T;
	int tipo_friccion;
	string fich_okada;
	string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int okada_flag, kajiura_flag;
	double depth_kajiura;
	int usar_ventana;
	int ratioRefNivel[MAX_LEVELS];
	bool haySubmallasAdyacentesNivel[MAX_LEVELS];
	string fich_ent;
	string fich_pesos("");
	string nombre_bati;
	string prefijo[MAX_LEVELS][MAX_GRIDS_LEVEL];  // Prefijos de los ficheros NetCDF
	int numNiveles, numFaults;
	int err, numProcsX, numProcsY;
	int numSubmallasNivel[MAX_LEVELS];
	int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL];
	// submallaNivelSuperior[l][i] indica la submalla del nivel l-1
	// (empieza por 0) que contiene la submalla i del nivel l.
	// posSubmallaNivelSuperior indica la posición en el vector datosVolumenesNivel
	// donde empiezan los datos de la malla del nivel superior
	int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL];
	int posSubmallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL];

	if (argc < 4) {
		mostrarFormatoProgramaGRD(argv, version);
		return EXIT_FAILURE;
	}

	// Fichero de datos
	fich_ent = argv[1];
	if (! existeFichero(fich_ent)) {
		cerr << "Error: File '" << fich_ent << "' not found" << endl;
		return EXIT_FAILURE;
	}

	// Número de divisiones en X e Y
	numProcsX = atoi(argv[2]);
	if (numProcsX < 1) {
		cerr << "Error: The number of divisions in X should be greater than 0" << endl;
		return EXIT_FAILURE;
	}
	numProcsY = atoi(argv[3]);
	if (numProcsY < 1) {
		cerr << "Error: The number of divisions in Y should be greater than 0" << endl;
		return EXIT_FAILURE;
	}

	// Fichero de pesos de las GPUs
	if (argc >= 5) {
		fich_pesos = argv[4];
		if (! existeFichero(fich_pesos)) {
			cerr << "Error: File '" << fich_pesos << "' not found" << endl;
			return EXIT_FAILURE;
		}
	}

	err = cargarDatosProblema(fich_ent, nombre_bati, prefijo, &no_hidros, &numPesosJacobi, &numNiveles, &okada_flag, &kajiura_flag,
			&depth_kajiura, fich_okada, &numFaults, &usar_ventana, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, submallasNivel,
			numSubmallasNivel, submallaNivelSuperior, numVolumenesNivel, posSubmallaNivelSuperior, &leer_fichero_puntos, &num_puntos_guardar,
			&borde_sup, &borde_inf, &borde_izq, &borde_der, &tiempo_tot, tiempoGuardarNetCDF, guardarVariables, &tiempoGuardarSeries,
			&CFL, &tipo_friccion, fich_friccion, &mf0, &vmax, &epsilon_h, ratioRefNivel, haySubmallasAdyacentesNivel, &difh_at,
			&xIniCluster, &yIniCluster, &L, &H, &Q, &T, numProcsX, numProcsY, fich_pesos);
	if (err > 0)
		return EXIT_FAILURE;

	mostrarDatosProblema(version, no_hidros, numPesosJacobi, numNiveles, okada_flag, kajiura_flag, depth_kajiura, fich_okada,
		numFaults, datosNivel, submallasNivel, numSubmallasNivel, tiempo_tot, tiempoGuardarNetCDF, leer_fichero_puntos,
		tiempoGuardarSeries, CFL, tipo_friccion, fich_friccion, mf0, vmax, epsilon_h, difh_at, ratioRefNivel, xIniCluster,
		yIniCluster, L, H, Q, T, numProcsX, numProcsY);

	escribirFicheroSalida(nombre_bati, &numNiveles, submallasNivel, numSubmallasNivel, submallaNivelSuperior,
		haySubmallasAdyacentesNivel, xIniCluster, yIniCluster, &numProcsX, &numProcsY);

	liberarMemoria(numNiveles, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, numSubmallasNivel, xIniCluster, yIniCluster);

	return EXIT_SUCCESS;
}

