#ifndef _NETCDF_SERIES_H_
#define _NETCDF_SERIES_H_

#include "../Constantes.hxx"
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <netcdf.h>
#include <netcdf_par.h>
#include <mpi.h>

bool ErrorEnNetCDF;
// Ids de los ficheros NetCDF de series de tiempos (de la simulación anterior y el actual)
int ncid_ts_old;
int ncid_ts;
// Ids de variables del fichero de series de tiempos
int time_ts_id;
int eta_ts_id;
int ux_ts_id;
int uy_ts_id;
int uz_ts_id;
int eta_max_ts_id;
int eta_min_ts_id;
int lon_ts_id;
int lat_ts_id;
// Ids de la batimetría deformada con Okada
int grid_ts_id_okada;

void check_err(int iret)
{
	if ((iret != NC_NOERR) && (! ErrorEnNetCDF)) {
		fprintf(stderr, "%s\n", nc_strerror(iret));
		ErrorEnNetCDF = true;
	}
}

/*********************/
/* Series de tiempos */
/*********************/

void escribirDatosKajiuraTimeSeries(int ncid_ts, int kajiura_flag, float depth_kajiura)
{
	float val_float;
	int iret;

	if (kajiura_flag == 1) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "kajiura_filter", 3, "yes");
		check_err(iret);
		val_float = depth_kajiura;
		iret = nc_put_att_float(ncid_ts, NC_GLOBAL, "kajiura_depth", NC_FLOAT, 1, &val_float);
		check_err(iret);
	}
	else {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "kajiura_filter", 2, "no");
		check_err(iret);
	}
}

int initTimeSeriesNC(int no_hidros, int numPesosJacobi, int numNiveles, char *nombre_bati, char *prefijo, int num_points, double *lonPuntos,
			double *latPuntos, double tiempo_tot, double CFL, double epsilon_h, int tipo_friccion, string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double mf0, double vmax, double difh_at, double borde_sup, double borde_inf, double borde_izq, double borde_der, int okada_flag,
			char *fich_okada, int numFaults, double *defTime, double *LON_C, double *LAT_C, double *DEPTH_C, double *FAULT_L, double *FAULT_W,
			double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *LONCTRI, double *LATCTRI, string *fich_def, int kajiura_flag,
			double depth_kajiura, double lonGauss, double latGauss, double heightGauss, double sigmaGauss, char *version)
{
	char nombre_fich[256];
	char cadena[256];
	// Dimensiones
	int grid_npoints_dim;
	int grid_dims[1];
	int var_dims[2];
	int time_dim;
	// Nivel de compresión (0-9)
	int deflate_level = DEFLATE_LEVEL;
	// Ids
	double fill_double;
	float val_float, fill_float;
	int val_int;
	struct timeval tv;
	char fecha_act[24];
	int i, iret;

	// Creamos el fichero y entramos en modo definición
	sprintf(nombre_fich, "%s_ts.nc", prefijo);
	iret = nc_create(nombre_fich, NC_CLOBBER|NC_NETCDF4, &ncid_ts);
	check_err(iret);

	// Definimos dimensiones
	iret = nc_def_dim(ncid_ts, "grid_npoints", num_points, &grid_npoints_dim);
	check_err(iret);
	iret = nc_def_dim(ncid_ts, "time", NC_UNLIMITED, &time_dim);
	check_err(iret);

	// Definimos variables
	fill_double = -9999.0;
	fill_float = -9999.0f;
	grid_dims[0] = grid_npoints_dim;
	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE) || (okada_flag == OKADA_TRIANGULAR) ||
		(okada_flag == OKADA_TRIANGULAR_FROM_FILE) || (okada_flag == DEFORMATION_FROM_FILE) || (okada_flag == DYNAMIC_DEFORMATION)) {
		// Definimos la batimetría deformada con Okada
		iret = nc_def_var(ncid_ts, "deformed_bathy", NC_FLOAT, 1, grid_dims, &grid_ts_id_okada);
		check_err(iret);
		iret = nc_def_var_deflate(ncid_ts, grid_ts_id_okada, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	else {
		// Definimos la batimetría original
		iret = nc_def_var(ncid_ts, "bathymetry", NC_FLOAT, 1, grid_dims, &grid_ts_id_okada);
		check_err(iret);
		iret = nc_def_var_deflate(ncid_ts, grid_ts_id_okada, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	var_dims[0] = time_dim;
	var_dims[1] = grid_npoints_dim;
	// Nota: reutilizamos el array grid_dims
	grid_dims[0] = grid_npoints_dim;
	iret = nc_def_var(ncid_ts, "time", NC_FLOAT, 1, &time_dim, &time_ts_id);
	check_err(iret);
	// longitude
	iret = nc_def_var(ncid_ts, "longitude", NC_DOUBLE, 1, grid_dims, &lon_ts_id);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, lon_ts_id, "units", 12, "degrees_east");
	check_err(iret);
	iret = nc_put_att_double(ncid_ts, lon_ts_id, "missing_value", NC_DOUBLE, 1, &fill_double);
	check_err(iret);
	iret = nc_put_att_double(ncid_ts, lon_ts_id, "_FillValue", NC_DOUBLE, 1, &fill_double);
	check_err(iret);
	iret = nc_def_var_deflate(ncid_ts, lon_ts_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	// latitude
	iret = nc_def_var(ncid_ts, "latitude", NC_DOUBLE, 1, grid_dims, &lat_ts_id);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, lat_ts_id, "units", 13, "degrees_north");
	check_err(iret);
	iret = nc_put_att_double(ncid_ts, lat_ts_id, "missing_value", NC_DOUBLE, 1, &fill_double);
	check_err(iret);
	iret = nc_put_att_double(ncid_ts, lat_ts_id, "_FillValue", NC_DOUBLE, 1, &fill_double);
	check_err(iret);
	iret = nc_def_var_deflate(ncid_ts, lat_ts_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	// eta_min
	iret = nc_def_var(ncid_ts, "min_height", NC_FLOAT, 1, grid_dims, &eta_min_ts_id);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, eta_min_ts_id, "long_name", 22, "Minimum wave amplitude");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, eta_min_ts_id, "units", 6, "meters");
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, eta_min_ts_id, "missing_value", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, eta_min_ts_id, "_FillValue", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_def_var_deflate(ncid_ts, eta_min_ts_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	// eta_max
	iret = nc_def_var(ncid_ts, "max_height", NC_FLOAT, 1, grid_dims, &eta_max_ts_id);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, eta_max_ts_id, "long_name", 22, "Maximum wave amplitude");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, eta_max_ts_id, "units", 6, "meters");
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, eta_max_ts_id, "missing_value", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, eta_max_ts_id, "_FillValue", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_def_var_deflate(ncid_ts, eta_max_ts_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	// eta
	iret = nc_def_var(ncid_ts, "eta", NC_FLOAT, 2, var_dims, &eta_ts_id);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, eta_ts_id, "units", 6, "meters");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, eta_ts_id, "long_name", 14, "Wave amplitude");
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, eta_ts_id, "missing_value", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, eta_ts_id, "_FillValue", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_def_var_deflate(ncid_ts, eta_ts_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	// ux
	iret = nc_def_var(ncid_ts, "ux", NC_FLOAT, 2, var_dims, &ux_ts_id);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, ux_ts_id, "units", 13, "meters/second");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, ux_ts_id, "long_name", 33, "Velocity of water along longitude");
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, ux_ts_id, "missing_value", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, ux_ts_id, "_FillValue", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_def_var_deflate(ncid_ts, ux_ts_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	// uy
	iret = nc_def_var(ncid_ts, "uy", NC_FLOAT, 2, var_dims, &uy_ts_id);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, uy_ts_id, "units", 13, "meters/second");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, uy_ts_id, "long_name", 32, "Velocity of water along latitude");
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, uy_ts_id, "missing_value", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_put_att_float(ncid_ts, uy_ts_id, "_FillValue", NC_FLOAT, 1, &fill_float);
	check_err(iret);
	iret = nc_def_var_deflate(ncid_ts, uy_ts_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	if (no_hidros) {
		// uz
		iret = nc_def_var(ncid_ts, "uz", NC_FLOAT, 2, var_dims, &uz_ts_id);
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, uz_ts_id, "units", 13, "meters/second");
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, uz_ts_id, "long_name", 26, "Vertical velocity of water");
		check_err(iret);
		iret = nc_put_att_float(ncid_ts, uz_ts_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid_ts, uz_ts_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_def_var_deflate(ncid_ts, uz_ts_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}

	// Asignamos attributos
	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE) || (okada_flag == OKADA_TRIANGULAR) ||
		(okada_flag == OKADA_TRIANGULAR_FROM_FILE) || (okada_flag == DEFORMATION_FROM_FILE) || (okada_flag == DYNAMIC_DEFORMATION)) {
		// Batimetría deformada con Okada
		iret = nc_put_att_text(ncid_ts, grid_ts_id_okada, "long_name", 24, "Grid deformed bathymetry");
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, grid_ts_id_okada, "standard_name", 14, "deformed depth");
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, grid_ts_id_okada, "units", 6, "meters");
		check_err(iret);
		iret = nc_put_att_float(ncid_ts, grid_ts_id_okada, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid_ts, grid_ts_id_okada, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
	}
	else {
		iret = nc_put_att_text(ncid_ts, grid_ts_id_okada, "long_name", 15, "Grid bathymetry");
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, grid_ts_id_okada, "standard_name", 5, "depth");
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, grid_ts_id_okada, "units", 6, "meters");
		check_err(iret);
		iret = nc_put_att_float(ncid_ts, grid_ts_id_okada, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid_ts, grid_ts_id_okada, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
	}

	iret = nc_put_att_text(ncid_ts, time_ts_id, "long_name", 4, "Time");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, time_ts_id, "units", 24, "seconds since 1970-01-01");
	check_err(iret);

	// Atributos globales
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "Conventions", 6, "CF-1.0");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "title", 57, "Time series output of Tsunami-HySEA Non-Hydrostatic model");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "Tsunami-HySEA_Non-Hydrostatic_version", strlen(version), version);
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "creator_name", 12, "EDANYA Group");
	check_err(iret);
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "institution", 20, "University of Malaga");
	check_err(iret);
	sprintf(cadena, " ");
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "comments", strlen(cadena), cadena);
	check_err(iret);
	sprintf(cadena, " ");
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "references", strlen(cadena), cadena);
	check_err(iret);

	gettimeofday(&tv, NULL);
	strftime(fecha_act, 24, "%Y-%m-%d %H:%M:%S", localtime(&(tv.tv_sec)));
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "history", strlen(fecha_act), fecha_act);
	check_err(iret);

	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "grid_name", strlen(nombre_bati), nombre_bati);
	check_err(iret);
	val_int = no_hidros;
	iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "non_hidrostatic", NC_INT, 1, &val_int);
	check_err(iret);
	if (no_hidros) {
		val_int = numPesosJacobi;
		iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "number_of_jacobi_weights", NC_INT, 1, &val_int);
		check_err(iret);
	}
	val_float = (float) tiempo_tot;
	iret = nc_put_att_float(ncid_ts, NC_GLOBAL, "simulation_time", NC_FLOAT, 1, &val_float);
	check_err(iret);

	sprintf(cadena, (fabs(borde_sup-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "upper_border", 4, cadena);
	check_err(iret);
	sprintf(cadena, (fabs(borde_inf-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "lower_border", 4, cadena);
	check_err(iret);
	sprintf(cadena, (fabs(borde_izq-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "left_border", 4, cadena);
	check_err(iret);
	sprintf(cadena, (fabs(borde_der-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "right_border", 4, cadena);
	check_err(iret);

	val_float = (float) CFL;
	iret = nc_put_att_float(ncid_ts, NC_GLOBAL, "CFL", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = (float) epsilon_h;
	iret = nc_put_att_float(ncid_ts, NC_GLOBAL, "epsilon_h", NC_FLOAT, 1, &val_float);
	check_err(iret);
	if (tipo_friccion == FIXED_FRICTION) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "friction_type", 5, "fixed");
		check_err(iret);
		val_float = (float) mf0;
		iret = nc_put_att_float(ncid_ts, NC_GLOBAL, "water_bottom_friction", NC_FLOAT, 1, &val_float);
		check_err(iret);
	}
	else {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "friction_type", 8, "variable");
		check_err(iret);
		val_int = strlen((fich_friccion[0][0]).c_str());
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "water_bottom_friction_file_level_0", val_int, (fich_friccion[0][0]).c_str());
		check_err(iret);
	}
	val_float = (float) vmax;
	iret = nc_put_att_float(ncid_ts, NC_GLOBAL, "max_speed_water", NC_FLOAT, 1, &val_float);
	check_err(iret);
	if (difh_at >= 0.0) {
		val_float = (float) difh_at;
		iret = nc_put_att_float(ncid_ts, NC_GLOBAL, "threshold_arrival_times", NC_FLOAT, 1, &val_float);
		check_err(iret);
	}
	// Flag de okada y parámetros de Okada o gaussiana
	if (okada_flag == SEA_SURFACE_FROM_FILE) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 21, "sea_surface_from_file");
		check_err(iret);
	}
	if (okada_flag == OKADA_STANDARD) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 14, "okada_standard");
		check_err(iret);
		escribirDatosKajiuraTimeSeries(ncid_ts, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "time: %.4f, lon: %.4f, lat: %.4f, depth: %.4f, length: %.4f, width: %.4f, strike: %.4f, dip: %.4f, rake: %.4f, slip: %.4f",
				defTime[i], LON_C[i], LAT_C[i], DEPTH_C[i], FAULT_L[i], FAULT_W[i], STRIKE[i], DIP[i], RAKE[i], SLIP[i]);
			iret = nc_put_att_text(ncid_ts, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == OKADA_STANDARD_FROM_FILE) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 24, "okada_standard_from_file");
		check_err(iret);
		escribirDatosKajiuraTimeSeries(ncid_ts, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "faults_file", strlen(fich_okada), fich_okada);
		check_err(iret);
	}
	else if (okada_flag == OKADA_TRIANGULAR) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 16, "okada_triangular");
		check_err(iret);
		escribirDatosKajiuraTimeSeries(ncid_ts, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "time: %.4f, lon_barycenter: %.4f, lat_barycenter: %.4f, rake: %.4f, slip: %.4f", defTime[i], LONCTRI[i], LATCTRI[i], RAKE[i], SLIP[i]);
			iret = nc_put_att_text(ncid_ts, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == OKADA_TRIANGULAR_FROM_FILE) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 26, "okada_triangular_from_file");
		check_err(iret);
		escribirDatosKajiuraTimeSeries(ncid_ts, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "faults_file", strlen(fich_okada), fich_okada);
		check_err(iret);
	}
	else if (okada_flag == DEFORMATION_FROM_FILE) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 31, "sea_floor_deformation_from_file");
		check_err(iret);
		escribirDatosKajiuraTimeSeries(ncid_ts, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "time: %.4f, file: %s", defTime[i], (char *) fich_def[i].c_str());
			iret = nc_put_att_text(ncid_ts, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == DYNAMIC_DEFORMATION) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 39, "sea_floor_dynamic_deformation_from_file");
		check_err(iret);
		escribirDatosKajiuraTimeSeries(ncid_ts, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid_ts, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "file: %s", (char *) fich_def[i].c_str());
			iret = nc_put_att_text(ncid_ts, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == GAUSSIAN) {
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "initialization_mode", 8, "gaussian");
		check_err(iret);
		sprintf(cadena, "lon: %.4f, lat: %.4f, height: %.4f, sigma: %.4f", lonGauss, latGauss, heightGauss, sigmaGauss);
		iret = nc_put_att_text(ncid_ts, NC_GLOBAL, "gaussian_data", strlen(cadena), cadena);
		check_err(iret);
	}

	// Fin del modo definición
	iret = nc_enddef(ncid_ts);
	check_err(iret);

	// Guardamos longitudes y latitudes de los puntos
	iret = nc_put_var_double(ncid_ts, lon_ts_id, lonPuntos);
	check_err(iret);
	iret = nc_put_var_double(ncid_ts, lat_ts_id, latPuntos);
	check_err(iret);

	return 0;
}

void writeStateTimeSeriesHidrosNC(int num, float tiempo_act, int num_points, float *eta, float *ux, float *uy)
{
	int iret;
	float t_act = tiempo_act;
	const size_t num_cst = num;
	const size_t npoints_cst = num_points;
	const size_t uno = 1;
	const size_t start[] = {num_cst, 0};
	const size_t count[] = {1, npoints_cst};

	// Tiempo
	iret = nc_put_vara_float(ncid_ts, time_ts_id, &num_cst, &uno, &t_act);
	check_err(iret);
	// eta
	iret = nc_put_vara_float(ncid_ts, eta_ts_id, start, count, eta);
	check_err(iret);
	// ux
	iret = nc_put_vara_float(ncid_ts, ux_ts_id, start, count, ux);
	check_err(iret);
	// uy
	iret = nc_put_vara_float(ncid_ts, uy_ts_id, start, count, uy);
	check_err(iret);

	iret = nc_sync(ncid_ts);
	check_err(iret);
}

void writeStateTimeSeriesNoHidrosNC(int num, float tiempo_act, int num_points, float *eta, float *ux, float *uy, float *uz)
{
	int iret;
	float t_act = tiempo_act;
	const size_t num_cst = num;
	const size_t npoints_cst = num_points;
	const size_t uno = 1;
	const size_t start[] = {num_cst, 0};
	const size_t count[] = {1, npoints_cst};

	// Tiempo
	iret = nc_put_vara_float(ncid_ts, time_ts_id, &num_cst, &uno, &t_act);
	check_err(iret);
	// eta
	iret = nc_put_vara_float(ncid_ts, eta_ts_id, start, count, eta);
	check_err(iret);
	// ux
	iret = nc_put_vara_float(ncid_ts, ux_ts_id, start, count, ux);
	check_err(iret);
	// uy
	iret = nc_put_vara_float(ncid_ts, uy_ts_id, start, count, uy);
	check_err(iret);
	// uz
	iret = nc_put_vara_float(ncid_ts, uz_ts_id, start, count, uz);
	check_err(iret);

	iret = nc_sync(ncid_ts);
	check_err(iret);
}

void guardarBatimetriaModificadaTimeSeriesNC(float *vec)
{
	int iret;

	iret = nc_put_var_float(ncid_ts, grid_ts_id_okada, vec);
	check_err(iret);
}

void guardarAmplitudesTimeSeriesNC(float *eta_min_puntos, float *eta_max_puntos)
{
	int iret;

	iret = nc_put_var_float(ncid_ts, eta_min_ts_id, eta_min_puntos);
	check_err(iret);
	iret = nc_put_var_float(ncid_ts, eta_max_ts_id, eta_max_puntos);
	check_err(iret);
}

void closeTimeSeriesNC()
{
	int iret;

	iret = nc_close(ncid_ts);
	check_err(iret);
}

// CONTINUACIÓN DE UNA SIMULACIÓN GUARDADA

void abrirTimeSeriesOldNC(const char *nombre_fich)
{
	int iret;

	iret = nc_open(nombre_fich, NC_NOWRITE, &ncid_ts_old);
	check_err(iret);
}

int obtenerNumPuntosTimeSeriesOldNC()
{
	int iret;
	size_t num;
	int npoints_dim;

	iret = nc_inq_dimid(ncid_ts_old, "grid_npoints", &npoints_dim);
	check_err(iret);
	iret = nc_inq_dimlen(ncid_ts_old, npoints_dim, &num);
	check_err(iret);

	return ((int) num);
}

int obtenerNumEstadosTimeSeriesOldNC()
{
	int iret;
	size_t num;
	int time_dim;

	iret = nc_inq_dimid(ncid_ts_old, "time", &time_dim);
	check_err(iret);
	iret = nc_inq_dimlen(ncid_ts_old, time_dim, &num);
	check_err(iret);

	return ((int) num);
}

void leerLongitudesYLatitudesTimeSeriesOldNC(double *lonPuntos, double *latPuntos)
{
	int iret;
	int lon_id, lat_id;

	iret = nc_inq_varid(ncid_ts_old, "longitude", &lon_id);
	check_err(iret);
	iret = nc_get_var_double(ncid_ts_old, lon_id, lonPuntos);
	check_err(iret);
	iret = nc_inq_varid(ncid_ts_old, "latitude", &lat_id);
	check_err(iret);
	iret = nc_get_var_double(ncid_ts_old, lat_id, latPuntos);
	check_err(iret);
}

void leerAmplitudesTimeSeriesOldNC(float *eta_min_puntos, float *eta_max_puntos)
{
	int iret;
	int eta_min_id, eta_max_id;

	iret = nc_inq_varid(ncid_ts_old, "min_height", &eta_min_id);
	check_err(iret);
	iret = nc_get_var_float(ncid_ts_old, eta_min_id, eta_min_puntos);
	check_err(iret);
	iret = nc_inq_varid(ncid_ts_old, "max_height", &eta_max_id);
	check_err(iret);
	iret = nc_get_var_float(ncid_ts_old, eta_max_id, eta_max_puntos);
	check_err(iret);
}

void readStateTimeSeriesOldNC(int num, int num_points, float *tiempo, float *eta, float *ux, float *uy)
{
	int iret, var_id;
	const size_t num_cst = num;
	const size_t npoints_cst = num_points;
	const size_t uno = 1;
	const size_t start[] = {num_cst, 0};
	const size_t count[] = {1, npoints_cst};

	// Tiempo
	iret = nc_inq_varid(ncid_ts_old, "time", &var_id);
	check_err(iret);
	iret = nc_get_vara_float(ncid_ts_old, var_id, &num_cst, &uno, tiempo);
	check_err(iret);
	// eta
	iret = nc_inq_varid(ncid_ts_old, "eta", &var_id);
	check_err(iret);
	iret = nc_get_vara_float(ncid_ts_old, var_id, start, count, eta);
	check_err(iret);
	// ux
	iret = nc_inq_varid(ncid_ts_old, "ux", &var_id);
	check_err(iret);
	iret = nc_get_vara_float(ncid_ts_old, var_id, start, count, ux);
	check_err(iret);
	// uy
	iret = nc_inq_varid(ncid_ts_old, "uy", &var_id);
	check_err(iret);
	iret = nc_get_vara_float(ncid_ts_old, var_id, start, count, uy);
	check_err(iret);
}

// Ambos ficheros de series de tiempos (anterior y actual) deben estar abiertos.
// v1, v2 y v3 son vectores temporales que se usarán para traspasar los datos. Su tamaño es numPuntosGuardarTotal
void traspasarDatosTimeSeriesNC(int numPuntosGuardarAnt, int numPuntosGuardarTotal, int numEstados,
			float *v1, float *v2, float *v3)
{
	float tiempo;
	float val = -9999.0f;
	int i, j;

	for (i=0; i<numEstados; i++) {
		readStateTimeSeriesOldNC(i, numPuntosGuardarAnt, &tiempo, v1, v2, v3);
		for (j=numPuntosGuardarAnt; j<numPuntosGuardarTotal; j++) {
			v1[j] = v2[j] = v3[j] = val;
		}
		writeStateTimeSeriesHidrosNC(i, tiempo, numPuntosGuardarTotal, v1, v2, v3);
	}
}

void cerrarTimeSeriesOldNC()
{
	int iret;

	iret = nc_close(ncid_ts_old);
	check_err(iret);
}

#endif
