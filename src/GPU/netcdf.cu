#ifndef _NETCDF_H_
#define _NETCDF_H_

#include "../Constantes.hxx"
#include <stdio.h>
#include <netcdf.h>
#include <mpi.h>
#include "netcdfSeries.cu"

// Id del fichero NetCDF principal
int ncid[MAX_LEVELS][MAX_GRIDS_LEVEL];
// Id del fichero NetCDF de fricciones
int ncid_fricciones[MAX_LEVELS][MAX_GRIDS_LEVEL];
// Id del fichero de deformación dinámica
int ncid_defDinamica;
// Ids de variables del fichero principal
int time_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int eta_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int ux_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int uy_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int uz_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int u_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int nhp_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int fluxx_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int fluxy_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int fluxz_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int eta_max_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int ux_max_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int uy_max_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int uz_max_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int u_max_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int q_max_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int flux_max_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
int arr_times_id[MAX_LEVELS][MAX_GRIDS_LEVEL];
// Ids de la batimetría deformada con Okada
int grid_id_okada;


/***********/
/* Lectura */
/***********/

int abrirGRD(const char *nombre_fich, int nivel, int submalla, MPI_Comm comm)
{
	int iret, formato;
	bool es_netcdf4;

	iret = nc_open_par(nombre_fich, NC_NOWRITE, comm, MPI_INFO_NULL, &(ncid[nivel][submalla]));
	check_err(iret);
	iret = nc_inq_format(ncid[nivel][submalla], &formato);
	check_err(iret);
	es_netcdf4 = ( ((formato == NC_FORMAT_NETCDF4) || (formato == NC_FORMAT_NETCDF4_CLASSIC)) ? true : false);

	return (es_netcdf4 ? 0 : 1);
}

void cerrarGRD(int nivel, int submalla)
{
	int iret;

	iret = nc_close(ncid[nivel][submalla]);
	check_err(iret);
}

int abrirGRDFricciones(const char *nombre_fich, int nivel, int submalla, MPI_Comm comm)
{
	int iret, formato;
	bool es_netcdf4;

	iret = nc_open_par(nombre_fich, NC_NOWRITE, comm, MPI_INFO_NULL, &(ncid_fricciones[nivel][submalla]));
	check_err(iret);
	iret = nc_inq_format(ncid_fricciones[nivel][submalla], &formato);
	check_err(iret);
	es_netcdf4 = ( ((formato == NC_FORMAT_NETCDF4) || (formato == NC_FORMAT_NETCDF4_CLASSIC)) ? true : false);

	return (es_netcdf4 ? 0 : 1);
}

void leerTamanoSubmallaGRDFricciones(int nivel, int submalla, int *nvx, int *nvy)
{
	int id, iret;
	int lon_dim, lat_dim;
	size_t nx, ny;

	id = ncid_fricciones[nivel][submalla];
	// El nombre de las dimensiones puede ser (lon,lat) o (x,y).
	iret = nc_inq_dimid(id, "lon", &lon_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "x", &lon_dim);
		check_err(iret);
	}
	iret = nc_inq_dimid(id, "lat", &lat_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "y", &lat_dim);
		check_err(iret);
	}
	iret = nc_inq_dimlen(id, lon_dim, &nx);
	check_err(iret);
	iret = nc_inq_dimlen(id, lat_dim, &ny);
	check_err(iret);
	*nvx = nx;
	*nvy = ny;
}

void leerFriccionesGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *fric)
{
	int id, iret;
	int z_id;
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};

	id = ncid_fricciones[nivel][submalla];
	iret = nc_inq_varid(id, "z", &z_id);
	check_err(iret);
	iret = nc_var_par_access(id, z_id, NC_COLLECTIVE);
	check_err(iret);
	iret = nc_get_vara_float(id, z_id, start, count, fric);
	check_err(iret);
}

void cerrarGRDFricciones(int nivel, int submalla)
{
	int iret;

	iret = nc_close(ncid_fricciones[nivel][submalla]);
	check_err(iret);
}

void leerTamanoSubmallaGRD(const char *nombre_fich, int nivel, int submalla, int *nvx, int *nvy)
{
	int id, iret;
	int lon_dim, lat_dim;
	size_t nx, ny;

	iret = nc_open(nombre_fich, NC_NOWRITE, &id);
	check_err(iret);
	// Leemos el tamaño de la malla.
	// El nombre de las dimensiones puede ser (lon,lat) o (x,y).
	iret = nc_inq_dimid(id, "lon", &lon_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "x", &lon_dim);
		check_err(iret);
	}
	iret = nc_inq_dimid(id, "lat", &lat_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "y", &lat_dim);
		check_err(iret);
	}
	iret = nc_inq_dimlen(id, lon_dim, &nx);
	check_err(iret);
	iret = nc_inq_dimlen(id, lat_dim, &ny);
	check_err(iret);
	iret = nc_close(id);
	check_err(iret);
	*nvx = nx;
	*nvy = ny;
}

void leerLongitudGRD(int nivel, int submalla, double *lon)
{
	int id, iret;
	int lon_id;

	id = ncid[nivel][submalla];
	// El nombre de la variable puede ser lon o x
	iret = nc_inq_varid(id, "lon", &lon_id);
	if (iret != NC_NOERR) {
		iret = nc_inq_varid(id, "x", &lon_id);
		check_err(iret);
	}
	iret = nc_get_var_double(id, lon_id, lon);
	check_err(iret);
}

void leerLatitudGRD(int nivel, int submalla, double *lat)
{
	int id, iret;
	int lat_id;

	id = ncid[nivel][submalla];
	// El nombre de la variable puede ser lat o y
	iret = nc_inq_varid(id, "lat", &lat_id);
	if (iret != NC_NOERR) {
		iret = nc_inq_varid(id, "y", &lat_id);
		check_err(iret);
	}
	iret = nc_get_var_double(id, lat_id, lat);
	check_err(iret);
}

void leerBatimetriaGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *bati)
{
	int id, iret;
	int z_id;
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};

	id = ncid[nivel][submalla];
	iret = nc_inq_varid(id, "z", &z_id);
	check_err(iret);
	iret = nc_var_par_access(id, z_id, NC_COLLECTIVE);
	check_err(iret);
	iret = nc_get_vara_float(id, z_id, start, count, bati);
	check_err(iret);
}

void leerEtaGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *eta)
{
	int id, iret;
	int var_id;
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};

	id = ncid[nivel][submalla];
	// El nombre de la variable puede ser eta o z
	iret = nc_inq_varid(id, "eta", &var_id);
	if (iret != NC_NOERR) {
		iret = nc_inq_varid(id, "z", &var_id);
		check_err(iret);
	}
	check_err(iret);
	iret = nc_var_par_access(id, var_id, NC_COLLECTIVE);
	check_err(iret);
	iret = nc_get_vara_float(id, var_id, start, count, eta);
	check_err(iret);
}

void leerUxGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *ux)
{
	int id, iret;
	int var_id;
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};

	id = ncid[nivel][submalla];
	iret = nc_inq_varid(id, "ux", &var_id);
	check_err(iret);
	iret = nc_var_par_access(id, var_id, NC_COLLECTIVE);
	check_err(iret);
	iret = nc_get_vara_float(id, var_id, start, count, ux);
	check_err(iret);
}

void leerUyGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *uy)
{
	int id, iret;
	int var_id;
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};

	id = ncid[nivel][submalla];
	iret = nc_inq_varid(id, "uy", &var_id);
	check_err(iret);
	iret = nc_var_par_access(id, var_id, NC_COLLECTIVE);
	check_err(iret);
	iret = nc_get_vara_float(id, var_id, start, count, uy);
	check_err(iret);
}

void leerUzGRD(int nivel, int submalla, int inix_cluster, int iniy_cluster, int num_volx, int num_voly, float *uz)
{
	int id, iret;
	int var_id;
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};

	id = ncid[nivel][submalla];
	iret = nc_inq_varid(id, "uz", &var_id);
	check_err(iret);
	iret = nc_var_par_access(id, var_id, NC_COLLECTIVE);
	check_err(iret);
	iret = nc_get_vara_float(id, var_id, start, count, uz);
	check_err(iret);
}

// LECTURA DE LA DEFORMACIÓN CON OKADA

void leerDeformacionGRD(const char *nombre_fich, int *nvx, int *nvy, double *lon, double *lat, float *def)
{
	int id, iret;
	int lon_dim, lat_dim;
	int lon_id, lat_id, z_id;
	size_t nx, ny;

	iret = nc_open(nombre_fich, NC_NOWRITE, &id);
	check_err(iret);
	// Leemos el tamaño de la malla.
	// El nombre de las dimensiones puede ser (lon,lat) o (x,y).
	iret = nc_inq_dimid(id, "lon", &lon_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "x", &lon_dim);
		check_err(iret);
	}
	iret = nc_inq_dimid(id, "lat", &lat_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "y", &lat_dim);
		check_err(iret);
	}
	iret = nc_inq_dimlen(id, lon_dim, &nx);
	check_err(iret);
	iret = nc_inq_dimlen(id, lat_dim, &ny);
	check_err(iret);
	*nvx = nx;
	*nvy = ny;
	// Leemos longitudes
	// El nombre de la variable puede ser lon o x
	iret = nc_inq_varid(id, "lon", &lon_id);
	if (iret != NC_NOERR) {
		iret = nc_inq_varid(id, "x", &lon_id);
		check_err(iret);
	}
	iret = nc_get_var_double(id, lon_id, lon);
	check_err(iret);
	// Leemos latitudes
	// El nombre de la variable puede ser lat o y
	iret = nc_inq_varid(id, "lat", &lat_id);
	if (iret != NC_NOERR) {
		iret = nc_inq_varid(id, "y", &lat_id);
		check_err(iret);
	}
	iret = nc_get_var_double(id, lat_id, lat);
	check_err(iret);
	// Leemos la deformación
	iret = nc_inq_varid(id, "z", &z_id);
	check_err(iret);
	iret = nc_get_var_float(id, z_id, def);
	check_err(iret);
	iret = nc_close(id);
	check_err(iret);
}

// LECTURA DE LA DEFORMACIÓN DINÁMICA

void leerTamanoYNumEstadosDefDinamicaGRD(const char *nombre_fich, int *nvx, int *nvy, int *num_estados)
{
	int id, iret;
	int lon_dim, lat_dim, time_dim;
	size_t nx, ny, num_est;

	iret = nc_open(nombre_fich, NC_NOWRITE, &id);
	check_err(iret);
	// Leemos el tamaño de la malla.
	// El nombre de las dimensiones puede ser (lon,lat) o (x,y).
	iret = nc_inq_dimid(id, "lon", &lon_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "x", &lon_dim);
		check_err(iret);
	}
	iret = nc_inq_dimid(id, "lat", &lat_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(id, "y", &lat_dim);
		check_err(iret);
	}
	iret = nc_inq_dimlen(id, lon_dim, &nx);
	check_err(iret);
	iret = nc_inq_dimlen(id, lat_dim, &ny);
	check_err(iret);
	*nvx = nx;
	*nvy = ny;
	// Leemos el número de estados en la dimensión tiempo
	iret = nc_inq_dimid(id, "time", &time_dim);
	check_err(iret);
	iret = nc_inq_dimlen(id, time_dim, &num_est);
	check_err(iret);
	iret = nc_close(id);
	check_err(iret);
	*num_estados = num_est;
}

void abrirGRDDefDinamica(const char *nombre_fich)
{
	int iret;

	iret = nc_open(nombre_fich, NC_NOWRITE, &ncid_defDinamica);
	check_err(iret);
}

void leerLongitudLatitudYTiemposDefDinamicaGRD(double *lon, double *lat, float *time)
{
	int id, iret;
	int lon_id, lat_id, time_id;

	id = ncid_defDinamica;
	// El nombre de la variable puede ser lon o x
	iret = nc_inq_varid(id, "lon", &lon_id);
	if (iret != NC_NOERR) {
		iret = nc_inq_varid(id, "x", &lon_id);
		check_err(iret);
	}
	iret = nc_get_var_double(id, lon_id, lon);
	check_err(iret);
	// El nombre de la variable puede ser lat o y
	iret = nc_inq_varid(id, "lat", &lat_id);
	if (iret != NC_NOERR) {
		iret = nc_inq_varid(id, "y", &lat_id);
		check_err(iret);
	}
	iret = nc_get_var_double(id, lat_id, lat);
	check_err(iret);
	// Leemos los tiempos
	iret = nc_inq_varid(id, "time", &time_id);
	check_err(iret);
	iret = nc_get_var_float(id, time_id, time);
	check_err(iret);
}

void leerEstadoDefDinamicaGRD(int nvx, int nvy, int num, float *def)
{
	int iret, var_id;
	int id = ncid_defDinamica;
	const size_t start[] = {(size_t) num, (size_t) 0, (size_t) 0};
	const size_t count[] = {1, (size_t) nvy, (size_t) nvx};

	iret = nc_inq_varid(id, "z", &var_id);
	check_err(iret);
	iret = nc_get_vara_float(id, var_id, start, count, def);
	check_err(iret);
}

void cerrarGRDDefDinamica()
{
	int iret;

	iret = nc_close(ncid_defDinamica);
	check_err(iret);
}

// CONTINUACIÓN DE UNA SIMULACIÓN GUARDADA

bool existeVariableNetCDF(int nivel, int submalla, const char *nombre_var)
{
	int iret;
	int id, var_id;

	id = ncid[nivel][submalla];
	iret = nc_inq_varid(id, nombre_var, &var_id);

	return ((iret == NC_NOERR) ? true : false);
}

/*************/
/* Escritura */
/*************/

void escribirDatosKajiuraNC(int ncid, int kajiura_flag, float depth_kajiura)
{
	float val_float;
	int iret;

	if (kajiura_flag == 1) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "kajiura_filter", 3, "yes");
		check_err(iret);
		val_float = depth_kajiura;
		iret = nc_put_att_float(ncid, NC_GLOBAL, "kajiura_depth", NC_FLOAT, 1, &val_float);
		check_err(iret);
	}
	else {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "kajiura_filter", 2, "no");
		check_err(iret);
	}
}

void fgennc(int no_hidros, int numPesosJacobi, int numNiveles, int nivel, int submalla, VariablesGuardado *var, double *lon_grid,
			double *lat_grid, double *lon, double *lat, char *nombre_bati, char *prefijo, int *p_ncid, int *var_time_id, int *var_eta_id,
			int *var_ux_id, int *var_uy_id, int *var_uz_id, int *var_u_id, int *var_nhp_id, int *var_mfx_id, int *var_mfy_id, int *var_mfz_id,
			int nx_nc, int ny_nc, int num_volx, int num_voly, int inix_cluster, int iniy_cluster, int num_volx_total, int num_voly_total,
			double tiempo_tot, double CFL, double epsilon_h, int tipo_friccion, string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL],
			double mf0, double vmax, double difh_at, int npics, double borde_sup, double borde_inf, double borde_izq, double borde_der,
			int okada_flag, char *fich_okada, int numFaults, double *defTime, double *LON_C, double *LAT_C, double *DEPTH_C, double *FAULT_L,
			double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *LONCTRI, double *LATCTRI, string *fich_def,
			int kajiura_flag, double depth_kajiura, double lonGauss, double latGauss, double heightGauss, double sigmaGauss,
			float *batiOriginal, char *version, MPI_Comm comm)
{
	char nombre_fich[256];
	char cadena[256];
	// Dimensiones
	int grid_lon_dim, grid_lat_dim;
	int grid_dims[2];
	int lon_dim, lat_dim;
	int var_dims[3];
	int time_dim;
	// Nivel de compresión (0-9)
	int deflate_level = DEFLATE_LEVEL;
	// Ids
	int ncid;
	int grid_id;
	int grid_lon_id, grid_lat_id;
	int lon_id, lat_id;
	int val_int;
	float val_float, fill_float;
	struct timeval tv;
	char fecha_act[24];
	int i, iret;

	// Creamos el fichero y entramos en modo definición
	sprintf(nombre_fich, "%s.nc", prefijo);
	iret = nc_create_par(nombre_fich, NC_CLOBBER|NC_NETCDF4, comm, MPI_INFO_NULL, p_ncid);
	check_err(iret);
	ncid = *p_ncid;

	// Definimos dimensiones
	iret = nc_def_dim(ncid, "lon", nx_nc, &lon_dim);
	check_err(iret);
	iret = nc_def_dim(ncid, "lat", ny_nc, &lat_dim);
	check_err(iret);
	iret = nc_def_dim(ncid, "grid_lon", num_volx_total, &grid_lon_dim);
	check_err(iret);
	iret = nc_def_dim(ncid, "grid_lat", num_voly_total, &grid_lat_dim);
	check_err(iret);
	iret = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
	check_err(iret);

	// Definimos variables
	fill_float = -9999.0f;
	iret = nc_def_var(ncid, "lon", NC_DOUBLE, 1, &lon_dim, &lon_id);
	check_err(iret);
	iret = nc_def_var_deflate(ncid, lon_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	iret = nc_def_var(ncid, "lat", NC_DOUBLE, 1, &lat_dim, &lat_id);
	check_err(iret);
	iret = nc_def_var_deflate(ncid, lat_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	iret = nc_def_var(ncid, "grid_lon", NC_DOUBLE, 1, &grid_lon_dim, &grid_lon_id);
	check_err(iret);
	iret = nc_def_var_deflate(ncid, grid_lon_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	iret = nc_def_var(ncid, "grid_lat", NC_DOUBLE, 1, &grid_lat_dim, &grid_lat_id);
	check_err(iret);
	iret = nc_def_var_deflate(ncid, grid_lat_id, NC_SHUFFLE, 1, deflate_level);
	check_err(iret);
	grid_dims[0] = grid_lat_dim;
	grid_dims[1] = grid_lon_dim;
	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE) || (okada_flag == OKADA_TRIANGULAR) ||
		(okada_flag == OKADA_TRIANGULAR_FROM_FILE) || (okada_flag == DEFORMATION_FROM_FILE) || (okada_flag == DYNAMIC_DEFORMATION)) {
		// Definimos la batimetría original y la deformada con Okada
		iret = nc_def_var(ncid, "original_bathy", NC_FLOAT, 2, grid_dims, &grid_id);
		check_err(iret);
		iret = nc_var_par_access(ncid, grid_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, grid_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
		iret = nc_def_var(ncid, "deformed_bathy", NC_FLOAT, 2, grid_dims, &grid_id_okada);
		check_err(iret);
		iret = nc_var_par_access(ncid, grid_id_okada, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, grid_id_okada, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	else {
		iret = nc_def_var(ncid, "bathymetry", NC_FLOAT, 2, grid_dims, &grid_id);
		check_err(iret);
		iret = nc_var_par_access(ncid, grid_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, grid_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	var_dims[0] = time_dim;
	var_dims[1] = lat_dim;
	var_dims[2] = lon_dim;
	// Nota: reutilizamos el array grid_dims
	grid_dims[0] = lat_dim;
	grid_dims[1] = lon_dim;
	iret = nc_def_var(ncid, "time", NC_FLOAT, 1, &time_dim, var_time_id);
	check_err(iret);
	iret = nc_var_par_access(ncid, *var_time_id, NC_COLLECTIVE);
	check_err(iret);
	if (var->eta_max != 0) {
		iret = nc_def_var(ncid, "max_height", NC_FLOAT, 2, grid_dims, &(eta_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_put_att_text(ncid, eta_max_id[nivel][submalla], "long_name", 22, "Maximum wave amplitude");
		check_err(iret);
		iret = nc_put_att_text(ncid, eta_max_id[nivel][submalla], "units", 6, "meters");
		check_err(iret);
		iret = nc_put_att_float(ncid, eta_max_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, eta_max_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, eta_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, eta_max_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->velocidades_max != 0) {
		// max_ux
		iret = nc_def_var(ncid, "max_ux", NC_FLOAT, 2, grid_dims, &(ux_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_put_att_text(ncid, ux_max_id[nivel][submalla], "long_name", 41, "Maximum velocity of water along longitude");
		check_err(iret);
		iret = nc_put_att_text(ncid, ux_max_id[nivel][submalla], "units", 13, "meters/second");
		check_err(iret);
		iret = nc_put_att_float(ncid, ux_max_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, ux_max_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, ux_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, ux_max_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
		// max_uy
		iret = nc_def_var(ncid, "max_uy", NC_FLOAT, 2, grid_dims, &(uy_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_put_att_text(ncid, uy_max_id[nivel][submalla], "long_name", 40, "Maximum velocity of water along latitude");
		check_err(iret);
		iret = nc_put_att_text(ncid, uy_max_id[nivel][submalla], "units", 13, "meters/second");
		check_err(iret);
		iret = nc_put_att_float(ncid, uy_max_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, uy_max_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, uy_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, uy_max_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
		if (no_hidros) {
			// max_uz
			iret = nc_def_var(ncid, "max_uz", NC_FLOAT, 2, grid_dims, &(uz_max_id[nivel][submalla]));
			check_err(iret);
			iret = nc_put_att_text(ncid, uz_max_id[nivel][submalla], "long_name", 34, "Maximum vertical velocity of water");
			check_err(iret);
			iret = nc_put_att_text(ncid, uz_max_id[nivel][submalla], "units", 13, "meters/second");
			check_err(iret);
			iret = nc_put_att_float(ncid, uz_max_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
			check_err(iret);
			iret = nc_put_att_float(ncid, uz_max_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
			check_err(iret);
			iret = nc_var_par_access(ncid, uz_max_id[nivel][submalla], NC_COLLECTIVE);
			check_err(iret);
			iret = nc_def_var_deflate(ncid, uz_max_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
			check_err(iret);
		}
	}
	if (var->modulo_velocidades_max != 0) {
		iret = nc_def_var(ncid, "max_u", NC_FLOAT, 2, grid_dims, &(u_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_put_att_text(ncid, u_max_id[nivel][submalla], "long_name", 27, "Maximum modulus of velocity");
		check_err(iret);
		iret = nc_put_att_text(ncid, u_max_id[nivel][submalla], "units", 13, "meters/second");
		check_err(iret);
		iret = nc_put_att_float(ncid, u_max_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, u_max_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, u_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, u_max_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->modulo_caudales_max != 0) {
		iret = nc_def_var(ncid, "max_q", NC_FLOAT, 2, grid_dims, &(q_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_put_att_text(ncid, q_max_id[nivel][submalla], "long_name", 28, "Maximum modulus of mass-flow");
		check_err(iret);
		iret = nc_put_att_text(ncid, q_max_id[nivel][submalla], "units", 15, "meters^2/second");
		check_err(iret);
		iret = nc_put_att_float(ncid, q_max_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, q_max_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, q_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, q_max_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->flujo_momento_max != 0) {
		iret = nc_def_var(ncid, "max_mom_flux", NC_FLOAT, 2, grid_dims, &(flux_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_put_att_text(ncid, flux_max_id[nivel][submalla], "long_name", 21, "Maximum momentum flux");
		check_err(iret);
		iret = nc_put_att_text(ncid, flux_max_id[nivel][submalla], "units", 18, "meters^3/seconds^2");
		check_err(iret);
		iret = nc_put_att_float(ncid, flux_max_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, flux_max_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, flux_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, flux_max_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->tiempos_llegada != 0) {
		iret = nc_def_var(ncid, "arrival_times", NC_FLOAT, 2, grid_dims, &(arr_times_id[nivel][submalla]));
		check_err(iret);
		iret = nc_put_att_text(ncid, arr_times_id[nivel][submalla], "long_name", 28, "Arrival times of the tsunami");
		check_err(iret);
		iret = nc_put_att_text(ncid, arr_times_id[nivel][submalla], "units", 7, "seconds");
		check_err(iret);
		iret = nc_put_att_float(ncid, arr_times_id[nivel][submalla], "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, arr_times_id[nivel][submalla], "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, arr_times_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, arr_times_id[nivel][submalla], NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->eta != 0) {
		// eta
		iret = nc_def_var(ncid, "eta", NC_FLOAT, 3, var_dims, var_eta_id);
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_eta_id, "units", 6, "meters");
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_eta_id, "long_name", 14, "Wave amplitude");
		check_err(iret);
		iret = nc_put_att_float(ncid, *var_eta_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, *var_eta_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, *var_eta_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, *var_eta_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->velocidades != 0) {
		// ux
		iret = nc_def_var(ncid, "ux", NC_FLOAT, 3, var_dims, var_ux_id);
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_ux_id, "units", 13, "meters/second");
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_ux_id, "long_name", 33, "Velocity of water along longitude");
		check_err(iret);
		iret = nc_put_att_float(ncid, *var_ux_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, *var_ux_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, *var_ux_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, *var_ux_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
		// uy
		iret = nc_def_var(ncid, "uy", NC_FLOAT, 3, var_dims, var_uy_id);
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_uy_id, "units", 13, "meters/second");
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_uy_id, "long_name", 32, "Velocity of water along latitude");
		check_err(iret);
		iret = nc_put_att_float(ncid, *var_uy_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, *var_uy_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_var_par_access(ncid, *var_uy_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, *var_uy_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
		if (no_hidros) {
			// uz
			iret = nc_def_var(ncid, "uz", NC_FLOAT, 3, var_dims, var_uz_id);
			check_err(iret);
			iret = nc_put_att_text(ncid, *var_uz_id, "units", 13, "meters/second");
			check_err(iret);
			iret = nc_put_att_text(ncid, *var_uz_id, "long_name", 26, "Vertical velocity of water");
			check_err(iret);
			iret = nc_put_att_float(ncid, *var_uz_id, "missing_value", NC_FLOAT, 1, &fill_float);
			check_err(iret);
			iret = nc_put_att_float(ncid, *var_uz_id, "_FillValue", NC_FLOAT, 1, &fill_float);
			check_err(iret);
			iret = nc_var_par_access(ncid, *var_uz_id, NC_COLLECTIVE);
			check_err(iret);
			iret = nc_def_var_deflate(ncid, *var_uz_id, NC_SHUFFLE, 1, deflate_level);
			check_err(iret);
		}
	}
	if (var->modulo_velocidades != 0) {
		// u (módulo de la velocidad)
		iret = nc_def_var(ncid, "u", NC_FLOAT, 3, var_dims, var_u_id);
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_u_id, "units", 13, "meters/second");
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_u_id, "long_name", 17, "Velocity of water");
		check_err(iret);
		iret = nc_var_par_access(ncid, *var_u_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, *var_u_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->presion_no_hidrostatica != 0) {
		// nhp
		iret = nc_def_var(ncid, "nhp", NC_FLOAT, 3, var_dims, var_nhp_id);
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_nhp_id, "long_name", 24, "Non-hydrostatic pressure");
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_nhp_id, "units", 17, "meters^2/second^2");
		check_err(iret);
		iret = nc_var_par_access(ncid, *var_nhp_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, *var_nhp_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
	}
	if (var->flujo_momento != 0) {
		// mom_flux_x
		iret = nc_def_var(ncid, "mom_flux_x", NC_FLOAT, 3, var_dims, var_mfx_id);
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_mfx_id, "units", 18, "meters^3/seconds^2");
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_mfx_id, "long_name", 29, "Momentum flux along longitude");
		check_err(iret);
		iret = nc_var_par_access(ncid, *var_mfx_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, *var_mfx_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
		// mom_flux_y
		iret = nc_def_var(ncid, "mom_flux_y", NC_FLOAT, 3, var_dims, var_mfy_id);
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_mfy_id, "units", 18, "meters^3/seconds^2");
		check_err(iret);
		iret = nc_put_att_text(ncid, *var_mfy_id, "long_name", 28, "Momentum flux along latitude");
		check_err(iret);
		iret = nc_var_par_access(ncid, *var_mfy_id, NC_COLLECTIVE);
		check_err(iret);
		iret = nc_def_var_deflate(ncid, *var_mfy_id, NC_SHUFFLE, 1, deflate_level);
		check_err(iret);
		if (no_hidros) {
			// mom_flux_z
			iret = nc_def_var(ncid, "mom_flux_z", NC_FLOAT, 3, var_dims, var_mfz_id);
			check_err(iret);
			iret = nc_put_att_text(ncid, *var_mfz_id, "units", 18, "meters^3/seconds^2");
			check_err(iret);
			iret = nc_put_att_text(ncid, *var_mfz_id, "long_name", 22, "Vertical momentum flux");
			check_err(iret);
			iret = nc_var_par_access(ncid, *var_mfz_id, NC_COLLECTIVE);
			check_err(iret);
			iret = nc_def_var_deflate(ncid, *var_mfz_id, NC_SHUFFLE, 1, deflate_level);
			check_err(iret);
		}
	}

	// Asignamos attributos
	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE) || (okada_flag == OKADA_TRIANGULAR) ||
		(okada_flag == OKADA_TRIANGULAR_FROM_FILE) || (okada_flag == DEFORMATION_FROM_FILE) || (okada_flag == DYNAMIC_DEFORMATION)) {
		// Batimetría original
		iret = nc_put_att_text(ncid, grid_id, "long_name", 24, "Grid original bathymetry");
		check_err(iret);
		iret = nc_put_att_text(ncid, grid_id, "standard_name", 14, "original depth");
		check_err(iret);
		iret = nc_put_att_text(ncid, grid_id, "units", 6, "meters");
		check_err(iret);
		iret = nc_put_att_float(ncid, grid_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, grid_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		// Batimetría deformada con Okada
		iret = nc_put_att_text(ncid, grid_id_okada, "long_name", 24, "Grid deformed bathymetry");
		check_err(iret);
		iret = nc_put_att_text(ncid, grid_id_okada, "standard_name", 14, "deformed depth");
		check_err(iret);
		iret = nc_put_att_text(ncid, grid_id_okada, "units", 6, "meters");
		check_err(iret);
		iret = nc_put_att_float(ncid, grid_id_okada, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, grid_id_okada, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
	}
	else {
		iret = nc_put_att_text(ncid, grid_id, "long_name", 15, "Grid bathymetry");
		check_err(iret);
		iret = nc_put_att_text(ncid, grid_id, "standard_name", 5, "depth");
		check_err(iret);
		iret = nc_put_att_text(ncid, grid_id, "units", 6, "meters");
		check_err(iret);
		iret = nc_put_att_float(ncid, grid_id, "missing_value", NC_FLOAT, 1, &fill_float);
		check_err(iret);
		iret = nc_put_att_float(ncid, grid_id, "_FillValue", NC_FLOAT, 1, &fill_float);
		check_err(iret);
	}

	iret = nc_put_att_text(ncid, grid_lon_id, "long_name", 14, "Grid longitude");
	check_err(iret);
	iret = nc_put_att_text(ncid, grid_lon_id, "units", 12, "degrees_east");
	check_err(iret);
	iret = nc_put_att_text(ncid, grid_lat_id, "long_name", 13, "Grid latitude");
	check_err(iret);
	iret = nc_put_att_text(ncid, grid_lat_id, "units", 13, "degrees_north");
	check_err(iret);

	iret = nc_put_att_text(ncid, lon_id, "standard_name", 9, "longitude");
	check_err(iret);
	iret = nc_put_att_text(ncid, lon_id, "long_name", 9, "longitude");
	check_err(iret);
	iret = nc_put_att_text(ncid, lon_id, "units", 12, "degrees_east");
	check_err(iret);

	iret = nc_put_att_text(ncid, lat_id, "standard_name", 8, "latitude");
	check_err(iret);
	iret = nc_put_att_text(ncid, lat_id, "long_name", 8, "latitude");
	check_err(iret);
	iret = nc_put_att_text(ncid, lat_id, "units", 13, "degrees_north");
	check_err(iret);

	iret = nc_put_att_text(ncid, *var_time_id, "long_name", 4, "Time");
	check_err(iret);
	iret = nc_put_att_text(ncid, *var_time_id, "units", 24, "seconds since 1970-01-01");
	check_err(iret);

	// Atributos globales
	iret = nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "CF-1.0");
	check_err(iret);
	iret = nc_put_att_text(ncid, NC_GLOBAL, "title", 42, "Tsunami-HySEA Non-Hydrostatic model output");
	check_err(iret);
	iret = nc_put_att_text(ncid, NC_GLOBAL, "Tsunami-HySEA_Non-Hydrostatic_version", strlen(version), version);
	check_err(iret);
	iret = nc_put_att_text(ncid, NC_GLOBAL, "creator_name", 12, "EDANYA Group");
	check_err(iret);
	iret = nc_put_att_text(ncid, NC_GLOBAL, "institution", 20, "University of Malaga");
	check_err(iret);
	sprintf(cadena, " ");
	iret = nc_put_att_text(ncid, NC_GLOBAL, "comments", strlen(cadena), cadena);
	check_err(iret);
	sprintf(cadena, " ");
	iret = nc_put_att_text(ncid, NC_GLOBAL, "references", strlen(cadena), cadena);
	check_err(iret);

	gettimeofday(&tv, NULL);
	strftime(fecha_act, 24, "%Y-%m-%d %H:%M:%S", localtime(&(tv.tv_sec)));
	MPI_Bcast(fecha_act, 24, MPI_CHAR, 0, comm);
	iret = nc_put_att_text(ncid, NC_GLOBAL, "history", strlen(fecha_act), fecha_act);
	check_err(iret);

	iret = nc_put_att_text(ncid, NC_GLOBAL, "grid_name", strlen(nombre_bati), nombre_bati);
	check_err(iret);
	val_int = no_hidros;
	iret = nc_put_att_int(ncid, NC_GLOBAL, "non_hidrostatic", NC_INT, 1, &val_int);
	check_err(iret);
	if (no_hidros) {
		val_int = numPesosJacobi;
		iret = nc_put_att_int(ncid, NC_GLOBAL, "number_of_jacobi_weights", NC_INT, 1, &val_int);
		check_err(iret);
	}
	val_float = (float) tiempo_tot;
	iret = nc_put_att_float(ncid, NC_GLOBAL, "simulation_time", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_int = npics;
	iret = nc_put_att_int(ncid, NC_GLOBAL, "output_grid_interval", NC_INT, 1, &val_int);
	check_err(iret);

	sprintf(cadena, (fabs(borde_sup-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid, NC_GLOBAL, "upper_border", 4, cadena);
	check_err(iret);
	sprintf(cadena, (fabs(borde_inf-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid, NC_GLOBAL, "lower_border", 4, cadena);
	check_err(iret);
	sprintf(cadena, (fabs(borde_izq-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid, NC_GLOBAL, "left_border", 4, cadena);
	check_err(iret);
	sprintf(cadena, (fabs(borde_der-1.0) < EPSILON) ? "open" : "wall");
	iret = nc_put_att_text(ncid, NC_GLOBAL, "right_border", 4, cadena);
	check_err(iret);

	val_float = (float) CFL;
	iret = nc_put_att_float(ncid, NC_GLOBAL, "CFL", NC_FLOAT, 1, &val_float);
	check_err(iret);
	val_float = (float) epsilon_h;
	iret = nc_put_att_float(ncid, NC_GLOBAL, "epsilon_h", NC_FLOAT, 1, &val_float);
	check_err(iret);
	if (tipo_friccion == FIXED_FRICTION) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "friction_type", 5, "fixed");
		check_err(iret);
		val_float = (float) mf0;
		iret = nc_put_att_float(ncid, NC_GLOBAL, "water_bottom_friction", NC_FLOAT, 1, &val_float);
		check_err(iret);
	}
	else {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "friction_type", 8, "variable");
		check_err(iret);
		val_int = strlen((fich_friccion[0][0]).c_str());
		iret = nc_put_att_text(ncid, NC_GLOBAL, "water_bottom_friction_file_level_0", val_int, (fich_friccion[0][0]).c_str());
		check_err(iret);
	}
	val_float = (float) vmax;
	iret = nc_put_att_float(ncid, NC_GLOBAL, "max_speed_water", NC_FLOAT, 1, &val_float);
	check_err(iret);
	if (difh_at >= 0.0) {
		val_float = (float) difh_at;
		iret = nc_put_att_float(ncid, NC_GLOBAL, "threshold_arrival_times", NC_FLOAT, 1, &val_float);
		check_err(iret);
	}
	// Flag de okada y parámetros de Okada o gaussiana
	if (okada_flag == SEA_SURFACE_FROM_FILE) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 21, "sea_surface_from_file");
		check_err(iret);
	}
	else if (okada_flag == OKADA_STANDARD) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 14, "okada_standard");
		check_err(iret);
		escribirDatosKajiuraNC(ncid, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "time: %.4f, lon: %.4f, lat: %.4f, depth: %.4f, length: %.4f, width: %.4f, strike: %.4f, dip: %.4f, rake: %.4f, slip: %.4f",
				defTime[i], LON_C[i], LAT_C[i], DEPTH_C[i], FAULT_L[i], FAULT_W[i], STRIKE[i], DIP[i], RAKE[i], SLIP[i]);
			iret = nc_put_att_text(ncid, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == OKADA_STANDARD_FROM_FILE) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 24, "okada_standard_from_file");
		check_err(iret);
		escribirDatosKajiuraNC(ncid, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		iret = nc_put_att_text(ncid, NC_GLOBAL, "faults_file", strlen(fich_okada), fich_okada);
		check_err(iret);
	}
	else if (okada_flag == OKADA_TRIANGULAR) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 16, "okada_triangular");
		check_err(iret);
		escribirDatosKajiuraNC(ncid, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "time: %.4f, lon_barycenter: %.4f, lat_barycenter: %.4f, rake: %.4f, slip: %.4f", defTime[i], LONCTRI[i], LATCTRI[i], RAKE[i], SLIP[i]);
			iret = nc_put_att_text(ncid, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == OKADA_TRIANGULAR_FROM_FILE) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 26, "okada_triangular_from_file");
		check_err(iret);
		escribirDatosKajiuraNC(ncid, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		iret = nc_put_att_text(ncid, NC_GLOBAL, "faults_file", strlen(fich_okada), fich_okada);
		check_err(iret);
	}
	else if (okada_flag == DEFORMATION_FROM_FILE) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 31, "sea_floor_deformation_from_file");
		check_err(iret);
		escribirDatosKajiuraNC(ncid, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "time: %.4f, file: %s", defTime[i], (char *) fich_def[i].c_str());
			iret = nc_put_att_text(ncid, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == DYNAMIC_DEFORMATION) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 39, "sea_floor_dynamic_deformation_from_file");
		check_err(iret);
		escribirDatosKajiuraNC(ncid, kajiura_flag, depth_kajiura);
		val_int = numFaults;
		iret = nc_put_att_int(ncid, NC_GLOBAL, "num_faults", NC_INT, 1, &val_int);
		check_err(iret);
		for (i=0; i<numFaults; i++) {
			sprintf(nombre_fich, "fault_%d", i+1);
			sprintf(cadena, "file: %s", (char *) fich_def[i].c_str());
			iret = nc_put_att_text(ncid, NC_GLOBAL, nombre_fich, strlen(cadena), cadena);
			check_err(iret);
		}
	}
	else if (okada_flag == GAUSSIAN) {
		iret = nc_put_att_text(ncid, NC_GLOBAL, "initialization_mode", 8, "gaussian");
		check_err(iret);
		sprintf(cadena, "lon: %.4f, lat: %.4f, height: %.4f, sigma: %.4f", lonGauss, latGauss, heightGauss, sigmaGauss);
		iret = nc_put_att_text(ncid, NC_GLOBAL, "gaussian_data", strlen(cadena), cadena);
		check_err(iret);
	}

	// Fin del modo definición
	iret = nc_enddef(ncid);
	check_err(iret);

	// Guardamos lon
	iret = nc_put_var_double(ncid, lon_id, lon);
	check_err(iret);
	// Guardamos lat
	iret = nc_put_var_double(ncid, lat_id, lat);
	check_err(iret);

	// Guardamos la batimetría original
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};
	iret = nc_put_var_double(ncid, grid_lon_id, lon_grid);
	check_err(iret);
	iret = nc_put_var_double(ncid, grid_lat_id, lat_grid);
	check_err(iret);
	iret = nc_put_vara_float(ncid, grid_id, start, count, batiOriginal);
	check_err(iret);
}

void freadnc(int nivel, int submalla, VariablesGuardado *var, char *prefijo, int *p_ncid, int okada_flag,
			double tiempo_tot, MPI_Comm comm)
{
	char nombre_fich[256];
	int ncid, iret;
	float val_float;

	// Abrimos el fichero
	sprintf(nombre_fich, "%s.nc", prefijo);
	iret = nc_open_par(nombre_fich, NC_WRITE, comm, MPI_INFO_NULL, p_ncid);
	check_err(iret);
	ncid = *p_ncid;

	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE) || (okada_flag == OKADA_TRIANGULAR) ||
		(okada_flag == OKADA_TRIANGULAR_FROM_FILE) || (okada_flag == DEFORMATION_FROM_FILE) || (okada_flag == DYNAMIC_DEFORMATION)) {
		// Batimetría deformada con Okada
		iret = nc_inq_varid(ncid, "deformed_bathy", &grid_id_okada);
		check_err(iret);
		iret = nc_var_par_access(ncid, grid_id_okada, NC_COLLECTIVE);
		check_err(iret);
	}
	iret = nc_inq_varid(ncid, "time", &(time_id[nivel][submalla]));
	check_err(iret);
	iret = nc_var_par_access(ncid, time_id[nivel][submalla], NC_COLLECTIVE);
	check_err(iret);
	if (var->eta_max != 0) {
		iret = nc_inq_varid(ncid, "max_height", &(eta_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, eta_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->velocidades_max != 0) {
		// max_ux
		iret = nc_inq_varid(ncid, "max_ux", &(ux_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, ux_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		// max_uy
		iret = nc_inq_varid(ncid, "max_uy", &(uy_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, uy_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->modulo_velocidades_max != 0) {
		iret = nc_inq_varid(ncid, "max_u", &(u_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, u_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->modulo_caudales_max != 0) {
		iret = nc_inq_varid(ncid, "max_q", &(q_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, q_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->flujo_momento_max != 0) {
		iret = nc_inq_varid(ncid, "max_mom_flux", &(flux_max_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, flux_max_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->tiempos_llegada != 0) {
		iret = nc_inq_varid(ncid, "arrival_times", &(arr_times_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, arr_times_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->eta != 0) {
		// eta
		iret = nc_inq_varid(ncid, "eta", &(eta_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, eta_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->velocidades != 0) {
		// ux
		iret = nc_inq_varid(ncid, "ux", &(ux_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, ux_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		// uy
		iret = nc_inq_varid(ncid, "uy", &(uy_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, uy_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->modulo_velocidades != 0) {
		// u (módulo de la velocidad)
		iret = nc_inq_varid(ncid, "u", &(u_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, u_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	if (var->flujo_momento != 0) {
		// mom_flux_x
		iret = nc_inq_varid(ncid, "mom_flux_x", &(fluxx_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, fluxx_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
		// mom_flux_y
		iret = nc_inq_varid(ncid, "mom_flux_y", &(fluxy_id[nivel][submalla]));
		check_err(iret);
		iret = nc_var_par_access(ncid, fluxy_id[nivel][submalla], NC_COLLECTIVE);
		check_err(iret);
	}
	val_float = (float) tiempo_tot;
	iret = nc_put_att_float(ncid, NC_GLOBAL, "simulation_time", NC_FLOAT, 1, &val_float);
	check_err(iret);
}

int initNC(int no_hidros, int numPesosJacobi, int numNiveles, int nivel, int submalla, VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL],
			char *nombre_bati, char *prefijo, int num_volx, int num_voly, int inix_cluster, int iniy_cluster, int num_volx_total, int num_voly_total,
			int *nx_nc, int *ny_nc, int npics, double *lon_grid, double *lat_grid, double tiempo_tot, double CFL, double epsilon_h, int tipo_friccion,
			string fich_friccion[MAX_LEVELS][MAX_GRIDS_LEVEL], double mf0, double vmax, double difh_at, double borde_sup, double borde_inf,
			double borde_izq, double borde_der, int okada_flag, char *fich_okada, int numFaults, double *defTime, double *LON_C, double *LAT_C,
			double *DEPTH_C, double *FAULT_L, double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *LONCTRI,
			double *LATCTRI, string *fich_def, int kajiura_flag, double depth_kajiura, double lonGauss, double latGauss, double heightGauss,
			double sigmaGauss, float *batiOriginal, char *version, MPI_Comm comm)
{
	VariablesGuardado var;
	double *lon, *lat;
	int i;
	// nivel y submalla empiezan por 0
	// Nueva simulación

	var = guardarVariables[nivel][submalla];
	ErrorEnNetCDF = false;
	*nx_nc = (num_volx_total-1)/npics + 1;
	*ny_nc = (num_voly_total-1)/npics + 1;
	lon = (double *) malloc((*nx_nc)*sizeof(double));
	lat = (double *) malloc((*ny_nc)*sizeof(double));
	if (lat == NULL) {
		free(lon);
		return 1;
	}

	for (i=0; i<(*nx_nc); i++)
		lon[i] = lon_grid[i*npics];
	for (i=0; i<(*ny_nc); i++)
		lat[i] = lat_grid[i*npics];

	fgennc(no_hidros, numPesosJacobi, numNiveles, nivel, submalla, &var, lon_grid, lat_grid, lon, lat, nombre_bati, prefijo,
		&(ncid[nivel][submalla]), &(time_id[nivel][submalla]), &(eta_id[nivel][submalla]), &(ux_id[nivel][submalla]), &(uy_id[nivel][submalla]),
		&(uz_id[nivel][submalla]), &(u_id[nivel][submalla]), &(nhp_id[nivel][submalla]), &(fluxx_id[nivel][submalla]), &(fluxy_id[nivel][submalla]),
		&(fluxz_id[nivel][submalla]), *nx_nc, *ny_nc, num_volx, num_voly, inix_cluster, iniy_cluster, num_volx_total, num_voly_total,
		tiempo_tot, CFL, epsilon_h, tipo_friccion, fich_friccion, mf0, vmax, difh_at, npics, borde_sup, borde_inf, borde_izq, borde_der,
		okada_flag, fich_okada, numFaults, defTime, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, LONCTRI, LATCTRI,
		fich_def, kajiura_flag, depth_kajiura, lonGauss, latGauss, heightGauss, sigmaGauss, batiOriginal, version, comm);

	free(lon);
	free(lat);

	return 0;
}

void readNC(int nivel, int submalla, VariablesGuardado guardarVariables[MAX_LEVELS][MAX_GRIDS_LEVEL],
			char *prefijo, int num_volx_total, int num_voly_total, int *nx_nc, int *ny_nc, int npics,
			int okada_flag, double tiempo_tot, MPI_Comm comm)
{
	VariablesGuardado var;
	// nivel y submalla empiezan por 0
	// Continuación de una simulación anterior

	var = guardarVariables[nivel][submalla];
	ErrorEnNetCDF = false;
	*nx_nc = (num_volx_total-1)/npics + 1;
	*ny_nc = (num_voly_total-1)/npics + 1;
	freadnc(nivel, submalla, &var, prefijo, &(ncid[nivel][submalla]), okada_flag, tiempo_tot, comm);
}

// ESCRIBIR TIEMPO

void writeTimeNC(int nivel, int submalla, int num, float tiempo_act)
{
	int iret;
	float t_act = tiempo_act;
	const size_t paso = num;
	const size_t uno = 1;

	iret = nc_put_vara_float(ncid[nivel][submalla], time_id[nivel][submalla], &paso, &uno, &t_act);
	check_err(iret);
}

// ESCRIBIR ETA

void writeEtaNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *eta)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, eta_id[nivel][submalla], start, count, eta);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

// ESCRIBIR VELOCIDADES

void writeUxNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *ux)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, ux_id[nivel][submalla], start, count, ux);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

void writeUyNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *uy)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, uy_id[nivel][submalla], start, count, uy);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

void writeUzNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *uz)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, uz_id[nivel][submalla], start, count, uz);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

void writeUNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *u)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, u_id[nivel][submalla], start, count, u);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

// ESCRIBIR PRESIÓN NO HIDROSTÁTICA

void writePresionNH(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *Pnh)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, nhp_id[nivel][submalla], start, count, Pnh);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

// ESCRIBIR FLUJO DE MOMENTO

void writeFluxxNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *flux)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, fluxx_id[nivel][submalla], start, count, flux);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

void writeFluxyNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *flux)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, fluxy_id[nivel][submalla], start, count, flux);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

void writeFluxzNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, int num, float *flux)
{
	int iret;
	int id = ncid[nivel][submalla];
	const size_t start[] = {(size_t) num, (size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {1, (size_t) ny_nc, (size_t) nx_nc};

	iret = nc_put_vara_float(id, fluxz_id[nivel][submalla], start, count, flux);
	check_err(iret);

	iret = nc_sync(id);
	check_err(iret);
}

// ESCRIBIR VALORES MÁXIMOS

void guardarAmplitudMaximaNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	const size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], eta_max_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

void guardarUxMaximaNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], ux_max_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

void guardarUyMaximaNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], uy_max_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

void guardarUzMaximaNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], uz_max_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

void guardarModuloVelocidadMaximaNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	const size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], u_max_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

void guardarModuloCaudalMaximoNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	const size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], q_max_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

void guardarFlujoMomentoMaximoNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	const size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], flux_max_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

// ESCRIBIR TIEMPOS DE LLEGADA DEL TSUNAMI

void guardarTiemposLlegadaNC(int nivel, int submalla, int nx_nc, int ny_nc, int inix_nc, int iniy_nc, float *vec)
{
	const size_t start[] = {(size_t) iniy_nc, (size_t) inix_nc};
	const size_t count[] = {(size_t) ny_nc, (size_t) nx_nc};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], arr_times_id[nivel][submalla], start, count, vec);
	check_err(iret);
}

// ESCRIBIR BATIMETRÍA DEFORMADA CON OKADA

void guardarBatimetriaModificadaNC(int nivel, int submalla, int num_volx, int num_voly, int inix_cluster, int iniy_cluster, float *vec)
{
	const size_t start[] = {(size_t) iniy_cluster, (size_t) inix_cluster};
	const size_t count[] = {(size_t) num_voly, (size_t) num_volx};
	int iret;

	iret = nc_put_vara_float(ncid[nivel][submalla], grid_id_okada, start, count, vec);
	check_err(iret);
}

void closeNC(int nivel, int submalla)
{
	int iret;

	iret = nc_close(ncid[nivel][submalla]);
	check_err(iret);
}

#endif
