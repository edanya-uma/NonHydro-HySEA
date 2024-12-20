#ifndef _NETCDF_H_
#define _NETCDF_H_

#include "Constantes.hxx"
#include <stdio.h>
#include <netcdf.h>

bool ErrorEnNetCDF;
// Id del fichero
int ncid[MAX_LEVELS][MAX_GRIDS_LEVEL];

void check_err(int iret)
{
	if ((iret != NC_NOERR) && (! ErrorEnNetCDF)) {
		fprintf(stderr, "%s\n", nc_strerror(iret));
		ErrorEnNetCDF = true;
	}
}

/***********/
/* Lectura */
/***********/

void abrirGRD(const char *nombre_fich, int nivel, int submalla, int *nvx, int *nvy)
{
	int iret;
	int *p_ncid;
	int lon_dim, lat_dim;
	size_t nx, ny;

	p_ncid = &(ncid[nivel][submalla]);
	iret = nc_open(nombre_fich, NC_NOWRITE, p_ncid);
	check_err(iret);
	// Leemos el tamaño de la malla.
	// El nombre de las dimensiones puede ser (lon,lat) o (x,y).
	iret = nc_inq_dimid(*p_ncid, "lon", &lon_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(*p_ncid, "x", &lon_dim);
		check_err(iret);
	}
	iret = nc_inq_dimid(*p_ncid, "lat", &lat_dim);
	if (iret != NC_NOERR) {
		iret = nc_inq_dimid(*p_ncid, "y", &lat_dim);
		check_err(iret);
	}
	iret = nc_inq_dimlen(*p_ncid, lon_dim, &nx);
	check_err(iret);
	iret = nc_inq_dimlen(*p_ncid, lat_dim, &ny);
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

void leerBatimetriaGRD(int nivel, int submalla, int num_volx, int num_voly, float *bati)
{
	int id, iret;
	int z_id;
	size_t start[] = {0, 0};
	size_t count[] = {(size_t) num_voly, (size_t) num_volx};

	id = ncid[nivel][submalla];
	iret = nc_inq_varid(id, "z", &z_id);
	check_err(iret);
	iret = nc_get_vara_float(id, z_id, start, count, bati);
	check_err(iret);
}

void cerrarGRD(int nivel, int submalla)
{
	int iret;

	iret = nc_close(ncid[nivel][submalla]);
	check_err(iret);
}

#endif
