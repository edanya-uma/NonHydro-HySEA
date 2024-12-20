#ifndef _DEFORMACION_H_
#define _DEFORMACION_H_

#include <stdio.h>
#include "Constantes.hxx"
#include "NestedMeshesDif.cu"
#include "prtoxy.cu"
#include "DC3D.cu"
#include <cufft.h>

/****************************/
/* Comprobación de warnings */
/****************************/

// Si hay algún NaN que se ha puesto a cero en la deformación
// Si x = 1, y indica la falla que da el warning
__device__ int2 d_avisoNaNEnDef;

// Si algún valor de la deformación está fuera de [-500,500] metros
// Si x = 1, y indica la falla que da el warning
__device__ int2 d_avisoValorGrandeEnDef;

__global__ void inicializarFlagsWarningsDef()
{
	d_avisoNaNEnDef = make_int2(0,0);
	d_avisoValorGrandeEnDef = make_int2(0,0);
}

/*****************/
/* Interpolación */
/*****************/

__device__ double interpolacionBilineal(double *d_deformacionNivel0, int nvxDef, int posx_izq,
					int posx_der, int posy_inf, int posy_sup, double distx, double disty)
{
	// distx es la distancia en x del punto a interpolar con respecto a la posición izquierda
	// disty es la distancia en y del punto a interpolar con respecto a la posición inferior
	// si: sup izq; sd: sup der; ii: inf izq; id: inf der
	int possi, possd, posii, posid;
	double defsi, defsd, defii, defid;
	double val;

	possi = posy_sup*nvxDef + posx_izq;
	possd = posy_sup*nvxDef + posx_der;
	posii = posy_inf*nvxDef + posx_izq;
	posid = posy_inf*nvxDef + posx_der;

	defsi = d_deformacionNivel0[possi];
	defsd = d_deformacionNivel0[possd];
	defii = d_deformacionNivel0[posii];
	defid = d_deformacionNivel0[posid];

	val = defii*(1.0-distx)*(1.0-disty) + defid*distx*(1.0-disty) + defsi*(1.0-distx)*disty + defsd*distx*disty;

	return val;
}

__global__ void interpolarDeformacionGPU(int l, int numNiveles, double2 *d_datosVolumenesNivel_1, double *d_eta1Inicial,
				double *d_deformacionNivel0, double *d_deformacionAcumuladaNivel, int inixSubmalla, int iniySubmalla,
				int nvxSubmalla, int nvySubmalla, int inixDef, int iniyDef, int nvxDef, int nvyDef,
				int inixSubmallaCluster, int iniySubmallaCluster, int ratio_ref, double *vccos)
{
	int pos_x_hebra, pos_y_hebra;
	int pos2, pos_datos;
	int posxNivel0, posyNivel0;
	int restox, restoy;
	int posx_izq, posx_der;
	int posy_sup, posy_inf;
	int j_global;
	double distx, disty;
	double factor, U_Z;
	// inixSubmalla e iniySubmalla se indican con respecto a la malla del nivel 0
	// y están en la resolución de la malla fina donde se aplica la deformación

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < nvxSubmalla+4) && (pos_y_hebra < nvySubmalla+4)) {
		// Aplicamos la deformación también en las celdas fantasma (si la deformación está definida para ellas)
		pos_datos = pos_y_hebra*(nvxSubmalla + 4) + pos_x_hebra;
		// Restamos 2 por la indexación debido a la inclusión de las celdas fantasma
		j_global = iniySubmallaCluster + pos_y_hebra-2;
		posxNivel0 = (inixSubmalla+inixSubmallaCluster+pos_x_hebra-2)/ratio_ref;
		posyNivel0 = (iniySubmalla+iniySubmallaCluster+pos_y_hebra-2)/ratio_ref;

		if ((posxNivel0 >= inixDef) && (posxNivel0 < inixDef+nvxDef) && (posyNivel0 >= iniyDef) && (posyNivel0 < iniyDef+nvyDef)) {
			// El punto está dentro de la ventana de computación de Okada
			restox = ((inixSubmalla+inixSubmallaCluster+pos_x_hebra-2)&(ratio_ref-1));
			restoy = ((iniySubmalla+iniySubmallaCluster+pos_y_hebra-2)&(ratio_ref-1));
			factor = 1.0/ratio_ref;
			distx = (restox + 0.5)*factor;
			disty = (restoy + 0.5)*factor;
			if (restox+0.5 < 0.5*ratio_ref) {
				posx_der = posxNivel0-inixDef;
				posx_izq = max(posx_der-1,0);
				distx += 0.5;
			}
			else {
				posx_izq = posxNivel0-inixDef;
				posx_der = min(posx_izq+1,nvxDef-1);
				distx -= 0.5;
			}
			if (restoy+0.5 < 0.5*ratio_ref) {
				posy_sup = posyNivel0-iniyDef;
				posy_inf = max(posy_sup-1,0);
				disty += 0.5;
			}
			else {
				posy_inf = posyNivel0-iniyDef;
				posy_sup = min(posy_inf+1,nvyDef-1);
				disty -= 0.5;
			}

			U_Z = interpolacionBilineal(d_deformacionNivel0, nvxDef, posx_izq, posx_der,
					posy_inf, posy_sup, distx, disty);
			U_Z *= vccos[j_global];
			d_datosVolumenesNivel_1[pos_datos].y -= U_Z;
			if ((pos_x_hebra > 1) && (pos_x_hebra < nvxSubmalla+2) && (pos_y_hebra > 1) && (pos_y_hebra < nvySubmalla+2)) {
				// Solo aplicamos la deformación en d_deformacionAcumuladaNivel y en d_eta1Inicial
				// si es una celda interna de la submalla (no fantasma)
				// Aplicamos la deformación también a d_eta1Inicial (para los tiempos de llegada
				// y la activación de las mallas anidadas)
				pos2 = (pos_y_hebra-2)*nvxSubmalla + pos_x_hebra-2;
				d_eta1Inicial[pos2] += U_Z;
				if (l < numNiveles-1)
					d_deformacionAcumuladaNivel[pos2] -= U_Z;
			}
		}
	}
}


/***********/
/* Kajiura */
/***********/

// Devuelve 0 si todo ha ido bien, 1 si no hay memoria suficiente
int crearDatosCPUKajiura(int okada_flag, tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int numFaults,
						int numEstadosDefDinamica, int4 *submallasDeformacion, double **F2Sx, double **F2Sy, double H)
{
	tipoDatosSubmalla *tds;
	double dxm = 1852.0/H;  // 1 arc-min en metros
	double dxmin, dymin;
	double fsx, fsy;
	double stepx, stepy;
	int nstepx, nstepy;
	int nvx, nvy;
	int i, j, k;
	int numDeformaciones = ((okada_flag == DYNAMIC_DEFORMATION) ? numEstadosDefDinamica : numFaults);

	for (k=0; k<numDeformaciones; k++) {
		if (okada_flag == DYNAMIC_DEFORMATION) {
			nvx = submallasDeformacion[0].z;
			nvy = submallasDeformacion[0].w;
		}
		else {
			nvx = submallasDeformacion[k].z;
			nvy = submallasDeformacion[k].w;
		}

		F2Sx[k] = (double *) malloc(nvx*sizeof(double));
		if (F2Sx[k] == NULL) {
			for (i=0; i<k; i++) {
				free(F2Sx[i]);
				free(F2Sy[i]);
			}
			return 1;
		}
		F2Sy[k] = (double *) malloc(nvy*sizeof(double));
		if (F2Sy[k] == NULL) {
			for (i=0; i<k; i++) {
				free(F2Sx[i]);
				free(F2Sy[i]);
			}
			free(F2Sx[k]);
			return 1;
		}
		tds = &(datosNivel[0][0]);
		dxmin = (tds->longitud[1] - tds->longitud[0])*60.0;
		dymin = (tds->latitud[1] - tds->latitud[0])*60.0;
		fsx = 1.0/(dxmin*dxm);
		fsy = 1.0/(dymin*dxm);
		// F2Sx
		stepx = 1.0/(0.5*nvx);
		nstepx = nvx/2 + 1;
		for (i=0; i<nstepx; i++)
			F2Sx[k][i] = 0.5*fsx*i*stepx;
		j = 2;
		for (i=nstepx; i<nvx; i++) {
			F2Sx[k][i] = F2Sx[k][nstepx-j];
			j++;
		}
		// F2Sy
		stepy = 1.0/(0.5*nvy);
		nstepy = nvy/2 + 1;
		for (i=0; i<nstepy; i++)
			F2Sy[k][i] = 0.5*fsy*i*stepy;
		j = 2;
		for (i=nstepy; i<nvy; i++) {
			F2Sy[k][i] = F2Sy[k][nstepy-j];
			j++;
		}
	}

	return 0;
}

__global__ void crearDatosKajiuraGPU(cuDoubleComplex *d_datosKajiura, double *d_datosDeformacion,
				int num_volx, int num_voly)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double def;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		def = d_datosDeformacion[pos];
		d_datosKajiura[pos] = make_cuDoubleComplex(def, 0.0);
	}
}

__global__ void operacionesKajiuraGPU(cuDoubleComplex *d_datosKajiura, double *d_F2Sx, double *d_F2Sy,
				int num_volx, int num_voly, double depth_kajiura)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double valx, valy, factor;
	cuDoubleComplex def;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		factor = depth_kajiura*2.0*M_PI;
		valx = cosh(factor*d_F2Sx[pos_x_hebra]);
		valy = cosh(factor*d_F2Sy[pos_y_hebra]);
		factor = (1.0 / valy) / valx;
		def = d_datosKajiura[pos];
		def = make_cuDoubleComplex(cuCreal(def)*factor, cuCimag(def)*factor);
		d_datosKajiura[pos] = def;
	}
}

__global__ void escribirKajiuraEnDeformacionGPU(cuDoubleComplex *d_datosKajiura, double *d_datosDeformacion,
				int num_volx, int num_voly)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double def;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		def = cuCreal(d_datosKajiura[pos]);
		// Normalizamos el resultado dividiendo por el número de elementos
		d_datosDeformacion[pos] = def/(num_volx*num_voly);
	}
}

void aplicarKajiura(double *d_deformacionNivel0, cuDoubleComplex *d_datosKajiura, double *d_F2Sx, double *d_F2Sy,
					int num_volx, int num_voly, double depthk, dim3 blockGridOkada, dim3 threadBlockEst)
{
	cufftHandle plan;

	crearDatosKajiuraGPU<<<blockGridOkada, threadBlockEst>>>(d_datosKajiura, d_deformacionNivel0, num_volx, num_voly);
	cufftPlan2d(&plan, num_voly, num_volx, CUFFT_Z2Z);
	cufftExecZ2Z(plan, d_datosKajiura, d_datosKajiura, CUFFT_FORWARD);
	operacionesKajiuraGPU<<<blockGridOkada, threadBlockEst>>>(d_datosKajiura, d_F2Sx, d_F2Sy, num_volx, num_voly, depthk);
	cufftExecZ2Z(plan, d_datosKajiura, d_datosKajiura, CUFFT_INVERSE);
	escribirKajiuraEnDeformacionGPU<<<blockGridOkada, threadBlockEst>>>(d_datosKajiura, d_deformacionNivel0, num_volx, num_voly);
	cufftDestroy(plan);
}


/******************/
/* Okada standard */
/******************/

__global__ void aplicarOkadaStandardGPU(double *d_deformacionNivel0, int fallaOkada, int num_volx, int num_voly,
				double lon_ini, double incx, double lat_ini, double incy, double LON_C_ent, double LAT_C_ent,
				double DEPTH_C_ent, double FAULT_L, double FAULT_W, double STRIKE, double DIP_ent,
				double RAKE, double SLIP, double H)
{
	double LON_P, LAT_P;
	double LON_C, LAT_C;
	double DEPTH_C, DIP;
	double S_RAKE;
	double C_RAKE;
	double S_STRIKE;
	double C_STRIKE;
	double Z;
	double AL1, AL2, AW1, AW2;
	double X_OKA, Y_OKA;
	double XP, YP;
	double RAD = M_PI/180.0;
	double alfa = 2.0/3.0;
	int i0 = 0;
	int IRET;
	double U_X, U_Y, U_Z, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ;
	double DISL1, DISL2, DISL3;
	int pos, pos_x_hebra, pos_y_hebra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		LON_C = LON_C_ent;
		LAT_C = LAT_C_ent;
		DEPTH_C = DEPTH_C_ent;
		DIP = DIP_ent;

		// Obtenemos longitud y latitud del punto asociado a la hebra
		LON_P = lon_ini + pos_x_hebra*incx;
		LAT_P = lat_ini + pos_y_hebra*incy;

		S_RAKE = sin(RAD*RAKE);
		C_RAKE = cos(RAD*RAKE);

		S_STRIKE = sin(RAD*STRIKE);
		C_STRIKE = cos(RAD*STRIKE);

		DISL2 = SLIP*S_RAKE;
		DISL1 = SLIP*C_RAKE;
		DISL3 = 0.0;

		Z = 0.0;
		AL1 = -0.5*FAULT_L;
		AL2 = 0.5*FAULT_L;
		AW1 = -0.5*FAULT_W;
		AW2 = 0.5*FAULT_W;

		prtoxy_(&LAT_P, &LON_P, &LAT_C, &LON_C, &XP, &YP, &i0);
		X_OKA = XP*S_STRIKE + YP*C_STRIKE;
		Y_OKA = -XP*C_STRIKE + YP*S_STRIKE;
		dc3d_(&alfa, &X_OKA, &Y_OKA, &Z, &DEPTH_C, &DIP, &AL1, &AL2, &AW1, &AW2, &DISL1, &DISL2, &DISL3,
			&U_X, &U_Y, &U_Z, &UXX, &UYX, &UZX, &UXY, &UYY, &UZY, &UXZ, &UYZ, &UZZ, &IRET);

		// Comprobación del valor U_Z obtenido
		if (U_Z != U_Z) {
			U_Z = 0.0;
			d_avisoNaNEnDef = make_int2(1, fallaOkada);
		}
		else if ((U_Z < -500.0) || (U_Z > 500.0)) {
			d_avisoValorGrandeEnDef = make_int2(1, fallaOkada);
		}

		// Escribimos la deformación U_Z en d_deformacionNivel0
		d_deformacionNivel0[pos] = U_Z/H;
	}
}

/*********************/
/* Sumar deformación */
/*********************/

__global__ void sumarDeformacionADatosGPU(int numNiveles, double2 *d_datosVolumenesNivel0_1, double *d_eta1Inicial,
				double *d_deformacionNivel0, double *d_deformacionAcumuladaNivel0, int nvxNivel0, int nvyNivel0,
				int inixDef, int iniyDef, int nvxDef, int nvyDef, int inix_cluster, int iniy_cluster, double *vccos)
{
	int pos_x_hebra, pos_y_hebra;
	int pos_def, pos_datos;
	double U_Z;
	// Coordenadas x, y globales del nivel 0 asociadas a la hebra
	int posx, posy;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < nvxNivel0) && (pos_y_hebra < nvyNivel0)) {
		posx = inix_cluster+pos_x_hebra;
		posy = iniy_cluster+pos_y_hebra;
		if ((posx >= inixDef) && (posx < inixDef+nvxDef) && (posy >= iniyDef) && (posy < iniyDef+nvyDef)) {
			pos_datos = (pos_y_hebra+2)*(nvxNivel0+4) + pos_x_hebra+2;
			pos_def = (posy-iniyDef)*nvxDef + (posx-inixDef);
			U_Z = d_deformacionNivel0[pos_def];
			U_Z *= vccos[posy];
			d_datosVolumenesNivel0_1[pos_datos].y -= U_Z;

			// Aplicamos la deformación también a d_eta1Inicial (para los tiempos de llegada
			// y la activación de las mallas anidadas)
			pos_datos = pos_y_hebra*nvxNivel0 + pos_x_hebra;
			d_eta1Inicial[pos_datos] += U_Z;
			if (numNiveles > 1)
				d_deformacionAcumuladaNivel0[pos_y_hebra*nvxNivel0 + pos_x_hebra] -= U_Z;
		}
	}
}


/********************/
/* Okada triangular */
/********************/

__device__ void d_DOT_PRODUCT(double *V1, double *V2, double& D)
{
	int i;

	D = 0.0;
	for (i=0; i<3; i++) {
		D += V1[i]*V2[i];
	}
}

__device__ void d_CROSS(double *X_VEC, double *Y_VEC, double *CROSS_VEC)
{
	CROSS_VEC[0] = X_VEC[1] * Y_VEC[2] - X_VEC[2] * Y_VEC[1];
	CROSS_VEC[1] = X_VEC[2] * Y_VEC[0] - X_VEC[0] * Y_VEC[2];
	CROSS_VEC[2] = X_VEC[0] * Y_VEC[1] - X_VEC[1] * Y_VEC[0];
}

__device__ void ROTATE(double X, double Y, double ALPHA, double& XP, double& YP)
{
	double AA = (M_PI/180.0)*ALPHA;
	XP = cos(AA)*X - sin(AA)*Y;
	YP = sin(AA)*X + cos(AA)*Y;
}

__device__ void calculo_UZ(double x, double y, double2 *d_vc, double* Z, double *d_SLIPVEC, double& UZ)
{
	double n[3];
	double v1[3], v2[3];
	double d;
	double z0;

	v1[0] = d_vc[1].x - d_vc[0].x;
	v1[1] = d_vc[1].y - d_vc[0].y;
	v1[2] = Z[1]-Z[0];
	v2[0] = d_vc[2].x - d_vc[0].x;
	v2[1] = d_vc[2].y - d_vc[0].y;
	v2[2] = Z[2]-Z[0];
	d_CROSS(v1,v2,n);
	d = n[0]*d_vc[0].x + n[1]*d_vc[0].y + n[2]*Z[0];
	if (fabs(n[2]) > 0.0) {
		z0 = (d-n[0]*x - n[1]*y)/n[2];
		if (z0 < 0.0) {
			UZ = UZ-d_SLIPVEC[2];
		}
	}
}

__device__ void in_triangle(double x, double y, double2 *d_vc, int& triangle)
{
	double angle, pi2, sum, theta, theta1, thetai;
	double tol, u, v;
	int i,m;
	int ll = 1;

    triangle = 0;
    pi2 = 2.0*M_PI;
    tol = 1e-12;
    m = 0;
    u = d_vc[0].x - x;
    v = d_vc[0].y - y;
    if ((fabs(u) < EPSILON) && (fabs(v) < EPSILON)) {
        ll = 0;
    }
    theta1 = atan2(v,u);

    sum = 0.0;
    theta = theta1;
    for (i=1; i<3; i++) {
        u = d_vc[i].x - x;
        v = d_vc[i].y - y;
        if ((fabs(u) < EPSILON) && (fabs(v) < EPSILON)) {
            ll = 0;
        }
        thetai = atan2(v, u);
    
        angle = fabs(thetai - theta);
        if (fabs(angle - M_PI) < tol) {
            ll = 0;
        }
        if (angle > M_PI) {
            angle = angle - pi2;
        }
        
        if (theta > thetai) {
            angle = -angle;
        }
        sum = sum + angle;
        theta = thetai;
    }
    
    angle = fabs(theta1 - theta);
    if (fabs(angle - M_PI) < tol) {
        ll = 0;
    }
    if (angle > M_PI) {
        angle = angle - pi2;
    }
    if (theta > theta1) {
        angle = -angle;
    }
    sum = sum + angle;
    
    m = int(fabs(sum)/pi2 + 0.02);
    if (m == 0) {
        triangle = 0;
    }
    else {
        triangle = 1;
    }
    if (ll == 0) {
        triangle = 1;
    }
}

__device__ void ADV(double Y1, double Y2, double Y3, double A, double BETA, double NU, double B1, double B2, double B3, double& V1, double& V2, double &V3)
{
    double sinBETA,cosBETA,COTBETA;
    double Z1,Z3,R2,R;
    double Y3BAR,Z1BAR,Z3BAR,R2BAR,RBAR,F,FBAR;
    double V1INFB1,V2INFB1,V3INFB1;
    double V1CB1,V2CB1,V3CB1;
    double V1B1,V2B1,V3B1;
    double V1INFB2,V2INFB2,V3INFB2;
    double V1CB2,V2CB2,V3CB2;
    double V1B2,V2B2,V3B2;
    double V1INFB3,V2INFB3,V3INFB3;
    double V1CB3,V2CB3,V3CB3;
    double V1B3,V2B3,V3B3;
    double factor1 = 1.0/(4.0*M_PI*(1.0-NU));
    double factor2 = 1.0/(8.0*M_PI*(1.0-NU));

    sinBETA=sin(BETA);
    cosBETA=cos(BETA);
    COTBETA=1.0/tan(BETA);

    Z1=Y1*cosBETA-Y3*sinBETA;
    Z3=Y1*sinBETA+Y3*cosBETA;
    R2=Y1*Y1+Y2*Y2+Y3*Y3;
    R=sqrt(R2);
    Y3BAR=Y3+2.0*A;
    Z1BAR=Y1*cosBETA+Y3BAR*sinBETA;
    Z3BAR=-Y1*sinBETA+Y3BAR*cosBETA;
    R2BAR=Y1*Y1+Y2*Y2+Y3BAR*Y3BAR;
    RBAR=sqrt(R2BAR);
    F=-atan2(Y2,Y1)+atan2(Y2,Z1)+atan2(Y2*R*sinBETA,Y1*Z1+(Y2*Y2)*cosBETA);

    FBAR=-atan2(Y2,Y1)+atan2(Y2,Z1BAR)+atan2(Y2*RBAR*sinBETA,Y1*Z1BAR+(Y2*Y2)*cosBETA);

	// Case I: Burgers vector (B1,0,0)
    V1INFB1=2.0*(1.0-NU)*(F+FBAR)-Y1*Y2*(1.0/(R*(R-Y3))+1.0/(RBAR*(RBAR+Y3BAR))) - Y2*cosBETA*((R*sinBETA-Y1)/(R*(R-Z3))+(RBAR*sinBETA-Y1)/(RBAR*(RBAR+Z3BAR)));
    V2INFB1=(1.0-2.0*NU)*(log(R-Y3)+log(RBAR+Y3BAR)-cosBETA*(log(R-Z3)+log(RBAR+Z3BAR))) -
    Y2*Y2*(1.0/(R*(R-Y3))+1.0/(RBAR*(RBAR+Y3BAR))-cosBETA*(1.0/(R*(R-Z3))+1.0/(RBAR*(RBAR+Z3BAR))));
    V3INFB1=Y2*(1.0/R-1.0/RBAR-cosBETA*((R*cosBETA-Y3)/(R*(R-Z3))-(RBAR*cosBETA+Y3BAR)/(RBAR*(RBAR+Z3BAR))));

    V1INFB1=V1INFB1*factor2;
    V2INFB1=V2INFB1*factor2;
    V3INFB1=V3INFB1*factor2;

    V1CB1=-2.0*(1.0-NU)*(1.0-2.0*NU)*FBAR*(COTBETA*COTBETA)+(1.0-2.0*NU)*Y2/(RBAR+Y3BAR)*((1.0-2.0*NU-A/RBAR)*COTBETA-Y1/(RBAR+Y3BAR)*(NU+A/RBAR))+(1.0-2.0*NU)*Y2*cosBETA*COTBETA/(RBAR+Z3BAR)*(cosBETA+A/RBAR)+ A*Y2*(Y3BAR-A)*COTBETA/(RBAR*RBAR*RBAR)+Y2*(Y3BAR-A)/(RBAR*(RBAR+Y3BAR))*(-(1.0-2.0*NU)*COTBETA+Y1/(RBAR+Y3BAR)*(2.0*NU+A/RBAR)+A*Y1/(RBAR*RBAR)+Y2*(Y3BAR-A)/(RBAR*(RBAR+Z3BAR))*(cosBETA/RBAR+Z3BAR)*((RBAR*cosBETA+Y3BAR)*((1.0-2.0*NU)*cosBETA-A/RBAR)*COTBETA+2.0*(1.0-NU)*(RBAR*sinBETA-Y1)*cosBETA)- A*Y3BAR*cosBETA*COTBETA/(RBAR*RBAR));
    V2CB1=(1.0-2.0*NU)*((2.0*(1.0-NU)*(COTBETA*COTBETA)-NU)*log(RBAR+Y3BAR)-(2.0*(1.0-NU)*(COTBETA*COTBETA)+1.0-2.0*NU)*cosBETA*log(RBAR+Z3BAR))-(1.0-2.0*NU)/(RBAR+Y3BAR)*(Y1*COTBETA*(1.0-2.0*NU-A/RBAR)+NU*Y3BAR-A+(Y2*Y2)/(RBAR+Y3BAR)*(NU+A/RBAR))-(1.0-2.0*NU)*Z1BAR*COTBETA/(RBAR+Z3BAR)*(cosBETA+A/RBAR)-A*Y1*(Y3BAR-A)*COTBETA/(RBAR*RBAR*RBAR)+(Y3BAR-A)/(RBAR+Y3BAR)*(-2.0*NU+1.0/RBAR*((1.0-2.0*NU)*Y1*COTBETA-A)+(Y2*Y2)/(RBAR*(RBAR+Y3BAR))*(2.0*NU+A/RBAR)+A*(Y2*Y2)/(RBAR*RBAR*RBAR))+(Y3BAR-A)/(RBAR+Z3BAR)*((cosBETA*cosBETA)-1.0/RBAR*((1.0-2.0*NU)*Z1BAR*COTBETA+A*cosBETA)+A*Y3BAR*Z1BAR*COTBETA/(RBAR*RBAR*RBAR)-1.0/(RBAR*(RBAR+Z3BAR))*((Y2*Y2)*(cosBETA*cosBETA)-A*Z1BAR*COTBETA/RBAR*(RBAR*cosBETA+Y3BAR)));
    V3CB1=2.0*(1.0-NU)*(((1.0-2.0*NU)*FBAR*COTBETA)+(Y2/(RBAR+Y3BAR)*(2.0*NU+A/RBAR))-(Y2*cosBETA/(RBAR+Z3BAR)*(cosBETA+A/RBAR)))+Y2*(Y3BAR-A)/RBAR*(2.0*NU/(RBAR+Y3BAR)+A/(RBAR*RBAR))+Y2*(Y3BAR-A)*cosBETA/(RBAR*(RBAR+Z3BAR))*(1.0-2.0*NU-(RBAR*cosBETA+Y3BAR)/(RBAR+Z3BAR)*(cosBETA+A/RBAR)-A*Y3BAR/(RBAR*RBAR));

    V1CB1=V1CB1*factor1;
    V2CB1=V2CB1*factor1;
    V3CB1=V3CB1*factor1;

    V1B1=V1INFB1+V1CB1;
    V2B1=V2INFB1+V2CB1;
    V3B1=V3INFB1+V3CB1;

	// Case II: Burgers vector (0,B2,0)
    V1INFB2=-(1.0-2.0*NU)*(log(R-Y3)+log(RBAR+Y3BAR)-cosBETA*(log(R-Z3)+log(RBAR+Z3BAR)))+Y1*Y1*(1.0/(R*(R-Y3))+1.0/(RBAR*(RBAR+Y3BAR)))+Z1*(R*sinBETA-Y1)/(R*(R-Z3))+Z1BAR*(RBAR*sinBETA-Y1)/(RBAR*(RBAR+Z3BAR));
    V2INFB2=2.0*(1.0-NU)*(F+FBAR)+Y1*Y2*(1.0/(R*(R-Y3))+1.0/(RBAR*(RBAR+Y3BAR)))-Y2*(Z1/(R*(R-Z3))+Z1BAR/(RBAR*(RBAR+ Z3BAR)));
    V3INFB2=-(1.0-2.0*NU)*sinBETA*(log(R-Z3)-log(RBAR+Z3BAR))-Y1*(1.0/R-1.0/RBAR)+Z1*(R*cosBETA-Y3)/(R*(R-Z3))-Z1BAR*(RBAR*cosBETA+Y3BAR)/(RBAR*(RBAR+Z3BAR));

    V1INFB2=V1INFB2*factor2;
    V2INFB2=V2INFB2*factor2;
    V3INFB2=V3INFB2*factor2;

    V1CB2=(1.0-2.0*NU)*((2.0*(1.0-NU)*(COTBETA*COTBETA)+NU)*log(RBAR+Y3BAR)-(2.0*(1.0-NU)*(COTBETA*COTBETA)+ 1.0)*cosBETA*log(RBAR+Z3BAR))+(1.0-2.0*NU)/(RBAR+Y3BAR)*(-(1.0-2.0*NU)*Y1*COTBETA+NU*Y3BAR-A+A*Y1*COTBETA/RBAR+(Y1*Y1)/(RBAR+Y3BAR)*(NU+A/RBAR))-(1.0-2.0*NU)*COTBETA/(RBAR+Z3BAR)*(Z1BAR*cosBETA-A*(RBAR*sinBETA-Y1)/(RBAR*cosBETA))-A*Y1*(Y3BAR-A)*COTBETA/(RBAR*RBAR*RBAR)+(Y3BAR-A)/(RBAR+Y3BAR)*(2.0*NU+1.0/RBAR*((1.0-2.0*NU)*Y1*COTBETA+A)-(Y1*Y1)/(RBAR*(RBAR+Y3BAR))*(2.0*NU+A/RBAR)-A*(Y1*Y1)/(RBAR*RBAR*RBAR))+(Y3BAR-A)*COTBETA/(RBAR+Z3BAR)*(-cosBETA*sinBETA+A*Y1*Y3BAR/(RBAR*RBAR*RBAR*cosBETA)+(RBAR*sinBETA-Y1)/RBAR*(2.0*(1.0-NU)*cosBETA-(RBAR*cosBETA+Y3BAR)/(RBAR+Z3BAR)*(1.0+A/(RBAR*cosBETA))));
    V2CB2=2.0*(1.0-NU)*(1.0-2.0*NU)*FBAR*COTBETA*COTBETA+(1.0-2.0*NU)*Y2/(RBAR+Y3BAR)*(-(1.0-2.0*NU-A/RBAR)*COTBETA+ Y1/(RBAR+Y3BAR)*(NU+A/RBAR))-(1.0-2.0*NU)*Y2*COTBETA/(RBAR+Z3BAR)*(1.0+A/(RBAR*cosBETA))-A*Y2*(Y3BAR-A)*COTBETA/(RBAR*RBAR*RBAR)+Y2*(Y3BAR-A)/(RBAR*(RBAR+Y3BAR))*((1.0-2.0*NU)*COTBETA-2.0*NU*Y1/(RBAR+Y3BAR)- A*Y1/RBAR*(1.0/RBAR+1.0/(RBAR+Y3BAR)))+Y2*(Y3BAR-A)*COTBETA/(RBAR*(RBAR+Z3BAR))*(-2.0*(1.0- NU)*cosBETA+(RBAR*cosBETA+Y3BAR)/(RBAR+Z3BAR)*(1.0+A/(RBAR*cosBETA))+A*Y3BAR/((RBAR*RBAR)*cosBETA));
    V3CB2=-2.0*(1.0-NU)*(1.0-2.0*NU)*COTBETA*(log(RBAR+Y3BAR)-cosBETA*log(RBAR+Z3BAR))-2.0*(1.0-NU)*Y1/(RBAR+Y3BAR)*(2*NU+A/RBAR)+2.0*(1.0-NU)*Z1BAR/(RBAR+Z3BAR)*(cosBETA+A/RBAR)+(Y3BAR-A)/RBAR*((1.0-2.0*NU)*COTBETA- 2.0*NU*Y1/(RBAR+Y3BAR)-A*Y1/(RBAR*RBAR))-(Y3BAR-A)/(RBAR+Z3BAR)*(cosBETA*sinBETA+(RBAR*cosBETA+ Y3BAR)*COTBETA/RBAR*(2.0*(1.0-NU)*cosBETA-(RBAR*cosBETA+Y3BAR)/(RBAR+Z3BAR))+A/RBAR*(sinBETA- Y3BAR*Z1BAR/(RBAR*RBAR)-Z1BAR*(RBAR*cosBETA+Y3BAR)/(RBAR*(RBAR+Z3BAR))));

    V1CB2=V1CB2*factor1;
    V2CB2=V2CB2*factor1;
    V3CB2=V3CB2*factor1;

    V1B2=V1INFB2+V1CB2;
    V2B2=V2INFB2+V2CB2;
    V3B2=V3INFB2+V3CB2;

	// Case III: Burgers vector (0,0,B3)
    V1INFB3=Y2*sinBETA*((R*sinBETA-Y1)/(R*(R-Z3))+(RBAR*sinBETA-Y1)/(RBAR*(RBAR+Z3BAR)));
    V2INFB3=(1.0-2.0*NU)*sinBETA*(log(R-Z3)+log(RBAR+Z3BAR))-(Y2*Y2)*sinBETA*(1.0/(R*(R-Z3))+1.0/(RBAR*(RBAR+Z3BAR)));
    V3INFB3=2.0*(1.0-NU)*(F-FBAR)+Y2*sinBETA*((R*cosBETA-Y3)/(R*(R-Z3))-(RBAR*cosBETA+Y3BAR)/(RBAR*(RBAR+Z3BAR)));

    V1INFB3=V1INFB3*factor2;
    V2INFB3=V2INFB3*factor2;
    V3INFB3=V3INFB3*factor2;

    V1CB3=(1.0-2.0*NU)*(Y2/(RBAR+Y3BAR)*(1+A/RBAR)-Y2*cosBETA/(RBAR+Z3BAR)*(cosBETA+A/RBAR))-Y2*(Y3BAR-A)/RBAR*(A/(RBAR*RBAR)+1.0/(RBAR+Y3BAR))+Y2*(Y3BAR-A)*cosBETA/(RBAR*(RBAR+Z3BAR))*((RBAR*cosBETA+Y3BAR)/(RBAR+Z3BAR)*(cosBETA+A/RBAR)+A*Y3BAR/(RBAR*RBAR));
    V2CB3=(1.0-2.0*NU)*(-sinBETA*log(RBAR+Z3BAR)-Y1/(RBAR+Y3BAR)*(1+A/RBAR)+Z1BAR/(RBAR+Z3BAR)*(cosBETA+A/RBAR))+Y1*(Y3BAR-A)/RBAR*(A/(RBAR*RBAR)+1.0/(RBAR+Y3BAR))-(Y3BAR-A)/(RBAR+Z3BAR)*(sinBETA*(cosBETA-A/RBAR)+ Z1BAR/RBAR*(1.0+A*Y3BAR/(RBAR*RBAR))-1.0/(RBAR*(RBAR+Z3BAR))*((Y2*Y2)*cosBETA*sinBETA- A*Z1BAR/RBAR*(RBAR*cosBETA+Y3BAR)));
    V3CB3=2.0*(1.0-NU)*FBAR+2.0*(1.0-NU)*(Y2*sinBETA/(RBAR+Z3BAR)*(cosBETA+A/RBAR))+Y2*(Y3BAR-A)*sinBETA/(RBAR*(RBAR+ Z3BAR))*(1.0+(RBAR*cosBETA+Y3BAR)/(RBAR+Z3BAR)*(cosBETA+A/RBAR)+A*Y3BAR/(RBAR*RBAR));

    V1CB3=V1CB3*factor1;
    V2CB3=V2CB3*factor1;
    V3CB3=V3CB3*factor1;

    V1B3=V1INFB3+V1CB3;
    V2B3=V2INFB3+V2CB3;
    V3B3=V3INFB3+V3CB3;

	// Sum for each component
    V1=B1*V1B1+B2*V1B2+B3*V1B3;
    V2=B1*V2B1+B2*V2B2+B3*V2B3;
    V3=B1*V3B1+B2*V3B2+B3*V3B3;
}

__device__ void PRTOXY(double ALATDG, double ALNGDG, double ALATO, double ALNGO, double& X, double& Y)
{
    double A,E2,E12,D,RD,RLAT,SLAT,V2,CLAT,AL,PH1,RPH1;
    double SRPH1,RPH2,SRPH2,R,AN,C1,C2;

    A = 6378.160;
    E2 = 6.6946053e-3;
    E12 = 6.7397251e-3;
    D = 57.29578;
    RD = 1.0/57.29578;

    RLAT = ALATDG*RD;
    SLAT = sin(RLAT);
    CLAT = cos(RLAT);
    V2 = 1.0 + E12*CLAT*CLAT;
    AL = ALNGDG-ALNGO;
    PH1 = ALATDG + 0.5*V2*AL*AL*SLAT*CLAT*RD;
    RPH1 = PH1*RD;
    RPH2 = (PH1+ALATO)*0.5*RD;
    SRPH1 = sin(RPH1);
    SRPH2 = sin(RPH2);

    R = A*(1.0 - E2) / pow(1.0 - E2*SRPH2*SRPH2, 1.5);
    AN = A / sqrt(1.0 - E2*SRPH1*SRPH1);
    C1 = D / R;
    C2 = D / AN;
    Y = (PH1-ALATO)/C1;
    X = AL*CLAT/C2*(1.0 + AL*AL*cos(2.0*RLAT)/(6.0*D*D));
}

// d_deformacionNivel0: deformación final
// num_volx, num_voly: tamaño de la malla
// d_vc: (x,y) de cada vértice del triángulo de Okada (4 componentes)
// Z: profundidad de cada vértice del triángulo de Okada (4 componentes)
// LATCTRI y LONCTRI: baricentro del triángulo de Okada
// d_SLIPVEC: vector para calcular la deformación
__global__  void aplicarOkadaTriangularGPU(double *d_deformacionNivel0, int fallaOkada, int num_volx, int num_voly,
						double lon_ini, double incx, double lat_ini, double incy, double2 *d_vc, double *d_vz,
						double LATCTRI, double LONCTRI, double *d_SLIPVEC, double H)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double LON_P, LAT_P;
	double STRIKE,RX,RY,DIP,BETA;
	double SSVEC[3];
	double TSVEC[3];
	double DSVEC[3];
	double LSS,LTS,LDS;
	double SX,SY,SZ;
	double UX,UY,UZ;
	double SX1,SY1,SX2,SY2,UX1,UY1,UZ1,UX2,UY2,UZ2,UXN,UYN,UZN;
	double grad2rad = M_PI/180.0;
	double rad2grad = 180.0/M_PI;
	int triangle = 0;
	int i;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
        pos = pos_y_hebra*num_volx + pos_x_hebra;

		// Obtenemos longitud y latitud del punto asociado a la hebra
		LON_P = lon_ini + pos_x_hebra*incx;
		LAT_P = lat_ini + pos_y_hebra*incy;

		PRTOXY(LAT_P, LON_P, LATCTRI, LONCTRI, SX, SY);
		SZ = 0.0;
		UZN = UXN = UYN = 0.0;
		UX = UY = UZ = 0.0;

        for (i=0; i<3; i++) {
			STRIKE = rad2grad*(atan2(d_vc[i+1].y-d_vc[i].y,d_vc[i+1].x-d_vc[i].x));
			ROTATE(d_vc[i+1].x-d_vc[i].x, d_vc[i+1].y-d_vc[i].y, -STRIKE, RX, RY);
			DIP = rad2grad*(atan2(d_vz[i+1] - d_vz[i], RX));

			if (DIP >= 0.0){
				BETA = grad2rad*(90.0-DIP);
				if (BETA > 0.5*M_PI) {
					BETA = 0.5*M_PI - BETA;
				}
			}
			else {
				BETA = -grad2rad*(90.0+DIP);
				if (BETA < -0.5*M_PI) {
					BETA = 0.5*M_PI - fabs(BETA);
				}
			}
			SSVEC[0] = cos(STRIKE*grad2rad);
			SSVEC[1] = sin(STRIKE*grad2rad);
			SSVEC[2] = 0.0;

			TSVEC[0] = -sin(STRIKE*grad2rad);
			TSVEC[1] = cos(STRIKE*grad2rad);
			TSVEC[2] = 0.0;

			d_CROSS(SSVEC,TSVEC,DSVEC);
			d_DOT_PRODUCT(d_SLIPVEC,SSVEC,LSS);
			d_DOT_PRODUCT(d_SLIPVEC,TSVEC,LTS);
			d_DOT_PRODUCT(d_SLIPVEC,DSVEC,LDS);

			if ((fabs(BETA) > 1e-6) && (fabs(BETA-M_PI) > 1e-6)) {
				// First angular dislocation
				ROTATE(SX-d_vc[i].x,SY-d_vc[i].y,-STRIKE,SX1,SY1);
				ADV(SX1,SY1,SZ-d_vz[i],d_vz[i],BETA,0.25,LSS,LTS,LDS,UX1,UY1,UZ1);

				// Second angular dislocation
				ROTATE(SX-d_vc[i+1].x,SY-d_vc[i+1].y,-STRIKE,SX2,SY2);
				ADV(SX2,SY2,SZ-d_vz[i+1],d_vz[i+1],BETA,0.25,LSS,LTS,LDS,UX2,UY2,UZ2);

				//  Rotate vectors to correct for strike
				ROTATE(UX1-UX2,UY1-UY2,STRIKE,UXN,UYN);
				UZN = UZ1-UZ2;

				// Add the displacements from current leg
				UX = UX+UXN;
				UY = UY+UYN;
				UZ = UZ+UZN;
			}
		}

		in_triangle(SX,SY,d_vc,triangle);
		if (triangle == 1) {
			calculo_UZ(SX,SY,d_vc,d_vz,d_SLIPVEC,UZ);
		}

		// Comprobación del valor UZ obtenido
		if (UZ != UZ) {
			UZ = 0.0;
			d_avisoNaNEnDef = make_int2(1, fallaOkada);
		}
		else if ((UZ < -500.0) || (UZ > 500.0)) {
			d_avisoValorGrandeEnDef = make_int2(1, fallaOkada);
		}
		else if (fabs(UZ) <= 1e-6) {
			UZ = 0.0;
		}
		d_deformacionNivel0[pos] = UZ/H;
	}
}

/*****************/
/* Aplicar Okada */
/*****************/

// Aplica Okada (estándar, triangular o leer de fichero). Si kajiura_flag=1, aplica el filtro de Kajiura
// al resultado obtenido con Okada
void aplicarOkada(int numNiveles, int okada_flag, int fallaOkada, int kajiura_flag, double depth_kajiura,
		TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL], tipoDatosSubmalla datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		tipoDatosSubmalla d_datosNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int *numSubmallasNivel, int4 submallaDeformacion, double2 **d_datosVolumenesNivel_1, double **d_eta1Inicial,
		double **d_deformacionNivel0, double **d_deformacionAcumuladaNivel, cuDoubleComplex *d_datosKajiura, double *d_F2Sx,
		double *d_F2Sy, int submallaNivelSuperior[MAX_LEVELS][MAX_GRIDS_LEVEL], int *ratioRefNivel, double LON_C, double LAT_C,
		double DEPTH_C, double FAULT_L, double FAULT_W, double STRIKE, double DIP, double RAKE, double SLIP, double2 *d_vc, double *d_vz,
		double LONCTRI, double LATCTRI, double *d_SLIPVEC, dim3 blockGridOkada, dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL],
		dim3 blockGridFanNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, double H, cudaStream_t *streams, int nstreams)
{
	int i, j, k, l;
	int nvx, nvy;
	int nvxDef, nvyDef;
	int nvxNivel0, nvyNivel0;
	int inix, iniy;
	int submallaSup;
	int pos, ratio;
	tipoDatosSubmalla *tds;
	double *d_def;
	double lon_ini, lat_ini;
	double incx, incy;

	nvxNivel0 = submallasNivel[0][0].z;
	nvyNivel0 = submallasNivel[0][0].w;
	nvxDef = submallaDeformacion.z;
	nvyDef = submallaDeformacion.w;
	// Variables de la ventana de computación
	tds = &(datosNivel[0][0]);
	lon_ini = tds->longitud[submallaDeformacion.x];
	lat_ini = tds->latitud[submallaDeformacion.y];
	incx = (tds->longitud[nvxNivel0-1] - tds->longitud[0])/(nvxNivel0-1);
	incy = (tds->latitud[nvyNivel0-1] - tds->latitud[0])/(nvyNivel0-1);

	if ((okada_flag == OKADA_STANDARD) || (okada_flag == OKADA_STANDARD_FROM_FILE)) {
		// Aplicamos Okada estándar en el nivel 0
		d_def = d_deformacionNivel0[0];
		aplicarOkadaStandardGPU<<<blockGridOkada, threadBlockEst>>>(d_def, fallaOkada, nvxDef, nvyDef, lon_ini,
			incx, lat_ini, incy, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, H);
	}
	else if (okada_flag == DEFORMATION_FROM_FILE) {
		// La deformación ya está en d_deformacionNivel0[fallaOkada]
		d_def = d_deformacionNivel0[fallaOkada];
	}
	else if ((okada_flag == OKADA_TRIANGULAR) || (okada_flag == OKADA_TRIANGULAR_FROM_FILE)) {
		// Aplicamos Okada triangular en el nivel 0
		d_def = d_deformacionNivel0[0];
		aplicarOkadaTriangularGPU<<<blockGridOkada, threadBlockEst>>>(d_def, fallaOkada, nvxDef, nvyDef, lon_ini,
			incx, lat_ini, incy, d_vc, d_vz, LATCTRI, LONCTRI, d_SLIPVEC, H);
	}
	if (kajiura_flag == 1) {
		aplicarKajiura(d_def, d_datosKajiura, d_F2Sx, d_F2Sy, nvxDef, nvyDef, depth_kajiura, blockGridOkada, threadBlockEst);
	}

	sumarDeformacionADatosGPU<<<blockGridEstNivel[0][0], threadBlockEst>>>(numNiveles, d_datosVolumenesNivel_1[0],
		d_eta1Inicial[0], d_def, d_deformacionAcumuladaNivel[0], datosClusterCPU[0][0].numVolx, datosClusterCPU[0][0].numVoly,
		submallaDeformacion.x, submallaDeformacion.y, nvxDef, nvyDef, datosClusterCPU[0][0].inix, datosClusterCPU[0][0].iniy,
		d_datosNivel[0][0].vccos);
	// Interpolamos la deformación en el resto de niveles
	for (l=1; l<numNiveles; l++) {
		pos = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			j = datosClusterCPU[l][i].iniy;
			if (j != -1) {
				// Ponemos en ratio el ratio acumulado, y en inix e iniy las coordenadas de inicio de
				// la submalla con respecto a la malla del nivel 0 y en la resolución de la malla fina
				inix = submallasNivel[l][i].x;
				iniy = submallasNivel[l][i].y;
				ratio = ratioRefNivel[l];
				submallaSup = submallaNivelSuperior[l][i];
				for (k=l-1; k>=0; k--) {
					inix += ratio*submallasNivel[k][submallaSup].x;
					iniy += ratio*submallasNivel[k][submallaSup].y;
					ratio *= ratioRefNivel[k];
					submallaSup = submallaNivelSuperior[k][submallaSup];
				}
				// Interpolamos
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				interpolarDeformacionGPU<<<blockGridFanNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(
					l, numNiveles, d_datosVolumenesNivel_1[l]+pos, d_eta1Inicial[l]+pos, d_def,
					d_deformacionAcumuladaNivel[l]+pos, inix, iniy, nvx, nvy, submallaDeformacion.x,
					submallaDeformacion.y, nvxDef, nvyDef, datosClusterCPU[l][i].inix, j, ratio,
					d_datosNivel[l][i].vccos);
				pos += (nvx + 4)*(nvy + 4);
			}
		}
	}
}


// Función que se llama tras aplicarOkada en el bucle principal. Para cada submalla,
// actualiza los valores medios de eta para cada celda gruesa.
void obtenerMediasEta(int numNiveles, TDatosClusterCPU datosClusterCPU[MAX_LEVELS][MAX_GRIDS_LEVEL],
		int4 submallasNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], int *numSubmallasNivel, double2 **d_datosVolumenesNivel_1,
		double **d_mediaEtaNivel, int *ratioRefNivel, double *factorCorreccionNivel,
		dim3 blockGridEstNivel[MAX_LEVELS][MAX_GRIDS_LEVEL], dim3 threadBlockEst, cudaStream_t *streams, int nstreams)
{
	int i, j, l, m;
	int nvx, nvy;
	int pos;

	for (l=1; l<numNiveles; l++) {
		pos = m = 0;
		for (i=0; i<numSubmallasNivel[l]; i++) {
			j = datosClusterCPU[l][i].iniy;
			if (j != -1) {
				nvx = datosClusterCPU[l][i].numVolx;
				nvy = datosClusterCPU[l][i].numVoly;
				// Actualizamos los valores medios de eta para cada celda gruesa
				obtenerMediaEtaSubmallaNivel1GPU<<<blockGridEstNivel[l][i], threadBlockEst, 0, streams[i&nstreams]>>>(d_mediaEtaNivel[l]+m,
					d_datosVolumenesNivel_1[l]+pos, nvx, nvy, ratioRefNivel[l], factorCorreccionNivel[l]);
				pos += (nvx + 4)*(nvy + 4);
				m += (nvx*nvy)/(ratioRefNivel[l]*ratioRefNivel[l]);
			}
		}
	}
}

#endif
