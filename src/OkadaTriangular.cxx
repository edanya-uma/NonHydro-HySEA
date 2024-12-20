#ifndef _OKADA_TRIANGULAR_H_
#define _OKADA_TRIANGULAR_H_

#include "Constantes.hxx"

void PRTOXYC(double ALATDG, double ALNGDG, double ALATO, double ALNGO, double& X, double& Y)
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

	R = A*(1.0 - E2) / pow(1.0 - E2*SRPH2*SRPH2,1.5);
	AN = A / sqrt(1.0 - E2*SRPH1*SRPH1);
	C1 = D / R;
	C2 = D / AN;
	Y = (PH1-ALATO)/C1;
	X = AL*CLAT/C2*(1.0 + AL*AL*cos(2.0*RLAT)/(6.0*D*D));
}

void CROSS(double* X_VEC,double *Y_VEC,double *CROSS_VEC)
{
	CROSS_VEC[0] = X_VEC[1] * Y_VEC[2] - X_VEC[2] * Y_VEC[1];
	CROSS_VEC[1] = X_VEC[2] * Y_VEC[0] - X_VEC[0] * Y_VEC[2];
	CROSS_VEC[2] = X_VEC[0] * Y_VEC[1] - X_VEC[1] * Y_VEC[0];
}

void NORM2(double *VEC, double& NRM)
{
	NRM = sqrt(VEC[0]*VEC[0]+ VEC[1]*VEC[1] + VEC[2]*VEC[2]);
}

// Obtiene los datos vc, vz (vectores de 4 componentes), LATCTRI, LONCTRI (baricentro del triángulo de Okada)
// y SLIPVEC (vector de 3 componentes). Datos de entrada: v (3 componentes), rake, slip y vz.
void obtenerDatosOkadaTriangular(double2 *v, double rake, double slip, double2 *vc, double *vz,
		double *LATCTRI, double *LONCTRI, double *SLIPVEC)
{
	double VEC1[3];
	double VEC2[3];
	double NORMVEC[3];
	double STRIKEVEC[3];
	double DIPVEC[3];
	double SLIPCOMP[3];
	double NORM;
	double RAD = 0.017453292519943;
	double ZCTRI = (vz[0] + vz[1] + vz[2])/3.0;
	double S_RAKE = sin(RAD*rake);
	double C_RAKE = cos(RAD*rake);
	int i;

	*LONCTRI = (v[0].x + v[1].x + v[2].x)/3.0;
	*LATCTRI = (v[0].y + v[1].y + v[2].y)/3.0;
	for (i=0; i<3; i++) {
		PRTOXYC(v[i].y, v[i].x, *LATCTRI, *LONCTRI, vc[i].x, vc[i].y);
	}

	VEC1[0] = vc[1].x - vc[0].x;
	VEC1[1] = vc[1].y - vc[0].y;
	VEC1[2] = vz[1] - vz[0];
	VEC2[0] = vc[2].x - vc[0].x;
	VEC2[1] = vc[2].y - vc[0].y;
	VEC2[2] = vz[2] - vz[0];

	CROSS(VEC1, VEC2, NORMVEC);
	NORM2(NORMVEC,NORM);
	for (i=0; i<3; i++) {
		NORMVEC[i] /= NORM;
	}
	if (NORMVEC[2] < 0.0) {
		for (i=0; i<3; i++) {
			NORMVEC[i] = -NORMVEC[i];
		}
		double2 aux2 = vc[1];
		vc[1] = vc[2];
		vc[2] = aux2;
		double aux = vz[1];
		vz[1] = vz[2];
		vz[2] = aux;
	}
    
	STRIKEVEC[0] = -sin(atan2(NORMVEC[1],NORMVEC[0]));
	STRIKEVEC[1] = cos(atan2(NORMVEC[1],NORMVEC[0]));
	STRIKEVEC[2] = 0.0;

	CROSS(NORMVEC, STRIKEVEC, DIPVEC);

	SLIPCOMP[0] = slip*C_RAKE;
	SLIPCOMP[1] = slip*S_RAKE;
	SLIPCOMP[2] = 0.0;
	for (i=0; i<3; i++) {
		SLIPVEC[i] = STRIKEVEC[i]*SLIPCOMP[0] + DIPVEC[i]*SLIPCOMP[1] + NORMVEC[i]*SLIPCOMP[2];
	}
	vc[3] = vc[0];
	vz[3] = vz[0];
}

#endif
