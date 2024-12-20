#ifndef _RECONSTRUCCION_KERNEL_H_
#define _RECONSTRUCCION_KERNEL_H_

#include <stdio.h>
#include "Constantes.hxx"

__device__ double minmod(double a, double b)
{
    double c = 0.5*(a+b);
    double d;

    if ((a > 0.0) && (b > 0.0) && (c > 0.0)) {
        d = min(a,min(b,c));
    }
    else {
        if ((a < 0.0) && (b < 0.0) && (c < 0.0)) {
            d = max(a,max(b,c));
        }
        else {
            d = 0.0;
        }
    }
    return d;
}

__device__ double avg(double a, double b)
{
    double d, aa, bb;
    
    aa = fabs(a);
    bb = fabs(b);
    
    if ((aa+bb) > EPSILON) {
        d = (a*bb+b*aa)/(aa+bb);
    }
    else {
        d = 0.0;
    }
    return d;
}

__device__ void reconstruye_h(double hl, double hc, double hr, double& Recl, double& Recr, double cosl, double cosr, double& a0)
{
    double dl = hc-hl;
    double dr = hr-hc;
    double dh, aux;

#if (OrdenTiempo == 1)
    dh = minmod(dl,dr);
#else
    dh = avg(dl,dr);
#endif
    aux = 0.5*fabs(dh);
    if (hc < 0.5*fabs(dh)) {
        a0 = hc/(aux+EPSILON);
        a0 = a0*(a0<1.0) + 1.0*(a0>=1.0);
    }
    else {
        a0 = 1.0;
    }
    Recl = (hc - 0.5*dh*a0)*cosl;
    Recr = (hc + 0.5*dh*a0)*cosr;
}

__device__ void reconstruye_eta(double hl, double hc, double hr, double Hl, double Hc, double Hr, double& Recl, double& Recr,
				double cosl, double cosr, double a0)
{
    double Hmin;
    double dl, dr;
    double deta, aux;

    Hmin = min(Hc,Hl);
    dl = max(hc-Hc+Hmin,0.0) - max(hl-Hl+Hmin,0.0);
    Hmin = min(Hc,Hr);
    dr = max(hr-Hr+Hmin,0.0)- max(hc-Hc+Hmin,0.0);
#if (OrdenTiempo == 1)
    deta = minmod(dl,dr);
#else
    deta = avg(dl,dr);
#endif
    aux = hc-Hc;
    Recl = (aux - 0.5*deta*a0)*cosl;
    Recr = (aux + 0.5*deta*a0)*cosr;
}

__device__ void reconstruye_Q(double3 Ql, double3 Qc, double3 Qr, double* Recl, double* Recr, double a0)
{
    double dl, dr;
    double dq;

    dl = Qc.x-Ql.x;
    dr = Qr.x-Qc.x;
#if (OrdenTiempo == 1)
    dq = minmod(dl,dr);
#else
    dq = avg(dl,dr);
#endif
    Recl[0] = Qc.x-0.5*dq*a0;
    Recr[0] = Qc.x+0.5*dq*a0;

    dl = Qc.y-Ql.y;
    dr = Qr.y-Qc.y;
#if (OrdenTiempo == 1)
    dq = minmod(dl,dr);
#else
    dq = avg(dl,dr);
#endif
    Recl[1] = Qc.y-0.5*dq*a0;
    Recr[1] = Qc.y+0.5*dq*a0;

    dl = Qc.z-Ql.z;
    dr = Qr.z-Qc.z;
#if (OrdenTiempo == 1)
    dq = minmod(dl,dr);
#else
    dq = avg(dl,dr);
#endif
    Recl[2] = Qc.z-0.5*dq*a0;
    Recr[2] = Qc.z+0.5*dq*a0;
}

__global__ void calcularCoeficientesReconstruccionNoCom(double2* d_datosVolumenesh, double3* d_datosVolumenesQ,
				double* d_vcos, double* d_vctan, double* d_vccos, double* d_aristaReconstruido, double3* d_acumuladorQ,
				int npx, int npy, int npyTotal, double dx, double dy, double radio_tierra, double vmax, double delta_T,
				int iniySubmallaCluster, bool ultima_hebraY)
{
    int i0, j0, im, ip, jm, jp;
    int posc, posl, posr;
    int idr;
    int pos_x_hebra, pos_y_hebra;
    int j0_global, jm_global, jp_global;
    double2 hc, hl, hr;
    double3 qc, ql, qr;
    double etac;
    double divcos, dimvcos, dipvcos;
    double dvcos, dlcos, drcos, dvtan;
    double a0;
    double detax, detay;
    double dtaux = delta_T/radio_tierra;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

    if ((pos_x_hebra > 0) && (pos_x_hebra < npx-1) && (pos_y_hebra > 0) && (pos_y_hebra < npy-1)) {
        j0_global = iniySubmallaCluster+pos_y_hebra;
        jm_global = ( ((j0_global == 0) && (iniySubmallaCluster == 0)) ? 0 : j0_global-1 );
        jp_global = ( ((j0_global == npyTotal-1) && ultima_hebraY) ? npyTotal-1 : j0_global+1 );

        dvcos = d_vccos[j0_global];
        divcos = 1.0/(dvcos+EPSILON);
        dvtan = d_vctan[j0_global];

        i0 = pos_x_hebra;
        j0 = pos_y_hebra;
		im = i0-1;
		ip = i0+1;
		jm = j0-1;
		jp = j0+1;

        dimvcos = 1.0/(d_vccos[jm_global]+EPSILON);
        dipvcos = 1.0/(d_vccos[jp_global]+EPSILON);
        dlcos = d_vcos[2*j0_global];
        drcos = d_vcos[2*j0_global+2];
		posc = (j0+2)*(npx+4) + i0+2;
        posl = (j0+2)*(npx+4) + im+2;
        posr = (j0+2)*(npx+4) + ip+2;
        idr = posc*4*5;

        // Reconstrucción en la dirección x
        hc = d_datosVolumenesh[posc];
        hl = d_datosVolumenesh[posl];
        hr = d_datosVolumenesh[posr];

        qc = d_datosVolumenesQ[posc];
        ql = d_datosVolumenesQ[posl];
        qr = d_datosVolumenesQ[posr];
        hc.x *= divcos;
        hl.x *= divcos;
        hr.x *= divcos;
        hc.y *= divcos;
        hl.y *= divcos;
        hr.y *= divcos;

        reconstruye_h(hl.x, hc.x, hr.x, d_aristaReconstruido[idr+3*5], d_aristaReconstruido[idr+1*5], dvcos, dvcos, a0);
        reconstruye_eta(hl.x, hc.x, hr.x, hl.y, hc.y, hr.y, d_aristaReconstruido[idr+3*5+1], d_aristaReconstruido[idr+1*5+1], dvcos, dvcos, a0);
        reconstruye_Q(ql, qc, qr, &(d_aristaReconstruido[idr+3*5+2]), &(d_aristaReconstruido[idr+1*5+2]), a0);
        detax = d_aristaReconstruido[idr+1*5+1] - d_aristaReconstruido[idr+3*5+1];

        // Reconstrucción en la dirección y
        posl = (jm+2)*(npx+4) + i0+2;
        posr = (jp+2)*(npx+4) + i0+2;

        hl = d_datosVolumenesh[posl];
        hr = d_datosVolumenesh[posr];

        ql = d_datosVolumenesQ[posl];
        qr = d_datosVolumenesQ[posr];
        hl.x /= dimvcos;
        hr.x /= dipvcos;
        hl.y /= dimvcos;
        hr.y /= dipvcos;
        
        reconstruye_h(hl.x, hc.x, hr.x, d_aristaReconstruido[idr+0*5], d_aristaReconstruido[idr+2*5], dlcos, drcos, a0);
        reconstruye_eta(hl.x, hc.x, hr.x,hl.y, hc.y, hr.y, d_aristaReconstruido[idr+0*5+1], d_aristaReconstruido[idr+2*5+1], dlcos, drcos, a0);
        reconstruye_Q(ql, qc, qr, &(d_aristaReconstruido[idr+0*5+2]), &(d_aristaReconstruido[idr+2*5+2]), a0);
        etac = hc.x-hc.y;
        detay = d_aristaReconstruido[idr+2*5+1] - d_aristaReconstruido[idr+0*5+1] - etac*(drcos-dlcos);

        d_acumuladorQ[posc].x -= dtaux*hc.x*divcos*detax/dx;
        d_acumuladorQ[posc].x += dtaux*qc.x*qc.y/(hc.x*dvcos+EPSILON)*dvtan;

        d_acumuladorQ[posc].y -= dtaux*hc.x*detay/dy;
        d_acumuladorQ[posc].y -= dtaux*qc.x*qc.x/(hc.x*dvcos+EPSILON)*dvtan;
    }
}

__global__ void calcularCoeficientesReconstruccionCom(double2* d_datosVolumenesh, double3* d_datosVolumenesQ,
				double* d_vcos, double* d_vctan, double* d_vccos, double* d_aristaReconstruido, double3* d_acumuladorQ,
				int npx, int npy, int npyTotal, double dx, double dy, double radio_tierra, double vmax, double delta_T,
				int inixSubmallaCluster, int iniySubmallaCluster, bool ultima_hebraX, bool ultima_hebraY)
{
    int i0, j0, im, ip, jm, jp;
    int posc, posl, posr;
    int idr;
    int pos_x_hebra, pos_y_hebra;
    int j0_global, jm_global, jp_global;
    double2 hc, hl, hr;
    double3 qc, ql, qr;
    double etac;
    double divcos, dimvcos, dipvcos;
    double dvcos, dlcos, drcos, dvtan;
    double a0;
    double detax, detay;
    double dtaux = delta_T/radio_tierra;
	// Si es true, son volúmenes que se salen del dominio global (no se procesan)
	bool fuera_izq, fuera_der;
	bool fuera_sup, fuera_inf;

	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	pos_y_hebra = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

    if ((pos_x_hebra < npx+2) && (pos_y_hebra < npy+2)) {
		if ((pos_x_hebra<2) || (pos_x_hebra>=npx) || (pos_y_hebra<2) || (pos_y_hebra>=npy)) {
			fuera_izq = ( ((pos_x_hebra == 0) && (inixSubmallaCluster == 0)) ? true : false);
			fuera_der = ( ((pos_x_hebra == npx+1) && ultima_hebraX) ? true : false);
			fuera_sup = ( ((pos_y_hebra == 0) && (iniySubmallaCluster == 0)) ? true : false);
			fuera_inf = ( ((pos_y_hebra == npy+1) && ultima_hebraY) ? true : false);

			if ((! fuera_izq) && (! fuera_der) && (! fuera_sup) && (! fuera_inf)) {
		        j0_global = iniySubmallaCluster+pos_y_hebra-1;
        		jm_global = ( ((j0_global == 0) && (iniySubmallaCluster == 0)) ? 0 : j0_global-1 );
		        jp_global = ( ((j0_global == npyTotal-1) && ultima_hebraY) ? npyTotal-1 : j0_global+1 );

		        dvcos = d_vccos[j0_global];
		        divcos = 1.0/(dvcos+EPSILON);
		        dvtan = d_vctan[j0_global];

		        i0 = pos_x_hebra;
		        j0 = pos_y_hebra;
				im = ( ((i0 == 1) && (inixSubmallaCluster == 0)) ? 1 : i0-1 );
				ip = ( ((i0 == npx) && ultima_hebraX) ? npx : i0+1 );
				jm = ( ((j0 == 1) && (iniySubmallaCluster == 0)) ? 1 : j0-1 );
				jp = ( ((j0 == npy) && ultima_hebraY) ? npy : j0+1 );

		        dimvcos = 1.0/(d_vccos[jm_global]+EPSILON);
        		dipvcos = 1.0/(d_vccos[jp_global]+EPSILON);
		        dlcos = d_vcos[2*j0_global];
		        drcos = d_vcos[2*j0_global+2];
				posc = (j0+1)*(npx+4) + i0+1;
		        posl = (j0+1)*(npx+4) + im+1;
		        posr = (j0+1)*(npx+4) + ip+1;
		        idr = posc*4*5;

		        // Reconstrucción en la dirección x
		        hc = d_datosVolumenesh[posc];
		        hl = d_datosVolumenesh[posl];
		        hr = d_datosVolumenesh[posr];

		        qc = d_datosVolumenesQ[posc];
		        ql = d_datosVolumenesQ[posl];
		        qr = d_datosVolumenesQ[posr];
		        hc.x *= divcos;
		        hl.x *= divcos;
		        hr.x *= divcos;
		        hc.y *= divcos;
		        hl.y *= divcos;
		        hr.y *= divcos;

		        reconstruye_h(hl.x, hc.x, hr.x, d_aristaReconstruido[idr+3*5], d_aristaReconstruido[idr+1*5], dvcos, dvcos, a0);
		        reconstruye_eta(hl.x, hc.x, hr.x, hl.y, hc.y, hr.y, d_aristaReconstruido[idr+3*5+1], d_aristaReconstruido[idr+1*5+1], dvcos, dvcos, a0);
		        reconstruye_Q(ql, qc, qr, &(d_aristaReconstruido[idr+3*5+2]), &(d_aristaReconstruido[idr+1*5+2]), a0);
		        detax = d_aristaReconstruido[idr+1*5+1] - d_aristaReconstruido[idr+3*5+1];

		        // Reconstrucción en la dirección y
		        posl = (jm+1)*(npx+4) + i0+1;
		        posr = (jp+1)*(npx+4) + i0+1;

		        hl = d_datosVolumenesh[posl];
		        hr = d_datosVolumenesh[posr];

		        ql = d_datosVolumenesQ[posl];
		        qr = d_datosVolumenesQ[posr];
		        hl.x /= dimvcos;
		        hr.x /= dipvcos;
		        hl.y /= dimvcos;
		        hr.y /= dipvcos;

		        reconstruye_h(hl.x, hc.x, hr.x, d_aristaReconstruido[idr+0*5], d_aristaReconstruido[idr+2*5], dlcos, drcos, a0);
		        reconstruye_eta(hl.x, hc.x, hr.x,hl.y, hc.y, hr.y, d_aristaReconstruido[idr+0*5+1], d_aristaReconstruido[idr+2*5+1], dlcos, drcos, a0);
		        reconstruye_Q(ql, qc, qr, &(d_aristaReconstruido[idr+0*5+2]), &(d_aristaReconstruido[idr+2*5+2]), a0);
		        etac = hc.x-hc.y;
		        detay = d_aristaReconstruido[idr+2*5+1] - d_aristaReconstruido[idr+0*5+1] - etac*(drcos-dlcos);

		        d_acumuladorQ[posc].x -= dtaux*hc.x*divcos*detax/dx;
		        d_acumuladorQ[posc].x += dtaux*qc.x*qc.y/(hc.x*dvcos+EPSILON)*dvtan;

		        d_acumuladorQ[posc].y -= dtaux*hc.x*detay/dy;
		        d_acumuladorQ[posc].y -= dtaux*qc.x*qc.x/(hc.x*dvcos+EPSILON)*dvtan;
		    }
		}
    }
}

#endif
