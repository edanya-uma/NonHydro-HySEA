#ifndef _NOHIDRO_KERNEL_H_
#define _NOHIDRO_KERNEL_H_

#include "Constantes.hxx"
#include "Reconstruccion.cu"
#include "Matriz.cu"

void cargarIndices(int N, double *omega, int *ord)
{
	// Pesos para la resolucion de los sistemas lineales (ScheduledRelaxationJacobi).
	if (N == 1) {
        // Jacobi Standard
		omega[0] = 1.0;

		ord[0] = 0;
	}
	else if (N == 16) {
		omega[0] = 32.60;
		omega[1] = 0.8630;

		ord[0] = 0;
		ord[1] = 1;
		ord[2] = 1;
		ord[3] = 1;
		ord[4] = 1;
		ord[5] = 1;
		ord[6] = 1;
		ord[7] = 1;
		ord[8] = 1;
		ord[9] = 1;
		ord[10] = 1;
		ord[11] = 1;
		ord[12] = 1;
		ord[13] = 1;
		ord[14] = 1;
		ord[15] = 1;
	}
	else if (N == 27) {
	    omega[0] = 64.66;
	    omega[1] = 6.215;
	    omega[2] = 0.7042;

		ord[0] = 0;
		ord[1] = 2;
		ord[2] = 1;
	    ord[3] = 2;
	    ord[4] = 2;
	    ord[5] = 1;
	    ord[6] = 2;
	    ord[7] = 2;
	    ord[8] = 2;
	    ord[9] = 1;
	    ord[10] = 2;
	    ord[11] = 2;
	    ord[12] = 2;
	    ord[13] = 2;
	    ord[14] = 1;
	    ord[15] = 2;
	    ord[16] = 2;
	    ord[17] = 2;
	    ord[18] = 2;
	    ord[19] = 2;
	    ord[20] = 1;
	    ord[21] = 2;
	    ord[22] = 2;
	    ord[23] = 2;
	    ord[24] = 2;
	    ord[25] = 2;
	    ord[26] = 2;
	}
}

__device__ double gradH(double h0, double h1, double H0, double H1, double cos0, double cos1)
{
    double HH0 = H0/cos0;
    double HH1 = H1/cos1;
    double hh0 = h0/cos0;
    double hh1 = h1/cos1;
    double dh = hh1-hh0;
    double Hm = min(HH0,HH1);
    double deta = max(hh1-HH1+Hm,0.0) - max(hh0-HH0+Hm,0.0);    
    double dH = 0.5*(cos0+cos1)*(dh-deta);
    // double dH = H1-H0;

    return dH;
}

__global__ void compute_NH_coefficientsNoCom(double2 *hH, double3 *Quvw, double *dtHs, double *vccos, double *vtan, int npx, int npy,
				int npyTotal, double coefdt, double deltat, double deltatheta, double deltaphi, double radio_tierra, double L,
				double H, double *RHS_dt, double *CoefPE, double *CoefPW, double *CoefPN, double *CoefPS, int inixSubmallaCluster,
				int iniySubmallaCluster, bool ultima_hebraX, bool ultima_hebraY)
{
    int i = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	int j = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;
	int ii, im, imm, ip;
	int jj, jm, jmm, jp;

    if ((i > 1) && (i < npx-1) && (j > 1) && (j < npy-1)) {
        // Cálculo de índices de volúmenes vecinos con respecto a la referencia en malla staggered
        int posC = (j+1)*(npx+3) + i+1;  // Vértice (i,j)
		ii = i;
		im = i-1;
		jj = j;
		jm = j-1;
        int NE = 2+ii + (2+jj)*(npx+4);
        int NW = 2+im + (2+jj)*(npx+4);
        int SE = 2+ii + (2+jm)*(npx+4);
        int SW = 2+im + (2+jm)*(npx+4);
        ////////////////////////////////////////////
        double hs_NE = hH[NE].x; double Hs_NE = hH[NE].y;
        double hs_NW = hH[NW].x; double Hs_NW = hH[NW].y;
        double hs_SE = hH[SE].x; double Hs_SE = hH[SE].y;
        double hs_SW = hH[SW].x; double Hs_SW = hH[SW].y;
        if ((hs_NE > EPSILON) || (hs_NW > EPSILON) || (hs_SE > EPSILON) || (hs_SW > EPSILON)) {
            int j_global = iniySubmallaCluster+j;
            double dt = coefdt*deltat;
            double div_deltatheta = 1.0/deltatheta;
            double div_deltaphi = 1.0/deltaphi;
            double R = radio_tierra;
            // Cálculo de índices de volúmenes vecinos con respecto a la referencia en malla staggered
            imm = i-2;
            ip  = i+1;
            jmm = j-2;
            jp  = j+1;
            int NNE = 2+ii  + (2+jp)*(npx+4);
            int NNW = 2+im  + (2+jp)*(npx+4);
            int NEE = 2+ip  + (2+jj)*(npx+4);
            int NWW = 2+imm + (2+jj)*(npx+4);
            int SEE = 2+ip  + (2+jm)*(npx+4);
            int SWW = 2+imm + (2+jm)*(npx+4);
            int SSE = 2+ii  + (2+jmm)*(npx+4);
            int SSW = 2+im  + (2+jmm)*(npx+4);
            ////////////////////////////////////////////
            double hs_NNE = hH[NNE].x; double Hs_NNE = hH[NNE].y;
            double hs_NNW = hH[NNW].x; double Hs_NNW = hH[NNW].y;
            double hs_NEE = hH[NEE].x; double Hs_NEE = hH[NEE].y;
            double hs_NWW = hH[NWW].x; double Hs_NWW = hH[NWW].y;
            double hs_SEE = hH[SEE].x; double Hs_SEE = hH[SEE].y;
            double hs_SWW = hH[SWW].x; double Hs_SWW = hH[SWW].y;
            double hs_SSE = hH[SSE].x; double Hs_SSE = hH[SSE].y;
            double hs_SSW = hH[SSW].x; double Hs_SSW = hH[SSW].y;
            ////////////////////////////////////////////
            double cosphi_NNE = vccos[min(j_global+1,npyTotal-1)];  double tanphi_NNE = vtan[min(j_global+1,npyTotal-1)];
            double cosphi_NNW = vccos[min(j_global+1,npyTotal-1)];  double tanphi_NNW = vtan[min(j_global+1,npyTotal-1)];
            double cosphi_NE  = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NE  = vtan[min(j_global  ,npyTotal-1)];
            double cosphi_NEE = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NEE = vtan[min(j_global  ,npyTotal-1)];
            double cosphi_NW  = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NW  = vtan[min(j_global  ,npyTotal-1)];
            double cosphi_NWW = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NWW = vtan[min(j_global  ,npyTotal-1)];
            double cosphi_SE  = vccos[    j_global-1*(j_global>0)]; double tanphi_SE  = vtan[    j_global-1*(j_global>0)];
            double cosphi_SEE = vccos[    j_global-1*(j_global>0)];
            double cosphi_SW  = vccos[    j_global-1*(j_global>0)]; double tanphi_SW  = vtan[    j_global-1*(j_global>0)];
            double cosphi_SWW = vccos[    j_global-1*(j_global>0)];
            double cosphi_SSE = vccos[    j_global-2*(j_global>1)];
            double cosphi_SSW = vccos[    j_global-2*(j_global>1)];
            // double cosphi_NNE = 1.0; double tanphi_NNE = 0.0;
            // double cosphi_NNW = 1.0; double tanphi_NNW = 0.0;
            // double cosphi_NE  = 1.0; double tanphi_NE  = 0.0;
            // double cosphi_NEE = 1.0; double tanphi_NEE = 0.0;
            // double cosphi_NW  = 1.0; double tanphi_NW  = 0.0;
            // double cosphi_NWW = 1.0; double tanphi_NWW = 0.0;
            // double cosphi_SE  = 1.0; double tanphi_SE  = 0.0;
            // double cosphi_SEE = 1.0; double tanphi_SEE = 0.0;
            // double cosphi_SW  = 1.0; double tanphi_SW  = 0.0;
            // double cosphi_SWW = 1.0; double tanphi_SWW = 0.0;
            // double cosphi_SSE = 1.0; double tanphi_SSE = 0.0;
            // double cosphi_SSW = 1.0; double tanphi_SSW = 0.0;
            ////////////////////////////////////////////
            double hs_C = 0.25*(hs_NE+hs_SE+hs_SW+hs_NW);
            double Hs_C = 0.25*(Hs_NE+Hs_SE+Hs_SW+Hs_NW);
            double Lambdas_C = 2.0*Hs_C-hs_C;
            double cosphi_C = 0.25*(cosphi_NE+cosphi_SE+cosphi_SW+cosphi_NW);
            double tanphi_C = 0.25*(tanphi_NE+tanphi_SE+tanphi_SW+tanphi_NW);        
            ////////////////////////////////////////////
            double dtheta_hs_NE = minmod(hs_NEE-hs_NE,hs_NE-hs_NW)*div_deltatheta;
            double dtheta_hs_SE = minmod(hs_SEE-hs_SE,hs_SE-hs_SW)*div_deltatheta;
            double dtheta_hs_SW = minmod(hs_SE-hs_SW,hs_SW-hs_SWW)*div_deltatheta;
            double dtheta_hs_NW = minmod(hs_NE-hs_NW,hs_NW-hs_NWW)*div_deltatheta;

            double dphi_hs_NE = minmod(hs_NNE-hs_NE,hs_NE-hs_SE)*div_deltaphi;
            double dphi_hs_SE = minmod(hs_NE-hs_SE,hs_SE-hs_SSE)*div_deltaphi;
            double dphi_hs_SW = minmod(hs_NW-hs_SW,hs_SW-hs_SSW)*div_deltaphi;
            double dphi_hs_NW = minmod(hs_NNW-hs_NW,hs_NW-hs_SW)*div_deltaphi;
            ////////////////////////////////////////////
            // double dtheta_Hs_NE = minmod(Hs_NEE-Hs_NE,Hs_NE-Hs_NW)*div_deltatheta;
            // double dtheta_Hs_SE = minmod(Hs_SEE-Hs_SE,Hs_SE-Hs_SW)*div_deltatheta;
            // double dtheta_Hs_SW = minmod(Hs_SE-Hs_SW,Hs_SW-Hs_SWW)*div_deltatheta;
            // double dtheta_Hs_NW = minmod(Hs_NE-Hs_NW,Hs_NW-Hs_NWW)*div_deltatheta;

            // double dphi_Hs_NE   = minmod(Hs_NNE-Hs_NE,Hs_NE-Hs_SE)*div_deltaphi;
            // double dphi_Hs_SE   = minmod(Hs_NE-Hs_SE,Hs_SE-Hs_SSE)*div_deltaphi;        
            // double dphi_Hs_SW   = minmod(Hs_NW-Hs_SW,Hs_SW-Hs_SSW)*div_deltaphi;
            // double dphi_Hs_NW   = minmod(Hs_NNW-Hs_NW,Hs_NW-Hs_SW)*div_deltaphi;

            double grad0, grad1;
            grad0 = gradH(hs_NE,hs_NEE,Hs_NE,Hs_NEE,cosphi_NE,cosphi_NEE);
            grad1 = gradH(hs_NW,hs_NE,Hs_NW,Hs_NE,cosphi_NW,cosphi_NE);
            double dtheta_Hs_NE = minmod(grad0,grad1)*div_deltatheta;

            grad0 = gradH(hs_SE,hs_SEE,Hs_SE,Hs_SEE,cosphi_SE,cosphi_SEE);
            grad1 = gradH(hs_SW,hs_SE,Hs_SW,Hs_SE,cosphi_SW,cosphi_SE);
            double dtheta_Hs_SE = minmod(grad0,grad1)*div_deltatheta;

            grad0 = gradH(hs_SW,hs_SE,Hs_SW,Hs_SE,cosphi_SW,cosphi_SE);
            grad1 = gradH(hs_SWW,hs_SW,Hs_SWW,Hs_SW,cosphi_SWW,cosphi_SW);
            double dtheta_Hs_SW = minmod(grad0,grad1)*div_deltatheta;

            grad0 = gradH(hs_NW,hs_NE,Hs_NW,Hs_NE,cosphi_NW,cosphi_NE);
            grad1 = gradH(hs_NWW,hs_NW,Hs_NWW,Hs_NW,cosphi_NWW,cosphi_NW);
            double dtheta_Hs_NW = minmod(grad0,grad1)*div_deltatheta;

            grad0 = gradH(hs_NE,hs_NNE,Hs_NE,Hs_NNE,cosphi_NE,cosphi_NNE);
            grad1 = gradH(hs_SE,hs_NE,Hs_SE,Hs_NE,cosphi_SE,cosphi_NE);
            double dphi_Hs_NE = minmod(grad0,grad1)*div_deltaphi;

            grad0 = gradH(hs_SE,hs_NE,Hs_SE,Hs_NE,cosphi_SE,cosphi_NE);
            grad1 = gradH(hs_SSE,hs_SE,Hs_SSE,Hs_SE,cosphi_SSE,cosphi_SE);
            double dphi_Hs_SE = minmod(grad0,grad1)*div_deltaphi;

            grad0 = gradH(hs_SW,hs_NW,Hs_SW,Hs_NW,cosphi_SW,cosphi_NW);
            grad1 = gradH(hs_SSW,hs_SW,Hs_SSW,Hs_SW,cosphi_SSW,cosphi_SW);
            double dphi_Hs_SW = minmod(grad0,grad1)*div_deltaphi;

            grad0 = gradH(hs_NW,hs_NNW,Hs_NW,Hs_NNW,cosphi_NW,cosphi_NNW);
            grad1 = gradH(hs_SW,hs_NW,Hs_SW,Hs_NW,cosphi_SW,cosphi_NW);
            double dphi_Hs_NW = minmod(grad0,grad1)*div_deltaphi;
            ////////////////////////////////////////////
            double dtheta_hs_C = 0.25*(dtheta_hs_NE+dtheta_hs_SE+dtheta_hs_SW+dtheta_hs_NW);        
            double dtheta_Hs_C = 0.25*(dtheta_Hs_NE+dtheta_Hs_SE+dtheta_Hs_SW+dtheta_Hs_NW);
            double dthetaLambdas_C = 2.0*dtheta_Hs_C - dtheta_hs_C;

            double dphi_hs_C = 0.25*(dphi_hs_NE+dphi_hs_SE+dphi_hs_SW+dphi_hs_NW);
            double dphi_Hs_C = 0.25*(dphi_Hs_NE+dphi_Hs_SE+dphi_Hs_SW+dphi_Hs_NW);
            double dphiLambdas_C = 2.0*dphi_Hs_C - dphi_hs_C;        
            ////////////////////////////////////////////
            double C = hs_C/(cosphi_C*R);
            double D = hs_C/R;
            double E = dthetaLambdas_C/(cosphi_C*R);
            double F = (dphiLambdas_C+tanphi_C*Lambdas_C)/R;
            ////////////////////////////////////////////
            double AE = C*div_deltatheta+0.5*E;
            double AW =-C*div_deltatheta+0.5*E;
            double AN = D*div_deltaphi+0.5*F;
            double AS =-D*div_deltaphi+0.5*F;
            ////////////////////////////////////////////
            double bthetaE = 0.5*(dtheta_hs_NE-GAMMA*dtheta_Hs_NE)/(cosphi_NE*cosphi_NE*R) + 0.5*(dtheta_hs_SE-GAMMA*dtheta_Hs_SE)/(cosphi_SE*cosphi_SE*R);
            double bthetaW = 0.5*(dtheta_hs_NW-GAMMA*dtheta_Hs_NW)/(cosphi_NW*cosphi_NW*R) + 0.5*(dtheta_hs_SW-GAMMA*dtheta_Hs_SW)/(cosphi_SW*cosphi_SW*R);
            double athetaE = 0.5*hs_NE/(cosphi_NE*cosphi_NE*R) + 0.5*hs_SE/(cosphi_SE*cosphi_SE*R);
            double athetaW = 0.5*hs_NW/(cosphi_NW*cosphi_NW*R) + 0.5*hs_SW/(cosphi_SW*cosphi_SW*R);
            double bphiN = 0.5*(dphi_hs_NE-GAMMA*dphi_Hs_NE)/(R*cosphi_NE) + 0.5*(dphi_hs_NW-GAMMA*dphi_Hs_NW)/(R*cosphi_NW);
            double bphiS = 0.5*(dphi_hs_SE-GAMMA*dphi_Hs_SE)/(R*cosphi_SE) + 0.5*(dphi_hs_SW-GAMMA*dphi_Hs_SW)/(R*cosphi_SW);
            double aphiN = 0.5*hs_NE/(cosphi_NE*R) + 0.5*hs_NW/(cosphi_NW*R);
            double aphiS = 0.5*hs_SE/(cosphi_SE*R) + 0.5*hs_SW/(cosphi_SW*R);
            ////////////////////////////////////////////
            double div_adim = L*L/(H*H);
            double CoefPC = AE*(0.5*bthetaE-athetaE*div_deltatheta);
            CoefPC += AW*(0.5*bthetaW+athetaW*div_deltatheta);
            CoefPC += AN*(0.5*bphiN-aphiN*div_deltaphi);
            CoefPC += AS*(0.5*bphiS+aphiS*div_deltaphi);
            CoefPC += -2.0*GAMMA*cosphi_C*div_adim;        
            ////////////////////////////////////////////
            CoefPE[posC] = AE*(0.5*bthetaE+athetaE*div_deltatheta)/CoefPC;
            CoefPW[posC] = AW*(0.5*bthetaW-athetaW*div_deltatheta)/CoefPC;
            CoefPN[posC] = AN*(0.5*bphiN+aphiN*div_deltaphi)/CoefPC;
            CoefPS[posC] = AS*(0.5*bphiS-aphiS*div_deltaphi)/CoefPC;
            ////////////////////////////////////////////
            double QthetaE = 0.5*(Quvw[NE].x+Quvw[SE].x);
            double QthetaW = 0.5*(Quvw[NW].x+Quvw[SW].x);
            double QphiN = 0.5*(Quvw[NW].y+Quvw[NE].y);
            double QphiS = 0.5*(Quvw[SW].y+Quvw[SE].y);
            double QwC = 0.25*(Quvw[NE].z+Quvw[SE].z+Quvw[SW].z+Quvw[NW].z);

            // Derivada temporal del fondo dt(H*cos(phi))
            double dtHsC = 0.25*(dtHs[NE]+dtHs[SE]+dtHs[SW]+dtHs[NW]);

            RHS_dt[posC] = (C*(QthetaE-QthetaW)*div_deltatheta + D*(QphiN-QphiS)*div_deltaphi + E*(QthetaE+QthetaW)/2.0 + F*(QphiN+QphiS)/2.0 + 2.0*cosphi_C*QwC - 2.0*hs_C*dtHsC)/(dt*CoefPC);
        }
        else {
            CoefPE[posC] = 0.0;
            CoefPW[posC] = 0.0;
            CoefPN[posC] = 0.0;
            CoefPS[posC] = 0.0;
            RHS_dt[posC] = 0.0;
        }
    }
}

__global__ void compute_NH_coefficientsCom(double2 *hH, double3 *Quvw, double *dtHs, double *vccos, double *vtan, int npx, int npy,
				int npyTotal, double coefdt, double deltat, double deltatheta, double deltaphi, double radio_tierra, double L,
				double H, double *RHS_dt, double *CoefPE, double *CoefPW, double *CoefPN, double *CoefPS, int inixSubmallaCluster,
				int iniySubmallaCluster, bool ultima_hebraX, bool ultima_hebraY)
{
    int i = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	int j = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;
	int ii, im, imm, ip;
	int jj, jm, jmm, jp;

    if ((i < npx+1) && (j < npy+1)) {
		if ((i < 2) || (i >= npx-1) || (j < 2) || (j >= npy-1)) {
            // Cálculo de índices de volúmenes vecinos con respecto a la referencia en malla staggered
            int posC = (j+1)*(npx+3) + i+1;  // Vértice (i,j)
            ii = ( ((i >= npx) && ultima_hebraX) ? npx-1 : i );
            im = ( ((i == 0) && (inixSubmallaCluster == 0)) ? 0 : i-1 );
            jj = ( ((j >= npy) && ultima_hebraY) ? npy-1 : j );
            jm = ( ((j == 0) && (iniySubmallaCluster == 0)) ? 0 : j-1 );
            int NE = 2+ii + (2+jj)*(npx+4);
            int NW = 2+im + (2+jj)*(npx+4);
            int SE = 2+ii + (2+jm)*(npx+4);
            int SW = 2+im + (2+jm)*(npx+4);
            ////////////////////////////////////////////
            double hs_NE = hH[NE].x; double Hs_NE = hH[NE].y;
            double hs_NW = hH[NW].x; double Hs_NW = hH[NW].y;
            double hs_SE = hH[SE].x; double Hs_SE = hH[SE].y;
            double hs_SW = hH[SW].x; double Hs_SW = hH[SW].y;
            if ((hs_NE > EPSILON) || (hs_NW > EPSILON) || (hs_SE > EPSILON) || (hs_SW > EPSILON)) {
                int j_global = iniySubmallaCluster+j;
                double dt = coefdt*deltat;
                double div_deltatheta = 1.0/deltatheta;
                double div_deltaphi = 1.0/deltaphi;
                double R = radio_tierra;
                // Cálculo de índices de volúmenes vecinos con respecto a la referencia en malla staggered
                imm = ( ((i < 2) && (inixSubmallaCluster == 0)) ? 0 : i-2 );
                ip  = ( ((i >= npx) && ultima_hebraX) ? npx-1 : i+1 );
                jmm = ( ((j < 2) && (iniySubmallaCluster == 0)) ? 0 : j-2 );
                jp  = ( ((j >= npy) && ultima_hebraY) ? npy-1 : j+1 );
                int NNE = 2+ii  + (2+jp)*(npx+4);
                int NNW = 2+im  + (2+jp)*(npx+4);
                int NEE = 2+ip  + (2+jj)*(npx+4);
                int NWW = 2+imm + (2+jj)*(npx+4);
                int SEE = 2+ip  + (2+jm)*(npx+4);
                int SWW = 2+imm + (2+jm)*(npx+4);
                int SSE = 2+ii  + (2+jmm)*(npx+4);
                int SSW = 2+im  + (2+jmm)*(npx+4);
                ////////////////////////////////////////////
                double hs_NNE = hH[NNE].x; double Hs_NNE = hH[NNE].y;
                double hs_NNW = hH[NNW].x; double Hs_NNW = hH[NNW].y;
                double hs_NEE = hH[NEE].x; double Hs_NEE = hH[NEE].y;
                double hs_NWW = hH[NWW].x; double Hs_NWW = hH[NWW].y;
                double hs_SEE = hH[SEE].x; double Hs_SEE = hH[SEE].y;
                double hs_SWW = hH[SWW].x; double Hs_SWW = hH[SWW].y;
                double hs_SSE = hH[SSE].x; double Hs_SSE = hH[SSE].y;
                double hs_SSW = hH[SSW].x; double Hs_SSW = hH[SSW].y;
                ////////////////////////////////////////////
                double cosphi_NNE = vccos[min(j_global+1,npyTotal-1)];  double tanphi_NNE = vtan[min(j_global+1,npyTotal-1)];
                double cosphi_NNW = vccos[min(j_global+1,npyTotal-1)];  double tanphi_NNW = vtan[min(j_global+1,npyTotal-1)];
                double cosphi_NE  = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NE  = vtan[min(j_global  ,npyTotal-1)];
                double cosphi_NEE = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NEE = vtan[min(j_global  ,npyTotal-1)];
                double cosphi_NW  = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NW  = vtan[min(j_global  ,npyTotal-1)];
                double cosphi_NWW = vccos[min(j_global  ,npyTotal-1)];  double tanphi_NWW = vtan[min(j_global  ,npyTotal-1)];
                double cosphi_SE  = vccos[    j_global-1*(j_global>0)]; double tanphi_SE  = vtan[    j_global-1*(j_global>0)];
                double cosphi_SEE = vccos[    j_global-1*(j_global>0)];
                double cosphi_SW  = vccos[    j_global-1*(j_global>0)]; double tanphi_SW  = vtan[    j_global-1*(j_global>0)];
                double cosphi_SWW = vccos[    j_global-1*(j_global>0)];
                double cosphi_SSE = vccos[    j_global-2*(j_global>1)];
                double cosphi_SSW = vccos[    j_global-2*(j_global>1)];
                // double cosphi_NNE = 1.0; double tanphi_NNE = 0.0;
                // double cosphi_NNW = 1.0; double tanphi_NNW = 0.0;
                // double cosphi_NE  = 1.0; double tanphi_NE  = 0.0;
                // double cosphi_NEE = 1.0; double tanphi_NEE = 0.0;
                // double cosphi_NW  = 1.0; double tanphi_NW  = 0.0;
                // double cosphi_NWW = 1.0; double tanphi_NWW = 0.0;
                // double cosphi_SE  = 1.0; double tanphi_SE  = 0.0;
                // double cosphi_SEE = 1.0; double tanphi_SEE = 0.0;
                // double cosphi_SW  = 1.0; double tanphi_SW  = 0.0;
                // double cosphi_SWW = 1.0; double tanphi_SWW = 0.0;
                // double cosphi_SSE = 1.0; double tanphi_SSE = 0.0;
                // double cosphi_SSW = 1.0; double tanphi_SSW = 0.0;
                ////////////////////////////////////////////
                double hs_C = 0.25*(hs_NE+hs_SE+hs_SW+hs_NW);
                double Hs_C = 0.25*(Hs_NE+Hs_SE+Hs_SW+Hs_NW);
                double Lambdas_C = 2.0*Hs_C-hs_C;
                double cosphi_C = 0.25*(cosphi_NE+cosphi_SE+cosphi_SW+cosphi_NW);
                double tanphi_C = 0.25*(tanphi_NE+tanphi_SE+tanphi_SW+tanphi_NW);        
                ////////////////////////////////////////////
                double dtheta_hs_NE = minmod(hs_NEE-hs_NE,hs_NE-hs_NW)*div_deltatheta;
                double dtheta_hs_SE = minmod(hs_SEE-hs_SE,hs_SE-hs_SW)*div_deltatheta;
                double dtheta_hs_SW = minmod(hs_SE-hs_SW,hs_SW-hs_SWW)*div_deltatheta;
                double dtheta_hs_NW = minmod(hs_NE-hs_NW,hs_NW-hs_NWW)*div_deltatheta;

                double dphi_hs_NE = minmod(hs_NNE-hs_NE,hs_NE-hs_SE)*div_deltaphi;
                double dphi_hs_SE = minmod(hs_NE-hs_SE,hs_SE-hs_SSE)*div_deltaphi;
                double dphi_hs_SW = minmod(hs_NW-hs_SW,hs_SW-hs_SSW)*div_deltaphi;
                double dphi_hs_NW = minmod(hs_NNW-hs_NW,hs_NW-hs_SW)*div_deltaphi;
                ////////////////////////////////////////////
                // double dtheta_Hs_NE = minmod(Hs_NEE-Hs_NE,Hs_NE-Hs_NW)*div_deltatheta;
                // double dtheta_Hs_SE = minmod(Hs_SEE-Hs_SE,Hs_SE-Hs_SW)*div_deltatheta;
                // double dtheta_Hs_SW = minmod(Hs_SE-Hs_SW,Hs_SW-Hs_SWW)*div_deltatheta;
                // double dtheta_Hs_NW = minmod(Hs_NE-Hs_NW,Hs_NW-Hs_NWW)*div_deltatheta;

                // double dphi_Hs_NE   = minmod(Hs_NNE-Hs_NE,Hs_NE-Hs_SE)*div_deltaphi;
                // double dphi_Hs_SE   = minmod(Hs_NE-Hs_SE,Hs_SE-Hs_SSE)*div_deltaphi;        
                // double dphi_Hs_SW   = minmod(Hs_NW-Hs_SW,Hs_SW-Hs_SSW)*div_deltaphi;
                // double dphi_Hs_NW   = minmod(Hs_NNW-Hs_NW,Hs_NW-Hs_SW)*div_deltaphi;

                double grad0, grad1;
                grad0 = gradH(hs_NE,hs_NEE,Hs_NE,Hs_NEE,cosphi_NE,cosphi_NEE);
                grad1 = gradH(hs_NW,hs_NE,Hs_NW,Hs_NE,cosphi_NW,cosphi_NE);
                double dtheta_Hs_NE = minmod(grad0,grad1)*div_deltatheta;

                grad0 = gradH(hs_SE,hs_SEE,Hs_SE,Hs_SEE,cosphi_SE,cosphi_SEE);
                grad1 = gradH(hs_SW,hs_SE,Hs_SW,Hs_SE,cosphi_SW,cosphi_SE);
                double dtheta_Hs_SE = minmod(grad0,grad1)*div_deltatheta;

                grad0 = gradH(hs_SW,hs_SE,Hs_SW,Hs_SE,cosphi_SW,cosphi_SE);
                grad1 = gradH(hs_SWW,hs_SW,Hs_SWW,Hs_SW,cosphi_SWW,cosphi_SW);
                double dtheta_Hs_SW = minmod(grad0,grad1)*div_deltatheta;

                grad0 = gradH(hs_NW,hs_NE,Hs_NW,Hs_NE,cosphi_NW,cosphi_NE);
                grad1 = gradH(hs_NWW,hs_NW,Hs_NWW,Hs_NW,cosphi_NWW,cosphi_NW);
                double dtheta_Hs_NW = minmod(grad0,grad1)*div_deltatheta;

                grad0 = gradH(hs_NE,hs_NNE,Hs_NE,Hs_NNE,cosphi_NE,cosphi_NNE);
                grad1 = gradH(hs_SE,hs_NE,Hs_SE,Hs_NE,cosphi_SE,cosphi_NE);
                double dphi_Hs_NE = minmod(grad0,grad1)*div_deltaphi;

                grad0 = gradH(hs_SE,hs_NE,Hs_SE,Hs_NE,cosphi_SE,cosphi_NE);
                grad1 = gradH(hs_SSE,hs_SE,Hs_SSE,Hs_SE,cosphi_SSE,cosphi_SE);
                double dphi_Hs_SE = minmod(grad0,grad1)*div_deltaphi;

                grad0 = gradH(hs_SW,hs_NW,Hs_SW,Hs_NW,cosphi_SW,cosphi_NW);
                grad1 = gradH(hs_SSW,hs_SW,Hs_SSW,Hs_SW,cosphi_SSW,cosphi_SW);
                double dphi_Hs_SW = minmod(grad0,grad1)*div_deltaphi;

                grad0 = gradH(hs_NW,hs_NNW,Hs_NW,Hs_NNW,cosphi_NW,cosphi_NNW);
                grad1 = gradH(hs_SW,hs_NW,Hs_SW,Hs_NW,cosphi_SW,cosphi_NW);
                double dphi_Hs_NW = minmod(grad0,grad1)*div_deltaphi;
                ////////////////////////////////////////////
                double dtheta_hs_C = 0.25*(dtheta_hs_NE+dtheta_hs_SE+dtheta_hs_SW+dtheta_hs_NW);        
                double dtheta_Hs_C = 0.25*(dtheta_Hs_NE+dtheta_Hs_SE+dtheta_Hs_SW+dtheta_Hs_NW);
                double dthetaLambdas_C = 2.0*dtheta_Hs_C - dtheta_hs_C;

                double dphi_hs_C = 0.25*(dphi_hs_NE+dphi_hs_SE+dphi_hs_SW+dphi_hs_NW);
                double dphi_Hs_C = 0.25*(dphi_Hs_NE+dphi_Hs_SE+dphi_Hs_SW+dphi_Hs_NW);
                double dphiLambdas_C = 2.0*dphi_Hs_C - dphi_hs_C;        
                ////////////////////////////////////////////
                double C = hs_C/(cosphi_C*R);
                double D = hs_C/R;
                double E = dthetaLambdas_C/(cosphi_C*R);
                double F = (dphiLambdas_C+tanphi_C*Lambdas_C)/R;
                ////////////////////////////////////////////
                double AE = C*div_deltatheta+0.5*E;
                double AW =-C*div_deltatheta+0.5*E;
                double AN = D*div_deltaphi+0.5*F;
                double AS =-D*div_deltaphi+0.5*F;
                ////////////////////////////////////////////
                double bthetaE = 0.5*(dtheta_hs_NE-GAMMA*dtheta_Hs_NE)/(cosphi_NE*cosphi_NE*R) + 0.5*(dtheta_hs_SE-GAMMA*dtheta_Hs_SE)/(cosphi_SE*cosphi_SE*R);
                double bthetaW = 0.5*(dtheta_hs_NW-GAMMA*dtheta_Hs_NW)/(cosphi_NW*cosphi_NW*R) + 0.5*(dtheta_hs_SW-GAMMA*dtheta_Hs_SW)/(cosphi_SW*cosphi_SW*R);
                double athetaE = 0.5*hs_NE/(cosphi_NE*cosphi_NE*R) + 0.5*hs_SE/(cosphi_SE*cosphi_SE*R);
                double athetaW = 0.5*hs_NW/(cosphi_NW*cosphi_NW*R) + 0.5*hs_SW/(cosphi_SW*cosphi_SW*R);
                double bphiN = 0.5*(dphi_hs_NE-GAMMA*dphi_Hs_NE)/(R*cosphi_NE) + 0.5*(dphi_hs_NW-GAMMA*dphi_Hs_NW)/(R*cosphi_NW);
                double bphiS = 0.5*(dphi_hs_SE-GAMMA*dphi_Hs_SE)/(R*cosphi_SE) + 0.5*(dphi_hs_SW-GAMMA*dphi_Hs_SW)/(R*cosphi_SW);
                double aphiN = 0.5*hs_NE/(cosphi_NE*R) + 0.5*hs_NW/(cosphi_NW*R);
                double aphiS = 0.5*hs_SE/(cosphi_SE*R) + 0.5*hs_SW/(cosphi_SW*R);
                ////////////////////////////////////////////
                double div_adim = L*L/(H*H);
                double CoefPC = AE*(0.5*bthetaE-athetaE*div_deltatheta);
                CoefPC += AW*(0.5*bthetaW+athetaW*div_deltatheta);
                CoefPC += AN*(0.5*bphiN-aphiN*div_deltaphi);
                CoefPC += AS*(0.5*bphiS+aphiS*div_deltaphi);
                CoefPC += -2*GAMMA*cosphi_C*div_adim;        
                ////////////////////////////////////////////
                CoefPE[posC] = AE*(0.5*bthetaE+athetaE*div_deltatheta)/CoefPC;
                CoefPW[posC] = AW*(0.5*bthetaW-athetaW*div_deltatheta)/CoefPC;
                CoefPN[posC] = AN*(0.5*bphiN+aphiN*div_deltaphi)/CoefPC;
                CoefPS[posC] = AS*(0.5*bphiS-aphiS*div_deltaphi)/CoefPC;
                ////////////////////////////////////////////
                double QthetaE = 0.5*(Quvw[NE].x+Quvw[SE].x);
                double QthetaW = 0.5*(Quvw[NW].x+Quvw[SW].x);
                double QphiN = 0.5*(Quvw[NW].y+Quvw[NE].y);
                double QphiS = 0.5*(Quvw[SW].y+Quvw[SE].y);
                double QwC = 0.25*(Quvw[NE].z+Quvw[SE].z+Quvw[SW].z+Quvw[NW].z);

                // Derivada temporal del fondo dt(H*cos(phi))
                double dtHsC = 0.25*(dtHs[NE]+dtHs[SE]+dtHs[SW]+dtHs[NW]);

                RHS_dt[posC] = (C*(QthetaE-QthetaW)*div_deltatheta + D*(QphiN-QphiS)*div_deltaphi + E*(QthetaE+QthetaW)/2.0 + F*(QphiN+QphiS)/2.0 + 2.0*cosphi_C*QwC - 2.0*hs_C*dtHsC)/(dt*CoefPC);
            }
            else {
                CoefPE[posC] = 0.0;
                CoefPW[posC] = 0.0;
                CoefPN[posC] = 0.0;
                CoefPS[posC] = 0.0;
                RHS_dt[posC] = 0.0;
            }
        }
    }
}

__global__ void compute_NH_pressureNoCom(double *d_Pnh0, double *d_Pnh1, double *errorNH, int npx, int npy, double *RHS_dt,
				double *CoefPE, double *CoefPW, double *CoefPN, double *CoefPS, double alfa)
{
	int i = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	int j = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;
	int pos_error;

    if ((i > 0) && (i < npx) && (j > 0) && (j < npy)) {
        int posC = (j+1)*(npx+3) + i+1;
		int posN = posC+(npx+3);
		int posS = posC-(npx+3);
		int posW = posC-1;
		int posE = posC+1;

        d_Pnh1[posC] = RHS_dt[posC] - (CoefPE[posC]*d_Pnh0[posE] + CoefPW[posC]*d_Pnh0[posW] + CoefPN[posC]*d_Pnh0[posN] + CoefPS[posC]*d_Pnh0[posS]);
        d_Pnh1[posC] = alfa*d_Pnh1[posC] + (1.0-alfa)*d_Pnh0[posC];

        pos_error = j*(npx+1)+i;
        errorNH[pos_error] = -fabs(d_Pnh1[posC] - d_Pnh0[posC]);
    }
}

__global__ void compute_NH_pressureCom(double *d_Pnh0, double *d_Pnh1, double *errorNH, int npx, int npy, double *RHS_dt,
				double *CoefPE, double *CoefPW, double *CoefPN, double *CoefPS, double alfa, int inixSubmallaCluster,
				int iniySubmallaCluster, bool ultima_hebraX, bool ultima_hebraY)
{
	int i = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	int j = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;
	int pos_error;

    if ((i < npx+1) && (j < npy+1)) {
		if ((i == 0) || (i == npx) || (j == 0) || (j == npy)) {
	        int posC = (j+1)*(npx+3) + i+1;
			int posN = ( ((j == npy) && ultima_hebraY) ? posC : posC+(npx+3) );
			int posS = ( ((j == 0) && (iniySubmallaCluster == 0)) ? posC : posC-(npx+3) );
			int posW = ( ((i == 0) && (inixSubmallaCluster == 0)) ? posC : posC-1 );
			int posE = ( ((i == npx) && ultima_hebraX) ? posC : posC+1 );

	        d_Pnh1[posC] = RHS_dt[posC] - (CoefPE[posC]*d_Pnh0[posE] + CoefPW[posC]*d_Pnh0[posW] + CoefPN[posC]*d_Pnh0[posN] + CoefPS[posC]*d_Pnh0[posS]);
	        d_Pnh1[posC] = alfa*d_Pnh1[posC] + (1.0-alfa)*d_Pnh0[posC];

	        pos_error = j*(npx+1)+i;
	        errorNH[pos_error] = -fabs(d_Pnh1[posC] - d_Pnh0[posC]);
	    }
    }
}

__global__ void NH_correction(double3 *Quvw0, double2 *hH, double3 *Quvw, double *vccos, double *vtan, double *d_Pnh, int npx,
				int npy, int npyTotal, double coefdt, double deltat, double deltatheta, double deltaphi, double radio_tierra,
				double L, double H, int it, int inixSubmallaCluster, int iniySubmallaCluster, bool ultima_hebraX, bool ultima_hebraY)
{
    int i = blockIdx.x*NUM_HEBRAS_ANCHO_EST + threadIdx.x;
	int j = blockIdx.y*NUM_HEBRAS_ALTO_EST + threadIdx.y;

    if ((i < npx) && (j < npy)) {
        int j_global = iniySubmallaCluster+j;
        double dt = coefdt*deltat;
        double div_deltatheta = 1.0/deltatheta;
        double div_deltaphi = 1.0/deltaphi;
        double R = radio_tierra;
        //////////////////////////////////
        int pos  = (j+2)*(npx+4) + i+2;
        int posE = ( ((i == npx-1) && ultima_hebraX) ? pos : pos+1 );
        int posW = ( ((i == 0) && (inixSubmallaCluster == 0)) ? pos : pos-1 );
        int posN = ( ((j == npy-1) && ultima_hebraY) ? pos : pos+(npx+4) );
        int posS = ( ((j == 0) && (iniySubmallaCluster == 0)) ? pos : pos-(npx+4) );
        //////////////////////////////////
        int posNHSW = (j+1)*(npx+3) + i+1;
        int posNHSE = posNHSW + 1;
        int posNHNW = posNHSW + (npx+3);
        int posNHNE = posNHNW + 1;
        //////////////////////////////////
        double hs   = hH[pos].x;
        double Hs   = hH[pos].y;
        double hs_N = hH[posN].x;
        double Hs_N = hH[posN].y;
        double hs_S = hH[posS].x;
        double Hs_S = hH[posS].y;
        double hs_E = hH[posE].x;
        double Hs_E = hH[posE].y;
        double hs_W = hH[posW].x;
        double Hs_W = hH[posW].y;

        double cosphi   = vccos[j_global];
        double cosphi_E = vccos[j_global];
        double cosphi_W = vccos[j_global];
        double cosphi_N = vccos[j_global+1*(j_global<npyTotal-1)];
        double cosphi_S = vccos[j_global-1*(j_global>0)];
        double tanphi   = vtan[j_global];
        // double cosphi   = 1.0;
        // double cosphi_E = 1.0;
        // double cosphi_W = 1.0;
        // double cosphi_N = 1.0;
        // double cosphi_S = 1.0;
        // double tanphi   = 0.0;
        //////////////////////////////////
        double dtheta_hs = minmod(hs_E-hs,hs-hs_W)*div_deltatheta;
        double dphi_hs = minmod(hs_N-hs,hs-hs_S)*div_deltaphi;
        // double dtheta_Hs = minmod(Hs_E-Hs,Hs-Hs_W)*div_deltatheta;
        // double dphi_Hs = minmod(Hs_N-Hs,Hs-Hs_S)*div_deltaphi;
        double grad0, grad1;
        grad0 = gradH(hs,hs_E,Hs,Hs_E,cosphi,cosphi_E);
        grad1 = gradH(hs_W,hs,Hs_W,Hs,cosphi_W,cosphi);
        double dtheta_Hs = minmod(grad0,grad1)*div_deltatheta;

        grad0 = gradH(hs,hs_N,Hs,Hs_N,cosphi,cosphi_N);
        grad1 = gradH(hs_S,hs,Hs_S,Hs,cosphi_S,cosphi);
        double dphi_Hs = minmod(grad0,grad1)*div_deltaphi;
        //////////////////////////////////
        double atheta = hs/(cosphi*cosphi*R);
        double btheta = (dtheta_hs-GAMMA*dtheta_Hs)/(cosphi*cosphi*R);
        double aphi   = hs/(cosphi*R);
        double bphi   = (dphi_hs-GAMMA*dphi_Hs+tanphi*(hs-GAMMA*Hs))/(cosphi*R);
        //////////////////////////////////
        double dtheta_ps = 0.5*(d_Pnh[posNHNE]-d_Pnh[posNHNW])*div_deltatheta + 0.5*(d_Pnh[posNHSE]-d_Pnh[posNHSW])*div_deltatheta;
        double dphi_ps   = 0.5*(d_Pnh[posNHNE]-d_Pnh[posNHSE])*div_deltaphi + 0.5*(d_Pnh[posNHNW]-d_Pnh[posNHSW])*div_deltaphi;
        double ps = 0.25*(d_Pnh[posNHNE]+d_Pnh[posNHSE]+d_Pnh[posNHSW]+d_Pnh[posNHNW]);
        //////////////////////////////////
        Quvw[pos].x += -dt*(atheta*dtheta_ps+btheta*ps)*(hs>0.0);
        Quvw[pos].y += -dt*(aphi*dphi_ps+bphi*ps)*(hs>0.0);
        double div_adim = L*L/(H*H);
        Quvw[pos].z += dt*GAMMA*ps*div_adim*(hs>0.0);
        //////////////////////////////////

        // Breaking
#ifdef BREAKING
        double div_hs = 1.0/(hs + 1e-6);
        double utheta = Quvw[pos].x*div_hs;
        double uphi   = Quvw[pos].y*div_hs;
        double modu   = sqrt(utheta*utheta + uphi*uphi);
        double Fr     = modu/(0.4*sqrt(hs/cosphi) + 1e-6);
        double Cbr    = 35.0*max(Fr-1.0, 0.0);
        double Cdt    = 1.0 + 2.0*dt*Cbr*abs(Quvw[pos].z)/(hs/cosphi + 1e-6);
        Quvw[pos].z = Quvw[pos].z/Cdt;
#endif

        if (it == OrdenTiempo) {
            Quvw0[pos].x = Quvw[pos].x;
            Quvw0[pos].y = Quvw[pos].y;
            Quvw0[pos].z = Quvw[pos].z;
        }
    }
}

#endif
