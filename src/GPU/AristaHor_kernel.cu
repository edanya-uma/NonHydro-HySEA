#ifndef _ARISTAHOR_KERNEL_H_
#define _ARISTAHOR_KERNEL_H_

#include "AristaCommon.cu"

__global__ void procesarFlujoLatGPU(double *d_vcos, double* aristaReconstruido, int npx, int npy, double delta_T, double2 *d_acumuladorh,
				double3 *d_acumuladorq, double epsh, double dyR, int tipo, double vmax, double borde_sup, double borde_inf,
				double borde_izq, double borde_der, int iniySubmallaCluster, int id_hebraY, int ultima_hebraY)
{
    int pos_x_hebra, pos_y_hebra;
    double aux0, aux1;
    double H0, H1;
    double3 U0, U1;
    double q0x, q0y, q0w, q1x, q1y, q1w;
    double h0, h1, eta0, eta1;
    int pos0, pos1, poscos;
    int frontera = 0;
    double epsilon_h;
    int idx_arista;
    
	// Multiplicamos pos_y_hebra por 2 porque se procesan aristas alternas
	pos_x_hebra = blockIdx.x*NUM_HEBRAS_ANCHO_ARI + threadIdx.x;
	pos_y_hebra = 2*(blockIdx.y*NUM_HEBRAS_ALTO_ARI + threadIdx.y);
	if (tipo == 2) pos_y_hebra++;
    
	if ((pos_x_hebra < npx) && (pos_y_hebra <= npy)) {
        poscos = 2*iniySubmallaCluster + pos_y_hebra*2;
		// Obtenemos los datos de los volumenes 0 y 1
		if ((pos_y_hebra == 0) && (id_hebraY == 0)) {
			// Frontera inferior			
            frontera = 1;
            epsilon_h = epsh*d_vcos[poscos];
            pos0 = (pos_y_hebra+2)*(npx+4) + pos_x_hebra+2;

            idx_arista = pos0*4*5;
            h0 = aristaReconstruido[idx_arista+0*5+0];
            eta0 = aristaReconstruido[idx_arista+0*5+1];
            q0x = aristaReconstruido[idx_arista+0*5+2];
            q0y = aristaReconstruido[idx_arista+0*5+3];
            q0w = aristaReconstruido[idx_arista+0*5+4];

            if (h0 > EPSILON) {
                filtro_estado(h0, q0x, q0y, q0w, epsilon_h, vmax);
                h1 = h0;
                H1 = H0 = h0 - eta0;

                aux0 = M_SQRT2*h0/sqrt(pow4(h0) + pow4(max(h0,epsilon_h)));
                U1.x = U0.x = q0x*aux0;
                U1.y = U0.y = q0y*aux0;
                U1.z = U0.z = q0w*aux0;
                procesarArista(d_vcos[poscos], h0, H0, U0, h1, H1, U1, 0.0, -1.0, dyR, delta_T, d_acumuladorh,
                	d_acumuladorq, pos0, -1, epsilon_h, frontera, borde_sup, borde_inf, borde_izq, borde_der);
            }
        }
		else {
            pos0 = (pos_y_hebra+1)*(npx+4) + pos_x_hebra+2;

            idx_arista = pos0*4*5;
            h0 = aristaReconstruido[idx_arista+2*5+0];
            eta0 = aristaReconstruido[idx_arista+2*5+1];
            q0x = aristaReconstruido[idx_arista+2*5+2];
            q0y = aristaReconstruido[idx_arista+2*5+3];
            q0w = aristaReconstruido[idx_arista+2*5+4];
            epsilon_h = epsh*d_vcos[poscos];

			if ((pos_y_hebra == npy) && ultima_hebraY) {
                frontera = 3;
				// Frontera superior. El volumen 1 es fantasma
                // dvcos1 = dvcos0;
                h1 = h0;
                eta1 = eta0;
                q1x = q0x;
                q1y = q0y;
                q1w = q0w;
                pos1 = -1;
            }
			else {
				// Arista interna
                pos1 = (pos_y_hebra+2)*(npx+4) + pos_x_hebra+2;

                idx_arista = pos1*4*5;
                h1 = aristaReconstruido[idx_arista+0*5+0];
                eta1 = aristaReconstruido[idx_arista+0*5+1];
                q1x = aristaReconstruido[idx_arista+0*5+2];
                q1y = aristaReconstruido[idx_arista+0*5+3];
                q1w = aristaReconstruido[idx_arista+0*5+4];
			}

            if ((h0 > EPSILON) || (h1 > EPSILON)) {
                filtro_estado(h0, q0x, q0y, q0w, epsilon_h, vmax);
                filtro_estado(h1, q1x, q1y, q1w, epsilon_h, vmax);
                H0 = h0 - eta0;
                H1 = h1 - eta1;

                aux0 = M_SQRT2*h0/sqrt(pow4(h0) + pow4(max(h0,epsilon_h)));
                aux1 = M_SQRT2*h1/sqrt(pow4(h1) + pow4(max(h1,epsilon_h)));
                U0.x = q0x*aux0;
                U0.y = q0y*aux0;
                U0.z = q0w*aux0;
                U1.x = q1x*aux1;
                U1.y = q1y*aux1;
                U1.z = q1w*aux1;

                procesarArista(d_vcos[poscos], h0, H0, U0, h1, H1, U1, 0.0, 1.0, dyR, delta_T, d_acumuladorh,
                	d_acumuladorq, pos0, pos1, epsilon_h, frontera, borde_sup, borde_inf, borde_izq, borde_der);
            }
		}
	}
}

#endif
