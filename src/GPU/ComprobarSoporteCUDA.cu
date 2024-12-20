#include <stdio.h>

// Funci�n que devuelve 0 si la tarjeta gr�fica soporta CUDA, 1 si no hay tarjeta gr�fica,
// y 2 si hay pero no soporta CUDA
extern "C" int comprobarSoporteCUDA()
{
    int valor, dev, deviceCount;

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
        valor = 1;
	else {
		for (dev=0; dev < deviceCount; ++dev) {
	        cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);
			if (deviceProp.major >= 1)
	            break;
	    }
		if (dev == deviceCount)
			valor = 2;
		else
			valor = 0;
	}
	return valor;
}
