INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
CUDA_INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR} ${MPI_INCLUDE_PATH})

CUDA_ADD_LIBRARY(GPU
			ComprobarSoporteCUDA.cu
			ShallowWater.cu
			Timestep.cu
			TimestepNoHidro.cu
			AristaCommon.cu
			AristaVer_kernel.cu
			AristaHor_kernel.cu
			Reduccion_kernel.cu
			Volumen_kernel.cu
			NoHidro_kernel.cu
			Reconstruccion.cu
			NestedMeshesCommon.cu
			NestedMeshesDif.cu
			NestedMeshesVal.cu
			FriccionVariable.cu
			Metrics.cu
			netcdf.cu
			netcdfSeries.cu
			Deformacion.cu
			DeformacionDinamica.cu
			Matriz.cu
			DC3D.cu
			prtoxy.cu)

TARGET_LINK_LIBRARIES(GPU
			${MPI_LIBRARIES}
			${CUDA_CUFFT_LIBRARIES}
			${NETCDF_LIBRARIES})

