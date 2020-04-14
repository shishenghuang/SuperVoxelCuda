
#ifndef _NORMALESTIMATION_H
#define _NORMALESTIMATION_H


#include <iostream>
#include <cuda_runtime.h>



namespace cl{

    class NormalEstimation{
        public:
            NormalEstimation(double* points_, int* neighbors_, int pnum_ , int k_neighbor_ , double sv_size);

            ~NormalEstimation(){
                cudaFree(points_device);
                cudaFree(neighbors_device);
                cudaFree(normals_device);
            }

        public:
            void eval(/*output*/double* normals);    // normals: pum X 3 dimensions
            void eval_new(/*output*/double* normals, double* dis); // normals: pum X 3 dimensions, dis
            void exchange(/*input-output*/int* labels, double* dis);

            __host__ __device__ int getExchange(){ return exchange_num;}

        private:
            double* points_device;
            int* neighbors_device;
            double* normals_device;
            int pnum, k_neighbor;
            double supervoxel_size;
            int exchange_num;
    };

}
#endif