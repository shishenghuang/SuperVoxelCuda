
#ifndef _NORMALESTIMATION_H
#define _NORMALESTIMATION_H


#include <iostream>
#include <cuda_runtime.h>



namespace cl{

    class NormalEstimation{
        public:
            NormalEstimation(double* points_, int* neighbors_, int pnum_ , int k_neighbor_);

            ~NormalEstimation(){
                cudaFree(points_device);
                cudaFree(neighbors_device);
                cudaFree(normals_device);
            }

        public:
            void eval(/*output*/double* normals);    // normals: pum X 3 dimensions

        private:
            double* points_device;
            int* neighbors_device;
            double* normals_device;
            int pnum, k_neighbor;
    };

}
#endif