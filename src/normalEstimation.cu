

#include "normalEstimation.h"

namespace cl{

    __global__ void normalEstimationKernel(double* points_device, int* neighbors_device, int pnum_ , int k_neighbor_, double* normal_device){
        
        int locId = (threadIdx.x + blockIdx.x * blockDim.x);
        if(locId >= pnum_) return;
        
        double3 center_point = make_double3(0 ,0 , 0);
        double cx = 0 , cy = 0 , cz = 0;
        for(int i = 0 ; i < k_neighbor_ ; i ++){
            int index = neighbors_device[locId * k_neighbor_ + i];
            cx += points_device[3*index+0];
            cy += points_device[3*index+1];
            cz += points_device[3*index+2];
        }
        cx /= k_neighbor_;
        cy /= k_neighbor_;
        cz /= k_neighbor_;

        double a00 = 0.0, a01 = 0.0, a02 = 0.0, a11 = 0.0, a12 = 0.0, a22 = 0.0;
        int i = 0;
        double sum = 0.0;
        for (int i = 0 ; i < k_neighbor_ ; i ++) {
            int index = neighbors_device[locId * k_neighbor_ + i];
            double x = points_device[3*index+0] - cx;
            double y = points_device[3*index+1] - cy;
            double z = points_device[3*index+2] - cz;
            double w = 1.0 ; //weights[i];

            a00 += w * x * x;
            a01 += w * x * y;
            a02 += w * x * z;
            a11 += w * y * y;
            a12 += w * y * z;
            a22 += w * z * z;

            sum += w;
        }

        double t = 1.0 / sum;
        a00 = a00 * t;
        a01 = a01 * t;
        a02 = a02 * t;
        a11 = a11 * t;
        a12 = a12 * t;
        a22 = a22 * t;

        // Computing the least eigenvalue of the covariance matrix.
        double q = (a00 + a11 + a22) / 3.0;
        double pq = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) +
                    (a22 - q) * (a22 - q) +
                    2.0 * (a01 * a01 + a02 * a02 + a12 * a12);
        pq = sqrt(pq / 6.0);
        double mpq = 1.0 / (pq * pq * pq);//std::pow(1.0 / pq, 3.0);
        double det_b = mpq * ((a00 - q) * ((a11 - q) * (a22 - q) - a12 * a12) -
                            a01 * ( a01 * (a22 - q) - a12 * a02) +
                            a02 * ( a01 * a12 - (a11 - q) * a02));
        double r = 0.5 * det_b;
        double phi = 0.0;
        if (r <= -1.0)
            phi = 3.1415926 / 3.0;
        else if (r >= 1.0)
            phi = 0.0;
        else
            phi = acos(r) / 3.0;
        double eig = q + 2.0 * pq * cos(phi + 3.1415926 * (2.0 / 3.0));

        // Computing the corresponding eigenvector.

        double nx =  a01 * a12 - a02 * (a11 - eig);
        double ny =  a01 * a02 - a12 * (a00 - eig);
        double nz = (a00 - eig) * (a11 - eig) - a01 * a01;
        // normal->x = a01 * a12 - a02 * (a11 - eig);
        // normal->y = a01 * a02 - a12 * (a00 - eig);
        // normal->z = (a00 - eig) * (a11 - eig) - a01 * a01;
        double len = sqrt(nx*nx+ny*ny+nz*nz);
        nx /= len;
        ny /= len;
        nz /= len;

        // Normalize.
        //double norm = normal->norm();
        if (len == 0.0) {
            //*normal = RVector3D(0.0, 0.0, 1.0);
            normal_device[3*locId+0] = 0.0;
            normal_device[3*locId+1] = 0.0;
            normal_device[3*locId+2] = 1.0;
        } else {
            //*normal *= 1.0 / norm;
            normal_device[3*locId+0] = nx;
            normal_device[3*locId+1] = ny;
            normal_device[3*locId+2] = nz;
        }
    }

    NormalEstimation::NormalEstimation(double* points_, int* neighbors_, int pnum_ , int k_neighbor_){

        pnum = pnum_;
        k_neighbor = k_neighbor_;
        // double* points_device;
        // int* neighbors_device;
        // double* normals_device;

        cudaMallocManaged(&points_device , sizeof(double)*3*pnum);
        cudaMallocManaged(&neighbors_device , sizeof(int)*k_neighbor*pnum);
        cudaMallocManaged(&normals_device , sizeof(double)*3*pnum);

        cudaMemcpy(points_device , points_ , sizeof(double)*3*pnum , cudaMemcpyHostToDevice);
        cudaMemcpy(neighbors_device , neighbors_ , sizeof(int)*k_neighbor*pnum , cudaMemcpyHostToDevice);
    }

    void NormalEstimation::eval(/*output*/double* normals){

        int cudaBlockSize = 32;
        int gridSize = (int)ceil((float)pnum / (float)cudaBlockSize) + 1;//((int)ceil((float)cam_width / (float)cudaBlockSize.x), (int)ceil((float)cam_height / (float)cudaBlockSize.y));

        normalEstimationKernel<<<cudaBlockSize, gridSize>>>(points_device , neighbors_device , pnum , k_neighbor, normals_device);

        cudaMemcpy(normals , normals_device , sizeof(double)*3*pnum , cudaMemcpyDeviceToHost);
    }
}