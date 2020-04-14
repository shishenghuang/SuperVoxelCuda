
#include <unistd.h>
#include "normalEstimation.h"

namespace cl{

    // __global__ double mySqrt(const double x)
    // {
    //     float xf = x;
    //     float a = x;
    //     unsigned int i = *(unsigned int *)&xf;
    //     i = (i + 0x3f76cf62) >> 1;
    //     xf = *(float *)&i;
    //     xf = (xf + a / xf) * 0.5;
    //     return xf;
    // }

    // __global__ double metric(double* p1, double* n1, double* p2, double* n2, const double& lamda){

    //     double x = (p1[0] - p2[0]);
    //     double y = (p1[1] - p2[1]);
    //     double z = (p1[2] - p2[2]);

    //     double dd = mySqrt(x*x+y*y+z*z);
    //     return 1.0 - fabs(n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2]) + dd * lamda;

    // }

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
        //if(locId % 1000 == 0) printf("%d %lf %lf %lf  " , locId,normal_device[3*locId+0],normal_device[3*locId+1],normal_device[3*locId+2]);
    }

    __global__ void distanceEstimationKernel(double* points_device, int* neighbors_device, double* normal_device, int pnum_ , int k_neighbor_, double thelta, double* dis_device){
        
        int locId = (threadIdx.x + blockIdx.x * blockDim.x);
        if(locId >= pnum_) return;
        
        double max_value = 9999;
        double x0 = points_device[3*locId+0];
        double y0 = points_device[3*locId+1];
        double z0 = points_device[3*locId+2];
        double nx0 = normal_device[3*locId+0];
        double ny0 = normal_device[3*locId+1];
        double nz0 = normal_device[3*locId+2];

        double x1,y1,z1,nx1,ny1,nz1;
        for (int i = 0 ; i < k_neighbor_ ; i ++) {
            int index = neighbors_device[locId * k_neighbor_ + i];
            if(index == locId) continue;

            x1 = points_device[3*index+0];
            y1 = points_device[3*index+1];
            z1 = points_device[3*index+2];
            nx1 = normal_device[3*index+0];
            ny1 = normal_device[3*index+1];
            nz1 = normal_device[3*index+2];
            double dis_i = ( 1.0 - fabs(nx0*nx1+ny0*ny1+nz0*nz1) +  sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1))*thelta );
            //double dis_i = metric(&points_device[locId] , &normal_device[locId] , & points_device[index] , &normal_device[index] , thelta );
            max_value = (max_value > dis_i) ? dis_i : max_value;
        }

        //if(locId % 1000 == 0) printf("%d %lf " , locId , max_value);

        dis_device[locId] = max_value;
    }

    __global__ void exchangeKernel(double* points_device, int* neighbors_device, double* normal_device, int* label_device, double* dis_device, int* visited_in , int* visited_out, int pnum_ , int k_neighbor_, double thelta, int* exchange_num){
        int locId = (threadIdx.x + blockIdx.x * blockDim.x);
        if(locId >= pnum_) return;
        if(!visited_in[locId]) return;

        double max_value = 9999;
        double x0 = points_device[3*locId+0];
        double y0 = points_device[3*locId+1];
        double z0 = points_device[3*locId+2];
        double nx0 = normal_device[3*locId+0];
        double ny0 = normal_device[3*locId+1];
        double nz0 = normal_device[3*locId+2];

        double x1,y1,z1,nx1,ny1,nz1;
        double dis_locId = dis_device[locId];
        int maxId = label_device[locId];
        for (int i = 0 ; i < k_neighbor_ ; i ++) {
            int index = neighbors_device[locId * k_neighbor_ + i];
            if(index == locId) continue;
            int n_index = label_device[index];

            x1 = points_device[3*n_index+0];
            y1 = points_device[3*n_index+1];
            z1 = points_device[3*n_index+2];
            nx1 = normal_device[3*n_index+0];
            ny1 = normal_device[3*n_index+1];
            nz1 = normal_device[3*n_index+2];
            double dis_i = ( 1.0 - fabs(nx0*nx1+ny0*ny1+nz0*nz1) +  sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1))*thelta );
            //double dis_i = metric(&points_device[locId] , &normal_device[locId] , & points_device[index] , &normal_device[index] , thelta );
            //max_value = (max_value > dis_i) ? dis_i : max_value;
            if(dis_locId > dis_i){
                dis_locId = dis_i;
                maxId = n_index;
            }
        }
        if(maxId != label_device[locId]){
            atomicAdd(exchange_num , 1);
            label_device[locId] = maxId;
            dis_device[locId] = dis_locId;
            for (int i = 0 ; i < k_neighbor_ ; i ++) {
                int index = neighbors_device[locId * k_neighbor_ + i];   
                atomicExch(visited_out+index , 1);
                //visited[index] = visited[index] | true;    
            }     
        }

    }


    NormalEstimation::NormalEstimation(double* points_, int* neighbors_, int pnum_ , int k_neighbor_ ,double sv_size){

        pnum = pnum_;
        k_neighbor = k_neighbor_;
        supervoxel_size = sv_size;
        exchange_num = 0;
        // double* points_device;
        // int* neighbors_device;
        // double* normals_device;

        cudaMallocManaged(&points_device , sizeof(double)*3*pnum);
        cudaMallocManaged(&neighbors_device , sizeof(int)*k_neighbor*pnum);
        cudaMallocManaged(&normals_device , sizeof(double)*3*pnum);
        cudaDeviceSynchronize();

        cudaMemcpy(points_device , points_ , sizeof(double)*3*pnum , cudaMemcpyHostToDevice);
        cudaMemcpy(neighbors_device , neighbors_ , sizeof(int)*k_neighbor*pnum , cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void NormalEstimation::eval(/*output*/double* normals){

        int cudaBlockSize = 32;
        int gridSize = (int)ceil((float)pnum / (float)cudaBlockSize) + 1;//((int)ceil((float)cam_width / (float)cudaBlockSize.x), (int)ceil((float)cam_height / (float)cudaBlockSize.y));

        // printf("%d %d\n" , cudaBlockSize , gridSize);

        normalEstimationKernel<<<cudaBlockSize, gridSize>>>(points_device , neighbors_device , pnum , k_neighbor, normals_device);

        cudaMemcpy(normals , normals_device , sizeof(double)*3*pnum , cudaMemcpyDeviceToHost);
    }

    void NormalEstimation::eval_new(/*output*/double* normals,  double* dis){
        int cudaBlockSize = 320;
        int gridSize = (int)ceil((float)pnum / (float)cudaBlockSize) + 1;//((int)ceil((float)cam_width / (float)cudaBlockSize.x), (int)ceil((float)cam_height / (float)cudaBlockSize.y));

        normalEstimationKernel<<<gridSize,cudaBlockSize>>>(points_device , neighbors_device , pnum , k_neighbor, normals_device);
        cudaDeviceSynchronize();

        double* dis_device;
        cudaMallocManaged(&dis_device , sizeof(double)*pnum);

        distanceEstimationKernel<<<gridSize, cudaBlockSize>>>(points_device , neighbors_device , normals_device ,  pnum , k_neighbor, 0.4 / supervoxel_size,  dis_device);
        cudaDeviceSynchronize();
        cudaMemcpy(normals , normals_device , sizeof(double)*3*pnum , cudaMemcpyDeviceToHost);
        cudaMemcpy(dis , dis_device , sizeof(double)*pnum , cudaMemcpyDeviceToHost);

        cudaFree(dis_device);

    }

    void NormalEstimation::exchange(/*input-output*/int* labels, double* dis){

        exchange_num = 0;
        int* ex_num_device;
        int* ex_num_host = new int[1];
        cudaMallocManaged(&ex_num_device , sizeof(int));
        cudaMemset(ex_num_device , 0 , sizeof(int));

        int cudaBlockSize = 320;
        int gridSize = (int)ceil((float)pnum / (float)cudaBlockSize) + 1;//((int)ceil((float)cam_width / (float)cudaBlockSize.x), (int)ceil((float)cam_height / (float)cudaBlockSize.y));

        int* label_device;
        double* dis_device;
        cudaMallocManaged(&label_device , sizeof(int)*pnum);
        cudaMallocManaged(&dis_device , sizeof(double)*pnum);

        cudaMemcpy(label_device , labels , sizeof(int)*pnum , cudaMemcpyHostToDevice);
        cudaMemcpy(dis_device , dis , sizeof(double)*pnum , cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        int* visited_in ;
        int* visited_out;
        cudaMallocManaged(&visited_in ,  sizeof(int)*pnum);
        cudaMallocManaged(&visited_out , sizeof(int)*pnum);
        cudaMemset(visited_in , 1 , sizeof(int)*pnum);

        exchangeKernel<<<gridSize , cudaBlockSize>>>(points_device , neighbors_device , normals_device , label_device, dis_device, visited_in , visited_out, pnum , k_neighbor,0.4 / supervoxel_size, ex_num_device);
        cudaDeviceSynchronize();
        cudaMemcpy(ex_num_host , ex_num_device , sizeof(int) , cudaMemcpyDeviceToHost);

        //printf("exchage num : %d\n" , ex_num_host[0]);
        while(ex_num_host[0] > 1000){
            cudaMemset(ex_num_device , 0 , sizeof(int));
            cudaMemset(visited_in , 0 , sizeof(int)*pnum);
            exchangeKernel<<<gridSize , cudaBlockSize>>>(points_device , neighbors_device , normals_device , label_device, dis_device, visited_out , visited_in, pnum , k_neighbor, 0.4 / supervoxel_size,ex_num_device);
            cudaDeviceSynchronize();
            cudaMemcpy(ex_num_host , ex_num_device , sizeof(int) , cudaMemcpyDeviceToHost);
            if(ex_num_host[0] <= 1000) break;

            cudaMemset(visited_out , 0 , sizeof(int)*pnum);
            exchangeKernel<<<gridSize , cudaBlockSize>>>(points_device , neighbors_device , normals_device , label_device, dis_device, visited_in , visited_out, pnum , k_neighbor, 0.4 / supervoxel_size,ex_num_device);
            cudaDeviceSynchronize();
            cudaMemcpy(ex_num_host , ex_num_device , sizeof(int) , cudaMemcpyDeviceToHost);
            if(ex_num_host[0] <= 1000) break;

            //printf("exchange num: %d\n" , ex_num_host[0]);
        }   

        cudaMemcpy(labels , label_device , sizeof(int)*pnum , cudaMemcpyDeviceToHost);
        cudaMemcpy(dis , dis_device , sizeof(double)*pnum , cudaMemcpyDeviceToHost);

        cudaFree(label_device);
        cudaFree(dis_device);
        cudaFree(visited_in);
        cudaFree(visited_out);

    }
}