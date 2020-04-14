

#ifndef _CLUSTER_CUDA_H
#define _CLUSTER_CUDA_H

#include <iostream>
#include <vector>


namespace cl{

    class cluster_CUDA{

        public:
            cluster_CUDA(){
                resolution = 0.2;
                k_neighbor = 15;
            }
            cluster_CUDA(const double& resolution_, const int& k_neighbor_){
                
            }
            ~cluster_CUDA();

        public:
            void cluster();


        private:
            double resolution;
            int k_neighbor;
    }


} // end namespace

#endif