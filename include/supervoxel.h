

#ifndef _SUPERVOXEL_H
#define _SUPERVOXEL_H

#include "algorithm.h"
#include <iostream>
#include <vector>

#include "disjoint_set.h"
#include "grid_sample.h"
#include "octree.h"


namespace cl{

    class SVCluster{

        public:
            SVCluster(){
                resolution = 0.2;
                k_neighbor = 15;
            }
            SVCluster(const double& resolution_, const int& k_neighbor_){
                resolution = resolution_;
                k_neighbor = k_neighbor_;
            }
            ~SVCluster(){};

        
        public:
            void segment(double* points , const int& pnum, /*output*/ int* labels , int& n_supervoxels);

        private:
            double resolution;
            int k_neighbor;
    };
}



#endif