

#include <iostream>
#include <vector>
#include <queue>
#include <stdlib.h>
#include <cfloat>
#include "supervoxel.h"
#include "normalEstimation.h"
#include "glog/logging.h"
#define FLANN_USE_CUDA
#include <flann/flann.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_functions.h>

using namespace flann;
using namespace std;
namespace cl
{
    inline double mySqrt(const double x)
    {
        float xf = x;
        float a = x;
        unsigned int i = *(unsigned int *)&xf;
        i = (i + 0x3f76cf62) >> 1;
        xf = *(float *)&i;
        xf = (xf + a / xf) * 0.5;
        return xf;
    }

    inline double metric(double* p1, double* n1, double* p2, double* n2, const double& lamda){

        double x = (p1[0] - p2[0]);
        double y = (p1[1] - p2[1]);
        double z = (p1[2] - p2[2]);

        double dd = mySqrt(x*x+y*y+z*z);
        return 1.0 - std::fabs(n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2]) + dd * lamda;

    }
    void SVCluster::segment(double* points , const int& pnum, /*output*/ int* labels , int& n_supervoxels)
    {

        n_supervoxels = 0;
        double thelta = 0.4 / resolution;
        int max_nn = k_neighbor;

        //============================================================================================        LOG(INFO) << "Begin to sample the points";
        //=======================================GridSample===========================================
        GridSample(points , pnum , resolution , n_supervoxels);
        LOG(INFO) << "Finish sample the points and obtain supervoxel number: " << n_supervoxels;
        //***1***
        //=======================================Finish GridSample====================================
        //============================================================================================

        //============================================================================================        LOG(INFO) << "Begin to compute the neighbors and normals";
        //=======================================Compute KNN using Flann Cuda=========================
        flann::Matrix<float> data = flann::Matrix<float>(new float[pnum*3],pnum,3);
        flann::Matrix<float> query = flann::Matrix<float>(new float[pnum*3],pnum,3);
        flann::Matrix<int> match = flann::Matrix<int>(new int[pnum*max_nn],pnum,max_nn);
        flann::Matrix<float> dists = flann::Matrix<float>(new float[pnum*max_nn],pnum,max_nn);
        flann::Matrix<int> indices = flann::Matrix<int>(new int[pnum*max_nn],pnum,max_nn);

        for(int i = 0 ; i < pnum ; i ++){
            data[i][0] = points[3*i+0];
            data[i][1] = points[3*i+1];
            data[i][2] = points[3*i+2];
        }

        flann::KDTreeCuda3dIndex<L2_Simple<float> > index(data, flann::KDTreeCuda3dIndexParams());
        double start_t,end_t;
        start_t = clock();
        index.buildIndex();
        end_t = clock();
        LOG(INFO) << "Build Index using " << (end_t - start_t) / CLOCKS_PER_SEC << " s" << std::endl;

        start_t = clock();
        flann::SearchParams sp;
        sp.checks = 8;
        sp.sorted = true;   
        index.knnSearch(data, indices , dists , max_nn , sp);
        end_t = clock();
        LOG(INFO) << "Search Index using " << (end_t - start_t) / CLOCKS_PER_SEC << " s" << std::endl;

        int* neighbors = new int[pnum * k_neighbor];
        for(int i = 0  ; i < pnum ; i ++){
            int index = i * k_neighbor;
            for(int j = 0 ; j < k_neighbor ; j ++){
                neighbors[index + j] = indices[i][j];
            }
        }
        //=========================================Finish Compute KNN using Flann Cuda================
        //============================================================================================

        //============================================================================================
        //===================================Estimate Normals using CUDA==============================        double* normals = new double[pnum];
        LOG(INFO) << "Begin to compute normal";
        double* normals = new double[pnum*3];
        start_t = clock();
        NormalEstimation nest(points , neighbors , pnum , k_neighbor);
        nest.eval(normals);
        end_t = clock();
        LOG(INFO) << "Compute Normal using " << (end_t - start_t) / CLOCKS_PER_SEC << " s" << std::endl;
        //==================================Finish Estimate Normals===================================
        //============================================================================================

        //============================================================================================
        //==================================Cluster SuperVoxels=======================================        
        LOG(INFO) << "now cluster the points";

        // At first, each point is a supervoxel.
        DisjointSet set(pnum);
        vector<int> supervoxels(pnum);
        for (int i = 0; i < pnum; ++i) {
            supervoxels[i] = i;
        }

        // The size of supervoxel.
        vector<int> sizes(pnum, 1);

        // Queue for region growing.
        int* queue = new int[pnum];//(n_points);
        vector<int>* adjacents = new vector<int>[pnum];//neighbors;
        for(int i = 0  ; i < pnum ; i ++){
            int index = i * k_neighbor;
            for(int j=0;j<k_neighbor;j++){
                adjacents[i].push_back(neighbors[index+j]);
            }
        }

        int number_of_supervoxels = pnum;
        bool* visited = new bool[pnum];
        memset(visited , false , sizeof(bool)*pnum);
        bool* ismerged = new bool[pnum];
        memset(ismerged , false, sizeof(bool)*pnum);

        // Compute the minimum value of lambda.
        start_t = clock();
        vector<double> dis(pnum , DBL_MAX);

        for (int i = 0; i < pnum; ++i) {
            for (int j = 0 ; j < adjacents[i].size() ; j ++) {
                int k = adjacents[i][j];
                if (i != k) {
                    dis[i] = std::min(dis[i], metric(&points[i*3], &normals[i*3], &points[k*3], &normals[k*3] ,thelta));//);
                }
            }
        }
        std::sort(dis.begin() , dis.end());
        double lambda = std::max(DBL_EPSILON, dis[pnum / 2 + 1]);
        end_t = clock();
        LOG(INFO) << "Step 0 using : " << (end_t - start_t ) / CLOCKS_PER_SEC << "s " << std::endl;
        
        // ------------------------------------------------------------------
        // ---------------- Step 1: Find supervoxels. -----------------------
        start_t = clock();
        double start_tt, end_tt;
        for (; ; lambda *= 2.0) {
            if (supervoxels.size() <= 1) break;

            start_tt = clock();
            for (int i : supervoxels) {
                if (adjacents[i].empty()) continue;

                visited[i] = true;
                int front = 0, back = 1;
                queue[front++] = i;
                for (int j : adjacents[i]) {
                    j = set.Find(j);
                    if (!visited[j]) {
                        visited[j] = true;
                        queue[back++] = j;
                    }
                }

                vector<int> adjacent;
                while (front < back) {
                    int j = queue[front++];

                    double loss = sizes[j] * metric(&points[i*3], &normals[i*3], &points[j*3] , &normals[j*3] ,thelta);
                    //double loss = sizes[j] * dhp.compute(i, j , points[i] , points[j] , metric);
                    double improvement = lambda - loss;
                    if (improvement > 0.0) {
                        set.Link(j, i);

                        sizes[i] += sizes[j];
                        sizes[j] = 0;
                        ismerged[j] = true;

                        for (int k : adjacents[j]) {
                            k = set.Find(k);
                            if (!visited[k]) {
                                visited[k] = true;
                                queue[back++] = k;
                            }
                        }
                        adjacents[j].clear();
                        if (--number_of_supervoxels == n_supervoxels) break;
                    } else {
                        adjacent.push_back(j);
                    }
                }
                adjacents[i].clear();
                for(int j = 0 ; j < adjacent.size() ; j ++){
                    adjacents[i].push_back(adjacent[j]);
                }

                for (int j = 0; j < back; ++j) {
                    visited[queue[j]] = false;
                }
                if (number_of_supervoxels == n_supervoxels) break;
            }

            // Update supervoxels.
            number_of_supervoxels = 0;
            for (int i : supervoxels) {
                if (set.Find(i) == i) {
                    (supervoxels)[number_of_supervoxels++] = i;
                }
            }
            supervoxels.resize(number_of_supervoxels);

            end_tt = clock();
            LOG(INFO) <<  "In Step 1 obtain number_of_supervoxels " << number_of_supervoxels << " using " << (end_tt - start_tt ) / CLOCKS_PER_SEC << "s " << std::endl;
            if (number_of_supervoxels == n_supervoxels) break;
        }
        end_t = clock();
        LOG(INFO) << "Step 1 using : " << (end_t - start_t ) / CLOCKS_PER_SEC << "s " << std::endl;

        // Assign the label to each point according to its supervoxel ID.
        //labels->resize(n_points);
        for (int i = 0; i < pnum; ++i) {
            labels[i] = set.Find(i);
        }

        // ------------------------------------------------------------------
        // ---------------- Step 2: Refine the boundaries. ------------------
        start_t = clock();
        for (int i = 0; i < pnum; ++i) {
            int j = labels[i];
            dis[i] = metric(&points[i*3], &normals[i*3], &points[j*3], &normals[j*3] , thelta);
        }

        std::queue<int> q;
        vector<bool> in_q(pnum, false);

        for (int i = 0; i < pnum; ++i) {
            int index = i * k_neighbor;
            for(int j = 0 ; j < k_neighbor; j ++){
                int n_j = neighbors[index+j];
                if (labels[i] != labels[n_j]) {
                    if (!in_q[i]) {
                        q.push(i);
                        in_q[i] = true;
                    }
                    if (!in_q[n_j]) {
                        q.push(n_j);
                        in_q[n_j] = true;
                    }
                }
            }
        }

        bool change = false;
        while (!q.empty()) {
            int i = q.front();
            q.pop();
            in_q[i] = false;
            change = false;

            int index = i * k_neighbor;
            for(int j = 0 ; j < k_neighbor; j ++){
            // for (int j : neighbors[i]) {
                int n_j = neighbors[index+j];
                int a = labels[i];
                int b = labels[n_j];
                if (a == b) continue;
                double d = metric(&points[i*3], &normals[i*3], &points[b*3], &normals[b*3] , thelta);
                if (d < dis[i]) {
                    labels[i] = b;
                    dis[i] = d;
                    change = true;
                }
            }

            if (change) {
                for(int j = 0 ;j < k_neighbor; j ++){
                    int n_j = neighbors[index+j];
                //for (int j : neighbors[i]) {
                    if (labels[i] != labels[n_j]) {
                        if (!in_q[n_j] ) {
                            q.push(n_j);
                            in_q[n_j] = true;
                        }
                    }
                }
            }
        }
        end_t = clock();
        LOG(INFO) << "Step 2 using : " << (end_t - start_t ) / CLOCKS_PER_SEC << "s " << std::endl;
        
        // ------------------------------------------------------------------
        // ---------------- Step 3: Relabel the supervoxels. ----------------
        start_t = clock();
        vector<int> map(pnum);
        for (int i = 0; i < number_of_supervoxels; ++i) {
            map[supervoxels[i]] = i;
        }
        for (int i = 0; i < pnum; ++i) {
            labels[i] = map[labels[i]];
            // std::cout << i << " " << labels[i] << ", ";
        }
        end_t = clock();
        LOG(INFO) << "Step 3 using : " << (end_t - start_t ) / CLOCKS_PER_SEC << "s " << std::endl;    
        //***3***
        //====================================Finish==================================================
        //============================================================================================

        //--------------------------------------------------------------------------------------------
        //------------------------------Release the ptr-----------------------------------------------
        delete queue ;
        for(int i = 0 ; i < pnum ; i ++){
            adjacents[i].clear();
        }
        //delete adjacents;
        delete visited;
        delete ismerged;
        delete neighbors;
        delete normals;

    }
} // namespace cl
