

#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#define FLANN_USE_CUDA
#include <flann/flann.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector_functions.h>
#include "glog/logging.h"

using namespace flann;
using namespace std;


bool readData(const char* filename , vector<double>& points, vector<int>& colors)
{
    std::ifstream in(filename);
    if (!in) {
        LOG(INFO) << "Cannot open XYZ file '" << filename << "' for reading.";
        return false;
    }

    int n_lines = 0;
    std::string line;
    double x, y, z;
    int r, g, b;
    while (std::getline(in, line)) {
        std::istringstream is(line);

        if (!(is >> x) || !(is >> y) || !(is >> z) ||
            !(is >> r) || !(is >> g) || !(is >> b)) {
            LOG(INFO) << "Invalid XYZ format at line: " << n_lines++;
            in.close();
            return false;
        }
        points.push_back(x);
        points.push_back(y);
        points.push_back(z);
        colors.push_back(r);
        colors.push_back(g);
        colors.push_back(b);
        // points->emplace_back(x, y, z);
        // colors->emplace_back(r, g, b);
    }

    in.close();

}

int main()
{
    char* filename = "../testdata/test.xyz";
    vector<double> points;
    vector<int> colors;

    readData(filename , points , colors);

    int point_num = points.size() / 3;

    int max_nn = 15;
    flann::Matrix<float> data = flann::Matrix<float>(new float[point_num*3],point_num,3);
    flann::Matrix<float> query = flann::Matrix<float>(new float[point_num*3],point_num,3);
    flann::Matrix<int> match = flann::Matrix<int>(new int[point_num*max_nn],point_num,max_nn);
    flann::Matrix<float> dists = flann::Matrix<float>(new float[point_num*max_nn],point_num,max_nn);
    flann::Matrix<int> indices = flann::Matrix<int>(new int[point_num*max_nn],point_num,max_nn);
    for(int i = 0 ; i < point_num ; i ++){
        data[i][0] = points[3*i];
        data[i][1] = points[3*i+1];
        data[i][2] = points[3*i+2];
        query[i][0] = points[3*i];
        query[i][1] = points[3*i+1];
        query[i][2] = points[3*i+2];
    }

    flann::KDTreeCuda3dIndex<L2_Simple<float> > index(data, flann::KDTreeCuda3dIndexParams());
    double start_t,end_t;
    start_t = clock();
    index.buildIndex();
    end_t = clock();
    std::cout << "Build Index using " << (end_t - start_t) / CLOCKS_PER_SEC << " s" << std::endl;

    start_t = clock();
	flann::SearchParams sp;
    sp.checks = 8;
    sp.sorted = true;
	//sp.matrices_in_gpu_ram=true;    
    index.knnSearch(data,indices , dists , max_nn , sp);
    end_t = clock();
    std::cout << "Search Index using " << (end_t - start_t) / CLOCKS_PER_SEC << " s" << std::endl;

    for(int i = 0 ; i< point_num ; i += 1000){
        std::cout << "i " << i << " " ;// data[i][0] << " " << data[i][1] << " " << data[i][2] << " ";
        for(int j = 0 ; j < max_nn ; j ++){
            std::cout << indices[i][j] << " ";
            //std::cout << dists[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

	// thrust::host_vector<float4> data_host(point_num);
	// for( int i=0; i<point_num; i++ )
	// {
	// 	data_host[i]=make_float4(data[i][0],data[i][1],data[i][2],0);
	// }
	// thrust::device_vector<float4> data_device = data_host;

	// flann::Matrix<float> data_device_matrix( (float*)thrust::raw_pointer_cast(&data_device[0]),data.rows,3,4*4);
	// //flann::Matrix<float> query_device_matrix( (float*)thrust::raw_pointer_cast(&query_device[0]),data.rows,3,4*4);
	
	// flann::KDTreeCuda3dIndexParams index_params;
	// index_params["input_is_gpu_float4"]=true;
	// flann::Index<L2_Simple<float> > index2(data_device_matrix, index_params);
    // //start_timer("Building kd-tree index...");
    // start_t = clock();
    // index2.buildIndex();
    // end_t = clock();
    // std::cout << "Buind Index2 using " << (end_t - start_t) / CLOCKS_PER_SEC << " s" << std::endl;
    // //printf("done (%g seconds)\n", stop_timer());

	
	// thrust::device_vector<int> indices_device(query.rows*max_nn);
	// thrust::device_vector<float> dists_device(query.rows*max_nn);
	// flann::Matrix<int> indices_device_matrix( (int*)thrust::raw_pointer_cast(&indices_device[0]),query.rows,max_nn);
	// flann::Matrix<float> dists_device_matrix( (float*)thrust::raw_pointer_cast(&dists_device[0]),query.rows,max_nn);
	
    // //start_timer("Searching KNN...");
	// // indices.cols=4;
	// // dists.cols=4;
	// flann::SearchParams sp;
	// sp.matrices_in_gpu_ram=true;
    // start_t = clock();
    // index2.knnSearch(data_device_matrix, indices_device_matrix, dists_device_matrix, max_nn, sp );
    // end_t = clock();
    // std::cout << "Search2 using " << (end_t - start_t) / CLOCKS_PER_SEC << " s" << std::endl;

    // //printf("done (%g seconds)\n", stop_timer());
	
	// flann::Matrix<int> indices_host( new int[ query.rows*max_nn],query.rows,max_nn );
	// flann::Matrix<float> dists_host( new float[ query.rows*max_nn],query.rows,max_nn );
	
	// thrust::copy( dists_device.begin(), dists_device.end(), dists_host.ptr() );
	// thrust::copy( indices_device.begin(), indices_device.end(), indices_host.ptr() );

    return 0;
}