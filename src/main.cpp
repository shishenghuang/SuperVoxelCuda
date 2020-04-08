

#include <iostream>
#include <fstream>
#include <vector>
#include "supervoxel.h"
#include "glog/logging.h"

using namespace std;
using namespace cl;


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

void colormap(int rgb , int& alpha_ , int& red_, int& green_ , int& blue_) {
    alpha_ =  rgb >> 24;
    red_   = (rgb >> 16) & 0xff;
    green_ = (rgb >> 8)  & 0xff;
    blue_  =  rgb        & 0xff;
}

void WritePoints(const char* filename,
                 int n_supervoxels,
                 const vector<double>& points,
                 int* labels) {

    vector<int> colors(points.size());
    vector<int> svcolors(n_supervoxels * 3);
    std::mt19937 random;
    for(int i = 0 ; i < n_supervoxels ; i ++){
        int r, g , b, alpha;
        colormap(random() , alpha , r, g, b);
        svcolors[3*i+0] = r;
        svcolors[3*i+1] = g;
        svcolors[3*i+2] = b;
    }
    for (int i = 0; i < points.size() / 3; ++i) {
        int l = labels[i];
        colors[3*i+0] = svcolors[3*l+0];
        colors[3*i+1] = svcolors[3*l+1];
        colors[3*i+2] = svcolors[3*l+2];                
    }

    FILE* fp = fopen(filename , "w+");
    for(int i = 0 ; i < points.size() / 3 ; i ++){
        fprintf(fp, "%lf %lf %lf %d %d %d\n" , points[3*i+0] , 
                points[3*i+1] , points[3*i+2] , colors[3*i+0] , colors[3*i+1], colors[3*i+2]);
    }
    fclose(fp);

//    system(filename);
}

int main()
{

    vector<double> points ;
    vector<int> colors;

    readData("../testdata/test.xyz" , points, colors);

    int point_num = points.size() / 3;
    double* points_ptr = new double[point_num * 3];
    int* labels = new int[point_num];
    for(int i = 0 ; i < points.size() ; i ++){
        points_ptr[i] = points[i];
    }

    double resolution = 1.0;
    int k_neighbor = 15;
    int n_supervoxel = 0;
    cl::SVCluster svcluster(resolution , k_neighbor);
    svcluster.segment(points_ptr , point_num , labels , n_supervoxel);

    WritePoints("out.xyz" ,  n_supervoxel , points , labels );

    delete labels;
    return 0;
}