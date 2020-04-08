//
// Copyright 2016 Yangbin Lin. All Rights Reserved.
//
// Author: yblin@jmu.edu.cn (Yangbin Lin)
//
// This file is part of the Code Library.
//

#ifndef GEOMETRY_POINT_CLOUD_GRID_SAMPLE_H_
#define GEOMETRY_POINT_CLOUD_GRID_SAMPLE_H_

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

// #include "codelibrary/base/array.h"
#include "algorithm.h"
#include "box_3d.h"
#include "octree.h"

namespace cl {


/**
 * It considers a regular grid covering the bounding box of the input point
 * cloud, and clusters all points sharing the same cell of the grid by picking
 * as representant one arbitrarily chosen point.
 */
inline void GridSample(double* points, const int& pnum, double resolution, int& sampelSize){
                // Array<int>* sampling ) {
    assert(resolution > 0.0);
    // assert(sampling);

    // sampling->clear();

    int n = pnum;//CountElements(first, last);
    assert(n > 0);

    //Array<int> random_seq(n);
    std::vector<int> random_seq(n);

    for (int i = 0; i < n; ++i) {
        random_seq[i] = i;
    }
    std::mt19937 random;
    std::shuffle(random_seq.begin(), random_seq.end(), random);

    RBox3D box(points , pnum);
    assert(box.x_length() / resolution < INT_MAX);
    assert(box.y_length() / resolution < INT_MAX);
    assert(box.z_length() / resolution < INT_MAX);
    int size1 = box.x_length() / resolution + 1;
    int size2 = box.y_length() / resolution + 1;
    int size3 = box.z_length() / resolution + 1;

    Octree<bool> octree(size1, size2, size3);
    //Octree<int> octree(size1, size2, size3);
    typedef typename Octree<bool>::LeafNode LeafNode;
    //typedef typename Octree<int>::LeafNode LeafNode;
    //typedef typename std::iterator_traits<Iterator>::value_type Point;

    // Add the voxels into the octree.
    int index = 0;
    sampelSize = 0;
    for (int s : random_seq) {
        double px = points[3*s+0];
        double py = points[3*s+1];
        double pz = points[3*s+2];
        int x = (px - box.x_min()) / resolution;
        int y = (py - box.y_min()) / resolution;
        int z = (pz - box.z_min()) / resolution;
        x = Clamp(x, 0, size1 - 1);
        y = Clamp(y, 0, size2 - 1);
        z = Clamp(z, 0, size3 - 1);

        std::pair<LeafNode*, bool> pair = octree.Insert(x, y, z, true);
        //std::pair<LeafNode*, bool> pair = octree.Insert(x, y, z, index);
        if (pair.second) {
            //sampling->push_back(s);
            sampelSize ++;
            index ++;
        }
    }

    // for(int i = 0 ; i < n ; i ++){
    //     const Point& p = first[i];
    //     int x = (p.x - box.x_min()) / resolution;
    //     int y = (p.y - box.y_min()) / resolution;
    //     int z = (p.z - box.z_min()) / resolution;
    //     x = Clamp(x, 0, size1 - 1);
    //     y = Clamp(y, 0, size2 - 1);
    //     z = Clamp(z, 0, size3 - 1);  

    //     LeafNode* node = octree.Find(x,y,z);
    //     printf("i %d --> %d," , i , node->data());      
    // }
    // printf("\n");
}

inline void GridSampleLabel(double* points, const int& pnum, double resolution,
                int& sampleSize , int* labels) {
    assert(resolution > 0.0);
    // assert(sampling);

    // sampling->clear();

    int n = pnum;//CountElements(first, last);
    assert(n > 0);

    //Array<int> random_seq(n);
    std::vector<int> random_seq(n);
    for (int i = 0; i < n; ++i) {
        random_seq[i] = i;
    }
    std::mt19937 random;
    std::shuffle(random_seq.begin(), random_seq.end(), random);

    RBox3D box(points , pnum);
    assert(box.x_length() / resolution < INT_MAX);
    assert(box.y_length() / resolution < INT_MAX);
    assert(box.z_length() / resolution < INT_MAX);
    int size1 = box.x_length() / resolution + 1;
    int size2 = box.y_length() / resolution + 1;
    int size3 = box.z_length() / resolution + 1;

    //Octree<bool> octree(size1, size2, size3);
    Octree<int> octree(size1, size2, size3);
    //typedef typename Octree<bool>::LeafNode LeafNode;
    typedef typename Octree<int>::LeafNode LeafNode;
    //typedef typename std::iterator_traits<Iterator>::value_type Point;

    // Add the voxels into the octree.
    int index = 0;
    sampleSize = 0;
    for (int s : random_seq) {
        //const Point& p = first[s];
        double px = points[3*s+0];
        double py = points[3*s+1];
        double pz = points[3*s+2];
        int x = (px - box.x_min()) / resolution;
        int y = (py - box.y_min()) / resolution;
        int z = (pz - box.z_min()) / resolution;
        x = Clamp(x, 0, size1 - 1);
        y = Clamp(y, 0, size2 - 1);
        z = Clamp(z, 0, size3 - 1);

        //std::pair<LeafNode*, bool> pair = octree.Insert(x, y, z, true);
        std::pair<LeafNode*, bool> pair = octree.Insert(x, y, z, index);
        if (pair.second) {
            //sampling->push_back(s);
            sampleSize ++;
            index ++;
        }
    }

    //labels.resize(n);
    labels = new int[n];
    for(int i = 0 ; i < n ; i ++){
        double px = points[3*i+0];
        double py = points[3*i+1];
        double pz = points[3*i+2];
        int x = (px - box.x_min()) / resolution;
        int y = (py - box.y_min()) / resolution;
        int z = (pz - box.z_min()) / resolution;
        x = Clamp(x, 0, size1 - 1);
        y = Clamp(y, 0, size2 - 1);
        z = Clamp(z, 0, size3 - 1);  

        LeafNode* node = octree.Find(x,y,z);
        labels[i] = node->data();
        //printf("i %d --> %d," , i , node->data());      
    }
    //printf("\n");
}

} // namespace cl

#endif // GEOMETRY_POINT_CLOUD_GRID_SAMPLE_H_
