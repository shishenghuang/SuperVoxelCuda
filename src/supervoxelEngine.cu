


#include "SuperVoxelEngine.h"

#include "../../Utils/ITMCUDAUtils.h"
#include "../../../ORUtils/CUDADefines.h"
#include "../../Objects/Scene/ITMRepresentationAccess.h"
#include "../Meshing/Shared/ITMMeshingEngine_Shared.h"


#include <sys/time.h>

#include <unistd.h>
#include <mutex>
#include <set>
#include <thread>
// #define VCCS

using namespace ITMLib;
using namespace std;
using namespace cl;
using namespace geometry;
using namespace point_cloud;

#define N 100000000
#define assert_thread assert(mutexSV[i] == ti + 1); \
                assert(mutexSV[j] == ti + 1);
#define assert_threadi assert(mutexSV[i] == ti + 1);

__global__ void cluster(int nsv, int n_thread, int n_points,
                            int *superVoxel,
                            Vector3f *involvedVoxelPos_device,
                            Vector3f *involvedVoxelColor_device,
                            Vector3f *involvedVoxelNormal_device,
                            int *mutexSV,
                            int *fa,
                            int *head,
                            int *next,
                            int *to,
                            int *queue_all,
                            int *visited_all,
                            int *stack_all,
                            int *adjacent_all,
                            int *sizes,
                            double resolution,
                            double lambda
                            )
{
    int ti = blockIdx.x + gridDim.x * blockIdx.y;
    int lb = (nsv + n_thread - 1) / n_thread * ti;
    int ub = (nsv + n_thread - 1) / n_thread * (ti + 1);
    if (ub > nsv) ub = nsv;
    int *queue = queue_all + n_points * ti;
    int *stack = stack_all + n_points * ti;
    int *visited = visited_all + n_points * ti;
    int *adjacent = adjacent_all + n_points * ti;
    // printf("ublb: %d %d\n", lb, ub);
    // memset(visited, 0, sizeof(int) * n_points);
    int stack_cnt = 0, front, back, adjacent_cnt;
    for (int i_ = lb; i_ < ub; i_++) {

        int i = superVoxel[i_];
        // printf("here %d\n", i);
        // int check = atomicAdd(mutexSV + i, 1);
        int check = atomicCAS(mutexSV + i, 0, ti + 1);
        if (check) {
            // atomicAdd(mutexSV + i, -1);
            continue;
        }
        // printf("here\n");
        if (head[i] == -1) {
            mutexSV[i] = 0;
            // atomicAdd(mutexSV + i, -1);
            continue;
        }
        // printf("get lock of %d\n", i);

        if (fa[i] != i) { // don't know why
            // atomicAdd(mutexSV + i, -1);
            // mutexSV[i] = 0;
            // continue;
            printf("fai, i, head[i]: %d, %d, %d\n", fa[i], i, head[i]);
            assert_threadi;
            while(head[i] != -1);
            assert_threadi;
            printf("fai, i, head[i]: %d, %d, %d\n", fa[i], i, head[i]);
        }

        assert_threadi;
        assert(fa[i] == i);

        // printf("start0 %d %d\n", i, ub);

        visited[i] = 1;
        
        assert(i >= 0 && i < n_points);

        front = 0; back = 1;
        queue[front++] = i;

        int deadcnt = 0;
        for (int j_ = head[i]; j_ != -1; j_ = next[j_]) {
            if (deadcnt != -1 && deadcnt++ > 10000) {
                printf("dead7 %d\n", i);
                deadcnt = -1;
            }

            int x = to[j_];
            assert(stack_cnt == 0);
            stack[++stack_cnt] = x;

            assert_threadi;
            while (fa[x] != x) {
                stack[++stack_cnt] = fa[x];
                if (stack_cnt >= n_points) {
                    printf("%d %d\n", stack_cnt, n_points);
                }
                assert(stack_cnt < n_points);
                x = fa[x];
            }

            stack_cnt = 0; // while (stack_cnt) fa[stack[stack_cnt--]] = x;
            int j = x;
            if (!visited[j]) {
                visited[j] = 1;
                assert(j >= 0 && j < n_points);
                queue[back++] = j;
                assert(back < n_points);
            }
        }
        adjacent_cnt = 0;
        // printf("start %d\n", i);
        assert(head[i] != -1);
        while (front < back) {
            assert(head[i] != -1);
            int j = queue[front++];
            int x = j;
            assert(stack_cnt == 0);
            stack[++stack_cnt] = x;
            while (fa[x] != x) {
                stack[++stack_cnt] = fa[x];
                assert(stack_cnt < n_points);
                x = fa[x];
            }
            stack_cnt = 0; // while (stack_cnt) fa[stack[stack_cnt--]] = x;
            j = x;
            // visited[j] = true;
            int check = atomicCAS(mutexSV + j, 0, ti + 1);
            // int check = atomicAdd(mutexSV + j, 1);
            if (check) {
                // atomicAdd(mutexSV + j, -1);
                adjacent[++adjacent_cnt] = j;
                assert(adjacent_cnt < n_points);
                continue;
            }

            x = j;
            assert(stack_cnt == 0);
            stack[++stack_cnt] = x;
            while (fa[x] != x) {
                stack[++stack_cnt] = fa[x];
                assert(stack_cnt < n_points);
                x = fa[x];
            }
            stack_cnt = 0; // while (stack_cnt) fa[stack[stack_cnt--]] = x;

            if (x != j) {
                // atomicAdd(mutexSV + j, -1);
                mutexSV[j] = 0;
                adjacent[++adjacent_cnt] = x;
                assert(adjacent_cnt < n_points);
                continue;
            }
            
            Vector3f v1 = (involvedVoxelPos_device[i] - involvedVoxelPos_device[j]), v2 = (involvedVoxelColor_device[i] - involvedVoxelColor_device[j]);
            double l1 = 1.0 - fabs(involvedVoxelNormal_device[i][0] * involvedVoxelNormal_device[j][0]
                                                + involvedVoxelNormal_device[i][1] * involvedVoxelNormal_device[j][1]
                                                + involvedVoxelNormal_device[i][2] * involvedVoxelNormal_device[j][2]);
            double l2 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]) / resolution * 0.2;
            double l3 = sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]) * 0.002;
            double loss = (sizes[j] + 1) * (l1 + l2 + l3);

            double improvement = lambda - loss;

            // printf("loss: %lf %lf %lf %d\n", l1, l2, l3, sizes[j]);
            
            bool toprint = false;
            int tmp_headj;
            if (improvement > 0.0) {
                tmp_headj = head[j];

                assert_thread;
                assert(head[i] != -1);
                assert(head[i] != -1);
                head[j] = -1;
                assert_thread;
                assert(head[i] != -1);
                assert(head[i] != -1);
                assert_thread;
                assert(fa[i] == i);
                assert(fa[j] == j);
                assert(i != j);
                fa[j] = i;
                assert(j >= 0 && j < n_points);
                sizes[i] += sizes[j] + 1;
                int last_next;
                deadcnt = 0;
                // FILE *file = fopen("link log.txt", "a");
                // printf("link %d(j)  to %d(i)\n", j, i);
                // fclose(file);



                for (int k_ = tmp_headj; k_ != -1; last_next = k_, k_ = next[k_]) {
                    if (deadcnt != -1 && deadcnt++ > 100000) {
                        printf("dead3 %d %d\n", j, k_);
                        deadcnt = -1;
                    }
                    int x = to[k_];
                    assert(stack_cnt == 0);
                    stack[++stack_cnt] = x;
                    int deadcnt2 = 0;
                    while (fa[x] != x) {
                        if (deadcnt2++ > 10000) printf("dead4 %d %d\n", front, back);
                        stack[++stack_cnt] = fa[x];
                        assert(stack_cnt < n_points);
                        x = fa[x];
                    }
                    stack_cnt = 0; // while (stack_cnt) fa[stack[stack_cnt--]] = x;
                    int k = x;
                    if (!visited[k]) {
                        visited[k] = true;
                        assert(k >= 0 && k < n_points);
                        queue[back++] = k;
                        assert(back < n_points);
                    }
                }
                assert_thread;
                // printf("prev head[%d(j)]: %d\n", j, head[j]);
                if (tmp_headj != -1) {
                    toprint = true;
                    int tmp = head[i];
                    // head[i] = head[j];
                    // next[last_next] = tmp;
                    // head[j] = -1;
                    assert(j >= 0 && j < n_points);
                    deadcnt = 0;
                    for (int k_ = tmp; k_ != -1; k_ = next[k_]) {
                        if (deadcnt != -1 && deadcnt++ > 1000000) {
                            deadcnt = -1;
                            printf("dead5 %d\n", i);
                        }
                    }
                    deadcnt = 0;
                    for (int k_ = head[i]; k_ != -1; k_ = next[k_]) {
                        if (deadcnt != -1 && deadcnt++ > 1000000) {
                            deadcnt = -1;
                            printf("dead6 %d\n", i);
                        }
                    }
                }
                assert_thread;
                // printf("head[%d(j)]: %d\n", j, head[j]);
                // if (--number_of_supervoxels <= n_supervoxels) break;

            } else {
                adjacent[++adjacent_cnt] = j;
                assert(adjacent_cnt < n_points);
            }
            // if (toprint) printf("release lock of %d\n", j);
            assert_thread;
            assert(head[i] != -1);

            // atomicAdd(mutexSV + j, -1);
            mutexSV[j] = 0;
        }
        // printf("end %d\n", i);

        int p = head[i];
        for (int ii = 1; ii <= adjacent_cnt; ii++) {
            to[p] = adjacent[ii];
            if (!(p >= 0 && p < N)) {
                printf("p: %d\n", p);
            }
            assert(p >= 0 && p < N);
            if (next[p] == -1) break;
            p = next[p];
        }
        next[p] = -1;
        assert(p >= 0 && p < N);

        // check

        for (int j = 0; j < back; ++j) {
            visited[queue[j]] = 0;
        }
        
        // atomicAdd(mutexSV + i, -1);
        mutexSV[i] = 0;

        // printf("end0 %d\n", i);

    }
    printf("end of thread\n");
}


template <typename Point, class Metric>
bool SupervoxelSegmentation0(const Array<Point>& points,
                            const Array<Array<int> >& neighbors,
                            int n_supervoxels,
                            const Metric& metric,
                            Array<int>* labels,
                            int *supervoxels_device,
                            int *fa_device,
                            int *head_device,
                            int *next_device,
                            int *to_device,
                            int *sizes_device,
                            Vector3f *involvedVoxelPos_device,
                            Vector3f *involvedVoxelColor_device,
                            Vector3f *involvedVoxelNormal_device,
                            double resolution,
                            int *queue_all_device,
                            int *visited_all_device,
                            int *stack_all_device,
                            int *adjacent_all_device,
                            int *mutexSV_device
                            ) {
    assert(points.size() == neighbors.size());
    assert(n_supervoxels > 0);
    assert(labels);

    if (points.empty()) {
        labels->clear();
        return false;
    }

    printf("start0\n");
    int n_points = points.size();
    int *supervoxels = new int[n_points];
    int *fa = new int[n_points];
    for (int i = 0; i < n_points; i++) fa[i] = supervoxels[i] = i;
    ORcudaSafeCall(cudaMemcpy(fa_device, fa, sizeof(int) * n_points, cudaMemcpyHostToDevice));
    ORcudaSafeCall(cudaMemcpy(supervoxels_device, supervoxels, sizeof(int) * n_points, cudaMemcpyHostToDevice));
    int number_of_supervoxels = n_points;
    ORcudaSafeCall(cudaMemset(visited_all_device, 0, sizeof(int) * N));
    ORcudaSafeCall(cudaMemset(sizes_device, 0, sizeof(int) * n_points));
    ORcudaSafeCall(cudaMemset(mutexSV_device, 0, sizeof(int) * n_points));
    printf("start1\n");

    int *next = new int[N], *to = new int[N];
    int *head = new int[n_points];
    // int *next0 = new int[N], *to0 = new int[N];
    // int *head0 = new int[n_points];
    printf("start2\n");

    int cnt = 0;
    memset(head, -1, n_points * sizeof(int));
    for (int i = 0; i < n_points; ++i) {
        for (int j : neighbors[i]) {
            to[++cnt] = j;
            next[cnt] = head[i];
            head[i] = cnt;
        }
    }
    printf("start3\n");

    ORcudaSafeCall(cudaMemcpy(head_device, head, sizeof(int) * n_points, cudaMemcpyHostToDevice));
    ORcudaSafeCall(cudaMemcpy(next_device, next, sizeof(int) * N, cudaMemcpyHostToDevice));
    ORcudaSafeCall(cudaMemcpy(to_device, to, sizeof(int) * N, cudaMemcpyHostToDevice));
    
    printf("start4\n");

    // Compute the minimum value of lambda.
    Array<double> dis(n_points, DBL_MAX);
    for (int i = 0; i < n_points; ++i) {
        for (int j : neighbors[i]) {
            if (i != j) {
                dis[i] = std::min(dis[i], metric(points[i], points[j]));
            }
        }
    }
    double lambda = std::max(DBL_EPSILON, Median(dis.begin(), dis.end()));

    printf("start5\n");

    // ------------------------------------------------------------------
    // ---------------- Step 1: Find supervoxels. -----------------------

    timeval t_start, t_end;
    gettimeofday( &t_start, NULL);

    // Array<mutex> MutexSV(n_points);
    // mutex MutexEdge;

    int cntit = 0;

    printf("start\n");
    lambda *= 8;
    for (; ; lambda *= 2.0) {
        cntit++;
        if (number_of_supervoxels <= 1) break;
        ORcudaSafeCall(cudaMemcpy(supervoxels_device, supervoxels, sizeof(int) * number_of_supervoxels, cudaMemcpyHostToDevice));

        int n_thread = 32;
        dim3 gridSize(1, n_thread);
        dim3 cudaBlockSize(1, 1, 1);
        cluster<<<gridSize, cudaBlockSize>>>(
                            number_of_supervoxels, n_thread, n_points,
                            supervoxels_device,
                            involvedVoxelPos_device,
                            involvedVoxelColor_device,
                            involvedVoxelNormal_device,
                            mutexSV_device,
                            fa_device,
                            head_device,
                            next_device,
                            to_device,
                            queue_all_device,
                            visited_all_device,
                            stack_all_device,
                            adjacent_all_device,
                            sizes_device,
                            resolution,
                            lambda
        );
        cout << "before check\n" << endl;
        ORcudaKernelCheck;
        cout << "before cpy\n" << endl;
        ORcudaSafeCall(cudaMemcpy(fa, fa_device, sizeof(int) * n_points, cudaMemcpyDeviceToHost));
        cout << "finish" << endl;
        Array<int> stack(n_points);
        int stack_cnt = 0;
        // Update supervoxels.
        int num_prev = number_of_supervoxels;
        number_of_supervoxels = 0;
        for (int i_ = 0; i_ < num_prev; i_++) {
            int i = supervoxels[i_];
            int x = i;
            stack[++stack_cnt] = x;
            while (fa[x] != x) {
                stack[++stack_cnt] = fa[x];
                x = fa[x];
            }
            stack_cnt = 0; // while (stack_cnt) fa[stack[stack_cnt--]] = x;
            if (x == i) {
                supervoxels[number_of_supervoxels++] = i;
            }
        }
        
        if (number_of_supervoxels <= n_supervoxels || cntit > 100) break;
        std::cout << "aaaaaaaa" << number_of_supervoxels << "/" << n_supervoxels << std::endl;
    }

    // Assign the label to each point according to its supervoxel ID.
    labels->resize(n_points);
    Array<int> stack(n_points);
    int stack_cnt = 0;
    for (int i = 0; i < n_points; ++i) {
        int x = i;
        stack[++stack_cnt] = x;
        while (fa[x] != x) {
            stack[++stack_cnt] = fa[x];
            x = fa[x];
        }
        stack_cnt = 0; // while (stack_cnt) fa[stack[stack_cnt--]] = x;
        (*labels)[i] = x;
    }
    std::cout << "step1 finish" << std::endl;

    gettimeofday( &t_end, NULL);
    double delta_t = (t_end.tv_sec-t_start.tv_sec) + 
                    (t_end.tv_usec-t_start.tv_usec)/1000000.0;
    printf("time of step1: %lf\n", delta_t);
    // gettimeofday( &t_start, NULL);


    // ------------------------------------------------------------------
    // ---------------- Step 2: Refine the boundaries. ------------------
    for (int i = 0; i < n_points; ++i) {
        int j = (*labels)[i];
        dis[i] = metric(points[i], points[j]);
    }

    std::queue<int> q;
    Array<bool> in_q(n_points, false);

    for (int i = 0; i < n_points; ++i) {
        for (int j : neighbors[i]) {
            if ((*labels)[i] != (*labels)[j]) {
                if (!in_q[i]) {
                    q.push(i);
                    in_q[i] = true;
                }
                if (!in_q[j]) {
                    q.push(j);
                    in_q[j] = true;
                }
            }
        }
    }

    while (!q.empty()) {
        int i = q.front();
        q.pop();
        in_q[i] = false;

        bool change = false;
        for (int j : neighbors[i]) {
            int a = (*labels)[i];
            int b = (*labels)[j];
            if (a == b) continue;
            double d = metric(points[i], points[b]);
            if (d < dis[i]) {
                (*labels)[i] = b;
                dis[i] = d;
                change = true;
            }
        }

        if (change) {
            for (int j : neighbors[i]) {
                if ((*labels)[i] != (*labels)[j]) {
                    if (!in_q[j]) {
                        q.push(j);
                        in_q[j] = true;
                    }
                }
            }
        }
    }
    std::cout << "step2 finish" << std::endl;

    gettimeofday( &t_end, NULL);
    delta_t = (t_end.tv_sec-t_start.tv_sec) + 
                    (t_end.tv_usec-t_start.tv_usec)/1000000.0;
    printf("time of step2: %lf\n", delta_t);

    // ------------------------------------------------------------------
    // ---------------- Step 3: Relabel the supervoxels. ----------------
    Array<int> map(n_points);
    for (int i = 0; i < number_of_supervoxels; ++i) {
        map[supervoxels[i]] = i;
    }
    for (int i = 0; i < n_points; ++i) {
        (*labels)[i] = map[(*labels)[i]];
    }

    delete[] fa;
    // delete[] sizes0;
    delete[] head;
    delete[] next;
    delete[] to;
    // delete[] head0;
    // delete[] next0;
    // delete[] to0;
    return true;
}





#define CHECKSDF(x, y, z, i)     localBlockLocation = blockLocation + Vector3i(x, y, z); \
    sdf[i] = TVoxel::valueToFloat(readVoxel(localVBA, hashTable, localBlockLocation, vmIndex).sdf); \
    if (!vmIndex || sdf[i] == 1.0f) return false;

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline bool pointOnSurface(THREADPTR(float) *sdf, Vector3i blockLocation,
        const CONSTPTR(TVoxel) *localVBA, const CONSTPTR(ITMHashEntry) *hashTable)
{
    int vmIndex; Vector3i localBlockLocation;

    CHECKSDF(0, 0, 0, 0);
    CHECKSDF(1, 0, 0, 1);
    CHECKSDF(1, 1, 0, 2);
    CHECKSDF(0, 1, 0, 3);
    CHECKSDF(0, 0, 1, 4);
    CHECKSDF(1, 0, 1, 5);
    CHECKSDF(1, 1, 1, 6);
    CHECKSDF(0, 1, 1, 7);

    int cubeIndex = 0;
    if (sdf[0] < 0) cubeIndex |= 1; if (sdf[1] < 0) cubeIndex |= 2;
    if (sdf[2] < 0) cubeIndex |= 4; if (sdf[3] < 0) cubeIndex |= 8;
    if (sdf[4] < 0) cubeIndex |= 16; if (sdf[5] < 0) cubeIndex |= 32;
    if (sdf[6] < 0) cubeIndex |= 64; if (sdf[7] < 0) cubeIndex |= 128;

    return edgeTable[cubeIndex] != 0;

}

#define DEFINE_DIRS const Vector3i dirs[] = {Vector3i(0, 0, 1), \
                    Vector3i(0, 0, -1), \
                    Vector3i(0, 1, 0), \
                    Vector3i(0, -1, 0), \
                    Vector3i(1, 0, 0), \
                    Vector3i(-1, 0, 0)};

template<class TVoxel>
__global__ void findNewBlocks(unsigned int* newBlock, unsigned int *labelTableAll, bool countingLabel, 
        const ITMHashEntry *hashTable, int noTotalEntries,
        unsigned int* numNewBlocks, TVoxel *localVBA, float factor)
{
    int entryId = blockIdx.x + gridDim.x * blockIdx.y;
    if (entryId > noTotalEntries - 1) return;

    const ITMHashEntry &currentHashEntry = hashTable[entryId];
    if (currentHashEntry.ptr >= 0) {

        TVoxel& reprVoxel = localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3 +
                                            threadIdx.x +
                                            threadIdx.y * SDF_BLOCK_SIZE +
                                            threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
        short SVLabel = reprVoxel.SVLabel;
        if (reprVoxel.SVLabelFlag) {
            SVLabel = -1;
        }
        if (countingLabel && SVLabel != -1) {
            atomicAdd(labelTableAll + SVLabel, 1); 
        }

        Vector3i voxelLocation = Vector3i(currentHashEntry.pos.x,
                currentHashEntry.pos.y, currentHashEntry.pos.z) * SDF_BLOCK_SIZE +
                        Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

        float sdfVals[8];
        bool isOnSurface = pointOnSurface(sdfVals, voxelLocation, localVBA, hashTable);
        if (isOnSurface) 
        {
            if (SVLabel == -1) {
                int voId = atomicAdd(newBlock + currentHashEntry.ptr, 1);
                if (voId > 0) return;
                atomicAdd(numNewBlocks, 1);
            }
        }

    }
}


template<class TVoxel>
__global__ void findInvolvedLabels(unsigned int *labelTable,    const unsigned int* newBlock,
        const ITMHashEntry *hashTable, int noTotalEntries,
        const TVoxel *localVBA, float factor, unsigned int *numInvolvedLabels)
{
    int entryId = blockIdx.x + gridDim.x * blockIdx.y;
    if (entryId > noTotalEntries - 1) return;
    if (*numInvolvedLabels > 200) return;

    const ITMHashEntry &currentHashEntry = hashTable[entryId];
    DEFINE_DIRS;

    if (currentHashEntry.ptr >= 0) {


        Vector3i voxelLocation = Vector3i(currentHashEntry.pos.x,
                currentHashEntry.pos.y, currentHashEntry.pos.z) * SDF_BLOCK_SIZE +
                        Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);
        float sdfVals[8];
        bool isOnSurface = pointOnSurface(sdfVals, voxelLocation, localVBA, hashTable);
        
        const TVoxel& reprVoxel = localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3 +
                                            threadIdx.x +
                                            threadIdx.y * SDF_BLOCK_SIZE +
                                            threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
        short SVLabel = reprVoxel.SVLabel;

        if (isOnSurface) 
        {
            bool flag = false;
            for (int step = 0; step < 3; step++) {
                for (int i = 0; i < 6; i++) {
                    Vector3i blockPos = Vector3i(currentHashEntry.pos.x, currentHashEntry.pos.y, currentHashEntry.pos.z) + dirs[i] * step;
                    unsigned int hashIdx = hashIndex(blockPos);
                    ITMHashEntry hashEntry = hashTable[hashIdx];
                    if (!IS_EQUAL3(hashEntry.pos, blockPos)) {
                        if (hashEntry.ptr >= -1)
                        {
                            bool isFound = false;
                            while (hashEntry.offset >= 1)
                            {
                                hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
                                hashEntry = hashTable[hashIdx];

                                if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= -1)
                                {
                                    isFound = true;
                                    break;
                                }
                            }
                            if (!isFound) continue;
                        }
                        else continue;
                    }
                    if (newBlock[hashEntry.ptr]) {
                        flag = true;
                        break;
                    }
                }
                if (flag) break;
            }
            if (!flag) return;
            if (SVLabel != -1) {
                int ck = atomicAdd(labelTable + SVLabel, 1);
                if (ck == 0) atomicAdd(numInvolvedLabels, 1);
            }
        }

    }
}

template<class TVoxel>
__global__ void findInvolvedVoxels(Vector3f *involvedVoxelPos, Vector3f *involvedVoxelColor, const unsigned int* labelTable,
        Vector3i* blockPos, Vector3i *localPos,
        const ITMHashEntry *hashTable, int noTotalEntries,
        unsigned int* numInvolvedVoxels, TVoxel *localVBA, float factor)
{
    int entryId = blockIdx.x + gridDim.x * blockIdx.y;
    if (entryId > noTotalEntries - 1) return;

    const ITMHashEntry &currentHashEntry = hashTable[entryId];

    if (currentHashEntry.ptr >= 0) {

        Vector3i voxelLocation = Vector3i(currentHashEntry.pos.x,
                currentHashEntry.pos.y, currentHashEntry.pos.z) * SDF_BLOCK_SIZE +
                        Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);
        float sdfVals[8];
        bool isOnSurface = pointOnSurface(sdfVals, voxelLocation, localVBA, hashTable);
        TVoxel& reprVoxel = localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3 +
                                            threadIdx.x +
                                            threadIdx.y * SDF_BLOCK_SIZE +
                                            threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
        if (isOnSurface)
        {

            short SVLabel = reprVoxel.SVLabel;
            if (reprVoxel.SVLabelFlag) {
                SVLabel = -1;
                reprVoxel.SVLabelFlag = 0;
            }
            if (SVLabel == -1 || labelTable[SVLabel]) {
                int Id = atomicAdd(numInvolvedVoxels, 1);
                if (Id >= MAX_N_VOXEL_SV - 1) {
                    atomicAdd(numInvolvedVoxels, -1);
                    return;
                }
                involvedVoxelPos[Id] = Vector3f(voxelLocation.x * factor,
                                                   voxelLocation.y * factor,
                                                   voxelLocation.z * factor);
                involvedVoxelColor[Id] = Vector3f(reprVoxel.clr.x, reprVoxel.clr.y, reprVoxel.clr.z);
                blockPos[Id] = Vector3i(currentHashEntry.pos.x,
                    currentHashEntry.pos.y, currentHashEntry.pos.z);
                localPos[Id] = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);


            }
        }
        else {
            reprVoxel.SVLabel = -1;
        }

    }
}


template <class TVoxel>
__global__ void updateSVLabel(const Vector3i *blockPos, const Vector3i *localPos, const short* SVLabel, int numInvolvedVoxels,
    TVoxel *localVBA, const ITMHashEntry *hashTable)
{
    int entryId = blockIdx.x + gridDim.x * blockIdx.y;
    if (entryId > numInvolvedVoxels - 1) return;
    unsigned int hashIdx = hashIndex(blockPos[entryId]);
    ITMHashEntry hashEntry = hashTable[hashIdx];

    if (!IS_EQUAL3(hashEntry.pos, blockPos[entryId])) {
        if (hashEntry.ptr >= -1)
        {
            bool isFound = false;
            while (hashEntry.offset >= 1)
            {
                hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
                hashEntry = hashTable[hashIdx];
                if (IS_EQUAL3(hashEntry.pos, blockPos[entryId]) && hashEntry.ptr >= -1)
                {
                    isFound = true;
                    break;
                }
            }
        }
    } else if (hashEntry.ptr < 0) {
        return;
    }
    TVoxel& tVoxel = localVBA[hashEntry.ptr * SDF_BLOCK_SIZE3 +
                              localPos[entryId][0] +
                              localPos[entryId][1] * SDF_BLOCK_SIZE +
                              localPos[entryId][2] * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
    tVoxel.SVLabel = SVLabel[entryId];
}

template<class TVoxel>
__global__ void writeSVLabelVis(
        const ITMHashEntry *hashTable, int noTotalEntries,
        TVoxel *localVBA)
{
    int entryId = blockIdx.x + gridDim.x * blockIdx.y;
    if (entryId > noTotalEntries - 1) return;
    const ITMHashEntry &currentHashEntry = hashTable[entryId];

    DEFINE_DIRS;
    if (currentHashEntry.ptr >= 0) {
        Vector3i voxelLocation = Vector3i(currentHashEntry.pos.x,
                currentHashEntry.pos.y, currentHashEntry.pos.z) * SDF_BLOCK_SIZE +
                        Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

        TVoxel& voxel = localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3 +
                                            threadIdx.x +
                                            threadIdx.y * SDF_BLOCK_SIZE +
                                            threadIdx.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
        short SVLabel = voxel.SVLabel;
        if (SVLabel != -1) {
            voxel.SVLabelForVis = voxel.SVLabel;
        }
        else {
            bool found = false;
            for (int step = 0; step < 10; step++) {
                for (int i = 0; i < 6; i++) {
                    Vector3i loc_neib = voxelLocation + dirs[i] * step;
                    short lx = (loc_neib[0] % SDF_BLOCK_SIZE + SDF_BLOCK_SIZE) % SDF_BLOCK_SIZE;
                    short ly = (loc_neib[1] % SDF_BLOCK_SIZE + SDF_BLOCK_SIZE) % SDF_BLOCK_SIZE;
                    short lz = (loc_neib[2] % SDF_BLOCK_SIZE + SDF_BLOCK_SIZE) % SDF_BLOCK_SIZE;

                    short bx = (loc_neib[0] - lx) / SDF_BLOCK_SIZE;
                    short by = (loc_neib[1] - ly) / SDF_BLOCK_SIZE;
                    short bz = (loc_neib[2] - lz) / SDF_BLOCK_SIZE;
                    Vector3i blockPos = Vector3i(bx, by, bz);

                    unsigned int hashIdx = hashIndex(blockPos);
                    ITMHashEntry hashEntry = hashTable[hashIdx];

                    if (!IS_EQUAL3(hashEntry.pos, blockPos)) {
                        if (hashEntry.ptr >= -1)
                        {
                            bool isFound = false;
                            while (hashEntry.offset >= 1)
                            {
                                hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
                                hashEntry = hashTable[hashIdx];

                                if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= -1)
                                {
                                    isFound = true;
                                    break;
                                }
                            }

                            if (!isFound) continue;
                        }
                        else continue;
                    }
                    if (hashEntry.ptr < 0) continue;
                    TVoxel& neibVoxel = localVBA[hashEntry.ptr * SDF_BLOCK_SIZE3 +
                                lx +
                                ly * SDF_BLOCK_SIZE +
                                lz * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];

                    if (neibVoxel.SVLabel > -1) {
                        voxel.SVLabelForVis = neibVoxel.SVLabel;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }
            if (!found) voxel.SVLabelForVis = 0;
        }

    }
}




template<class TVoxel>
SuperVoxelEngine_CUDA<TVoxel>::SuperVoxelEngine_CUDA() {
    
    ORcudaSafeCall(cudaMalloc((void**)&newBlock_device, SDF_LOCAL_BLOCK_NUM * sizeof(unsigned int)));
    ORcudaSafeCall(cudaMalloc((void**)&labelTable_device, MAX_N_VOXEL_SV * sizeof(unsigned int)));
    ORcudaSafeCall(cudaMalloc((void**)&labelTableAll_device, MAX_N_VOXEL_SV * sizeof(unsigned int)));
    ORcudaSafeCall(cudaMalloc((void**)&involvedVoxelPos_device, MAX_N_VOXEL_SV * sizeof(Vector3f)));
    ORcudaSafeCall(cudaMalloc((void**)&involvedVoxelColor_device, MAX_N_VOXEL_SV * sizeof(Vector3f)));
    ORcudaSafeCall(cudaMalloc((void**)&involvedVoxelNormal_device, MAX_N_VOXEL_SV * sizeof(Vector3f)));
    ORcudaSafeCall(cudaMalloc((void**)&involvedVoxelPtr_device, MAX_N_VOXEL_SV * sizeof(unsigned int)));
    ORcudaSafeCall(cudaMalloc((void**)&localPos_device, MAX_N_VOXEL_SV * sizeof(Vector3i)));
    ORcudaSafeCall(cudaMalloc((void**)&blockPos_device, MAX_N_VOXEL_SV * sizeof(Vector3i)));
    ORcudaSafeCall(cudaMalloc((void**)&SVLabel_device, MAX_N_VOXEL_SV * sizeof(short)));
    
    ORcudaSafeCall(cudaMalloc((void**)&fa_device, MAX_N_VOXEL_SV * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&head_device, MAX_N_VOXEL_SV * sizeof(int)));
    // ORcudaSafeCall(cudaMalloc((void**)&head0_device, MAX_N_VOXEL_SV * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&sizes_device, MAX_N_VOXEL_SV * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&next_device, N * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&to_device, N * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&supervoxels_device, MAX_N_VOXEL_SV * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&mutexSV_device, MAX_N_VOXEL_SV * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&queue_all_device, N * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&visited_all_device, N * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&stack_all_device, N * sizeof(int)));
    ORcudaSafeCall(cudaMalloc((void**)&adjacent_all_device, N * sizeof(int)));
    // ORcudaSafeCall(cudaMalloc((void**)&to0_device, N * sizeof(int)));
    // ORcudaSafeCall(cudaMalloc((void**)&next0_device, N * sizeof(int)));



}

template<class TVoxel>
SuperVoxelEngine_CUDA<TVoxel>::~SuperVoxelEngine_CUDA() {

    ORcudaSafeCall(cudaFree(newBlock_device));
    ORcudaSafeCall(cudaFree(labelTable_device));
    ORcudaSafeCall(cudaFree(labelTableAll_device));
    ORcudaSafeCall(cudaFree(involvedVoxelPos_device));
    ORcudaSafeCall(cudaFree(involvedVoxelColor_device));
    ORcudaSafeCall(cudaFree(involvedVoxelPtr_device));
    ORcudaSafeCall(cudaFree(involvedVoxelNormal_device));
    ORcudaSafeCall(cudaFree(localPos_device));
    ORcudaSafeCall(cudaFree(blockPos_device));
    ORcudaSafeCall(cudaFree(SVLabel_device));

    ORcudaSafeCall(cudaFree(fa_device));
    ORcudaSafeCall(cudaFree(head_device));
    // ORcudaSafeCall(cudaFree(head0_device));
    ORcudaSafeCall(cudaFree(next_device));
    // ORcudaSafeCall(cudaFree(next0_device));
    // ORcudaSafeCall(cudaFree(to0_device));
    ORcudaSafeCall(cudaFree(to_device));
    ORcudaSafeCall(cudaFree(sizes_device));
    ORcudaSafeCall(cudaFree(supervoxels_device));
    ORcudaSafeCall(cudaFree(queue_all_device));
    ORcudaSafeCall(cudaFree(visited_all_device));
    ORcudaSafeCall(cudaFree(stack_all_device));
    ORcudaSafeCall(cudaFree(adjacent_all_device));
    ORcudaSafeCall(cudaFree(mutexSV_device));

    delete[] normalSV;

}

/// Point with Normal.
struct PointWithNormal : cl::RPoint3D {
    PointWithNormal() {}

    cl::RVector3D normal;
    cl::RPoint3D color;
};

/**
 * Metric used in VCCS supervoxel segmentation.
 *
 * Reference:
 *   Rusu, R.B., Cousins, S., 2011. 3d is here: Point cloud library (pcl),
 *   IEEE International Conference on Robotics and Automation, pp. 1–4.
 */
class VCCSMetric {
public:
    explicit VCCSMetric(double resolution)
        : resolution_(resolution) {}

    double operator() (const PointWithNormal& p1,
                       const PointWithNormal& p2) const {
        return 1.0 - std::fabs(p1.normal * p2.normal) +
               cl::geometry::Distance(p1, p2) / resolution_ * 0.2 // 0.4
               + cl::geometry::Distance(p1.color, p2.color) * 0.002; // 0.02
    }

private:
    double resolution_;
};

inline int norm2(Vector3i a) {
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}


#define TIMING(info)     gettimeofday( &t_end, NULL); \
    delta_t = (t_end.tv_sec-t_start.tv_sec) + \
                    (t_end.tv_usec-t_start.tv_usec)/1000000.0; \
    fprintf(SVlog, info, delta_t); \
    gettimeofday( &t_start, NULL);


template<class TVoxel>
void SuperVoxelEngine_CUDA<TVoxel>::performSVSeg(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, std::recursive_timed_mutex &mMutexScene) {
    
    unsigned int numUsedBlocks = 0, numNewBlocks = 0;
    TVoxel *localVBA;
    float factor = scene->sceneParams->voxelSize;
    unsigned int numInvolvedVoxels = 0;
    
    Vector3f* VPos;
    Vector3f* VColor_vx;
    Vector3f* VNormal;
    short* SVLabel;
    unsigned int *labelTableAll, *labelTable;

    FILE *SVlog = fopen("SVlog.txt", "a");
    fprintf(SVlog, "-----------------------------------------------------\n");

    timeval t_start, t_end;
    double delta_t;
    gettimeofday( &t_start, NULL);


    int noTotalEntries;
    {
        std::unique_lock<std::recursive_timed_mutex> lock(mMutexScene);
        
        localVBA = scene->localVBA.GetVoxelBlocks();
        const ITMHashEntry *hashTable = scene->index.GetEntries();
        noTotalEntries = scene->index.noTotalEntries;
        printf("tot: %d -----------------------------------------------------\n", noTotalEntries);

        ORcudaSafeCall(cudaMemset(newBlock_device, 0, sizeof(unsigned int) * SDF_LOCAL_BLOCK_NUM));
        unsigned int* numNewBlocks_device;
        ORcudaSafeCall(cudaMalloc((void**)&numNewBlocks_device, sizeof(unsigned int)));
        ORcudaSafeCall(cudaMemset(numNewBlocks_device, 0, sizeof(unsigned int)));
        ORcudaSafeCall(cudaMemset(labelTableAll_device, 0, sizeof(unsigned int) * MAX_N_VOXEL_SV));

        dim3 gridSize((int)ceil((float)noTotalEntries / 256.0f), 256);
        dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
        findNewBlocks<TVoxel><<<gridSize, cudaBlockSize>>>(
                newBlock_device, labelTableAll_device, true, 
                hashTable, noTotalEntries, numNewBlocks_device, localVBA, factor);
        ORcudaKernelCheck;

        ORcudaSafeCall(cudaMemcpy(&numNewBlocks, numNewBlocks_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // std::cout << "Find new blocks: " << numNewBlocks << std::endl; 
        unsigned int* numInvolvedLabels_device;
        ORcudaSafeCall(cudaMalloc((void**)&numInvolvedLabels_device, sizeof(unsigned int)));
        ORcudaSafeCall(cudaMemset(numInvolvedLabels_device, 0, sizeof(unsigned int)));
        ORcudaSafeCall(cudaMemset(labelTable_device, 0, sizeof(unsigned int) * MAX_N_VOXEL_SV));
        findInvolvedLabels<TVoxel><<<gridSize, cudaBlockSize>>>(
                labelTable_device, newBlock_device,
                hashTable, noTotalEntries, localVBA, factor, numInvolvedLabels_device);
        ORcudaKernelCheck;

        unsigned int numInvolvedLabels;
        ORcudaSafeCall(cudaMemcpy(&numInvolvedLabels, numInvolvedLabels_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // printf("num involved labels: %d\n", numInvolvedLabels);
        
        unsigned int* numInvolvedVoxels_device;
        ORcudaSafeCall(cudaMalloc((void**)&numInvolvedVoxels_device, sizeof(unsigned int)));
        ORcudaSafeCall(cudaMemset(numInvolvedVoxels_device, 0, sizeof(unsigned int)));

        findInvolvedVoxels<TVoxel><<<gridSize, cudaBlockSize>>>(
                involvedVoxelPos_device, involvedVoxelColor_device, labelTable_device, blockPos_device, localPos_device,
                hashTable, noTotalEntries, numInvolvedVoxels_device, localVBA, factor);
        ORcudaKernelCheck;

        ORcudaSafeCall(cudaMemcpy(&numInvolvedVoxels, numInvolvedVoxels_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));

        std::cout << "Find involved voxels: " << numInvolvedVoxels << std::endl;

        if (numInvolvedVoxels == 0) {
            mMutexScene.unlock();
            usleep(5000000);
            return;
        }

        VPos = new Vector3f[numInvolvedVoxels];
        VColor_vx = new Vector3f[numInvolvedVoxels];
        VNormal = new Vector3f[numInvolvedVoxels];

        ORcudaSafeCall(cudaMemcpy(VPos, involvedVoxelPos_device, sizeof(Vector3f) * numInvolvedVoxels, cudaMemcpyDeviceToHost));
        ORcudaSafeCall(cudaMemcpy(VColor_vx, involvedVoxelColor_device, sizeof(Vector3f) * numInvolvedVoxels, cudaMemcpyDeviceToHost));
        
        labelTableAll = new unsigned int[MAX_N_VOXEL_SV];
        ORcudaSafeCall(cudaMemcpy(labelTableAll, labelTableAll_device, sizeof(unsigned int) * MAX_N_VOXEL_SV, cudaMemcpyDeviceToHost));
        labelTable = new unsigned int[MAX_N_VOXEL_SV];
        ORcudaSafeCall(cudaMemcpy(labelTable, labelTable_device, sizeof(unsigned int) * MAX_N_VOXEL_SV, cudaMemcpyDeviceToHost));

        mMutexScene.unlock();
    }



    int maxLabel = 0;
    for (int i = 0; i < MAX_N_VOXEL_SV; i++) {
        if (labelTableAll[i] != 0) maxLabel = i;
    }
    cl::Array<cl::RPoint3D> points;
    points.clear();
    for (int i = 0; i < numInvolvedVoxels; i++) {
        points.emplace_back(VPos[i][0], VPos[i][1], VPos[i][2]);
    }
    int n_points = points.size();


    TIMING("time of traversing scene: %lf\n");
    fprintf(SVlog, "tot points: %d\n", n_points);


    const int k_neighbors = 50;
    const double voxel_resolution = 0.03;
    const double resolution = 0.2;
    VCCSMetric metric(resolution);

    if (k_neighbors < n_points) {

        cl::KDTree<cl::RPoint3D> kdtree;
        kdtree.SwapPoints(&points);

        TIMING("time of  building kdtree: %lf\n");

        // cl::Array<cl::RVector3D> normals(n_points);
        cl::Array<cl::Array<int> > neighbors(n_points);
        
        int n_thread = 10;
        thread threads[30];

        for (int ti = 0; ti < n_thread; ti++)
            threads[ti] = thread([=, &kdtree, &neighbors](){
                int lb = n_points / n_thread * ti;
                int ub = n_points / n_thread * (ti + 1);
                if (ub > n_points) ub = n_points;

                for (int i = lb; i < ub; ++i) {
                    cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
                    kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors,
                                                &neighbors[i]);
                    for (int k = 0; k < k_neighbors; ++k) {
                        neighbor_points[k] = kdtree.points()[neighbors[i][k]];
                    }
                    cl::RVector3D normal0;
                    cl::geometry::point_cloud::PCAEstimateNormal(neighbor_points.begin(),
                                                                neighbor_points.end(),
                                                                &normal0);
                    VNormal[i] = Vector3f(normal0[0], normal0[1], normal0[2]);
                }
            });

        for (int ti = 0; ti < n_thread; ti++)
            threads[ti].join();


        kdtree.SwapPoints(&points);
        cl::Array<PointWithNormal> oriented_points(n_points);
        for (int i = 0; i < n_points; ++i) {
            oriented_points[i].x = points[i].x;
            oriented_points[i].y = points[i].y;
            oriented_points[i].z = points[i].z;
            oriented_points[i].normal.x = VNormal[i][0];
            oriented_points[i].normal.y = VNormal[i][1];
            oriented_points[i].normal.z = VNormal[i][2];

            oriented_points[i].color.x = VColor_vx[i][0];
            oriented_points[i].color.y = VColor_vx[i][1];
            oriented_points[i].color.z = VColor_vx[i][2];
        }

        TIMING("time of computing neighbors: %lf\n");
        

#ifndef VCCS

        ORcudaSafeCall(cudaMemcpy(involvedVoxelNormal_device, VNormal, sizeof(Vector3f) * n_points, cudaMemcpyHostToDevice));

        cl::Array<int> labels;
        
        ///////////// uncomment these
        // Array<int> sampling;
        // GridSample(oriented_points.begin(), oriented_points.end(), resolution, &sampling);

        // TIMING("time of grid sampling: %lf\n");


        // bool ret = SupervoxelSegmentation0(oriented_points, neighbors, sampling.size(), metric, &labels,
        //     supervoxels_device, fa_device, head_device, next_device, to_device, sizes_device,
        //     involvedVoxelPos_device, involvedVoxelColor_device, involvedVoxelNormal_device,
        //     resolution, queue_all_device, visited_all_device, stack_all_device, adjacent_all_device, mutexSV_device);
        ///////////
        
        // //////and comment these
        cl::Array<int> supervoxels;
        bool ret = cl::geometry::point_cloud::SupervoxelSegmentation(oriented_points,
                                                        neighbors,
                                                        resolution,
                                                        metric,
                                                        &supervoxels,
                                                        &labels);
        //////////////// to switch between

        if (!ret) return;
        // int n_supervoxels = supervoxels.size();
        // LOG(INFO) << n_supervoxels << " supervoxels computed.";
#endif



#ifdef VCCS
        cl::VCCSSupervoxel vccs(points.begin(), points.end(),
                            voxel_resolution,
                            resolution);
        cl::Array<int> labels;
        cl::Array<cl::VCCSSupervoxel::Supervoxel> vccs_supervoxels;
        vccs.Segment(&labels, &vccs_supervoxels);
        // printf("size: %d\n", vccs_supervoxels.size());
        // printf("size: %d npoint: %d\n", vccs_knn_supervoxels.size(), n_points);
        if (labels.size() != n_points) return;
#endif

        TIMING("time of computing SV segmentation: %lf\n");

        int startLabel = 0;
        std::map<short, short> mapLabel;
        short *SVLabel = new short[MAX_N_VOXEL_SV];

        for (int i = 0; i < n_points; i++) {
            if (labels[i] == -1) {
                labels[i] = 0;
            }
            if (mapLabel.count(labels[i])) {
                SVLabel[i] = mapLabel[labels[i]];
            }
            else if (labelTableAll[labels[i]] == 0) {
                mapLabel[labels[i]] = labels[i];
                labelTableAll[labels[i]] = 1;
                SVLabel[i] = labels[i];
            } else {
                while (labelTableAll[startLabel]) startLabel++;
                labelTableAll[startLabel] = 1;                
                mapLabel[labels[i]] = startLabel;
                SVLabel[i] = startLabel;
            }
        }
        
        TIMING("for loop 1: %lf\n");


        short *count = new short[MAX_N_LABEL]{0};
        for (int i = 0; i < n_points; i++) {
            if (count[SVLabel[i]] == 0) {
                normalSV[SVLabel[i]] = VNormal[i];
                count[SVLabel[i]] = 1;
            }
            else {
                normalSV[SVLabel[i]] += VNormal[i];
                count[SVLabel[i]]++;
            }
        }
        for (int i = 0; i < n_points; i++) {
            if (count[SVLabel[i]] != 0) {
                normalSV[SVLabel[i]] /= count[SVLabel[i]];
                count[SVLabel[i]] = 0;
            }
        }

        TIMING("for loop 2: %lf\n");
        


        {
            std::unique_lock<std::recursive_timed_mutex> lock(mMutexScene);
            localVBA = scene->localVBA.GetVoxelBlocks();
            const ITMHashEntry *hashTable = scene->index.GetEntries();
            ORcudaSafeCall(cudaMemcpy(SVLabel_device, SVLabel, sizeof(short) * numInvolvedVoxels, cudaMemcpyHostToDevice));
            dim3 gridSize((int)ceil((float)numInvolvedVoxels / 1024.0f), 1024);
            dim3 one(1);
            dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);

            updateSVLabel<TVoxel><<<gridSize, one>>>(
                    blockPos_device, localPos_device, SVLabel_device, numInvolvedVoxels, localVBA, hashTable);
            
            noTotalEntries = scene->index.noTotalEntries;
            gridSize = dim3((int)ceil((float)noTotalEntries / 256.0f), 256);
            writeSVLabelVis<TVoxel><<<gridSize, cudaBlockSize>>>(hashTable, noTotalEntries, localVBA);
            ORcudaKernelCheck;
            mMutexScene.unlock();
        }
    }

    TIMING("time of labeling: %lf\n");
    fclose(SVlog);

    if (SVLabel) delete [] SVLabel;
    if (VPos) delete [] VPos;
    if (VColor_vx) delete [] VColor_vx;
    if (VNormal) delete [] VNormal;
    if (labelTable) delete [] labelTable;
    if (labelTableAll) delete [] labelTableAll;

}