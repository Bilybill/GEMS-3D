// include/mesh3d.h
#ifndef MESH3D_H
#define MESH3D_H

#include <vector>

// 定义 3D 结构化网格 (Hexahedral Mesh for Edge Elements)
struct HostMesh3D {
    int NX; // 网格在X方向的划分数
    int NY; // 网格在Y方向的划分数
    int NZ; // 网格在Z方向的划分数
    
    int NE;     // 总单元数 (NX * NY * NZ)
    int NP;     // 总节点数 ((NX+1) * (NY+1) * (NZ+1))
    int NEdges; // 总边缘数 (也就是全局方程组的未知数个数 / 自由度)
    
    // 💡 升级为张量网格：存储每个网格的真实尺寸
    std::vector<double> hx, hy, hz; 
    
    // ME_Edges: 单元到边缘的映射，长度为 12 * NE
    std::vector<int> ME_Edges; 
    
    // ME_Nodes: 单元到节点的映射，长度为 8 * NE
    std::vector<int> ME_Nodes; 
};

HostMesh3D generateMesh3D(int NX, int NY, int NZ, 
                          const std::vector<double>& hx, 
                          const std::vector<double>& hy, 
                          const std::vector<double>& hz);

#endif // MESH3D_H