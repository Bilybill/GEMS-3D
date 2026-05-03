// src/mesh3d.cpp
#include "../include/mesh3d.h"
#include <iostream>

HostMesh3D generateMesh3D(int NX, int NY, int NZ, 
                          const std::vector<double>& hx, 
                          const std::vector<double>& hy, 
                          const std::vector<double>& hz) {
    HostMesh3D mesh;
    mesh.NX = NX; mesh.NY = NY; mesh.NZ = NZ;
    mesh.hx = hx; mesh.hy = hy; mesh.hz = hz;

    mesh.NE = NX * NY * NZ;
    mesh.NP = (NX + 1) * (NY + 1) * (NZ + 1);

    int nEx = NX * (NY + 1) * (NZ + 1); 
    int nEy = (NX + 1) * NY * (NZ + 1); 
    int nEz = (NX + 1) * (NY + 1) * NZ; 
    mesh.NEdges = nEx + nEy + nEz;

    mesh.ME_Edges.resize(12 * mesh.NE, 0);
    mesh.ME_Nodes.resize(8 * mesh.NE, 0);

    int offset_X = 0;
    int offset_Y = nEx;
    int offset_Z = nEx + nEy;

    auto getNodeId = [&](int i, int j, int k) {
        return i + j * (NX + 1) + k * (NX + 1) * (NY + 1);
    };
    auto getEdgeXId = [&](int i, int j, int k) { 
        return offset_X + i + j * NX + k * NX * (NY + 1);
    };
    auto getEdgeYId = [&](int i, int j, int k) { 
        return offset_Y + i + j * (NX + 1) + k * (NX + 1) * NY;
    };
    auto getEdgeZId = [&](int i, int j, int k) { 
        return offset_Z + i + j * (NX + 1) + k * (NX + 1) * (NY + 1);
    };

    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                int elemIdx = i + j * NX + k * NX * NY;

                mesh.ME_Nodes[8 * elemIdx + 0] = getNodeId(i,   j,   k);
                mesh.ME_Nodes[8 * elemIdx + 1] = getNodeId(i+1, j,   k);
                mesh.ME_Nodes[8 * elemIdx + 2] = getNodeId(i+1, j+1, k);
                mesh.ME_Nodes[8 * elemIdx + 3] = getNodeId(i,   j+1, k);
                mesh.ME_Nodes[8 * elemIdx + 4] = getNodeId(i,   j,   k+1);
                mesh.ME_Nodes[8 * elemIdx + 5] = getNodeId(i+1, j,   k+1);
                mesh.ME_Nodes[8 * elemIdx + 6] = getNodeId(i+1, j+1, k+1);
                mesh.ME_Nodes[8 * elemIdx + 7] = getNodeId(i,   j+1, k+1);

                mesh.ME_Edges[12 * elemIdx + 0]  = getEdgeXId(i,   j,   k);
                mesh.ME_Edges[12 * elemIdx + 1]  = getEdgeYId(i+1, j,   k);
                mesh.ME_Edges[12 * elemIdx + 2]  = getEdgeXId(i,   j+1, k);
                mesh.ME_Edges[12 * elemIdx + 3]  = getEdgeYId(i,   j,   k);
                mesh.ME_Edges[12 * elemIdx + 4]  = getEdgeXId(i,   j,   k+1);
                mesh.ME_Edges[12 * elemIdx + 5]  = getEdgeYId(i+1, j,   k+1);
                mesh.ME_Edges[12 * elemIdx + 6]  = getEdgeXId(i,   j+1, k+1);
                mesh.ME_Edges[12 * elemIdx + 7]  = getEdgeYId(i,   j,   k+1);
                mesh.ME_Edges[12 * elemIdx + 8]  = getEdgeZId(i,   j,   k);
                mesh.ME_Edges[12 * elemIdx + 9]  = getEdgeZId(i+1, j,   k);
                mesh.ME_Edges[12 * elemIdx + 10] = getEdgeZId(i+1, j+1, k);
                mesh.ME_Edges[12 * elemIdx + 11] = getEdgeZId(i,   j+1, k);
            }
        }
    }
    return mesh;
}