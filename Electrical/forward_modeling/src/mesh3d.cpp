// src/mesh3d.cpp
#include "../include/mesh3d.h"
#include <cmath>
#include <limits>
#include <stdexcept>

HostMesh3D generateMesh3D(int NX, int NY, int NZ,
                          const std::vector<double>& hx,
                          const std::vector<double>& hy,
                          const std::vector<double>& hz) {
    if (NX <= 0 || NY <= 0 || NZ <= 0) {
        throw std::invalid_argument("mesh dimensions must be positive");
    }
    if (hx.size() != static_cast<size_t>(NX)
        || hy.size() != static_cast<size_t>(NY)
        || hz.size() != static_cast<size_t>(NZ)) {
        throw std::invalid_argument("cell-width arrays must match the mesh dimensions");
    }
    auto invalidWidth = [](double width) {
        return !std::isfinite(width) || width <= 0.0;
    };
    for (double width : hx) {
        if (invalidWidth(width)) {
            throw std::invalid_argument("x cell widths must be positive and finite");
        }
    }
    for (double width : hy) {
        if (invalidWidth(width)) {
            throw std::invalid_argument("y cell widths must be positive and finite");
        }
    }
    for (double width : hz) {
        if (invalidWidth(width)) {
            throw std::invalid_argument("z cell widths must be positive and finite");
        }
    }

    constexpr long long maxInt = std::numeric_limits<int>::max();
    long long elementCount = static_cast<long long>(NX) * NY * NZ;
    long long nodeCount =
        static_cast<long long>(NX + 1LL) * (NY + 1LL) * (NZ + 1LL);
    long long xEdgeCount =
        static_cast<long long>(NX) * (NY + 1LL) * (NZ + 1LL);
    long long yEdgeCount =
        static_cast<long long>(NX + 1LL) * NY * (NZ + 1LL);
    long long zEdgeCount =
        static_cast<long long>(NX + 1LL) * (NY + 1LL) * NZ;
    long long edgeCount = xEdgeCount + yEdgeCount + zEdgeCount;
    if (elementCount > maxInt || nodeCount > maxInt || edgeCount > maxInt
        || elementCount > maxInt / 12) {
        throw std::overflow_error("mesh exceeds 32-bit topology capacity");
    }

    HostMesh3D mesh;
    mesh.NX = NX; mesh.NY = NY; mesh.NZ = NZ;
    mesh.hx = hx; mesh.hy = hy; mesh.hz = hz;

    mesh.NE = static_cast<int>(elementCount);
    mesh.NP = static_cast<int>(nodeCount);

    int nEx = static_cast<int>(xEdgeCount);
    int nEy = static_cast<int>(yEdgeCount);
    int nEz = static_cast<int>(zEdgeCount);
    mesh.NEdges = static_cast<int>(edgeCount);

    mesh.ME_Edges.resize(static_cast<size_t>(12) * mesh.NE, 0);
    mesh.ME_Nodes.resize(static_cast<size_t>(8) * mesh.NE, 0);

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
