#include "../include/femAssemble3d.h"
#include "../include/mesh3d.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

namespace {

bool checkMesh(int nx, int ny, int nz)
{
    std::vector<double> hx(nx, 1.0);
    std::vector<double> hy(ny, 1.0);
    std::vector<double> hz(nz, 1.0);
    HostMesh3D mesh = generateMesh3D(nx, ny, nz, hx, hy, hz);

    std::vector<std::set<int>> expected(mesh.NEdges);
    for (int element = 0; element < mesh.NE; ++element) {
        const int* edges = mesh.ME_Edges.data() + 12 * element;
        for (int localRow = 0; localRow < 12; ++localRow) {
            for (int localCol = 0; localCol < 12; ++localCol) {
                expected[edges[localRow]].insert(edges[localCol]);
            }
        }
    }

    std::vector<int> colInd;
    std::vector<int> rowPtr;
    femBuild3D_CSRStructure(mesh, colInd, rowPtr);
    if (rowPtr.size() != static_cast<size_t>(mesh.NEdges + 1)
        || rowPtr.front() != 0
        || rowPtr.back() != static_cast<int>(colInd.size())) {
        return false;
    }

    for (int row = 0; row < mesh.NEdges; ++row) {
        std::vector<int> actual(
            colInd.begin() + rowPtr[row], colInd.begin() + rowPtr[row + 1]
        );
        std::vector<int> reference(expected[row].begin(), expected[row].end());
        if (actual != reference || !std::binary_search(actual.begin(), actual.end(), row)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    for (const auto& shape : std::vector<std::vector<int>>{
             {1, 1, 1}, {2, 3, 2}, {3, 2, 4}}) {
        if (!checkMesh(shape[0], shape[1], shape[2])) {
            std::cerr << "CSR topology mismatch for "
                      << shape[0] << "x" << shape[1] << "x" << shape[2] << '\n';
            return 1;
        }
    }
    return 0;
}
