#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>


using namespace batchlas;
// Test fixture for SYEVX operations
class LanczosTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a SYCL queue
        ctx = std::make_shared<Queue>(Device::default_device());
        
        // Initialize test matrices
        A_data = UnifiedVector<float>(rows * rows * batch_size);
        W_data = UnifiedVector<float>(rows * batch_size);
        
        // Set up CSR format data for symmetric matrices with known eigenvalues
        setupCSRMatrices();
    }
    
    void TearDown() override {
    }
    
    // Set up CSR matrices with known eigenvalues
    void setupCSRMatrices() {
        // For each batch, create a sparse symmetric matrix with known eigenvalues
        // We'll use a simple tridiagonal matrix with known eigenvalues
        
        // For a tridiagonal matrix with diagonal=2 and off-diagonals=-1,
        // the eigenvalues are: 2 - 2*cos(k*pi/(n+1)) where k=1,2,...,n
        // The largest 3 eigenvalues for n=10 will be approximately:
        // 3.90, 3.62, 3.17
        
        int nnz_per_matrix = 30*rows;
        total_nnz = nnz_per_matrix;  // Per matrix, not total
        
        // Allocate CSR format arrays
        csr_values.resize(total_nnz * batch_size);
        csr_col_indices.resize(total_nnz * batch_size);
        csr_row_offsets.resize((rows + 1) * batch_size);

        
        // Known largest eigenvalues for our test matrix
        known_eigenvalues = {1087.76, 1087.76, 1087.76};
        
        for (int b = 0; b < batch_size; ++b) {
            int base_idx = b * nnz_per_matrix;
            int base_row_ptr = b * (rows + 1);
            
            // Set row pointers
            csr_row_offsets[base_row_ptr] = 0;  // Each batch matrix starts at 0
            
            int nnz_count = 0;
            //Pre-computed Hessian of the C20 fullerene potential
            std::array<float, 60*30> Avals = {
                467.283, 0.000186032, -110.893, -184.505, 32.8416, 7.14737, -184.505, -32.8418, 7.14751, -131.095, -3.61167e-06, 130.224, 11.5192, -9.2696, -3.96905, -10.1164, -33.6383, 1.68713, 15.0079, 4.16187, -14.5311, -10.1165, 33.6384, 1.68711, 11.5193, 9.2696, -3.96906, 15.0078, -4.16194, -14.5312, 0.000186032, 551.997, 0.000310918, 83.1742, -222.201, 55.5253, -83.1743, -222.201, -55.5255, -8.67248e-06, -166.104, 1.55717e-05, 7.1753, -6.38028, -3.32003, 7.97274, 25.2092, -0.353045, 23.5979, 10.4255, -25.9265, -7.97279, 25.2092, 0.353043, -7.1754, -6.38037, 3.32008, -23.5978, 10.4254, 25.9264, -110.893, 0.000310918, 406.836, 55.0163, -10.3607, -103.94, 55.0165, 10.3604, -103.94, 34.4854, -7.25407e-05, -213.448, -25.059, 28.6987, 12.4979, -1.1574, -1.27729, 2.54408, 9.40371, 2.17753, -7.79633, -1.15741, 1.27731, 2.54423, -25.0591, -28.6987, 12.4979, 9.40358, -2.17751, -7.79625, 330.214, -0.000191346, -42.3564, -131.095, -8.67248e-06, 34.4854, -95.1882, -5.41977, 10.3579, -95.1882, 5.41984, 10.3579, -5.28651, -3.80906, 19.5504, 0.690935, -21.5233, -19.6449, 0.224137, 16.1862, -6.32783, -5.28656, 3.8091, 19.5504, 0.224171, -16.1862, -6.32784, 0.690949, 21.5234, -19.6449, -0.000191346, 551.997, -3.16855e-05, -3.61167e-06, -166.104, -7.25407e-05, -86.86, -222.201, -49.5619, 86.86, -222.201, 49.5619, 12.6358, 10.4255, -32.701, -0.239174, -6.38013, -7.90226, -3.24978, 25.2092, -7.28899, -12.6357, 10.4255, 32.7009, 3.24976, 25.2091, 7.28898, 0.239205, -6.38016, 7.90232, -42.3564, -3.16855e-05, 543.907, 130.224, 1.55717e-05, -213.448, -37.5114, -34.0083, -193.257, -37.5114, 34.0083, -193.257, -4.38401, -2.74882, 12.4979, 1.44487, 21.1254, 23.3261, -3.4832, 29.5161, -7.7963, -4.3841, 2.7489, 12.498, -3.48322, -29.5161, -7.79631, 1.44492, -21.1254, 23.3261, 406.838, -105.464, 34.267, -95.1882, 86.86, -37.5114, -143.19, 16.648, -27.8993, -182.952, 33.9358, 20.7521, 15.0078, -23.5978, 9.40358, 4.33918, -0.612875, 3.40455, -12.1008, 17.3671, 20.5379, 8.5966, -10.642, 11.2482, -4.05554, -14.3302, -35.0377, 2.70461, -0.164353, 0.834938, -105.464, 475.373, 24.8964, 5.41984, -222.201, 34.0083, 16.648, -154.009, -20.27, 115.376, -134.437, -46.1846, -4.16194, 10.4254, -2.17751, -17.0578, 0.799764, -37.947, -3.91707, 6.41154, 5.1539, 10.6421, -14.2858, 17.9401, 2.11474, 9.19448, 14.9642, -19.6003, 22.7287, 9.61626, 34.267, 24.8964, 543.906, 10.3579, 49.5619, -193.257, -105.353, -76.5437, -213.448, 50.337, -5.46455, -193.257, -14.5312, 25.9264, -7.79625, 5.16253, 0.353012, 12.4979, -13.5861, 16.2415, 23.3261, 11.2483, -17.9401, 23.3261, 1.93105, 4.80077, 12.4979, 20.1671, -21.8316, -7.79627, 530.818, -65.1803, -13.0889, -182.952, 115.376, 50.337, -237.192, -40.7201, -43.9354, -162.761, 10.2891, 10.6565, 0.690949, 0.239205, 1.44492, 19.0214, -2.85781, 4.97684, 6.33101, -16.4106, 37.1419, 26.6252, -11.828, -8.88766, -12.1008, 3.91708, -13.5861, 11.5192, 7.1753, -25.059, -65.1803, 351.392, -40.2834, 33.9358, -134.437, -5.46455, 40.7201, -80.1965, 25.1664, 10.2891, -134.438, 32.7976, 21.5234, -6.38016, -21.1254, -22.2938, 6.41191, -8.27054, 0.0343012, -1.19207, 8.48838, 7.60796, -1.19192, -3.76571, -17.3671, 6.41157, -16.2415, -9.2696, -6.38028, 28.6987, -13.0889, -40.2834, 543.907, 20.7521, -46.1846, -193.257, -43.9354, -25.1664, -193.257, 40.2415, 123.85, -213.448, -19.6449, 7.90232, 23.3261, -29.1479, 5.80824, -7.7963, 1.25957, -5.01896, 12.498, 26.9951, -12.4337, -7.79631, 20.5379, -5.15391, 23.3261, -3.96905, -3.32003, 12.4979, 543.908, -24.8967, -34.2676, -162.761, 10.2891, 40.2415, -252.697, -61.0172, -50.5991, -184.505, 83.1742, 55.0163, 2.70461, -19.6003, 20.1671, -4.05556, -2.11475, 1.93107, 29.3788, -20.8057, 0.857094, 19.0214, 22.2938, -29.1479, 14.2929, 0.04166, 0.185582, -5.28651, 12.6358, -4.38401, -24.8967, 475.373, -105.465, 10.2891, -134.438, 123.85, -10.6847, -154.008, 23.956, 32.8416, -222.201, -10.3607, -0.164353, 22.7287, -21.8316, 14.3302, 9.19449, -4.80079, 20.8056, -14.2862, 1.49543, 2.85781, 6.4119, -5.80822, -41.5695, 0.799756, 1.71366, -3.80906, 10.4255, -2.74882, -34.2676, -105.465, 406.836, 10.6565, 32.7976, -213.448, 26.8545, 49.1222, -103.94, 7.14737, 55.5253, -103.94, 0.834938, 9.61626, -7.79627, -35.0377, -14.9642, 12.498, 0.857125, -1.49547, 2.54428, 4.97685, 8.27055, -7.79632, -1.57241, -0.706041, 2.54419, 19.5504, -32.701, 12.4979, 496.551, 40.2839, 89.7141, -252.697, -10.6847, 26.8545, -143.19, -16.648, -105.353, -142.36, -25.1663, -38.4193, -10.1165, -7.97279, -1.15741, 26.6252, -7.60794, 26.9951, 4.33922, -17.0578, 5.16249, 6.33101, 0.0343012, 1.25957, 0.224104, 3.24975, -3.48315, 14.2929, 41.5696, -1.57242, 40.2839, 522.729, -65.1813, -61.0172, -154.008, 49.1222, -16.648, -154.009, 76.5437, 25.1663, -264.346, -40.7197, 33.6384, 25.2092, 1.27731, 11.828, -1.1919, 12.4338, -0.612872, 0.799761, 0.353012, -16.4106, -1.19207, -5.01896, -16.1862, 25.2092, -29.5161, -0.0416718, 0.799741, 0.706041, 89.7141, -65.1813, 406.836, -50.5991, 23.956, -103.94, -27.8993, 20.2701, -213.448, -38.4194, 40.7198, -103.94, 1.68711, 0.353043, 2.54423, -8.88768, 3.7657, -7.79632, 3.40452, -37.9471, 12.4979, 37.1419, 8.48838, 12.498, -6.32779, 7.28897, -7.79628, 0.185587, -1.71366, 2.54429, 496.551, -40.2839, 89.714, -142.36, 25.1663, -38.4194, -143.19, 16.648, -105.353, -252.697, 10.6847, 26.8546, 14.2929, -41.5695, -1.57241, 0.224107, -3.24976, -3.48317, 6.33095, -0.0342829, 1.25955, 4.33922, 17.0578, 5.16249, 26.6252, 7.60793, 26.9951, -10.1164, 7.97274, -1.1574, -40.2839, 522.729, 65.1812, -25.1663, -264.346, 40.7198, 16.648, -154.009, -76.5437, 61.0172, -154.008, -49.1222, 0.04166, 0.799756, -0.706041, 16.1862, 25.2092, 29.5161, 16.4107, -1.19205, 5.01896, 0.612868, 0.79974, -0.353017, -11.828, -1.19191, -12.4337, -33.6383, 25.2092, -1.27729, 89.714, 65.1812, 406.836, -38.4193, -40.7197, -103.94, -27.8993, -20.27, -213.448, -50.599, -23.9561, -103.94, 0.185582, 1.71366, 2.54419, -6.32779, -7.28897, -7.79627, 37.1418, -8.48835, 12.4979, 3.40452, 37.947, 12.4979, -8.88768, -3.7657, -7.79632, 1.68713, -0.353045, 2.54408, 543.908, 24.8968, -34.2678, -252.697, 61.0172, -50.599, -162.761, -10.289, 40.2414, -184.505, -83.1743, 55.0165, 14.2929, -0.0416718, 0.185587, 19.0214, -22.2938, -29.1478, -5.28656, -12.6357, -4.3841, -4.05549, 2.11469, 1.93105, 2.70463, 19.6003, 20.1671, 29.3788, 20.8056, 0.857125, 24.8968, 475.373, 105.465, 10.6847, -154.008, -23.9561, -10.289, -134.438, -123.85, -32.8418, -222.201, 10.3604, 41.5696, 0.799741, -1.71366, -2.85781, 6.41193, 5.80824, 3.8091, 10.4255, 2.7489, -14.3302, 9.19443, 4.80081, 0.164287, 22.7287, 21.8317, -20.8057, -14.2862, -1.49547, -34.2678, 105.465, 406.835, 26.8546, -49.1222, -103.94, 10.6566, -32.7975, -213.448, 7.14751, -55.5255, -103.94, -1.57242, 0.706041, 2.54429, 4.97683, -8.27055, -7.7963, 19.5504, 32.7009, 12.498, -35.0377, 14.9642, 12.4979, 0.834974, -9.61632, -7.7963, 0.857094, 1.49543, 2.54428, 530.818, 65.18, -13.0888, -162.761, -10.289, 10.6566, -237.192, 40.7202, -43.9355, -182.952, -115.376, 50.337, 11.5193, -7.1754, -25.0591, -12.1007, -3.91705, -13.5861, 26.6252, 11.828, -8.88762, 6.33095, 16.4107, 37.1418, 19.0214, 2.8578, 4.97685, 0.690935, -0.239174, 1.44487, 65.18, 351.393, 40.2831, -10.289, -134.438, -32.7975, -40.7201, -80.1966, -25.1663, -33.9358, -134.437, 5.46461, 9.2696, -6.38037, -28.6987, 17.3671, 6.41153, 16.2415, -7.60795, -1.19191, 3.76569, -0.0342829, -1.19205, -8.48835, 22.2938, 6.4119, 8.27056, -21.5233, -6.38013, 21.1254, -13.0888, 40.2831, 543.906, 40.2414, -123.85, -213.448, -43.9354, 25.1664, -193.257, 20.7521, 46.1846, -193.257, -3.96906, 3.32008, 12.4979, 20.538, 5.15389, 23.3261, 26.9951, 12.4337, -7.79629, 1.25955, 5.01896, 12.4979, -29.1479, -5.8082, -7.79631, -19.6449, -7.90226, 23.3261, 406.838, 105.464, 34.2668, -182.952, -33.9358, 20.7521, -143.19, -16.648, -27.8992, -95.1882, -86.86, -37.5114, 2.70463, 0.164287, 0.834974, -4.0556, 14.3302, -35.0377, 8.5966, 10.6421, 11.2483, -12.1008, -17.3671, 20.5379, 4.3392, 0.612873, 3.40452, 15.0079, 23.5979, 9.40371, 105.464, 475.373, -24.8963, -115.376, -134.437, 46.1846, -16.6479, -154.009, 20.2699, -5.41977, -222.201, -34.0083, 19.6003, 22.7287, -9.61632, -2.11478, 9.19453, -14.9642, -10.642, -14.2858, -17.9401, 3.9171, 6.41155, -5.15393, 17.0578, 0.799706, 37.947, 4.16187, 10.4255, 2.17753, 34.2668, -24.8963, 543.907, 50.337, 5.46461, -193.257, -105.353, 76.5436, -213.448, 10.3579, -49.5619, -193.257, 20.1671, 21.8317, -7.7963, 1.93106, -4.80074, 12.4979, 11.2482, 17.9401, 23.3261, -13.5861, -16.2415, 23.3261, 5.16253, -0.353032, 12.4979, -14.5311, -25.9265, -7.79633, 496.551, 40.2838, 89.7141, -143.19, -16.6479, -105.353, -252.697, -10.6847, 26.8545, -142.36, -25.1662, -38.4195, 0.224171, 3.24976, -3.48322, 6.33104, 0.0343216, 1.25959, 14.293, 41.5695, -1.5724, 26.6252, -7.60795, 26.9951, -10.1165, -7.97282, -1.1574, 4.33918, -17.0578, 5.16253, 40.2838, 522.729, -65.181, -16.648, -154.009, 76.5436, -61.0172, -154.008, 49.1222, 25.1663, -264.346, -40.7198, -16.1862, 25.2091, -29.5161, -16.4106, -1.19211, -5.01897, -0.0416719, 0.799728, 0.706046, 11.828, -1.19191, 12.4337, 33.6384, 25.2092, 1.27728, -0.612875, 0.799764, 0.353012, 89.7141, -65.181, 406.836, -27.8992, 20.2699, -213.448, -50.5991, 23.956, -103.94, -38.4195, 40.7197, -103.94, -6.32784, 7.28898, -7.79631, 37.1418, 8.48836, 12.4979, 0.185589, -1.71363, 2.54422, -8.88762, 3.76569, -7.79629, 1.6871, 0.353036, 2.54423, 3.40455, -37.947, 12.4979, 496.551, -40.284, 89.7143, -142.36, 25.1663, -38.4195, -252.697, 10.6847, 26.8545, -143.19, 16.648, -105.353, 4.3392, 17.0578, 5.16253, -10.1165, 7.97279, -1.1574, 26.6252, 7.60796, 26.9951, 14.293, -41.5695, -1.57245, 6.33099, -0.0343014, 1.25956, 0.224137, -3.24978, -3.4832, -40.284, 522.729, 65.1813, -25.1662, -264.346, 40.7197, 61.0172, -154.008, -49.1222, 16.648, -154.009, -76.5437, 0.612873, 0.799706, -0.353032, -33.6383, 25.2092, -1.2773, -11.828, -1.19192, -12.4337, 0.0416647, 0.799745, -0.706053, 16.4106, -1.19207, 5.01895, 16.1862, 25.2092, 29.5161, 89.7143, 65.1813, 406.836, -38.4195, -40.7198, -103.94, -50.5991, -23.956, -103.94, -27.8993, -20.27, -213.448, 3.40452, 37.947, 12.4979, 1.68713, -0.353053, 2.54419, -8.88766, -3.76571, -7.79631, 0.18558, 1.71367, 2.54418, 37.1418, -8.48838, 12.4979, -6.32783, -7.28899, -7.7963, 543.908, 24.8967, -34.2677, -252.697, 61.0172, -50.5991, -184.505, -83.1742, 55.0164, -162.761, -10.289, 40.2414, -4.05554, 2.11474, 1.93105, 29.3788, 20.8056, 0.857112, 2.70462, 19.6003, 20.1671, 14.293, -0.0416719, 0.185589, -5.28654, -12.6358, -4.38407, 19.0214, -22.2938, -29.1479, 24.8967, 475.373, 105.465, 10.6847, -154.008, -23.956, -32.8417, -222.201, 10.3606, -10.289, -134.438, -123.85, -14.3302, 9.19448, 4.80077, -20.8056, -14.2862, -1.49545, 0.164319, 22.7287, 21.8317, 41.5695, 0.799728, -1.71363, 3.80909, 10.4255, 2.74887, -2.85781, 6.41191, 5.80824, -34.2677, 105.465, 406.836, 26.8545, -49.1222, -103.94, 7.14743, -55.5254, -103.94, 10.6566, -32.7975, -213.448, -35.0377, 14.9642, 12.4979, 0.85712, 1.49545, 2.54426, 0.834957, -9.6163, -7.79629, -1.5724, 0.706046, 2.54422, 19.5504, 32.701, 12.4979, 4.97684, -8.27054, -7.7963, 530.818, 65.1801, -13.0888, -162.761, -10.289, 10.6566, -182.952, -115.376, 50.3369, -237.192, 40.7201, -43.9354, 6.33099, 16.4106, 37.1418, 0.690957, -0.239207, 1.44491, 19.0214, 2.85781, 4.97685, 11.5193, -7.17535, -25.059, 26.6252, 11.828, -8.88768, -12.1008, -3.91707, -13.5861, 65.1801, 351.393, 40.2832, -10.289, -134.438, -32.7975, -33.9358, -134.437, 5.46459, -40.7201, -80.1965, -25.1664, -0.0343014, -1.19207, -8.48838, -21.5234, -6.38018, 21.1254, 22.2938, 6.4119, 8.27055, 9.26959, -6.38032, -28.6987, -7.60794, -1.1919, 3.7657, 17.3671, 6.41154, 16.2415, -13.0888, 40.2832, 543.907, 40.2414, -123.85, -213.448, 20.7521, 46.1846, -193.257, -43.9354, 25.1664, -193.257, 1.25956, 5.01895, 12.4979, -19.6449, -7.90234, 23.3261, -29.1479, -5.80822, -7.79632, -3.96905, 3.32005, 12.4979, 26.9951, 12.4338, -7.79632, 20.5379, 5.1539, 23.3261, 406.838, 105.464, 34.2671, -182.952, -33.9358, 20.7521, -95.1882, -86.86, -37.5114, -143.19, -16.648, -27.8993, -12.1008, -17.3671, 20.5379, 15.0079, 23.5979, 9.40365, 4.33922, 0.612868, 3.40452, 2.70462, 0.164319, 0.834957, 8.59664, 10.6421, 11.2482, -4.05556, 14.3302, -35.0377, 105.464, 475.373, -24.8964, -115.376, -134.437, 46.1846, -5.41982, -222.201, -34.0083, -16.648, -154.009, 20.2701, 3.91708, 6.41157, -5.15391, 4.16193, 10.4255, 2.17754, 17.0578, 0.79974, 37.947, 19.6003, 22.7287, -9.6163, -10.6421, -14.2859, -17.9401, -2.11475, 9.19449, -14.9642, 34.2671, -24.8964, 543.906, 50.3369, 5.46459, -193.257, 10.3579, -49.5619, -193.257, -105.353, 76.5437, -213.448, -13.5861, -16.2415, 23.3261, -14.5312, -25.9264, -7.79629, 5.16249, -0.353017, 12.4979, 20.1671, 21.8317, -7.79629, 11.2482, 17.9401, 23.3262, 1.93107, -4.80079, 12.498, 330.214, -1.59815e-06, -42.3566, -95.1882, -5.41982, 10.3579, -131.095, -8.99612e-06, 34.4855, -95.1883, 5.41978, 10.3579, 0.224104, -16.1862, -6.32779, -5.28654, 3.80909, 19.5504, 0.690954, 21.5234, -19.6449, 0.690957, -21.5234, -19.6449, -5.28653, -3.80907, 19.5504, 0.224107, 16.1862, -6.32779, -1.59815e-06, 551.997, 3.88715e-05, -86.86, -222.201, -49.5619, -4.10112e-06, -166.104, -2.1165e-05, 86.86, -222.201, 49.5619, 3.24975, 25.2092, 7.28897, -12.6358, 10.4255, 32.701, 0.239219, -6.38017, 7.90233, -0.239207, -6.38018, -7.90234, 12.6358, 10.4255, -32.701, -3.24976, 25.2092, -7.28897, -42.3566, 3.88715e-05, 543.907, -37.5114, -34.0083, -193.257, 130.224, -2.7682e-05, -213.448, -37.5114, 34.0082, -193.257, -3.48315, -29.5161, -7.79628, -4.38407, 2.74887, 12.4979, 1.44493, -21.1254, 23.3261, 1.44491, 21.1254, 23.3261, -4.38406, -2.74885, 12.4979, -3.48317, 29.5161, -7.79627, 406.838, -105.464, 34.2671, -95.1883, 86.86, -37.5114, -182.952, 33.9359, 20.7521, -143.19, 16.648, -27.8993, 8.59664, -10.6421, 11.2482, 2.70461, -0.164355, 0.834941, -4.05549, -14.3302, -35.0377, 15.0078, -23.5979, 9.40365, -12.1007, 17.3671, 20.538, 4.33922, -0.612872, 3.40452, -105.464, 475.373, 24.8965, 5.41978, -222.201, 34.0082, 115.376, -134.437, -46.1846, 16.648, -154.009, -20.27, 10.6421, -14.2859, 17.9401, -19.6003, 22.7287, 9.61628, 2.11469, 9.19443, 14.9642, -4.16192, 10.4254, -2.17753, -3.91705, 6.41153, 5.15389, -17.0578, 0.799761, -37.9471, 34.2671, 24.8965, 543.907, 10.3579, 49.5619, -193.257, 50.3369, -5.46454, -193.257, -105.353, -76.5437, -213.448, 11.2482, -17.9401, 23.3262, 20.1671, -21.8317, -7.79628, 1.93105, 4.80081, 12.4979, -14.5311, 25.9264, -7.79629, -13.5861, 16.2415, 23.3261, 5.16249, 0.353012, 12.4979, 530.818, -65.1803, -13.0888, -182.952, 115.376, 50.3369, -162.761, 10.2891, 10.6566, -237.192, -40.7201, -43.9354, 26.6252, -11.828, -8.88768, 11.5193, 7.17533, -25.0591, -12.1008, 3.9171, -13.5861, 0.690954, 0.239219, 1.44493, 6.33104, -16.4106, 37.1418, 19.0214, -2.85781, 4.97683, -65.1803, 351.393, -40.2834, 33.9359, -134.437, -5.46454, 10.2891, -134.438, 32.7976, 40.7202, -80.1966, 25.1664, 7.60793, -1.19191, -3.7657, -9.26961, -6.3803, 28.6987, -17.3671, 6.41155, -16.2415, 21.5234, -6.38017, -21.1254, 0.0343216, -1.19211, 8.48836, -22.2938, 6.41193, -8.27055, -13.0888, -40.2834, 543.907, 20.7521, -46.1846, -193.257, 40.2414, 123.85, -213.448, -43.9355, -25.1663, -193.257, 26.9951, -12.4337, -7.79632, -3.96904, -3.32003, 12.4979, 20.5379, -5.15393, 23.3261, -19.6449, 7.90233, 23.3261, 1.25959, -5.01897, 12.4979, -29.1478, 5.80824, -7.7963, 543.908, -24.8967, -34.2676, -162.761, 10.2891, 40.2414, -184.505, 83.1742, 55.0164, -252.697, -61.0172, -50.5991, 19.0214, 22.2938, -29.1479, -5.28653, 12.6358, -4.38406, 14.293, 0.0416647, 0.18558, 2.70461, -19.6003, 20.1671, 29.3788, -20.8056, 0.85712, -4.0556, -2.11478, 1.93106, -24.8967, 475.373, -105.465, 10.2891, -134.438, 123.85, 32.8417, -222.201, -10.3606, -10.6847, -154.008, 23.956, 2.8578, 6.4119, -5.8082, -3.80907, 10.4255, -2.74885, -41.5695, 0.799745, 1.71367, -0.164355, 22.7287, -21.8317, 20.8056, -14.2862, 1.49545, 14.3302, 9.19453, -4.80074, -34.2676, -105.465, 406.836, 10.6566, 32.7976, -213.448, 7.14743, 55.5254, -103.94, 26.8545, 49.1222, -103.94, 4.97685, 8.27056, -7.79631, 19.5504, -32.701, 12.4979, -1.57245, -0.706053, 2.54418, 0.834941, 9.61628, -7.79628, 0.857112, -1.49545, 2.54426, -35.0377, -14.9642, 12.4979, 467.283, 5.61103e-05, -110.893, -184.505, 32.8417, 7.14743, -131.095, -4.10112e-06, 130.224, -184.505, -32.8417, 7.14743, -10.1165, 33.6384, 1.6871, 15.0078, -4.16192, -14.5311, 11.5193, 9.26959, -3.96905, 11.5193, -9.26961, -3.96904, 15.0079, 4.16193, -14.5312, -10.1165, -33.6383, 1.68713, 5.61103e-05, 551.997, -2.72747e-05, 83.1742, -222.201, 55.5254, -8.99612e-06, -166.104, -2.7682e-05, -83.1742, -222.201, -55.5254, -7.97282, 25.2092, 0.353036, -23.5979, 10.4254, 25.9264, -7.17535, -6.38032, 3.32005, 7.17533, -6.3803, -3.32003, 23.5979, 10.4255, -25.9264, 7.97279, 25.2092, -0.353053, -110.893, -2.72747e-05, 406.836, 55.0164, -10.3606, -103.94, 34.4855, -2.1165e-05, -213.448, 55.0164, 10.3606, -103.94, -1.1574, 1.27728, 2.54423, 9.40365, -2.17753, -7.79629, -25.059, -28.6987, 12.4979, -25.0591, 28.6987, 12.4979, 9.40365, 2.17754, -7.79629, -1.1574, -1.2773, 2.54419,  
            };

            std::array<int, 60*30> Acols = {
                0, 1, 2, 12, 13, 14, 21, 22, 23, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 28, 29, 15, 16, 17, 24, 25, 26, 6, 7, 8, 0, 1, 2, 12, 13, 14, 21, 22, 23, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 28, 29, 15, 16, 17, 24, 25, 26, 6, 7, 8, 0, 1, 2, 12, 13, 14, 21, 22, 23, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 28, 29, 15, 16, 17, 24, 25, 26, 6, 7, 8, 3, 4, 5, 0, 1, 2, 27, 28, 29, 6, 7, 8, 12, 13, 14, 24, 25, 26, 33, 34, 35, 21, 22, 23, 30, 31, 32, 9, 10, 11, 3, 4, 5, 0, 1, 2, 27, 28, 29, 6, 7, 8, 12, 13, 14, 24, 25, 26, 33, 34, 35, 21, 22, 23, 30, 31, 32, 9, 10, 11, 3, 4, 5, 0, 1, 2, 27, 28, 29, 6, 7, 8, 12, 13, 14, 24, 25, 26, 33, 34, 35, 21, 22, 23, 30, 31, 32, 9, 10, 11, 6, 7, 8, 3, 4, 5, 33, 34, 35, 9, 10, 11, 0, 1, 2, 30, 31, 32, 39, 40, 41, 27, 28, 29, 36, 37, 38, 12, 13, 14, 6, 7, 8, 3, 4, 5, 33, 34, 35, 9, 10, 11, 0, 1, 2, 30, 31, 32, 39, 40, 41, 27, 28, 29, 36, 37, 38, 12, 13, 14, 6, 7, 8, 3, 4, 5, 33, 34, 35, 9, 10, 11, 0, 1, 2, 30, 31, 32, 39, 40, 41, 27, 28, 29, 36, 37, 38, 12, 13, 14, 9, 10, 11, 6, 7, 8, 39, 40, 41, 12, 13, 14, 3, 4, 5, 36, 37, 38, 15, 16, 17, 33, 34, 35, 42, 43, 44, 0, 1, 2, 9, 10, 11, 6, 7, 8, 39, 40, 41, 12, 13, 14, 3, 4, 5, 36, 37, 38, 15, 16, 17, 33, 34, 35, 42, 43, 44, 0, 1, 2, 9, 10, 11, 6, 7, 8, 39, 40, 41, 12, 13, 14, 3, 4, 5, 36, 37, 38, 15, 16, 17, 33, 34, 35, 42, 43, 44, 0, 1, 2, 12, 13, 14, 9, 10, 11, 15, 16, 17, 0, 1, 2, 6, 7, 8, 42, 43, 44, 21, 22, 23, 39, 40, 41, 18, 19, 20, 3, 4, 5, 12, 13, 14, 9, 10, 11, 15, 16, 17, 0, 1, 2, 6, 7, 8, 42, 43, 44, 21, 22, 23, 39, 40, 41, 18, 19, 20, 3, 4, 5, 12, 13, 14, 9, 10, 11, 15, 16, 17, 0, 1, 2, 6, 7, 8, 42, 43, 44, 21, 22, 23, 39, 40, 41, 18, 19, 20, 3, 4, 5, 15, 16, 17, 12, 13, 14, 42, 43, 44, 18, 19, 20, 0, 1, 2, 39, 40, 41, 48, 49, 50, 9, 10, 11, 45, 46, 47, 21, 22, 23, 15, 16, 17, 12, 13, 14, 42, 43, 44, 18, 19, 20, 0, 1, 2, 39, 40, 41, 48, 49, 50, 9, 10, 11, 45, 46, 47, 21, 22, 23, 15, 16, 17, 12, 13, 14, 42, 43, 44, 18, 19, 20, 0, 1, 2, 39, 40, 41, 48, 49, 50, 9, 10, 11, 45, 46, 47, 21, 22, 23, 18, 19, 20, 15, 16, 17, 48, 49, 50, 21, 22, 23, 12, 13, 14, 45, 46, 47, 24, 25, 26, 42, 43, 44, 51, 52, 53, 0, 1, 2, 18, 19, 20, 15, 16, 17, 48, 49, 50, 21, 22, 23, 12, 13, 14, 45, 46, 47, 24, 25, 26, 42, 43, 44, 51, 52, 53, 0, 1, 2, 18, 19, 20, 15, 16, 17, 48, 49, 50, 21, 22, 23, 12, 13, 14, 45, 46, 47, 24, 25, 26, 42, 43, 44, 51, 52, 53, 0, 1, 2, 21, 22, 23, 18, 19, 20, 24, 25, 26, 0, 1, 2, 15, 16, 17, 51, 52, 53, 3, 4, 5, 48, 49, 50, 27, 28, 29, 12, 13, 14, 21, 22, 23, 18, 19, 20, 24, 25, 26, 0, 1, 2, 15, 16, 17, 51, 52, 53, 3, 4, 5, 48, 49, 50, 27, 28, 29, 12, 13, 14, 21, 22, 23, 18, 19, 20, 24, 25, 26, 0, 1, 2, 15, 16, 17, 51, 52, 53, 3, 4, 5, 48, 49, 50, 27, 28, 29, 12, 13, 14, 24, 25, 26, 21, 22, 23, 51, 52, 53, 27, 28, 29, 0, 1, 2, 48, 49, 50, 30, 31, 32, 18, 19, 20, 54, 55, 56, 3, 4, 5, 24, 25, 26, 21, 22, 23, 51, 52, 53, 27, 28, 29, 0, 1, 2, 48, 49, 50, 30, 31, 32, 18, 19, 20, 54, 55, 56, 3, 4, 5, 24, 25, 26, 21, 22, 23, 51, 52, 53, 27, 28, 29, 0, 1, 2, 48, 49, 50, 30, 31, 32, 18, 19, 20, 54, 55, 56, 3, 4, 5, 27, 28, 29, 24, 25, 26, 30, 31, 32, 3, 4, 5, 21, 22, 23, 54, 55, 56, 6, 7, 8, 51, 52, 53, 33, 34, 35, 0, 1, 2, 27, 28, 29, 24, 25, 26, 30, 31, 32, 3, 4, 5, 21, 22, 23, 54, 55, 56, 6, 7, 8, 51, 52, 53, 33, 34, 35, 0, 1, 2, 27, 28, 29, 24, 25, 26, 30, 31, 32, 3, 4, 5, 21, 22, 23, 54, 55, 56, 6, 7, 8, 51, 52, 53, 33, 34, 35, 0, 1, 2, 30, 31, 32, 27, 28, 29, 54, 55, 56, 33, 34, 35, 3, 4, 5, 51, 52, 53, 36, 37, 38, 24, 25, 26, 57, 58, 59, 6, 7, 8, 30, 31, 32, 27, 28, 29, 54, 55, 56, 33, 34, 35, 3, 4, 5, 51, 52, 53, 36, 37, 38, 24, 25, 26, 57, 58, 59, 6, 7, 8, 30, 31, 32, 27, 28, 29, 54, 55, 56, 33, 34, 35, 3, 4, 5, 51, 52, 53, 36, 37, 38, 24, 25, 26, 57, 58, 59, 6, 7, 8, 33, 34, 35, 30, 31, 32, 36, 37, 38, 6, 7, 8, 27, 28, 29, 57, 58, 59, 9, 10, 11, 54, 55, 56, 39, 40, 41, 3, 4, 5, 33, 34, 35, 30, 31, 32, 36, 37, 38, 6, 7, 8, 27, 28, 29, 57, 58, 59, 9, 10, 11, 54, 55, 56, 39, 40, 41, 3, 4, 5, 33, 34, 35, 30, 31, 32, 36, 37, 38, 6, 7, 8, 27, 28, 29, 57, 58, 59, 9, 10, 11, 54, 55, 56, 39, 40, 41, 3, 4, 5, 36, 37, 38, 33, 34, 35, 57, 58, 59, 39, 40, 41, 6, 7, 8, 54, 55, 56, 42, 43, 44, 30, 31, 32, 45, 46, 47, 9, 10, 11, 36, 37, 38, 33, 34, 35, 57, 58, 59, 39, 40, 41, 6, 7, 8, 54, 55, 56, 42, 43, 44, 30, 31, 32, 45, 46, 47, 9, 10, 11, 36, 37, 38, 33, 34, 35, 57, 58, 59, 39, 40, 41, 6, 7, 8, 54, 55, 56, 42, 43, 44, 30, 31, 32, 45, 46, 47, 9, 10, 11, 39, 40, 41, 36, 37, 38, 42, 43, 44, 9, 10, 11, 33, 34, 35, 45, 46, 47, 12, 13, 14, 57, 58, 59, 15, 16, 17, 6, 7, 8, 39, 40, 41, 36, 37, 38, 42, 43, 44, 9, 10, 11, 33, 34, 35, 45, 46, 47, 12, 13, 14, 57, 58, 59, 15, 16, 17, 6, 7, 8, 39, 40, 41, 36, 37, 38, 42, 43, 44, 9, 10, 11, 33, 34, 35, 45, 46, 47, 12, 13, 14, 57, 58, 59, 15, 16, 17, 6, 7, 8, 42, 43, 44, 39, 40, 41, 45, 46, 47, 15, 16, 17, 9, 10, 11, 57, 58, 59, 18, 19, 20, 36, 37, 38, 48, 49, 50, 12, 13, 14, 42, 43, 44, 39, 40, 41, 45, 46, 47, 15, 16, 17, 9, 10, 11, 57, 58, 59, 18, 19, 20, 36, 37, 38, 48, 49, 50, 12, 13, 14, 42, 43, 44, 39, 40, 41, 45, 46, 47, 15, 16, 17, 9, 10, 11, 57, 58, 59, 18, 19, 20, 36, 37, 38, 48, 49, 50, 12, 13, 14, 45, 46, 47, 42, 43, 44, 57, 58, 59, 48, 49, 50, 15, 16, 17, 36, 37, 38, 51, 52, 53, 39, 40, 41, 54, 55, 56, 18, 19, 20, 45, 46, 47, 42, 43, 44, 57, 58, 59, 48, 49, 50, 15, 16, 17, 36, 37, 38, 51, 52, 53, 39, 40, 41, 54, 55, 56, 18, 19, 20, 45, 46, 47, 42, 43, 44, 57, 58, 59, 48, 49, 50, 15, 16, 17, 36, 37, 38, 51, 52, 53, 39, 40, 41, 54, 55, 56, 18, 19, 20, 48, 49, 50, 45, 46, 47, 51, 52, 53, 18, 19, 20, 42, 43, 44, 54, 55, 56, 21, 22, 23, 57, 58, 59, 24, 25, 26, 15, 16, 17, 48, 49, 50, 45, 46, 47, 51, 52, 53, 18, 19, 20, 42, 43, 44, 54, 55, 56, 21, 22, 23, 57, 58, 59, 24, 25, 26, 15, 16, 17, 48, 49, 50, 45, 46, 47, 51, 52, 53, 18, 19, 20, 42, 43, 44, 54, 55, 56, 21, 22, 23, 57, 58, 59, 24, 25, 26, 15, 16, 17, 51, 52, 53, 48, 49, 50, 54, 55, 56, 24, 25, 26, 18, 19, 20, 57, 58, 59, 27, 28, 29, 45, 46, 47, 30, 31, 32, 21, 22, 23, 51, 52, 53, 48, 49, 50, 54, 55, 56, 24, 25, 26, 18, 19, 20, 57, 58, 59, 27, 28, 29, 45, 46, 47, 30, 31, 32, 21, 22, 23, 51, 52, 53, 48, 49, 50, 54, 55, 56, 24, 25, 26, 18, 19, 20, 57, 58, 59, 27, 28, 29, 45, 46, 47, 30, 31, 32, 21, 22, 23, 54, 55, 56, 51, 52, 53, 57, 58, 59, 30, 31, 32, 24, 25, 26, 45, 46, 47, 33, 34, 35, 48, 49, 50, 36, 37, 38, 27, 28, 29, 54, 55, 56, 51, 52, 53, 57, 58, 59, 30, 31, 32, 24, 25, 26, 45, 46, 47, 33, 34, 35, 48, 49, 50, 36, 37, 38, 27, 28, 29, 54, 55, 56, 51, 52, 53, 57, 58, 59, 30, 31, 32, 24, 25, 26, 45, 46, 47, 33, 34, 35, 48, 49, 50, 36, 37, 38, 27, 28, 29, 57, 58, 59, 54, 55, 56, 45, 46, 47, 36, 37, 38, 30, 31, 32, 48, 49, 50, 39, 40, 41, 51, 52, 53, 42, 43, 44, 33, 34, 35, 57, 58, 59, 54, 55, 56, 45, 46, 47, 36, 37, 38, 30, 31, 32, 48, 49, 50, 39, 40, 41, 51, 52, 53, 42, 43, 44, 33, 34, 35, 57, 58, 59, 54, 55, 56, 45, 46, 47, 36, 37, 38, 30, 31, 32, 48, 49, 50, 39, 40, 41, 51, 52, 53, 42, 43, 44, 33, 34, 35};
            
            // Fill the tridiagonal matrix
            for (int i = 0; i < rows; ++i) {

                
                csr_row_offsets[base_row_ptr + i + 1] = 30*(i+1);  // Cumulative count for this matrix
            }

            memcpy(csr_values.data() + nnz_per_matrix * b, Avals.data(), nnz_per_matrix * sizeof(float));
            memcpy(csr_col_indices.data() + nnz_per_matrix * b, Acols.data(), nnz_per_matrix * sizeof(int));
        }
        // Also initialize the dense matrix for comparison
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < 30; ++j) {
                    A_data[b * rows * ld + i * ld + csr_col_indices[b * nnz_per_matrix + i*30 + j]] = csr_values[b * nnz_per_matrix + i*30 + j];
                }
            }
        }
    }
    
    std::shared_ptr<Queue> ctx;
    const int rows = 60;
    const int ld = 60;
    const int batch_size = 5;
    int total_nnz;
    
    UnifiedVector<float> A_data;
    UnifiedVector<float> W_data;
    
    // CSR format data
    UnifiedVector<float> csr_values;
    UnifiedVector<int> csr_col_indices;
    UnifiedVector<int> csr_row_offsets;
    
    // Known eigenvalues for verification
    std::vector<float> known_eigenvalues;

    void printMatrix(UnifiedVector<float>& matrix_data, int rows, int cols, int ld){
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                std::cout << matrix_data[i * ld + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Test LANCZOS operation with sparse matrix
TEST_F(LanczosTestBase, LanczosTest) {
    const int neig = 3;

    SparseMatHandle<float, Format::CSR, BatchType::Batched> sparse_matrix(
        csr_values.data(),
        csr_row_offsets.data(),
        csr_col_indices.data(),
        total_nnz,
        rows,
        rows,
        total_nnz,
        rows+1,
        batch_size);

    
    size_t buffer_size = lanczos_buffer_size<Backend::CUDA>(*ctx, sparse_matrix, W_data, JobType::NoEigenVectors);

    UnifiedVector<std::byte> workspace(buffer_size);

    lanczos<Backend::CUDA>(
        *ctx, sparse_matrix, W_data, workspace, JobType::NoEigenVectors);

    ctx->wait();
    
    // Verify that the computed eigenvalues match the expected ones
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < neig; ++i) {
            //Values are sorted in ascending order
            EXPECT_NEAR(W_data[rows*b + rows - i], known_eigenvalues[i], 0.1f)
                << "Eigenvalue mismatch at batch " << b << ", index " << i;
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}