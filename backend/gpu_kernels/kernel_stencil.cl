__kernel void stencil(__global double *n,
                      __global const int *index,
                      __global double *n_out)
{
        int ind = get_global_id(0);
        int i = index[ind];
        n_out[i] = n[i+1] + n[i-1] - 2 * n[i];
}


__kernel void stencil_1(const int N,
                        const float a,
                        __global float *grid,
                        __global float *grid_out)
{
    // Define the stencil size (3x3 in this case)
    const int stencil_size = 3;
//    const int global_index = ind ;

    // Define the tile size (specified in kernel arguments)
    const int tile_size = get_local_size(0);

    // Calculate the global ID of the current work item (i.e., output element)
    const int global_i = get_global_id(0);
    const int global_j = get_global_id(1);

    // Calculate the local ID of the current work item (i.e., output element within the tile)
    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);

    // Define shared memory for the tile (with padding for stencil access)
    __local float tile[16+3-1][16+3-1];

    // Load tile data from global memory to shared memory
    // Global tile index
    int tile_i = local_i + get_group_id(0) * tile_size;
    int tile_j = local_j + get_group_id(1) * tile_size;
    for (int i=-1; i<tile_size+1; i++)
    {
        for (int j=-1; j<tile_size+1; j++)
        {
            // Calculate the input indices for the current tile element
            int input_i = tile_i + i;
            int input_j = tile_j + j;

            // Check if the input indices are within bounds
            if (input_i >= 0 && input_i < N && input_j >= 0 && input_j < N)
            {
                // Calculate the indices of the current input and stencil elements
                int input_idx = input_i * N + input_j;
                // int stencil_idx = (i + 1) * stencil_size + (j + 1);

                // Load the input element into the shared memory tile
                tile[local_i + i][local_j + j] = grid[input_idx];
            }
            else
            {
                // Pad the shared memory tile with zeroes for out-of-bounds elements
                tile[local_i + i][local_j + j] = 0.0f;
            }
        }
    }
    // Wait for all threads to finish loading data into the shared memory tile
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate the output index for the current work item
    int output_idx = global_i * N + global_j;

    // Apply the stencil to the tile and store the result in the output buffer
    float output_value = -4 * tile[local_i][local_j] + tile[local_i-1][local_j] + tile[local_i+1][local_j] +
                                                       tile[local_i][local_j-1] + tile[local_i][local_j+1];

    grid_out[output_idx] = output_value;

}


__kernel void stencil_2(int input_size, __global double *input, __global double *output)
{
    const int tile_size = 16;
    const int stencil_size = 3;
    // Define the local input array
    __local float local_input[16 + 3 - 1];
    // Get the global and local IDs and sizes
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    // Calculate the number of tiles
    int tiles = (input_size + tile_size - 1) / tile_size;
    // Loop over the tiles
    for (int tile = 0; tile < tiles; tile++)
    {
        // Calculate the tile start and end indices
        int tile_start = tile * tile_size;
        int tile_end = min(tile_start + tile_size, input_size);
        // Calculate the tile size with stencil and start and end indices with stencil
        int tile_size_with_stencil = tile_size + stencil_size - 1;
        int tile_start_with_stencil = tile_start - (stencil_size - 1) / 2;
        int tile_end_with_stencil = tile_end + (stencil_size - 1) / 2;
        // Load the input array into local memory
        for (int i = tile_start_with_stencil + lid; i < tile_end_with_stencil; i += lsize)
        {
            // Clamp the index to the input array bounds
            int idx = max(0, min(i, input_size - 1));
            // Store the input value in local memory
            local_input[i - tile_start_with_stencil] = input[idx];
        }
        // Synchronize local memory
        barrier(CLK_LOCAL_MEM_FENCE);
        // Compute the stencil operation
        for (int i = tile_start + lid; i < tile_end; i += lsize)
        {
            float output_value = local_input[i-1] - 2 * local_input[i] + local_input[i+1];
            // Store the output value
            output[i] = output_value;
        }
    // Synchronize global memory
    barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


__kernel void stencil_3(int N, __global double *grid, __global double *grid_out)
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int tile_size = get_local_size(0);

    __local float tile[64+3-1];
    __local float tile_out[64+3-1];

    const int tiles = N;

    int tile_start = gid-1;
    int tile_end = gid + tile_size + 1;

    for (int i = 0; i<tile_size+2; i++)
    {
        tile[i] = grid[gid-1+i];
    }

    // Wait for all threads to finish loading shared memory
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i<tile_size; i++)
    {
        tile_out[i] = tile[i] - 2 * tile[i+1] + tile[i+2];
    }
    for (int i = 0; i<tile_size; i++)
    {
        grid_out[gid+i] = tile_out[i];
    }


}


__kernel void stencil_4(const int size, __global const float* grid, __global float* grid_out)
{

        // Determine the global ID of this thread
        int gid = get_global_id(0);

        // Define the shared memory for the tile
        __local float tile[8 + 2];

        // Load the grid values for the tile into shared memory
        int tile_index = get_local_id(0) + 1;
        int grid_index = gid + get_local_id(0) - 1;
        tile[tile_index] = grid[grid_index];

        // Load the grid values for the ghost cells into shared memory
        if (get_local_id(0) == 0)
        {
            tile[0] = grid[gid - 1];
        }
        if (get_local_id(0) == 7 - 1)
        {
            tile[7 + 1] = grid[gid + 2];
        }

        // Wait for all threads to finish loading shared memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute the stencil operation in a single line
        float out = tile[get_local_id(0)] - 2 * tile[get_local_id(0) + 1] + tile[get_local_id(0) + 2];

        // Write the output value to global memory
        grid_out[gid] = out;
    }


__kernel void reaction_equation(__global double *n,
               const double s,
               const double F,
               const double n0,
               const double tau,
               const double sigma,
               __global const double *f,
               const double D,
               __global const double *n_D,
               const double step,
               const double dt)
{
    int gid = get_global_id(0);
    const int tile_size = get_local_size(0);

    __local double tile[64];

    for (int i = 0; i<tile_size; i++)
    {
        tile[i] = n[gid+i];
    }

    // Wait for all threads to finish loading shared memory
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i<tile_size; i++)
    {
        double n_val = tile[i];
        tile[i] += (s*F*(1-n_val/n0) - n_val/tau - sigma*f[i]*n_val + n_D[i]*D/step/step)*dt;
    }

    for (int i = 0; i<tile_size; i++)
    {
        n[gid+i] = tile[i];
    }

}


__kernel void stencil_1D(__global double *grid,
                        const double a,
                        __global double *grid_out)
{
    double sum = 0;
    int i = get_global_id(0) + 1;
    sum += a*(-2*grid[i] + \
               grid[i+1] + grid[i-1]);
    grid_out[i] = sum;

}

__kernel void stencil_2D(__global double *grid,
                        const double a,
                        __global double *grid_out)
{
    double sum = 0;
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;
    sum += a*(-4*grid[i,j] + \
               grid[i+1,j] + grid[i-1,j] + \
               grid[i, j+1] + grid[i, j-1]);
    grid_out[i,j] = sum;
}

__kernel void stencil_3D(__global double *grid,
                        const double a,
                        __global double *grid_out)
{
    double sum = 0;
    int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;
    int k = get_global_id(2) + 1;
    sum += a*(-6*grid[i,j, k] + \
               grid[i+1,j, k] + grid[i-1,j, k] + \
               grid[i, j+1, k] + grid[i, j-1, k]) + \
               grid[i, j, k+1] + grid[i, j , k-1];
    grid_out[i, j, k] = sum;
}
