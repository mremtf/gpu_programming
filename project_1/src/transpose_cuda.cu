
__global__ void slow_transpose(float *odata, const float *idata)
{
	const size_t x = blockIdx.x * TILE_DIM + threadIdx.x;
	const size_t y = blockIdx.y * TILE_DIM + threadIdx.y;
	const size_t width = gridDim.x * TILE_DIM;
	
	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
		odata[x*width + (y+j)] = idata[(y+j)*width + x];
}
