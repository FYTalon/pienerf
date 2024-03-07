#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
    float c = a; a = b; b = c;
}

inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __device__ int mip_from_dt(const float dt, const float H, const float max_cascade) {
    const float mx = dt * H * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}

inline __host__ __device__ uint32_t __expand_bits(uint32_t v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t xx = __expand_bits(x);
	uint32_t yy = __expand_bits(y);
	uint32_t zz = __expand_bits(z);
	return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x)
{
	x = x & 0x49249249;
	x = (x | (x >> 2)) & 0xc30c30c3;
	x = (x | (x >> 4)) & 0x0f00f00f;
	x = (x | (x >> 8)) & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}


////////////////////////////////////////////////////
/////////////           utils          /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// nears/fars: [N]
// scalar_t should always be float in use.
template <typename scalar_t>
__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    scalar_t * nears, scalar_t * fars
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);
//     float far_x = far;
//     float near_x = near;

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    if (near > far_z || near_z > far) {
//         printf("-------------------%f,%f,%f,%f,%f,%f,\n",near_x,near_y,near_z,far_x,far_y,far_z);
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;

    if (near < min_near) near = min_near;

    nears[n] = near;
    fars[n] = far;
}


void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "near_far_from_aabb", ([&] {
        kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), aabb.data_ptr<scalar_t>(), N, min_near, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>());
    }));
}


// rays_o/d: [N, 3]
// radius: float
// coords: [N, 2]
template <typename scalar_t>
__global__ void kernel_sph_from_ray(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const float radius,
    const uint32_t N,
    scalar_t * coords
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;
    coords += n * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    // const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // solve t from || o + td || = radius
    const float A = dx * dx + dy * dy + dz * dz;
    const float B = ox * dx + oy * dy + oz * dz; // in fact B / 2
    const float C = ox * ox + oy * oy + oz * oz - radius * radius;

    const float t = (- B + sqrtf(B * B - A * C)) / A; // always use the larger solution (positive)

    // solve theta, phi (assume y is the up axis)
    const float x = ox + t * dx, y = oy + t * dy, z = oz + t * dz;
    const float theta = atan2(sqrtf(x * x + z * z), y); // [0, PI)
    const float phi = atan2(z, x); // [-PI, PI)

    // normalize to [-1, 1]
    coords[0] = 2 * theta * RPI() - 1;
    coords[1] = phi * RPI();
}


void sph_from_ray(const at::Tensor rays_o, const at::Tensor rays_d, const float radius, const uint32_t N, at::Tensor coords) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "sph_from_ray", ([&] {
        kernel_sph_from_ray<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), radius, N, coords.data_ptr<scalar_t>());
    }));
}


// coords: int32, [N, 3]
// indices: int32, [N]
__global__ void kernel_morton3D(
    const int * __restrict__ coords,
    const uint32_t N,
    int * indices
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;
    indices[n] = __morton3D(coords[0], coords[1], coords[2]);
}


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D<<<div_round_up(N, N_THREAD), N_THREAD>>>(coords.data_ptr<int>(), N, indices.data_ptr<int>());
}


// indices: int32, [N]
// coords: int32, [N, 3]
__global__ void kernel_morton3D_invert(
    const int * __restrict__ indices,
    const uint32_t N,
    int * coords
) {
    // parallel
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    coords += n * 3;

    const int ind = indices[n];

    coords[0] = __morton3D_invert(ind >> 0);
    coords[1] = __morton3D_invert(ind >> 1);
    coords[2] = __morton3D_invert(ind >> 2);
}


void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords) {
    static constexpr uint32_t N_THREAD = 128;
    kernel_morton3D_invert<<<div_round_up(N, N_THREAD), N_THREAD>>>(indices.data_ptr<int>(), N, coords.data_ptr<int>());
}


// grid: float, [C, H, H, H]
// N: int, C * H * H * H / 8
// density_thresh: float
// bitfield: uint8, [N]
template <typename scalar_t>
__global__ void kernel_packbits(
    const scalar_t * __restrict__ grid,
    const uint32_t N,
    const float density_thresh,
    uint8_t * bitfield
) {
    // parallel per byte
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    grid += n * 8;

    uint8_t bits = 0;

    #pragma unroll
    for (uint8_t i = 0; i < 8; i++) {
        bits |= (grid[i] > density_thresh) ? ((uint8_t)1 << i) : 0;
    }

    bitfield[n] = bits;
}


void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grid.scalar_type(), "packbits", ([&] {
        kernel_packbits<<<div_round_up(N, N_THREAD), N_THREAD>>>(grid.data_ptr<scalar_t>(), N, density_thresh, bitfield.data_ptr<uint8_t>());
    }));
}

////////////////////////////////////////////////////
/////////////         training         /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, deltas: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const uint8_t * __restrict__ grid,
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M,
    const scalar_t* __restrict__ nears, 
    const scalar_t* __restrict__ fars,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * rays,
    int * counter,
    const scalar_t* __restrict__ noises
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    // ray marching
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;
    
    float t0 = near;
    
    // perturb
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max) * noise;

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    //if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);
    
    while (t < far && num_steps < max_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        //if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, num_steps);

        if (occ) {
            num_steps++;
            t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;

            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }

    //printf("[n=%d] num_steps=%d, near=%f, far=%f, dt=%f, max_steps=%f\n", n, num_steps, near, far, dt_min, (far - near) / dt_min);

    // second pass: really locate and write points & dirs
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);
    
    //printf("[n=%d] num_steps=%d, point_index=%d, ray_index=%d\n", n, num_steps, point_index, ray_index);

    // write rays
    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps > M) return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 2;

    t = t0;
    uint32_t step = 0;

    float last_t = t;

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1.0f, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        // query grid
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max); 
            } while (t < tt);
        }
    }
}

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises) {

    static constexpr uint32_t N_THREAD = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<uint8_t>(), bound, dt_gamma, max_steps, N, C, H, M, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), counter.data_ptr<int>(), noises.data_ptr<scalar_t>());
    }));
}


// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t M, const uint32_t N, const float T_thresh, 
    scalar_t * weights_sum,
    scalar_t * depth,
    scalar_t * image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > M) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image[index * 3] = 0;
        image[index * 3 + 1] = 0;
        image[index * 3 + 2] = 0;
        return;
    }

    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;

    // accumulate 
    uint32_t step = 0;

    scalar_t T = 1.0f;
    scalar_t r = 0, g = 0, b = 0, ws = 0, t = 0, d = 0;

    while (step < num_steps) {

        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];
        
        t += deltas[1]; // real delta
        d += weight * t;
        
        ws += weight;
        
        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;
    image[index * 3] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;
}


void composite_rays_train_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor image) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), M, N, T_thresh, weights_sum.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}


// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here 
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_image,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs, 
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image,
    const uint32_t M, const uint32_t N, const float T_thresh,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps > M) return;

    grad_weights_sum += index;
    grad_image += index * 3;
    weights_sum += index;
    image += index * 3;
    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;
    grad_sigmas += offset;
    grad_rgbs += offset * 3;

    // accumulate 
    uint32_t step = 0;
    
    scalar_t T = 1.0f;
    const scalar_t r_final = image[0], g_final = image[1], b_final = image[2], ws_final = weights_sum[0];
    scalar_t r = 0, g = 0, b = 0, ws = 0;

    while (step < num_steps) {
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];
        ws += weight;

        T *= 1.0f - alpha;
        
        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.
        // write grad_rgbs
        grad_rgbs[0] = grad_image[0] * weight;
        grad_rgbs[1] = grad_image[1] * weight;
        grad_rgbs[2] = grad_image[2] * weight;

        // write grad_sigmas
        grad_sigmas[0] = deltas[0] * (
            grad_image[0] * (T * rgbs[0] - (r_final - r)) + 
            grad_image[1] * (T * rgbs[1] - (g_final - g)) + 
            grad_image[2] * (T * rgbs[2] - (b_final - b)) +
            grad_weights_sum[0] * (1 - ws_final)
        );

        //printf("[n=%d] num_steps=%d, T=%f, grad_sigmas=%f, r_final=%f, r=%f\n", n, step, T, grad_sigmas[0], r_final, r);
        // minimal remained transmittence
        if (T < T_thresh) break;
        
        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        grad_sigmas++;
        grad_rgbs += 3;

        step++;
    }
}


void composite_rays_train_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor image, const uint32_t M, const uint32_t N, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_image.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(), grad_image.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), weights_sum.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), M, N, T_thresh, grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////          inference        ////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    const scalar_t* __restrict__ rays_t, 
    const scalar_t* __restrict__ rays_o, 
    const scalar_t* __restrict__ rays_d, 
    const float bound,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t C, const uint32_t H,
    const uint8_t * __restrict__ grid,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas,
    const scalar_t* __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    const float noise = noises[n];
    
    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 2;
    
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;
    
    float t = rays_t[index]; // current ray's t
    // const float near = nears[index];
    const float far = fars[index];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness
    t += clamp(t * dt_gamma, dt_min, dt_max) * noise;

    float last_t = t;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1, level), bound);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}


void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor near, const at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises) {
    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays", ([&] {
        kernel_march_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(
            n_alive, n_step, rays_alive.data_ptr<int>(), 
            rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), 
            bound, dt_gamma, max_steps, C, H, grid.data_ptr<uint8_t>(), 
            near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), 
            dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), noises.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const float T_thresh,
    int* rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ rgbs, 
    const scalar_t* __restrict__ deltas, 
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    
    // locate 
    sigmas += n * n_step;
    rgbs += n * n_step * 3;
    deltas += n * n_step * 2;
    
    rays_t += index;
    weights_sum += index;
    depth += index;
    image += index * 3;

    scalar_t t = rays_t[0]; // current ray's t
    
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t r = image[0];
    scalar_t g = image[1];
    scalar_t b = image[2];

    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

//         printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, weight_sum, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;
    image[0] = r;
    image[1] = g;
    image[2] = b;
}


void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    image.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////  inference with bending   ////////////
////////////////////////////////////////////////////

__device__ double dot(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ void dot31(const float* T, const float* V, float* M) // 3x3x3 tensor dot 3x1 vector
{
    M[0] = T[0] * V[0] + T[9] * V[1] + T[18] * V[2];
    M[1] = T[1] * V[0] + T[10] * V[1] + T[19] * V[2];
    M[2] = T[2] * V[0] + T[11] * V[1] + T[20] * V[2];
    M[3] = T[3] * V[0] + T[12] * V[1] + T[21] * V[2];
    M[4] = T[4] * V[0] + T[13] * V[1] + T[22] * V[2];
    M[5] = T[5] * V[0] + T[14] * V[1] + T[23] * V[2];
    M[6] = T[6] * V[0] + T[15] * V[1] + T[24] * V[2];
    M[7] = T[7] * V[0] + T[16] * V[1] + T[25] * V[2];
    M[8] = T[8] * V[0] + T[17] * V[1] + T[26] * V[2];
}

__device__ void mul31(const float* M, const float* V, float* R) // 3x3 matrix multiply 3x1 vector
{
    R[0] = M[0] * V[0] + M[3] * V[1] + M[6] * V[2];
    R[1] = M[1] * V[0] + M[4] * V[1] + M[7] * V[2];
    R[2] = M[2] * V[0] + M[5] * V[1] + M[8] * V[2];
}

__device__ float det3x3(const float* A) {
    return A[0] * (A[4] * A[8] - A[5] * A[7])
         - A[1] * (A[3] * A[8] - A[5] * A[6])
         + A[2] * (A[3] * A[7] - A[4] * A[6]);
}

__device__ void inv3x3(const float* A, float* A_inv)
{
    float det = det3x3(A);
    if (det == 0) return;
    float inv_det = 1.0f / det;
    A_inv[0] = inv_det * (A[4] * A[8] - A[5] * A[7]);
    A_inv[1] = inv_det * (A[2] * A[7] - A[1] * A[8]);
    A_inv[2] = inv_det * (A[1] * A[5] - A[2] * A[4]);
    A_inv[3] = inv_det * (A[5] * A[6] - A[3] * A[8]);
    A_inv[4] = inv_det * (A[0] * A[8] - A[2] * A[6]);
    A_inv[5] = inv_det * (A[2] * A[3] - A[0] * A[5]);
    A_inv[6] = inv_det * (A[3] * A[7] - A[4] * A[6]);
    A_inv[7] = inv_det * (A[1] * A[6] - A[0] * A[7]);
    A_inv[8] = inv_det * (A[0] * A[4] - A[1] * A[3]);
}

__device__ int find_three_IPs(
    const int n_grid,
    const int g0, const int g1, const int g2, const float res,
    const int* __restrict__ pig_cnt, const int* __restrict__ pig_bgn, const int* __restrict__ pig_idx,
    int* IPs)
{
    int gid = g2 * res * res + g1 * res + g0;
    int count = 0;
    while(count < 3 && count < pig_cnt[gid])
    {
        IPs[count] = pig_idx[pig_bgn[gid] + count];
        count++;
    }
    if(count == 3)  return count;

    int nb = 1;
    const int nns = (2*nb+1)*(2*nb+1)*(2*nb+1); //number of neighbors, 1, 27, 125
    int fghs[27];//adjust according to the actual size used
    int cnt = 0;
    for(int f=-nb; f<=nb; f++)
        for(int g=-nb; g<=nb; g++)
            for(int h=-nb; h<=nb; h++)
            {
//                 assert(3*cnt+2<nns*3);
                if(f==0 && g==0 && h==0) continue;
                fghs[3*cnt] = f;
                fghs[3*cnt+1] = g;
                fghs[3*cnt+2] = h;
                cnt++;
            }
//     printf("cnt=%i\n", cnt);
//     assert(cnt==nns-1);

    for (int k=0; k<cnt; k++)
    {
        int f = fghs[3 * k];
        int g = fghs[3 * k + 1];
        int h = fghs[3 * k + 2];
        gid = (g2 + f) * res * res + (g1 + g) * res + g0 + h;
        if(gid >= n_grid || gid < 0)  continue;
        for(int i=0; i<pig_cnt[gid]; i++)
        {
            IPs[count] = pig_idx[pig_bgn[gid] + i];
            count++;
            if(count==3) return count;
        }
    }
    return count;
}

template <typename scalar_t>
__global__ void kernel_march_rays_quadratic_bending(
    const int* __restrict__ pig_cnt, // IP in grid
    const int* __restrict__ pig_bgn,
    const int* __restrict__ pig_idx,
    const int n_vtx,
    const int n_grid,
    const float* __restrict__ p_ori, // rest positions of IPs
    const float* __restrict__ p_def, // deformed positions of IPs
    const float* __restrict__ F_IP, // deformation gradients at IPs
    const float* __restrict__ dF_IP, // nabla deformation gradients at IPs

    const float* __restrict__ bbmin,
    const float hgs,
    const int resolution,

    const float def_margin,

    const uint32_t n_alive,
    const uint32_t n_step,
    const int* __restrict__ rays_alive,
    const scalar_t* __restrict__ rays_t,
    const scalar_t* __restrict__ rays_o,
    const scalar_t* __restrict__ rays_d,
    const float bound,//
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t C, const uint32_t H,
    const uint8_t * __restrict__ grid,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas,
    const scalar_t* __restrict__ noises
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    const float noise = noises[n];

    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)H;
    const float H3 = H * H * H;

    float t = rays_t[index]; // current ray's t
    // const float near = nears[index];
    const float far = fars[index];

    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (C - 1)) / H;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness
    t += clamp(t * dt_gamma, dt_min, dt_max) * noise;

    float last_t = t;

//     const float margined_bound = def_margin * bound;
    const float margined_bound = bound;

    while (t < far && step < n_step)
    {
        bool found = true;

        float x = clamp(ox + t * dx, -margined_bound, margined_bound);
        float y = clamp(oy + t * dy, -margined_bound, margined_bound);
        float z = clamp(oz + t * dz, -margined_bound, margined_bound);

        //raybending-------------kernel_march_rays_qb-----------------------------------------------------------------
        float x_map = 0.0;
        float y_map = 0.0;
        float z_map = 0.0;
        int g0 = floor((x - bbmin[0])/hgs);
        int g1 = floor((y - bbmin[1])/hgs);
        int g2 = floor((z - bbmin[2])/hgs);
        if (g0 == resolution)  g0 = resolution-1;
        if (g1 == resolution)  g1 = resolution-1;
        if (g2 == resolution)  g2 = resolution-1;
        if (g0 < 0 || g1 < 0 ||g2 < 0 || g0 >= resolution || g1 >= resolution ||g2 >= resolution)
            printf("ERROR: g0=%i, g1=%i, g2=%i, xyz:(%f,%f,%f), bbmin:(%f,%f,%f)\n", g0, g1, g2, x, y, z, bbmin[0], bbmin[1], bbmin[2]);
        int gid = g2 * resolution * resolution + g1 * resolution + g0;
        if (gid < 0 || gid >= n_grid)
            printf("ERROR: g0=%i, g1=%i, g2=%i, gid=%i, n_grid=%i\n", g0, g1, g2, gid, n_grid);
        int IPs[3] = {-1,-1,-1};
//         printf("IPs: %i,%i,%i\n", IPs[0], IPs[1], IPs[2]);

        int n_IP = find_three_IPs(n_grid, g0, g1, g2, resolution, pig_cnt, pig_bgn, pig_idx, IPs);
//         printf("IPs[0]: %i\n", IPs[0]);
//         printf("IPs: %i,%i,%i, n_IP=%i\n", IPs[0], IPs[1], IPs[2], n_IP);

        if(n_IP == 0)
            found = false;
        else
            found = true;
        if(found)
        {
            float ip_weight[3] = {0.0, 0.0, 0.0};
            if (n_IP == 1)
            {
                ip_weight[0] = 1.0;
            }
            else if (n_IP == 2)
            {
                float dist[2] = {0.0, 0.0};
                for(int k=0; k<2; k++)
                {
                    const float* pk_ = &p_def[IPs[k] * 3];
                    dist[k] = sqrt((pk_[0]-x)*(pk_[0]-x) + (pk_[1]-y)*(pk_[1]-y) + (pk_[2]-z)*(pk_[2]-z));
                }
                float dist_sum = dist[0] + dist[1];
                ip_weight[0] = dist[1] / dist_sum;
                ip_weight[1] = dist[0] / dist_sum;
            }
            else if (n_IP == 3)
            {
                float dist[3] = {0.0, 0.0, 0.0};
                for(int k=0; k<3; k++)
                {
                    const float* pk_ = &p_def[IPs[k] * 3];
                    dist[k] = sqrt((pk_[0]-x)*(pk_[0]-x) + (pk_[1]-y)*(pk_[1]-y) + (pk_[2]-z)*(pk_[2]-z));
                }
                float dist_sum = dist[0] * dist[1] + dist[1] * dist[2] + dist[2] * dist[0];
                ip_weight[0] = dist[1] * dist[2] / dist_sum;
                ip_weight[1] = dist[0] * dist[2] / dist_sum;
                ip_weight[2] = dist[0] * dist[1] / dist_sum;
            }

            const float p_[3] = {x, y, z};//deformed position
            for(int k=0; k<n_IP; k++)
            {
                assert(k < 3);
                assert(IPs[k] != -1);
                assert(IPs[k] < n_vtx);
                const float* pk = &p_ori[IPs[k] * 3];
                const float* pk_ = &p_def[IPs[k] * 3];
                const float* Fk = &F_IP[IPs[k] * 9];
                const float* dFk = &dF_IP[IPs[k] * 27];
                float p[3] = {pk[0], pk[1], pk[2]};//initial rest position, using deformed position as initial guess
                int num_itr = 0;
                while(num_itr < 10)
                {
                    // left hand side
                    float A[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    float dF_x[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};//dF dot x
                    dot31(dFk, p, dF_x);
                    float dF_xk[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};//dF dot xk
                    dot31(dFk, pk, dF_xk);
                    for(int j=0; j<9; j++)
                    {
                        A[j] = Fk[j] + 0.5 * dF_x[j] - dF_xk[j];
                    }
                    // right hand side
                    float b[3] = {0.0, 0.0, 0.0};
                    float F_xk[3] = {0.0, 0.0, 0.0};//F * xk
                    mul31(Fk, pk, F_xk);
                    float dF_xk_xk[3] = {0.0, 0.0, 0.0};
                    mul31(dF_xk, pk, dF_xk_xk);
                    for(int i=0; i<3; i++)
                    {
                        b[i] = p_[i] - pk_[i] + F_xk[i] - 0.5 * dF_xk_xk[i];
                    }
                    float A_inv[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    inv3x3(A, A_inv);
                    float p_last[3] = {p[0], p[1], p[2]};
                    mul31(A_inv, b, p);
                    if ((p_last[0]-p[0])*(p_last[0]-p[0]) + (p_last[1]-p[1])*(p_last[1]-p[1]) + (p_last[2]-p[2])*(p_last[2]-p[2]) < 1e-6)
                    {
//                         if (num_itr>3)
//                             printf("converged itr=%i\n", num_itr);
                        x_map += ip_weight[k] * p[0];
                        y_map += ip_weight[k] * p[1];
                        z_map += ip_weight[k] * p[2];
                        break;
                    }
//                     if (num_itr>3)
//                         printf("itr=%i\n", num_itr);
                    num_itr++;
                }


//                     x = p[0];
//                     y = p[1];
//                     z = p[2];
            }
        }
        else // not found
        {

        }


        //end raybending--------------------------------------------------------------------------------

        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, C), mip_from_dt(dt, H, C)); // range in [0, C - 1]

        const float mip_bound = fminf(scalbnf(1, level), margined_bound);
        const float mip_rbound = 1 / mip_bound;

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ && found) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do {
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}

void march_rays_quadratic_bending(
    const at::Tensor pig_cnt, const at::Tensor pig_bgn, const at::Tensor pig_idx,
    const int n_vtx, const int n_grid,
    const at::Tensor p_def, const at::Tensor p_ori,
    const at::Tensor F_IP, const at::Tensor dF_IP,
    const at::Tensor bbmin, const float hgs, const int resolution,
    const float def_margin,

    const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive,
    const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d,
    const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H,
    const at::Tensor grid, const at::Tensor near, const at::Tensor far,
    at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises
    )
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        at::ScalarType::Float, "march_rays_quadratic_bending", ([&] {
        kernel_march_rays_quadratic_bending<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(
            pig_cnt.data_ptr<int>(), pig_bgn.data_ptr<int>(), pig_idx.data_ptr<int>(),
            n_vtx, n_grid,
            p_ori.data_ptr<float>(), p_def.data_ptr<float>(),
            F_IP.data_ptr<float>(), dF_IP.data_ptr<float>(),
            bbmin.data_ptr<float>(), hgs, resolution,
            def_margin,

            n_alive, n_step, rays_alive.data_ptr<int>(),
            rays_t.data_ptr<float>(), rays_o.data_ptr<float>(), rays_d.data_ptr<float>(),
            bound, dt_gamma, max_steps, C, H, grid.data_ptr<uint8_t>(),
            near.data_ptr<float>(), far.data_ptr<float>(), xyzs.data_ptr<float>(),
            dirs.data_ptr<float>(), deltas.data_ptr<float>(), noises.data_ptr<float>());
    }));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

