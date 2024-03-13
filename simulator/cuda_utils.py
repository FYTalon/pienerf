from simulator.func_utils import *

@wp.kernel
def collect_param(
        topo: wp.array(dtype=wp.int32),
        mu: wp.array(dtype=wpfloat),
        lam: wp.array(dtype=wpfloat),
        mass: wp.array(dtype=wpfloat),
        IP_mu: wp.array(dtype=wpfloat),
        IP_lam: wp.array(dtype=wpfloat),
        IP_rho: wp.array(dtype=wpfloat)
):
    vid = wp.tid()

    IP = topo[vid]

    wp.atomic_add(IP_mu, IP, mu[vid] * mass[vid])
    wp.atomic_add(IP_lam, IP, lam[vid] * mass[vid])
    wp.atomic_add(IP_rho, IP, mass[vid])


@wp.kernel
def build_IP_global(
        dx: wpfloat,
        dt: wpfloat,
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        mu: wp.array(dtype=wpfloat),
        lam: wp.array(dtype=wpfloat),
        rho: wp.array(dtype=wpfloat),
        Nx: wp.array(shape=(0, 0), dtype=vec10),
        dNx: wp.array(shape=(0, 0, 0), dtype=vec10),
        ddNx: wp.array(shape=(0, 0, 0, 0), dtype=vec10),
        mat: wp.array(shape=(0, 0), dtype=wpfloat)
):
    idx = wp.tid()
    vid = idx // 6400
    i = idx % 6400 // 800
    x = idx % 800 // 80
    j = idx % 80 // 10
    y = idx % 10
    rho_v = rho[vid]
    mu_v = mu[vid]
    lam_v = lam[vid]

    r = topo[vid, i] * 10 + x
    c = topo[vid, j] * 10 + y

    wp.atomic_add(mat, r, c, rho_v * (dx ** wpfloat(3)) * Nx[vid, i][x] * Nx[vid, j][y] / (dt ** wpfloat(2)))

    for p in range(3):
        wp.atomic_add(mat, r, c, (dx ** wpfloat(3)) * (rho_v * (dx ** wpfloat(2)) / wpfloat(12) /
                                          (dt ** wpfloat(2)) + mu_v + lam_v) * dNx[vid, i, p][x] * dNx[vid, j, p][y])
        for q in range(3):
            wp.atomic_add(mat, r, c, (dx ** wpfloat(5)) * (mu_v + lam_v) / wpfloat(12) * \
                        ddNx[vid, i, p, q][x] * ddNx[vid, j, p, q][y])


@wp.kernel
def build_pin_global(
        stiff: wpfloat,
        vidx: wp.array(dtype=wp.int32),
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        Nx: wp.array(shape=(0, 0), dtype=vec10),
        mat: wp.array(shape=(0, 0), dtype=wpfloat)
):
    idx = wp.tid()
    vid = idx // 64
    vvid = vidx[vid]
    i = idx % 64 // 8
    j = idx % 8

    kidi = topo[vvid, i]
    kidj = topo[vvid, j]
    iNx = Nx[vvid, i]
    jNx = Nx[vvid, j]

    for x in range(10):
        for y in range(10):
            r = kidi * 10 + x
            c = kidj * 10 + y
            wp.atomic_add(mat, r, c, stiff * iNx[x] * jNx[y])

@wp.kernel
def calc_elastic(
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        dNx: wp.array(shape=(0, 0, 0), dtype=vec10),
        dof: wp.array(dtype=vec3),
        RF: wp.array(dtype=mat3),
        VF: wp.array(dtype=mat3),
        FF: wp.array(dtype=mat3)
):
    vid = wp.tid()

    F = mat3(wpfloat(0.0))

    for i in range(8):
        kernel = topo[vid, i]
        for x in range(10):
            dN = vec3(dNx[vid, i, 0][x], dNx[vid, i, 1][x], dNx[vid, i, 2][x])
            F += wp.outer(dof[kernel * 10 + x], dN)

    FF[vid] = F

    U = mat3()
    sig = vec3()
    V = mat3()
    wp.svd3(F, U, sig, V)
    VT = wp.transpose(V)
    S = mat3(
        sig[0], wpfloat(0.0), wpfloat(0.0),
        wpfloat(0.0), sig[1], wpfloat(0.0),
        wpfloat(0.0), wpfloat(0.0), sig[2])
    FF[vid] = U * S * VT

    RF[vid] = U * VT
    sig_ = volume_invariant_project(sig)
    S_ = mat3(
        sig_[0], wpfloat(0.0), wpfloat(0.0),
        wpfloat(0.0), sig_[1], wpfloat(0.0),
        wpfloat(0.0), wpfloat(0.0), sig_[2])
    VF[vid] = U * S_ * VT


@wp.kernel
def collect_rhs_IP(
        dx: wpfloat,
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        mu: wp.array(dtype=wpfloat),
        lam: wp.array(dtype=wpfloat),
        dNx: wp.array(shape=(0, 0, 0), dtype=vec10),
        rhs: wp.array(dtype=vec3),
        RF: wp.array(dtype=mat3),
        VF: wp.array(dtype=mat3)
):
    vid = wp.tid()

    mu_v = mu[vid]
    lam_v = lam[vid]

    R = RF[vid]
    V = VF[vid]

    for dir in range(8):
        kid = topo[vid, dir]
        for x in range(10):
            wp.atomic_add(rhs, kid * 10 + x, (dx ** wpfloat(3)) * (mu_v * R + lam_v * V) * \
                                 vec3(
                                     dNx[vid, dir, 0][x],
                                     dNx[vid, dir, 1][x],
                                     dNx[vid, dir, 2][x]
                                 ))

@wp.kernel
def collect_rhs_kernel(
        dx: wpfloat,
        buffer: wp.array(dtype=wp.vec2i),
        bg: wp.array(dtype=wp.int32),
        cnt: wp.array(dtype=wp.int32),
        mu: wp.array(dtype=wpfloat),
        lam: wp.array(dtype=wpfloat),
        dNx: wp.array(shape=(0, 0, 0), dtype=vec10),
        rhs: wp.array(dtype=vec3),
        RF: wp.array(dtype=mat3),
        VF: wp.array(dtype=mat3)
):
    tid = wp.tid()
    kid = tid // 10
    x = tid % 10

    bg_k = bg[kid]
    cnt_k = cnt[kid]

    for i in range(cnt_k):
        vid = buffer[bg_k + i][0]
        dir = buffer[bg_k + i][1]

        mu_v = mu[vid]
        lam_v = lam[vid]

        R = RF[vid]
        V = VF[vid]

        rhs[tid] += (dx ** wpfloat(3)) * (mu_v * R + lam_v * V) * \
                             vec3(
                                 dNx[vid, dir, 0][x],
                                 dNx[vid, dir, 1][x],
                                 dNx[vid, dir, 2][x]
                             )


@wp.kernel
def update_pos_kernel(
        pos: wp.array(dtype=vec3),
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        dof: wp.array(dtype=vec3),
        Nx: wp.array(shape=(0, 0), dtype=vec10)
):
    vid = wp.tid()

    for i in range(8):
        kid = topo[vid, i]
        for j in range(10):
            pos[vid] += Nx[vid, i][j] * dof[kid * 10 + j]


@wp.kernel
def update_F_kernel(
        pos: wp.array(dtype=vec3),
        F: wp.array(dtype=mat3),
        dF: wp.array(shape=(0, 0), dtype=mat3),
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        dof: wp.array(dtype=vec3),
        Nx: wp.array(shape=(0, 0), dtype=vec10),
        dNx: wp.array(shape=(0, 0, 0), dtype=vec10),
        ddNx: wp.array(shape=(0, 0, 0, 0), dtype=vec10)
):
    vid = wp.tid()

    for i in range(8):
        kid = topo[vid, i]
        for x in range(10):
            pos[vid] += Nx[vid, i][x] * dof[kid * 10 + x]
            F[vid] += wp.outer(dof[kid * 10 + x], vec3(
                dNx[vid, i, 0][x],
                dNx[vid, i, 1][x],
                dNx[vid, i, 2][x],
            ))
            for j in range(3):
                dF[vid, j] += wp.outer(dof[kid * 10 + x], vec3(
                    ddNx[vid, i, j, 0][x],
                    ddNx[vid, i, j, 1][x],
                    ddNx[vid, i, j, 2][x],
                ))

@wp.kernel
def count_IP_kernel(
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        cnt: wp.array(dtype=wp.int32)
):
    vid = wp.tid()

    for i in range(8):
        kid = topo[vid, i]
        wp.atomic_add(cnt, kid, wp.int32(1))


@wp.kernel
def allocate_IP_kernel(
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        cnt: wp.array(dtype=wp.int32),
        bg: wp.array(dtype=wp.int32),
        buffer: wp.array(dtype=wp.vec2i)
):
    vid = wp.tid()

    for i in range(8):
        kid = topo[vid, i]
        idx = wp.atomic_add(cnt, kid, wp.int32(1))
        buffer[bg[kid] + idx] = wp.vec2i(vid, i)


@wp.kernel
def collect_gravity(
        dx: wpfloat,
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        Nx: wp.array(shape=(0, 0), dtype=vec10),
        gravity: vec3,
        rho: wp.array(dtype=wpfloat),
        rhs: wp.array(dtype=vec3)
):

    vid = wp.tid()

    m = rho[vid] * dx * dx * dx

    for i in range(8):
        kid = topo[vid, i]
        for x in range(10):
            wp.atomic_add(rhs, kid * 10 + x, m * Nx[vid, i][x] * gravity)




