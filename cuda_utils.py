from func_utils import *

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

    IP_mu[IP] += mu[vid] * mass[vid]
    IP_lam[IP] += lam[vid] * mass[vid]
    IP_rho[IP] += mass[vid]


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

    mat[r, c] += rho_v * (dx ** wpfloat(3)) * Nx[vid, i][x] * Nx[vid, j][y] / (dt ** wpfloat(2))

    for p in range(3):
        mat[r, c] += (dx ** wpfloat(3)) * (rho_v * (dx ** wpfloat(2)) / wpfloat(12) /
                                          (dt ** wpfloat(2)) + mu_v + lam_v) * dNx[vid, i, p][x] * dNx[vid, j, p][y]
        for q in range(3):
            mat[r, c] += (dx ** wpfloat(5)) * (mu_v + lam_v) / wpfloat(12) * \
                        ddNx[vid, i, p, q][x] * ddNx[vid, j, p, q][y]


@wp.kernel
def build_pin_global(
        stiff: wpfloat,
        offset: wp.int32,
        vidx: wp.array(dtype=wp.int32),
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        Nx: wp.array(shape=(0, 0), dtype=vec10),
        mat: wp.array(shape=(0, 0), dtype=wpfloat)
):
    idx = wp.tid()
    vid = idx // 6400
    vvid = vidx[vid]
    i = idx % 6400 // 800
    x = idx % 800 // 80
    j = idx % 80 // 10
    y = idx % 10
    r = topo[vvid, i] * 10 + x
    c = topo[vvid, j] * 10 + y

    mat[r, c] += stiff * Nx[vvid, i][x] * Nx[vvid, j][y]

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
        dir: wp.int32,
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        mu: wp.array(dtype=wpfloat),
        lam: wp.array(dtype=wpfloat),
        dNx: wp.array(shape=(0, 0, 0), dtype=vec10),
        rhs: wp.array(dtype=vec3),
        RF: wp.array(dtype=mat3),
        VF: wp.array(dtype=mat3)
):
    vid = wp.tid()

    kid = topo[vid, dir]

    mu_v = mu[vid]
    lam_v = lam[vid]

    R = RF[vid]
    V = VF[vid]

    for x in range(10):
        rhs[kid * 10 + x] += (dx ** wpfloat(3)) * (mu_v * R + lam_v * V) * \
                             vec3(
                                 dNx[vid, dir, 0][x],
                                 dNx[vid, dir, 1][x],
                                 dNx[vid, dir, 2][x]
                             )


@wp.kernel
def collect_diag(
        diag: wp.array(dtype=wpfloat),
        row: wp.array(dtype=wp.int32),
        val: wp.array(dtype=wpfloat)
):
    vid = wp.tid()

    wp.atomic_add(diag, row[vid], val[vid])


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



