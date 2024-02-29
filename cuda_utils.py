import warp as wp
import torch
import numpy as np

torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda")
wp.set_device("cuda:0")

torchfloat = torch.float64
npfloat = np.float64
wpfloat = wp.float64

vec3 = wp.vec(length=3, dtype=wpfloat)
vec8i = wp.vec(length=8, dtype=wp.int32)
vec10 = wp.vec(length=10, dtype=wpfloat)

mat3 = wp.mat(shape=(3, 3), dtype=wpfloat)
mat10 = wp.mat(shape=(10, 10), dtype=wpfloat)


@wp.func
def w(r: wpfloat, p: vec3, q: vec3) -> wpfloat:
    d = wp.length(p - q) / r
    if d >= 1:
        return wpfloat(0.0)
    else :
        return (1.0 - d ** 2) ** 3


@wp.func
def dw(r: wpfloat, p: vec3, q: vec3) -> vec3:
    d = wp.length(p - q) / r
    if d >= 1:
        return vec3(0.0)
    else :
        return -6.0 * ((1.0 - d ** 2) ** 2) * (p - q) / (r ** 2)


@wp.func
def ddw(r: wpfloat, p: vec3, q: vec3) -> mat3:
    d = wp.length(p - q) / r
    if d >= 1:
        return mat3(0.0)
    else :
        return -6.0 * ((1.0 - d ** 2) ** 2) * wp.identity(n=3, dtype=wpfloat) / (r ** 2) + \
            24 * (1.0 - d ** 2) * wp.outer((p - q) / (r ** 2), (p - q) / (r ** 2))


@wp.func
def idx(x: wp.int32, y: wp.int32) -> wp.int32:
    if x > y:
        x, y = y, x

    if x == 0:
        return wp.int32(4 + y)
    else :
        return wp.int32(5 + x + y)


@wp.func
def P(p: vec3) -> vec10:
    return vec10(
        1,
        p[0], p[1], p[2],
        p[0] * p[0], p[0] * p[1], p[0] * p[2],
        p[1] * p[1], p[1] * p[2],
        p[2] * p[2]
    )


@wp.func
def Pj(p: vec3, j: wp.int32) -> vec10:
    a = vec10(0.0)
    a[j + 1] = wpfloat(1.0)
    for i in range(3):
        a[idx(i, j)] = p[i]
    a[idx(j, j)] += p[j]

    return a


@wp.func
def Pjk(p: vec3, j: wp.int32, k: wp.int32) -> vec10:
    a = vec10(0.0)
    a[idx(j, k)] = 1
    if j == k:
        a[idx(j, k)] += 1
    return a


@wp.kernel
def calc_G(
        r: wpfloat,
        pos: wp.array(dtype=vec3),
        topo: wp.array(dtype=vec8i),
        kernel_pos: wp.array(dtype=vec3),
        G: wp.array(dtype=mat10),
        dG: wp.array(dtype=mat10),
        ddG: wp.array(dtype=mat10)
):
    vid = wp.tid()
    G[vid] = mat10(0.0)
    for x in range(3):
        dG[vid, x] = mat10(0.0)
        for y in range(3):
            ddG[vid, x, y] = mat10(0.0)

    p = pos[vid]

    for i in range(8):
        q = kernel_pos[topo[vid][i]]
        Pi = P(q)
        weight = w(r, p, q)
        dweight = dw(r, p, q)
        ddweight = ddw(r, p, q)
        if weight <= 0.0:
            continue

        primitive = mat10(0.0)
        primitive += wp.outer(Pi, Pi)
        for j in range(3):
            Pij = Pj(q, j)
            primitive += wp.outer(Pij, Pij)
            for k in range(3):
                Pijk = Pjk(q, j, k)
                primitive += wp.outer(Pijk, Pijk)

        G[vid] += weight * primitive
        for x in range(3):
            dG[vid, x] += dweight[x] * primitive
            for y in range(3):
                ddG[vid, x, y] += ddweight[x, y] * primitive


@wp.kernel
def calc_Gp(
        pos: wp.array(dtype=vec3),
        G: wp.array(dtype=mat10),
        G_inv: wp.array(dtype=mat10),
        dG: wp.array(dtype=mat10),
        ddG: wp.array(dtype=mat10),
        Gp: wp.array(dtype=vec10),
        dGp: wp.array(dtype=vec10),
        ddGp: wp.array(dtype=vec10)
):
    vid = wp.tid()
    Gp[vid] = vec10(0.0)
    for x in range(3):
        dGp[vid, x] = vec10(0.0)
        for y in range(3):
            ddGp[vid, x, y] = vec10(0.0)

    p = pos[vid]
    Pv = P(p)

    Gp[vid] = G_inv[vid] * Pv
    for x in range(3):
        dPv = Pj(p, x)
        dGp[vid, x] = G_inv[vid] * dPv - G_inv[vid] * dG[vid, x] * G_inv[vid] * Pv

    for x in range(3):
        for y in range(3):
            ddPv = Pjk(p, x, y)
            ddGp[vid, x, y] = G[vid] * ddPv - \
                              G_inv[vid] * dG[vid, x] * G_inv[vid] * Pj(p, y) - \
                              G_inv[vid] * dG[vid, y] * G_inv[vid] * Pj(p, x) - \
                              G_inv[vid] * ddG[vid, x, y] * G_inv[vid] * Pv + \
                              G_inv[vid] * dG[vid, y] * G_inv[vid] * dG[vid, x] * G_inv[vid] * Pv + \
                              G_inv[vid] * dG[vid, x] * G_inv[vid] * dG[vid, y] * G_inv[vid] * Pv

@wp.kernel
def calc_weight(
        r: wpfloat,
        pos: wp.array(dtype=vec3),
        topo: wp.array(dtype=vec8i),
        kernel_pos: wp.array(dtype=vec3),
        Gp: wp.array(dtype=vec10),
        dGp: wp.array(dtype=vec10),
        ddGp: wp.array(dtype=vec10),
        Nx: wp.array(dtype=vec10),
        dNx: wp.array(dtype=vec10),
        ddNx: wp.array(dtype=vec10)
):
    vid = wp.tid()

    p = pos[vid]

    for i in range(8):
        q = kernel_pos[topo[vid][i]]
        Pi = P(q)
        weight = w(r, p, q)
        dweight = dw(r, p, q)
        ddweight = ddw(r, p, q)
        if weight <= 0.0:
            continue
        Nx[vid, i][0] = wp.dot(Gp[vid], Pi) * weight
        for j in range(3):
            dNx[vid, i, j][0] = wp.dot(Gp[vid], Pi) * dweight[j] + \
                                wp.dot(dGp[vid, j], Pi) * weight
            for k in range(3):
                ddNx[vid, i, j, k][0] = wp.dot(Gp[vid], Pi) * ddweight[j, k] + \
                                        wp.dot(dGp[vid, k], Pi) * dweight[j] + \
                                        wp.dot(dGp[vid, j], Pi) * dweight[k] + \
                                        wp.dot(ddGp[vid, j, k], Pi) * weight

        for x in range(3):
            Pix = P(q, x)
            Nx[vid, i][1 + x] = wp.dot(Gp[vid], Pix) * weight
            for j in range(3):
                dNx[vid, i, j][1 + x] = wp.dot(Gp[vid], Pix) * dweight[j] + \
                                    wp.dot(dGp[vid, j], Pix) * weight
                for k in range(3):
                    ddNx[vid, i, j, k][1 + x] = wp.dot(Gp[vid], Pix) * ddweight[j, k] + \
                                            wp.dot(dGp[vid, k], Pix) * dweight[j] + \
                                            wp.dot(dGp[vid, j], Pix) * dweight[k] + \
                                            wp.dot(ddGp[vid, j, k], Pix) * weight

            for y in range(3):
                Pixy = P(q, x, y)
                id = idx(x, y)
                Nx[vid, i][id] += wp.dot(Gp[vid], Pixy) * weight
                for j in range(3):
                    dNx[vid, i, j][id] += wp.dot(Gp[vid], Pixy) * dweight[j] + \
                                        wp.dot(dGp[vid, j], Pixy) * weight
                    for k in range(3):
                        ddNx[vid, i, j, k][id] += wp.dot(Gp[vid], Pixy) * ddweight[j, k] + \
                                                wp.dot(dGp[vid, k], Pixy) * dweight[j] + \
                                                wp.dot(dGp[vid, j], Pixy) * dweight[k] + \
                                                wp.dot(ddGp[vid, j, k], Pixy) * weight


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
