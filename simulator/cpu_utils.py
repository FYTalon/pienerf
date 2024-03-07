from .func_utils import *

@wp.kernel
def calc_G(
        r: wpfloat,
        pos: wp.array(dtype=vec3),
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        kernel_pos: wp.array(dtype=vec3),
        G: wp.array(dtype=mat10),
        dG: wp.array(shape=(0, 0), dtype=mat10),
        ddG: wp.array(shape=(0, 0, 0), dtype=mat10)
):
    vid = wp.tid()
    G[vid] = mat10(wpfloat(0.0))
    for x in range(3):
        dG[vid, x] = mat10(wpfloat(0.0))
        for y in range(3):
            ddG[vid, x, y] = mat10(wpfloat(0.0))

    p = pos[vid]

    for i in range(8):
        q = kernel_pos[topo[vid, i]]
        Pi = P(q)
        weight = w(r, p, q)
        dweight = dw(r, p, q)
        ddweight = ddw(r, p, q)
        if weight <= 0.0:
            continue

        primitive = mat10(wpfloat(0.0))
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
        dG: wp.array(shape=(0, 0), dtype=mat10),
        ddG: wp.array(shape=(0, 0, 0), dtype=mat10),
        Gp: wp.array(dtype=vec10),
        dGp: wp.array(shape=(0, 0), dtype=vec10),
        ddGp: wp.array(shape=(0, 0, 0), dtype=vec10)
):
    vid = wp.tid()
    Gp[vid] = vec10(wpfloat(0.0))
    for x in range(3):
        dGp[vid, x] = vec10(wpfloat(0.0))
        for y in range(3):
            ddGp[vid, x, y] = vec10(wpfloat(0.0))

    p = pos[vid]
    Pv = P(p)

    G_i = G_inv[vid]
    G_ = G[vid]

    Gp[vid] = G_i * Pv

    for x in range(3):
        dPv = Pj(p, x)
        dGp[vid, x] = G_i * dPv - G_i * dG[vid, x] * G_i * Pv

    for x in range(3):
        dPv = Pj(p, x)
        dGx = dG[vid, x]
        for y in range(3):
            ddPv = Pjk(p, x, y)
            ddGp[vid, x, y] = G_i * ddPv - \
                              G_i * dGx * G_i * Pj(p, y) - \
                              G_i * dG[vid, y] * G_i * dPv - \
                              G_i * ddG[vid, x, y] * G_i * Pv + \
                              G_i * dG[vid, y] * G_i * dGx * G_i * Pv + \
                              G_i * dGx * G_i * dG[vid, y] * G_i * Pv

@wp.kernel
def calc_weight(
        r: wpfloat,
        pos: wp.array(dtype=vec3),
        topo: wp.array(shape=(0, 0), dtype=wp.int32),
        kernel_pos: wp.array(dtype=vec3),
        Gp: wp.array(dtype=vec10),
        dGp: wp.array(shape=(0, 0), dtype=vec10),
        ddGp: wp.array(shape=(0, 0, 0), dtype=vec10),
        Nx: wp.array(shape=(0, 0), dtype=vec10),
        dNx: wp.array(shape=(0, 0, 0), dtype=vec10),
        ddNx: wp.array(shape=(0, 0, 0, 0), dtype=vec10)
):
    vid = wp.tid()

    p = pos[vid]

    Gp_ = Gp[vid]

    for i in range(8):
        q = kernel_pos[topo[vid, i]]
        Pi = P(q)
        weight = w(r, p, q)
        dweight = dw(r, p, q)
        ddweight = ddw(r, p, q)
        if weight <= 0.0:
            continue
        Nx[vid, i][0] = wp.dot(Gp_, Pi) * weight
        for j in range(3):
            dGpj = dGp[vid, j]
            dNx[vid, i, j][0] = wp.dot(Gp_, Pi) * dweight[j] + \
                                wp.dot(dGpj, Pi) * weight
            for k in range(3):
                ddNx[vid, i, j, k][0] = wp.dot(Gp_, Pi) * ddweight[j, k] + \
                                        wp.dot(dGp[vid, k], Pi) * dweight[j] + \
                                        wp.dot(dGpj, Pi) * dweight[k] + \
                                        wp.dot(ddGp[vid, j, k], Pi) * weight

        for x in range(3):
            Pix = Pj(q, x)
            Nx[vid, i][1 + x] = wp.dot(Gp_, Pix) * weight
            for j in range(3):
                dGpj = dGp[vid, j]
                dNx[vid, i, j][1 + x] = wp.dot(Gp_, Pix) * dweight[j] + \
                                    wp.dot(dGpj, Pix) * weight
                for k in range(3):
                    ddNx[vid, i, j, k][1 + x] = wp.dot(Gp_, Pix) * ddweight[j, k] + \
                                            wp.dot(dGp[vid, k], Pix) * dweight[j] + \
                                            wp.dot(dGpj, Pix) * dweight[k] + \
                                            wp.dot(ddGp[vid, j, k], Pix) * weight

            for y in range(3):
                Pixy = Pjk(q, x, y)
                id = idx(x, y)
                Nx[vid, i][id] += wp.dot(Gp_, Pixy) * weight
                for j in range(3):
                    dGpj = dGp[vid, j]
                    dNx[vid, i, j][id] += wp.dot(Gp_, Pixy) * dweight[j] + \
                                        wp.dot(dGpj, Pixy) * weight
                    for k in range(3):
                        ddNx[vid, i, j, k][id] += wp.dot(Gp_, Pixy) * ddweight[j, k] + \
                                                wp.dot(dGp[vid, k], Pixy) * dweight[j] + \
                                                wp.dot(dGpj, Pixy) * dweight[k] + \
                                                wp.dot(ddGp[vid, j, k], Pixy) * weight
