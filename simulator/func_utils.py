import warp as wp
import torch
import numpy as np

# torch.set_default_dtype(torch.float64)
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
def volume_invariant_project(sig: vec3) -> vec3:
    sig_ = vec3()
    D = vec3(wpfloat(0.0))
    for i in range(3):
        C = (sig[0] + D[0]) * (sig[1] + D[1]) * (sig[2] + D[2]) - wpfloat(1.0)
        dC = vec3(
            (sig[1] + D[1]) * (sig[2] + D[2]),
            (sig[0] + D[0]) * (sig[2] + D[2]),
            (sig[0] + D[0]) * (sig[1] + D[1]),
        )
        dCTD = wp.dot(dC, D)
        coef = (dCTD - C) / wp.dot(dC, dC)
        D[0] = coef * dC[0]
        D[1] = coef * dC[1]
        D[2] = coef * dC[2]
    sig_[0] = sig[0] + D[0]
    sig_[1] = sig[1] + D[1]
    sig_[2] = sig[2] + D[2]
    return sig_


@wp.func
def w(r: wpfloat, p: vec3, q: vec3) -> wpfloat:
    d = wp.length(p - q) / r
    if d >= 1:
        return wpfloat(0.0)
    else :
        return (wpfloat(1.0) - d ** wpfloat(2)) ** wpfloat(3)


@wp.func
def dw(r: wpfloat, p: vec3, q: vec3) -> vec3:
    d = wp.length(p - q) / r
    if d >= 1:
        return vec3(wpfloat(0.0))
    else :
        return -wpfloat(6.0) * ((wpfloat(1.0) - d ** wpfloat(2)) ** wpfloat(2)) * (p - q) / (r ** wpfloat(2))


@wp.func
def ddw(r: wpfloat, p: vec3, q: vec3) -> mat3:
    d = wp.length(p - q) / r
    if d >= 1:
        return mat3(wpfloat(0.0))
    else :
        return -wpfloat(6.0) * ((wpfloat(1.0) - d ** wpfloat(2)) ** wpfloat(2)) * \
            wp.identity(n=3, dtype=wpfloat) / (r ** wpfloat(2)) + \
            wpfloat(24) * (wpfloat(1.0) - d ** wpfloat(2)) * \
            wp.outer((p - q) / (r ** wpfloat(2)), (p - q) / (r ** wpfloat(2)))


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
        wpfloat(1),
        p[0], p[1], p[2],
        p[0] * p[0], p[0] * p[1], p[0] * p[2],
        p[1] * p[1], p[1] * p[2],
        p[2] * p[2]
    )


@wp.func
def Pj(p: vec3, j: wp.int32) -> vec10:
    a = vec10(wpfloat(0.0))
    a[j + 1] = wpfloat(1.0)
    for i in range(3):
        a[idx(i, j)] = p[i]
    a[idx(j, j)] += p[j]

    return a


@wp.func
def Pjk(p: vec3, j: wp.int32, k: wp.int32) -> vec10:
    a = vec10(wpfloat(0.0))
    a[idx(j, k)] = wpfloat(1)
    if j == k:
        a[idx(j, k)] += wpfloat(1)
    return a
