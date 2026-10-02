import math
from math import comb
from flint import arb, acb, arb_series, ctx
from kern import C1, C2, C3, C4

def theta2_coef(n):
    return arb(2)**(n+1)/(arb(n)**2*arb(comb(2*n, n)))

def _horner(s, coefs, N):
    out = arb_series([coefs[-1]], N)
    for m in range(len(coefs)-2, -1, -1):
        out = out*s + arb_series([coefs[m]], N)
    return out

def AB(c1, c2, c3, c4, N):
    T  = arb_series([arb(0)] + [theta2_coef(n) for n in range(1, N)], N)  # theta^2
    h  = arb_series([theta2_coef(n)/2 for n in range(1, N+1)], N).sqrt()  # sqrt(theta^2/(2v))
    qe = arb_series([c2], N)*T + arb_series([c4], N)*T*T                  # even part of q
    r  = arb_series([c1], N) + arb_series([c3], N)*T
    s  = T*r*r                                                            # (odd part of q)^2
    ch = _horner(s, [1/arb.fac_ui(2*m)   for m in range(N+1)], N)         # cosh(sqrt s)
    sh = _horner(s, [1/arb.fac_ui(2*m+1) for m in range(N+1)], N)         # sinh(sqrt s)/sqrt s
    E  = (-qe).exp()
    A  = E*ch
    B  = arb_series([-arb(2).sqrt()], N)*h*r*E*sh
    pad = lambda S: list(S.coeffs()) + [arb(0)]*(N - len(S.coeffs()))
    return pad(A), pad(B)

_A = []
def _acoef(N):
    if len(_A) < N+2:
        _A[:] = [arb(0)] + [theta2_coef(n) for n in range(1, N+2)]
    return _A

def AB_at(z, c1, c2, c3, c4, rho, N=200, M=260):
    # rigorous tails: a_{n+1}/a_n <= 1/2 gives a geometric bound for the omitted terms
    a = _acoef(N)
    t2 = acb(0); p = acb(1)
    for n in range(1, N+1):
        p = p*z; t2 = t2 + acb(a[n])*p
    tl = (a[N+1]*arb(rho)**(N+1)/(1 - arb(rho)/2)).upper()
    t2 = t2 + acb(arb(0, tl), arb(0, tl))
    qe = acb(c2)*t2 + acb(c4)*t2*t2
    r  = acb(c1) + acb(c3)*t2
    s  = t2*r*r
    def powsum(coefs):
        out = acb(0); p = acb(1)
        for c in coefs: out = out + acb(c)*p; p = p*s
        return out
    ch = powsum([1/arb.fac_ui(2*m)   for m in range(M)])
    sc = powsum([1/arb.fac_ui(2*m+1) for m in range(M)])
    E = (-qe).exp()
    h2 = acb(0); p = acb(1)
    for n in range(1, N+1): h2 = h2 + acb(a[n]/2)*p; p = p*z
    tl = (a[N+1]*arb(rho)**N/(2*(1 - arb(rho)/2))).upper()
    h2 = h2 + acb(arb(0, tl), arb(0, tl))
    return E*ch, -acb(arb(2).sqrt())*h2.sqrt()*r*E*sc

def circle_max(fn, rho, M=1440):
    best = arb(0)
    for i in range(M):
        lo = 2*math.pi*i/M; hi = 2*math.pi*(i+1)/M; mid = (lo+hi)/2
        cx = arb(rho)*arb(math.cos(mid)); cy = arb(rho)*arb(math.sin(mid))
        rad = float(arb(rho))*2*math.sin((hi-lo)/4)*1.0001
        m = abs(fn(acb(arb(cx.mid(), rad), arb(cy.mid(), rad))))
        if m.upper() > best.upper(): best = arb(m.upper())
    return best

def coeffs_and_tail(c1, c2, c3, c4, NT, rho, scale=arb(1)):
    ctx.cap = NT
    A, B = AB(c1, c2, c3, c4, NT)
    A = [abs(x)*scale for x in A]; B = [abs(x)*scale for x in B]
    MA = circle_max(lambda z: AB_at(z, c1, c2, c3, c4, float(rho))[0], rho, 720)*scale
    MB = circle_max(lambda z: AB_at(z, c1, c2, c3, c4, float(rho))[1], rho, 720)*scale
    return A, B, MA, MB

def ffact(a, r):
    p = arb(1)
    for i in range(r): p *= (a-i)
    return p

def afact(a, r):
    p = arb(1)
    for i in range(r): p *= abs(a-i)
    return p

def cf(lst, M, rho, k):
    return lst[k] if k < len(lst) else M/arb(rho)**k

def L1_origin(K, delta, A, B, MA, MB, rho, KMAX):
    d = arb(delta)
    a_part = arb(0)
    for k in range(K, KMAX):
        a_part += ffact(arb(k), K)*cf(A, MA, rho, k)*d**(k-K+1)/(k-K+1)
    P = [arb(1)]
    for j in range(1, K+1): P.append(P[-1]*(arb(2*K+1)/2 - (j-1)))
    b_part = arb(0)
    for m in range(0, KMAX-K):
        W = arb(0)
        for j in range(max(0, K-m), K+1):
            W += arb(math.comb(K, j))*P[j]*ffact(arb(m), K-j)
        b_part += cf(B, MB, rho, K+m)*W*d**(arb(2*m+3)/2)/(arb(2*m+3)/2)
    return a_part + b_part

def L1_model_far(K, B, lo, hi):
    tot = arb(0)
    for k in range(K):
        e = arb(2*k+1)/2 - K
        I = (arb(hi)**(e+1) - arb(lo)**(e+1))/(e+1)
        tot += B[k]*afact(arb(2*k+1)/2, K)*abs(I)
    return tot

def L1_model_end(K, B, width, at):
    # exponents k+1/2-K are negative, so the sup over [at, 2] is attained at v = at
    return sum(B[k]*afact(arb(2*k+1)/2, K)*width*at**(arb(2*k+1)/2-K) for k in range(K))

def L1_f(K, ulo, uhi, M):
    ctx.cap = K+1
    tot = arb(0); h = (arb(uhi) - arb(ulo))/M
    S = lambda c: arb_series([c], K+1)
    for i in range(M):
        c = arb(ulo) + h*(i + arb('0.5')); c = arb(c.mid(), (h/2).mid()*1.000001)
        th = arb_series([c, arb(1)], K+1).acos()
        co = list((-(th*(S(C1) + th*(S(C2) + th*(S(C3) + th*S(C4)))))).exp().coeffs())
        d = abs(co[K])*arb.fac_ui(K) if len(co) > K else arb(0)
        tot += h*arb(d.upper())
    return tot

def tail_norm(K, delta, dp, origin, antipode, KMAX, KMAXA, Mf, antipode_allowance=True):
    A, B, MA, MB, rho = origin
    At, Gm, MAt, MGm, rhoA = antipode
    def geo(d, r0, KM):   # geometric allowance for the series terms beyond KM
        r = (arb(d)/arb(r0))*arb(KM+1)/(KM+1-K); assert r < 1, (d, r0, KM); return 1 + r/(1-r)
    delta, dp = arb(delta), arb(dp)
    P = {}
    P["I  origin cancelled"]        = L1_origin(K, delta, A, B, MA, MB, rho, KMAX)*geo(delta, rho, KMAX)
    P["II origin model far"]        = L1_model_far(K, B, delta, 2-dp)
    P["IIf |f^(K)|"]                = L1_f(K, float(-1+dp), float(1-delta), Mf)
    P["III antipode cancelled"]     = L1_origin(K, dp, At, Gm, MAt, MGm, rhoA, KMAXA)
    if antipode_allowance:
        P["III antipode cancelled"] *= geo(dp, rhoA, KMAXA)
    P["IIg antipode model far"]     = L1_model_far(K, Gm, dp, 2-delta)
    P["IIIb origin model near pi"]  = L1_model_end(K, B, dp, 2-dp)
    P["IIIc antipode model near 0"] = L1_model_end(K, Gm, delta, 2-delta)
    total = arb(0)
    for v in P.values(): total += v
    return total, P

def bk(n, k):   # closed form of b_n[(1-u)^{k+1/2}]
    s = arb(2*k+1)/2; num = arb(2)**s*(2*n+1)*(1+s).gamma()/(-s).gamma()
    den = arb(1)
    for j in range(2*k+3): den *= (arb(n) - s + j)
    return num/den

def Phi(n, K):
    p = arb(1)
    for i in range(K): p *= (2*n-2*i-1)
    return (2*n+1)*arb(2)**(K-1)/p

def ineq9(n, K, c1, psip_abs, Rn, eps):
    # for n >= K every term on the right decreases in n, so once (9) holds it holds for all larger n
    lhs = c1 - psip_abs
    rhs = Phi(n, K)*Rn + sum(eps[k]*abs(bk(n, k)) for k in range(1, K))
    rhs = (arb(n) + arb(3)/2)*(arb(n) - arb(1)/2)*rhs
    return bool(lhs > rhs), lhs, rhs

def least_N0(K, c1, psip_abs, Rn, eps, nmax=40000):
    for n in range(K, nmax):
        if ineq9(n, K, c1, psip_abs, Rn, eps)[0]: return n
    return None
