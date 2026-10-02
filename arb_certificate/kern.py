# C1..C4 and PI are created at the ctx.prec active when this module is first imported,
# so each script deliberately sets its precision before or after `import kern`.
import os
from flint import arb, acb

RE    = arb(6371)
BETA  = arb('2.4541e-4');  GAMMA = arb('5.3881e-7')
EPSI  = arb('-2.7452e-11'); THETA = arb('8.3007e-14')
C1 = BETA*RE;  C2 = GAMMA*RE**2;  C3 = EPSI*RE**3;  C4 = THETA*RE**4
PI = arb.pi()

def q(z):
    c1, c2, c3, c4 = acb(C1), acb(C2), acb(C3), acb(C4)
    return z*(c1 + z*(c2 + z*(c3 + z*c4)))

def psi(z): return (-q(z)).exp()
def g(z):   return psi(z)*z.sin()

def at_pi():
    # returns q(pi), e^{-q(pi)} and the coefficients of Q(tau) = q(pi - tau)
    qpi = C1*PI + C2*PI**2 + C3*PI**3 + C4*PI**4
    Q1 = -(C1 + 2*C2*PI + 3*C3*PI**2 + 4*C4*PI**3)
    Q2 = (2*C2 + 6*C3*PI + 12*C4*PI**2)/2
    Q3 = -(6*C3 + 24*C4*PI)/6
    return qpi, (-qpi).exp(), (Q1, Q2, Q3, C4)

def int0pi(f, pieces):
    pts = [acb(PI*i/pieces) for i in range(pieces+1)]
    return sum((acb.integral(f, a, b) for a, b in zip(pts[:-1], pts[1:])), acb(0))

def bn(n):
    f = lambda z, _: g(z)*acb.legendre_p(z.cos(), n)
    return (acb(2*n+1)/2*int0pi(f, max(1, n))).real

def main_term(n):
    return C1/((arb(n) + arb(3)/2)*(arb(n) - arb(1)/2))

OUT = os.environ.get("CERT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)
def outpath(name): return os.path.join(OUT, name)
