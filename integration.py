import sympy as sp
x=sp.Symbol('x')
fx=3*x**2
print("F(x)=", sp.integrate(fx, x))
