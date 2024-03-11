import sympy as sp
x=sp.Symbol('x')
fx=x**2+4*x+3
print("f(x)=", sp.factor(fx))

fx=sp.sin(x)
print("f(x)=", sp.diff(fx,x))
