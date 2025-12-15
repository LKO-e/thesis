import sympy as sy


# Rotation matrices
def Rot(axis: str, angle: sy.Symbol):
    if axis == "x":
        R = [
            [1, 0, 0],
            [0, sy.cos(angle), sy.sin(angle)],
            [0, -sy.sin(angle), sy.cos(angle)],
        ]
    elif axis == "y":
        R = [
            [sy.cos(angle), 0, -sy.sin(angle)],
            [0, 1, 0],
            [sy.sin(angle), 0, sy.cos(angle)],
        ]
    elif axis == "z":
        R = [
            [sy.cos(angle), sy.sin(angle), 0],
            [-sy.sin(angle), sy.cos(angle), 0],
            [0, 0, 1],
        ]
    else:
        R = sy.zeros(3, 3)
    return sy.Matrix(R)


# Variables
U, phi, ω_e = sy.symbols("U φ V/Re")
dphi = sy.symbols("dφ")
psi, theta, gamma = sy.symbols("ψ ϑ γ")
dpsi, dtheta, dgamma = sy.symbols("dψ dϑ dγ")
ddpsi, ddtheta, ddgamma = sy.symbols("ddψ ddϑ ddγ")
β, α, dβ, dα = sy.symbols("β α dβ dα")
H = sy.symbols("H")
g = sy.symbols("g")
dV = sy.symbols("dV")
VRe1 = sy.symbols("V/Re") * sy.cos(theta)
dphi = VRe1 * sy.cos(-psi)
dlam = VRe1 * sy.sin(psi) / sy.cos(phi)
V = sy.symbols("V")
print("Angular velocity projections onto the b-frame")
W = (
    sy.Matrix([[dgamma, 0, 0]]).T
    + Rot("x", gamma) * sy.Matrix([[0, 0, dtheta]]).T
    + Rot("x", gamma) * Rot("z", theta) * sy.Matrix([[0, -dpsi, 0]]).T
    + Rot("x", gamma) * Rot("z", theta) * Rot("y", -psi) * sy.Matrix([[0, 0, -dphi]]).T
    + (
        Rot("x", gamma)
        * Rot("z", theta)
        * Rot("y", -psi)
        * Rot("z", -phi)
        * sy.Matrix([[dlam + U, 0, 0]]).T
    )
)
print("Wx = ")
print(sy.simplify(W[0]))
print("Wy = ")
print(sy.simplify(W[1]))
print("Wz = ")
print(sy.simplify(W[2]))
print()
# Acceleration calculation
print("Acceleration projections onto the OX''Y''Z'' frame")
W_ = (
    sy.Matrix([[0, 0, dtheta]]).T
    + Rot("z", theta) * sy.Matrix([[0, -dpsi, 0]]).T
    + Rot("z", theta) * Rot("y", -psi) * sy.Matrix([[0, 0, -dphi]]).T
    + (
        Rot("z", theta)
        * Rot("y", -psi)
        * Rot("z", -phi)
        * sy.Matrix([[dlam + 2 * U, 0, 0]]).T
    )
)
a = (
    sy.Matrix([[dV, 0, 0]]).T
    + Rot("z", theta) * sy.Matrix([[0, g, 0]]).T
    + W_.cross(sy.Matrix([[V, 0, 0]]).T)
)
print("a_x =")
print(sy.simplify(a[0]))
print("a_y =")
print(sy.simplify(a[1]))
print("a_z =")
print(sy.simplify(a[2]))
print()
print("Torques along the gyro pendulum suspension axis")
Hm = sy.symbols("Hm")
ml = sy.symbols("ml")
T_gp = (
    sy.Matrix([[0, 0, Hm]]).T.cross(W)[0]
    + (a[2] * sy.cos(gamma) - a[1] * sy.sin(gamma)) * ml
)
print(sy.simplify(T_gp))
