using LinearAlgebra
using Printf
using Plots

gr()

Id(n) = Matrix{Float64}(I, n, n)
spectral_radius(M) = maximum(abs.(eigvals(Matrix(M))))

function eq_matrix(A, B, Γ)
    n = size(A, 1)
    S = B' * B + Γ
    P = B * (S \ B')
    return (Id(n) - P) * A
end

function iterative_matrices(A, B, Γ, protocol)
    n = size(A, 1)
    S = B' * B + Γ
    D = Matrix(Diagonal(diag(S)))
    L = tril(S, -1)

    if protocol == :PBR
        M = D
        N = D - S
    elseif protocol == :SBR
        M = D + L
        N = -L'
    else
        error("protocol must be :PBR or :SBR")
    end

    T = M \ N
    Q = B * (M \ B')

    Aaug = [
        (Id(n) - Q) * A     B * T
        -(M \ (B' * A))     T
    ]

    return T, Aaug
end

mkpath("figures")

# ------------------------------------------------------------
# Experiment 1: primitive marginal network
# ------------------------------------------------------------

Apr = [
    0.5  0.5  0.0
    0.2  0.5  0.3
    0.1  0.3  0.6
]

Bpr = [
    1.0  0.2
    0.5  0.8
    0.2  1.0
]

ts = 10.0 .^ range(-2, 3, length=400)

rho_primitive = [
    spectral_radius(eq_matrix(Apr, Bpr, t * Id(2)))
    for t in ts
]

# ------------------------------------------------------------
# Experiment 2: periodic counterexample
# ------------------------------------------------------------

Aper = [
    0.0  0.0  1.0
    0.0  0.0  1.0
    0.9  0.1  0.0
]

Bper = reshape([0.0, 10.0, 5.0], 3, 1)

gammas_periodic = 10.0 .^ range(-1, 4, length=400)

rho_periodic = [
    spectral_radius(eq_matrix(Aper, Bper, γ * Id(1)))
    for γ in gammas_periodic
]

p1 = plot(
    ts,
    rho_primitive;
    xscale=:log10,
    xlabel="penalty scale t",
    ylabel="spectral radius",
    label="primitive network",
    linewidth=2
)
hline!(p1, [1.0]; label="stability boundary", linestyle=:dash)

p2 = plot(
    gammas_periodic,
    rho_periodic;
    xscale=:log10,
    xlabel="penalty γ",
    ylabel="spectral radius",
    label="periodic network",
    linewidth=2
)
hline!(p2, [1.0]; label="stability boundary", linestyle=:dash)

plot(
    p1,
    p2;
    layout=(1, 2),
    size=(1000, 400),
    margin=5Plots.mm
)
savefig("figures/eq_stability_sweeps.pdf")

# ------------------------------------------------------------
# Experiment 3: frozen versus augmented convergence
# ------------------------------------------------------------

Bit = ones(3, 3)
gammas_iterative = 10.0 .^ range(-1, 2, length=400)

rho_T_PBR = Float64[]
rho_A_PBR = Float64[]
rho_T_SBR = Float64[]
rho_A_SBR = Float64[]

for γ in gammas_iterative
    Γ = γ * Id(3)

    TPBR, APBR = iterative_matrices(Apr, Bit, Γ, :PBR)
    TSBR, ASBR = iterative_matrices(Apr, Bit, Γ, :SBR)

    push!(rho_T_PBR, spectral_radius(TPBR))
    push!(rho_A_PBR, spectral_radius(APBR))
    push!(rho_T_SBR, spectral_radius(TSBR))
    push!(rho_A_SBR, spectral_radius(ASBR))
end

p3 = plot(
    gammas_iterative,
    rho_T_PBR;
    xscale=:log10,
    xlabel="penalty γ",
    ylabel="spectral radius",
    label="frozen PBR",
    linewidth=2
)
plot!(p3, gammas_iterative, rho_A_PBR;
      label="augmented PBR", linewidth=2)
hline!(p3, [1.0]; label="stability boundary", linestyle=:dash)

p4 = plot(
    gammas_iterative,
    rho_T_SBR;
    xscale=:log10,
    xlabel="penalty γ",
    ylabel="spectral radius",
    label="frozen SBR",
    linewidth=2
)
plot!(p4, gammas_iterative, rho_A_SBR;
      label="augmented SBR", linewidth=2)
hline!(p4, [1.0]; label="stability boundary", linestyle=:dash)

plot(
    p3,
    p4;
    layout=(1, 2),
    size=(1000, 400),
    margin=5Plots.mm
)
savefig("figures/iterative_stability_sweeps.pdf")

# ------------------------------------------------------------
# Experiment 4: competition and structural metrics
# ------------------------------------------------------------

Γ = Id(2)
G = Bpr' * Bpr
S = G + Γ

g1 = [1.0, 0.0, 0.0]
g2 = [0.0, 0.0, 1.0]
goals = [g1, g2]

v = [
    dot(Bpr[:, 1], g1),
    dot(Bpr[:, 2], g2)
]

gsum = g1 + g2
q = Bpr' * gsum

P = Bpr * (S \ Bpr')
Fne = (Id(3) - P) * Apr
K = Id(3) - Fne
H = K \ (Bpr / S)

xne = H * v
zne = Apr * xne

une = S \ (v - Bpr' * zne)

R = 2.0 * G + Γ
uso_same_state = R \ (q - 2.0 * Bpr' * zne)

function social_cost(u, z, B, goals, Γ)
    value = 0.0
    for m in eachindex(goals)
        value += norm(z + B * u - goals[m])^2
        value += Γ[m, m] * u[m]^2
    end
    return value
end

Jne = social_cost(une, zne, Bpr, goals, Γ)
Jso = social_cost(uso_same_state, zne, Bpr, goals, Γ)

Delta_comp = Jne - Jso
Pi_comp = Jne / Jso

Pso = 2.0 * Bpr * (R \ Bpr')
Fso = (Id(3) - Pso) * Apr
cso = Bpr * (R \ q)
xso = (Id(3) - Fso) \ cso

xdes = 0.5 * (g1 + g2)
xclosest = H * pinv(H) * xdes

dstr = norm(xdes - xclosest)
dwithin = norm(xne - xclosest)
dtotal = norm(xne - xdes)
dcomp_ss = norm(xne - xso)

@printf("rho(F_NE)  = %.9f\n", spectral_radius(Fne))
@printf("rho(F_SO)  = %.9f\n", spectral_radius(Fso))
@printf("Delta_comp = %.9f\n", Delta_comp)
@printf("Pi_comp    = %.9f\n", Pi_comp)
@printf("d_comp_ss  = %.9f\n", dcomp_ss)
@printf("d_str      = %.9f\n", dstr)
@printf("d_within   = %.9f\n", dwithin)
@printf("d_total    = %.9f\n", dtotal)
@printf(
    "Pythagorean residual = %.3e\n",
    dtotal^2 - dstr^2 - dwithin^2
)