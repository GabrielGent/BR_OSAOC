#=
#!/usr/bin/env julia

"""
Reproduce all numerical results and parameter sweeps in Section VII of

    Competitive One-Step-Ahead Control of Friedkin--Johnsen Networks:
    Potential Games, Stability, and the Price of Competition.

Numerical calculations require only Julia's standard libraries:

    julia reproduce_section_VII.jl

To also save the penalty-sweep plot for Section VII-A, install Plots.jl and run

    import Pkg; Pkg.add("Plots")
    julia reproduce_section_VII.jl --plots

The two manuscript figures are saved as

    Figure_1_exact_feedback_sweeps.pdf
    Figure_2_best_response_sweeps.pdf

in the current directory.
"""
=#

using LinearAlgebra
using Printf

spectral_radius(M) = maximum(abs.(eigvals(M)))

# Julia's array-level `isapprox` compares a norm of the full error array.
# For manuscript values rounded entrywise to six decimals, the appropriate
# test is instead that every printed component is within half a unit in the
# last displayed decimal.
componentwise_isapprox(x, y; atol, rtol=0.0) =
    all(isapprox.(x, y; atol=atol, rtol=rtol))

function left_perron_vector(A)
    E = eigen(copy(transpose(A)))
    j = argmin(abs.(E.values .- 1))
    pi = real.(E.vectors[:, j])
    pi ./= sum(pi)
    if any(pi .< 0)
        pi .*= -1
        pi ./= sum(pi)
    end
    return pi
end

function ne_operators(A, B, Gamma; d=zeros(size(A, 1)))
    n = size(A, 1)
    In = Matrix{Float64}(I, n, n)
    G = transpose(B) * B
    S = G + Gamma
    P = B * (S \ transpose(B))
    F = (In - P) * A
    K = In - F
    xbase = K \ ((In - P) * d)
    H = K \ (B / S)
    return (; G, S, P, F, K, xbase, H)
end

function social_cost(u, z, B, goals, Gamma)
    predicted_state = z + B * u
    tracking_cost = sum(
        norm(predicted_state - goals[:, m])^2
        for m in axes(goals, 2)
    )
    return tracking_cost + dot(u, Gamma * u)
end

function section_VII_A(A, B)
    n, p = size(B)
    pi = left_perron_vector(A)
    alpha = dot(pi, B * transpose(B) * ones(n))
    eigA = sort(real.(eigvals(A)); rev=true)

    penalties = [0.1, 1.0, 10.0, 100.0, 1000.0]
    radii = Float64[]
    scaled_gaps = Float64[]

    for t in penalties
        Gamma = t * Matrix{Float64}(I, p, p)
        F = ne_operators(A, B, Gamma).F
        r = spectral_radius(F)
        push!(radii, r)
        push!(scaled_gaps, t * (1 - r))
    end

    println("\nSECTION VII-A: Primitive marginal network")
    @printf(
        "spectrum(A) = {%0.6f, %0.6f, %0.6f}\n",
        eigA[1], eigA[2], eigA[3],
    )
    @printf(
        "left Perron vector pi = (%0.6f, %0.6f, %0.6f)'\n",
        pi[1], pi[2], pi[3],
    )
    @printf("damping coefficient pi' B B' 1 = %0.6f\n\n", alpha)
    println("       t       rho(F_EQ(t))       t[1-rho(F_EQ(t))]")
    for (t, r, s) in zip(penalties, radii, scaled_gaps)
        @printf("%8.1f        %0.6f               %0.6f\n", t, r, s)
    end

    # Values printed in Section VII-A.
    @assert componentwise_isapprox(
        pi,
        [0.239130, 0.434783, 0.326087];
        atol=5e-7,
    )
    @assert isapprox(alpha, 2.330435; atol=5e-7)
    @assert componentwise_isapprox(
        radii,
        [0.140270, 0.299121, 0.810623, 0.977221, 0.997675];
        atol=5e-7,
    )

    # Dense sweep used for the optional plot.
    plot_penalties = 10.0 .^ range(-3, 3; length=800)
    plot_radii = [
        spectral_radius(
            ne_operators(
                A,
                B,
                t * Matrix{Float64}(I, p, p),
            ).F,
        ) for t in plot_penalties
    ]

    return (; pi, alpha, penalties, radii, scaled_gaps,
            plot_penalties, plot_radii)
end

function section_VII_B()
    Aper = [
        0.0  0.0  1.0
        0.0  0.0  1.0
        0.9  0.1  0.0
    ]
    bper = [0.0, 10.0, 5.0]
    Bper = reshape(bper, :, 1)

    w1 = [1.0, 1.0, 1.0]
    wminus1 = [1.0, 1.0, -1.0]
    yminus1 = [0.45, 0.05, -0.5]

    alignment_1 = dot(bper, w1)
    alignment_minus1 = dot(bper, wminus1)
    alpha_minus1 = dot(yminus1, bper) * dot(bper, wminus1)
    q_minus1(gamma) = -20 / (gamma + dot(bper, bper))

    penalties = [0.1, 10.0, 1000.0, 10000.0]
    radii = Float64[]
    for gamma in penalties
        Gamma = reshape([gamma], 1, 1)
        push!(radii, spectral_radius(ne_operators(Aper, Bper, Gamma).F))
    end

    println("\nSECTION VII-B: Imprimitive period-two network")
    @printf("b' w_1 = %0.6f\n", alignment_1)
    @printf("b' w_-1 = %0.6f\n", alignment_minus1)
    @printf("alpha_-1 = (y_-1' b)(b' w_-1) = %0.6f\n", alpha_minus1)
    println("q_gamma(-1) = -20/(gamma + 125)")
    println("\n   gamma       rho(F_EQ(gamma))       q_gamma(-1)")
    for (gamma, r) in zip(penalties, radii)
        @printf("%8.1f           %0.6f             % .6e\n",
                gamma, r, q_minus1(gamma))
    end

    # Values printed in Section VII-B.
    @assert isapprox(sort(real.(eigvals(Aper))), [-1.0, 0.0, 1.0]; atol=1e-12)
    @assert isapprox(alignment_1, 15.0; atol=1e-12)
    @assert isapprox(alignment_minus1, 5.0; atol=1e-12)
    @assert isapprox(alpha_minus1, -10.0; atol=1e-12)
    @assert componentwise_isapprox(
        radii,
        [1.096492, 1.088145, 1.009069, 1.000990];
        atol=5e-7,
    )

    plot_penalties = 10.0 .^ range(-1, 4; length=800)
    plot_radii = [
        spectral_radius(
            ne_operators(Aper, Bper, reshape([gamma], 1, 1)).F,
        ) for gamma in plot_penalties
    ]

    return (; Aper, Bper, w1, wminus1, yminus1, alignment_1,
            alignment_minus1, alpha_minus1, q_minus1, penalties, radii,
            plot_penalties, plot_radii)
end

function splitting_and_augmented_radii(A, B, gamma, protocol)
    n, p = size(B)
    In = Matrix{Float64}(I, n, n)
    Ip = Matrix{Float64}(I, p, p)
    S = transpose(B) * B + gamma * Ip
    D = Matrix(Diagonal(diag(S)))
    L = tril(S, -1)

    if protocol == :PBR
        M = D
        N = -(L + transpose(L))
    elseif protocol == :SBR
        M = D + L
        N = -transpose(L)
    else
        error("protocol must be :PBR or :SBR")
    end

    T = M \ N
    Q = B * (M \ transpose(B))
    Aaug = vcat(
        hcat((In - Q) * A, B * T),
        hcat(-(M \ (transpose(B) * A)), T),
    )

    return (; S, D, L, M, N, T, Q, Aaug,
            frozen_radius=spectral_radius(T),
            augmented_radius=spectral_radius(Aaug))
end

function section_VII_C(A)
    Biter = ones(3, 3)
    gamma_check = 6.0
    pbr_check = splitting_and_augmented_radii(A, Biter, gamma_check, :PBR)
    sbr_check = splitting_and_augmented_radii(A, Biter, gamma_check, :SBR)

    println("\nSECTION VII-C: Frozen-state versus closed-loop stability")
    @printf("At gamma = %0.1f:\n", gamma_check)
    @printf("  rho(T_PBR) = %0.6f\n", pbr_check.frozen_radius)
    @printf("  rho(A_PBR) = %0.6f\n", pbr_check.augmented_radius)
    @printf("  rho(T_SBR) = %0.6f\n", sbr_check.frozen_radius)
    @printf("  rho(A_SBR) = %0.6f\n", sbr_check.augmented_radius)

    # Values printed in Section VII-C.
    @assert isapprox(
        pbr_check.frozen_radius,
        6 / (3 + gamma_check);
        atol=1e-12,
    )
    @assert isapprox(pbr_check.frozen_radius, 0.666667; atol=5e-7)
    @assert isapprox(pbr_check.augmented_radius, 1.215250; atol=5e-7)
    @assert isapprox(sbr_check.frozen_radius, 0.192450; atol=5e-7)
    @assert isapprox(sbr_check.augmented_radius, 0.521584; atol=5e-7)

    plot_penalties = 10.0 .^ range(-1, 2; length=800)
    pbr_frozen = Float64[]
    pbr_augmented = Float64[]
    sbr_frozen = Float64[]
    sbr_augmented = Float64[]

    for gamma in plot_penalties
        pbr = splitting_and_augmented_radii(A, Biter, gamma, :PBR)
        sbr = splitting_and_augmented_radii(A, Biter, gamma, :SBR)
        push!(pbr_frozen, pbr.frozen_radius)
        push!(pbr_augmented, pbr.augmented_radius)
        push!(sbr_frozen, sbr.frozen_radius)
        push!(sbr_augmented, sbr.augmented_radius)
    end

    # The analytic PBR frozen-state sweep is rho(T_PBR)=6/(3+gamma).
    @assert isapprox(
        pbr_frozen,
        6.0 ./ (3.0 .+ plot_penalties);
        atol=1e-11,
    )

    return (; Biter, gamma_check, pbr_check, sbr_check, plot_penalties,
            pbr_frozen, pbr_augmented, sbr_frozen, sbr_augmented)
end

function section_VII_D(A, B)
    n, p = size(B)
    In = Matrix{Float64}(I, n, n)
    Gamma = Matrix{Float64}(I, p, p)
    d = zeros(n)                       # Theta = 0

    g1 = [1.0, 0.0, 0.0]
    g2 = [0.0, 0.0, 1.0]
    goals = hcat(g1, g2)

    ops = ne_operators(A, B, Gamma; d=d)
    G, S, F, K, xbase, H =
        ops.G, ops.S, ops.F, ops.K, ops.xbase, ops.H

    # v_m = b_m' g_m and q = B' sum_m g_m.
    v = [dot(B[:, m], goals[:, m]) for m in 1:p]
    gsum = vec(sum(goals; dims=2))
    q = transpose(B) * gsum

    # Nash-equilibrium feedback and its closed-loop equilibrium.
    xNE = K \ ((In - ops.P) * d + B * (S \ v))
    zNE = A * xNE + d
    uNE = S \ (v - transpose(B) * zNE)

    # Centralized social-feedback benchmark.
    SSO = p * G + Gamma
    PSO = p * B * (SSO \ transpose(B))
    FSO = (In - PSO) * A
    cSO = (In - PSO) * d + B * (SSO \ q)
    xSO = (In - FSO) \ cSO

    # Same-state comparison: both actions below use zNE.
    uSO_at_zNE = SSO \ (q - p * transpose(B) * zNE)
    JNE = social_cost(uNE, zNE, B, goals, Gamma)
    JSO = social_cost(uSO_at_zNE, zNE, B, goals, Gamma)
    Delta_comp = JNE - JSO
    Pi_comp = JNE / JSO
    dss_comp = norm(xNE - xSO)

    # Projection onto the equilibrium-reachable affine family.
    xdes = (g1 + g2) / 2
    xclosest = xbase + H * pinv(H) * (xdes - xbase)
    dstr = norm(xdes - xclosest)
    dwithin = norm(xNE - xclosest)
    dtotal = norm(xNE - xdes)

    println("\nSECTION VII-D: Structural and competitive inefficiency")
    @printf("v = (%0.6f, %0.6f)'\n", v[1], v[2])
    @printf("q = (%0.6f, %0.6f)'\n", q[1], q[2])
    @printf("rho(F_NE) = %0.6f\n", spectral_radius(F))
    @printf("rho(F_SO) = %0.6f\n", spectral_radius(FSO))
    @printf("steady-state displacement d_ss_comp = %0.6f\n", dss_comp)
    @printf("J_NE(z_NE) = %0.6f\n", JNE)
    @printf("J_SO(z_NE) = %0.6f\n", JSO)
    @printf("Delta_comp^* = %0.6f\n", Delta_comp)
    @printf("Pi_comp^* = %0.6f\n", Pi_comp)
    @printf("d_str = %0.6f\n", dstr)
    @printf("d_within = %0.6f\n", dwithin)
    @printf("total target error = %0.6f\n", dtotal)
    @printf(
        "Pythagorean residual = %.3e\n",
        dtotal^2 - dstr^2 - dwithin^2,
    )

    # Values printed in Section VII-D.
    @assert isapprox(v, [1.0, 1.0]; atol=1e-12)
    @assert isapprox(q, [1.2, 1.2]; atol=1e-12)
    @assert isapprox(spectral_radius(F), 0.299121; atol=5e-7)
    @assert isapprox(spectral_radius(FSO), 0.221680; atol=5e-7)
    @assert isapprox(dss_comp, 0.371726; atol=5e-7)
    @assert isapprox(JNE, 1.575607; atol=5e-7)
    @assert isapprox(JSO, 1.348770; atol=5e-7)
    @assert isapprox(Delta_comp, 0.226837; atol=5e-7)
    @assert isapprox(Pi_comp, 1.168181; atol=5e-7)
    @assert isapprox(dstr, 0.401416; atol=5e-7)
    @assert isapprox(dwithin, 0.354051; atol=5e-7)
    @assert isapprox(dtotal, 0.535244; atol=5e-7)
    @assert abs(dtotal^2 - dstr^2 - dwithin^2) < 1e-12

    return (; Gamma, goals, v, q, S, SSO, F, FSO, xNE, xSO, zNE,
            uNE, uSO_at_zNE, JNE, JSO, Delta_comp, Pi_comp,
            dss_comp, xbase, H, xdes, xclosest, dstr, dwithin, dtotal)
end

function section_VII_E(A, B, section_D)
    n, p = size(B)
    @assert p == 2

    pi = left_perron_vector(A)
    Gamma = section_D.Gamma
    S = section_D.S
    SSO = section_D.SSO
    xbase = section_D.xbase
    H = section_D.H

    Uperp = [1.0, -1.0] / sqrt(2)
    epsilon = 1 / sqrt(2)
    gbar = 0.5 * ones(n)
    goals0 = hcat(gbar, gbar)
    v0 = [dot(B[:, m], goals0[:, m]) for m in 1:p]
    x0 = xbase + H * v0
    z0 = A * x0                    # Theta = 0

    kappas = zeros(n)
    endpoint_displacements = zeros(n)
    endpoint_losses = zeros(n)

    for i in 1:n
        Di = Diagonal(vec(B[i, :]))
        conflict_direction = Di * Uperp

        kappas[i] = norm(H * conflict_direction)
        endpoint_displacements[i] = epsilon * kappas[i]

        action_direction = S \ conflict_direction
        beta_i = dot(action_direction, SSO * action_direction)
        endpoint_losses[i] = beta_i * epsilon^2

        # Direct endpoint check for eta = +epsilon.
        ei = zeros(n)
        ei[i] = 1.0
        g1 = gbar + (epsilon / sqrt(2)) * ei
        g2 = gbar - (epsilon / sqrt(2)) * ei
        goals_eta = hcat(g1, g2)
        v_eta = [dot(B[:, m], goals_eta[:, m]) for m in 1:p]
        x_eta = xbase + H * v_eta

        q_eta = transpose(B) * (g1 + g2)
        uNE_eta = S \ (v_eta - transpose(B) * z0)
        uSO_eta = SSO \ (q_eta - p * transpose(B) * z0)
        direct_loss = social_cost(uNE_eta, z0, B, goals_eta, Gamma) -
                      social_cost(uSO_eta, z0, B, goals_eta, Gamma)

        @assert isapprox(
            norm(x_eta - x0),
            endpoint_displacements[i];
            atol=1e-12,
        )
        @assert isapprox(direct_loss, endpoint_losses[i]; atol=1e-12)
    end

    println("\nSECTION VII-E: Goal conflicts restricted to selected nodes")
    @printf("epsilon = %0.6f\n", epsilon)
    @printf("v0 = (%0.6f, %0.6f)'\n", v0[1], v0[2])
    @printf(
        "x*(v0) = (%0.6f, %0.6f, %0.6f)'\n\n",
        x0[1], x0[2], x0[3],
    )
    println(" node       pi_i       kappa_i    epsilon*kappa_i   Delta_comp(+-epsilon)")
    for i in 1:n
        @printf(
            "  %d       %0.6f     %0.6f        %0.6f             %0.6f\n",
            i,
            pi[i],
            kappas[i],
            endpoint_displacements[i],
            endpoint_losses[i],
        )
    end

    # Values printed in Table I.
    @assert isapprox(v0, [0.85, 1.0]; atol=1e-12)
    @assert isapprox(x0, 0.5 * ones(n); atol=1e-12)
    @assert componentwise_isapprox(
        kappas,
        [0.489183, 0.449150, 0.480138];
        atol=5e-7,
    )
    @assert componentwise_isapprox(
        endpoint_displacements,
        [0.345905, 0.317597, 0.339509];
        atol=5e-7,
    )
    @assert componentwise_isapprox(
        endpoint_losses,
        [0.201380, 0.178809, 0.182720];
        atol=5e-7,
    )

    return (; pi, epsilon, Uperp, gbar, v0, x0, z0, kappas,
            endpoint_displacements, endpoint_losses)
end

function save_sweep_plots(results_A, results_B, results_C)
    @eval import Plots

    primitive_plot = Plots.plot(
        results_A.plot_penalties,
        results_A.plot_radii;
        xscale=:log10,
        xlabel="penalty t",
        ylabel="spectral radius",
        label="primitive network",
        linewidth=2,
        legend=:bottomright,
        grid=true,
    )
    Plots.hline!(
        primitive_plot,
        [1.0];
        label="stability boundary",
        linestyle=:dash,
        color=:red,
    )

    periodic_plot = Plots.plot(
        results_B.plot_penalties,
        results_B.plot_radii;
        xscale=:log10,
        xlabel="penalty gamma",
        ylabel="spectral radius",
        label="periodic network",
        linewidth=2,
        legend=:topright,
        grid=true,
    )
    Plots.hline!(
        periodic_plot,
        [1.0];
        label="stability boundary",
        linestyle=:dash,
        color=:red,
    )

    figure_1 = Plots.plot(
        primitive_plot,
        periodic_plot;
        layout=(1, 2),
        size=(1000, 400),
    )
    output_1 = joinpath(pwd(), "Figure_1_exact_feedback_sweeps.pdf")
    Plots.savefig(figure_1, output_1)

    pbr_plot = Plots.plot(
        results_C.plot_penalties,
        results_C.pbr_frozen;
        xscale=:log10,
        xlabel="penalty gamma",
        ylabel="spectral radius",
        label="frozen PBR",
        linewidth=2,
        legend=:topright,
        grid=true,
    )
    Plots.plot!(
        pbr_plot,
        results_C.plot_penalties,
        results_C.pbr_augmented;
        label="augmented PBR",
        linewidth=2,
    )
    Plots.hline!(
        pbr_plot,
        [1.0];
        label="stability boundary",
        linestyle=:dash,
        color=:green,
    )

    sbr_plot = Plots.plot(
        results_C.plot_penalties,
        results_C.sbr_frozen;
        xscale=:log10,
        xlabel="penalty gamma",
        ylabel="spectral radius",
        label="frozen SBR",
        linewidth=2,
        legend=:bottomright,
        grid=true,
    )
    Plots.plot!(
        sbr_plot,
        results_C.plot_penalties,
        results_C.sbr_augmented;
        label="augmented SBR",
        linewidth=2,
    )
    Plots.hline!(
        sbr_plot,
        [1.0];
        label="stability boundary",
        linestyle=:dash,
        color=:green,
    )

    figure_2 = Plots.plot(
        pbr_plot,
        sbr_plot;
        layout=(1, 2),
        size=(1000, 400),
    )
    output_2 = joinpath(pwd(), "Figure_2_best_response_sweeps.pdf")
    Plots.savefig(figure_2, output_2)

    println("\nSaved plots:")
    println("  $output_1")
    println("  $output_2")
end

function main()
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

    results_A = section_VII_A(Apr, Bpr)
    results_B = section_VII_B()
    results_C = section_VII_C(Apr)
    results_D = section_VII_D(Apr, Bpr)
    section_VII_E(Apr, Bpr, results_D)

#    if "--plots" in ARGS
        save_sweep_plots(results_A, results_B, results_C)
#    end

    println("\nAll numerical checks passed.")
end

main()
