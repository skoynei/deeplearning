import math
import random
from typing import List, Tuple


def mat_mul(a, b):
    n = len(a)
    m = len(b[0])
    p = len(b)
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(m):
                out[i][j] += aik * b[k][j]
    return out


def mat_add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_scale(a, c):
    return [[c * a[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_eye(n):
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        out[i][i] = 1.0
    return out


def mat_exp_series(a, n_terms=30):
    n = len(a)
    out = mat_eye(n)
    term = mat_eye(n)
    for k in range(1, n_terms + 1):
        term = mat_mul(term, a)
        term = mat_scale(term, 1.0 / k)
        out = mat_add(out, term)
    return out


def solve_linear(amat, bvec):
    n = len(amat)
    a = [row[:] for row in amat]
    b = bvec[:]
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(a[r][i]) > abs(a[pivot][i]):
                pivot = r
        if abs(a[pivot][i]) < 1e-12:
            # ridge regularization fallback
            a[i][i] += 1e-8
            pivot = i
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]

        piv = a[i][i]
        for j in range(i, n):
            a[i][j] /= piv
        b[i] /= piv

        for r in range(n):
            if r == i:
                continue
            fac = a[r][i]
            if fac == 0.0:
                continue
            for j in range(i, n):
                a[r][j] -= fac * a[i][j]
            b[r] -= fac * b[i]
    return b


def normalize_rows(p):
    for i in range(len(p)):
        p[i] = [max(0.0, x) for x in p[i]]
        s = sum(p[i])
        if s <= 0:
            p[i] = [0.0] * len(p[i])
            p[i][i] = 1.0
        else:
            p[i] = [x / s for x in p[i]]
    return p


def stationary_distribution_3state(q):
    # solve pi Q = 0 with pi0+pi1+pi2=1
    a = [
        [q[0][0], q[1][0], q[2][0]],
        [q[0][1], q[1][1], q[2][1]],
        [1.0, 1.0, 1.0],
    ]
    b = [0.0, 0.0, 1.0]
    pi = solve_linear(a, b)
    pi = [max(0.0, x) for x in pi]
    s = sum(pi)
    return [x / s for x in pi]


def compute_mu_star(params):
    beta = params["beta"]
    gamma = params["gamma"]
    u = params["u"]
    delta = params["delta"]
    omega = params["omega"]
    theta = params["theta"]
    mu = params["mu"]

    a = gamma + u + delta
    c = gamma / (omega + u)
    i_over_n = (beta - a) / (beta * (1.0 + c))
    i_over_n = max(i_over_n, 0.0)
    return mu - theta * i_over_n


def v4_closed_form_single_threshold(x0, params, q, y):
    r = params["r"]
    tau = params["tau"]
    c = params["c"]
    L = params["L"]
    sigma = params["sigma"]
    mu_star = compute_mu_star(params)

    pi = stationary_distribution_3state(q)
    y_bar = sum(pi[k] * y[k] for k in range(3))

    disc = (mu_star - 0.5 * sigma * sigma) ** 2 + 2.0 * sigma * sigma * r
    m_minus = (-(mu_star - 0.5 * sigma * sigma) - math.sqrt(disc)) / (sigma * sigma)

    if (1.0 - tau) * y_bar <= 1e-12 or r <= mu_star:
        return L

    x_star = ((r - mu_star) / ((1.0 - tau) * y_bar)) * (m_minus / (m_minus - 1.0)) * (L + c / r)
    A = -((1.0 - tau) * y_bar) / ((r - mu_star) * m_minus) * (x_star ** (1.0 - m_minus))

    if x0 <= x_star:
        return L
    return A * (x0**m_minus) + ((1.0 - tau) * y_bar / (r - mu_star)) * x0 - c / r


def v4_closed_form_from_ybar(x0, params, y_bar):
    """Closed-form V4 under the same aggregated single-state approximation used by v4_lsm_aligned."""
    r = params["r"]
    tau = params["tau"]
    c = params["c"]
    L = params["L"]
    sigma = params["sigma"]
    mu_star = compute_mu_star(params)

    disc = (mu_star - 0.5 * sigma * sigma) ** 2 + 2.0 * sigma * sigma * r
    m_minus = (-(mu_star - 0.5 * sigma * sigma) - math.sqrt(disc)) / (sigma * sigma)

    if (1.0 - tau) * y_bar <= 1e-12 or r <= mu_star:
        return L

    x_star = ((r - mu_star) / ((1.0 - tau) * y_bar)) * (m_minus / (m_minus - 1.0)) * (L + c / r)
    A = -((1.0 - tau) * y_bar) / ((r - mu_star) * m_minus) * (x_star ** (1.0 - m_minus))

    if x0 <= x_star:
        return L
    return A * (x0**m_minus) + ((1.0 - tau) * y_bar / (r - mu_star)) * x0 - c / r


def sample_next_state(curr_state: int, p_row: List[float], u: float) -> int:
    cum = 0.0
    for s, prob in enumerate(p_row):
        cum += prob
        if u <= cum:
            return s
    return len(p_row) - 1


def lsm_backward(cashflow, x, j, rho, params):
    r = params["r"]
    dt = params["dt"]
    L = params["L"]
    n_paths = len(x)
    m_steps = len(x[0]) - 1

    v_next = [L for _ in range(n_paths)]

    for m in range(m_steps - 1, -1, -1):
        y_target = [cashflow[n][m] + math.exp(-r * dt) * v_next[n] for n in range(n_paths)]
        cont_hat = [0.0 for _ in range(n_paths)]

        for state in (0, 1, 2):
            idx = [n for n in range(n_paths) if j[n][m] == state]
            if len(idx) < 8:
                for n in idx:
                    cont_hat[n] = y_target[n]
                continue

            # basis: 1, x, x^2, rho, rho^2, x*rho
            k = 6
            xtx = [[0.0] * k for _ in range(k)]
            xty = [0.0] * k
            for n in idx:
                xm = x[n][m]
                rm = rho[n][m]
                f = [1.0, xm, xm * xm, rm, rm * rm, xm * rm]
                yv = y_target[n]
                for a in range(k):
                    xty[a] += f[a] * yv
                    for b in range(k):
                        xtx[a][b] += f[a] * f[b]

            for a in range(k):
                xtx[a][a] += 1e-8

            beta = solve_linear(xtx, xty)

            for n in idx:
                xm = x[n][m]
                rm = rho[n][m]
                f = [1.0, xm, xm * xm, rm, rm * rm, xm * rm]
                cont_hat[n] = sum(beta[a] * f[a] for a in range(k))

        v_curr = [L if L >= cont_hat[n] else y_target[n] for n in range(n_paths)]
        v_next = v_curr

    return sum(v_next) / n_paths


def lsm_backward_1d(cashflow, x, params):
    """LSM for 1D state (single aggregated V4 control model)."""
    r = params["r"]
    dt = params["dt"]
    L = params["L"]
    n_paths = len(x)
    m_steps = len(x[0]) - 1

    v_next = [L for _ in range(n_paths)]
    disc = math.exp(-r * dt)

    for m in range(m_steps - 1, -1, -1):
        y_target = [cashflow[n][m] + disc * v_next[n] for n in range(n_paths)]

        # basis: 1, x, x^2
        k = 3
        xtx = [[0.0] * k for _ in range(k)]
        xty = [0.0] * k
        for n in range(n_paths):
            xm = x[n][m]
            f = [1.0, xm, xm * xm]
            yv = y_target[n]
            for a in range(k):
                xty[a] += f[a] * yv
                for b in range(k):
                    xtx[a][b] += f[a] * f[b]

        for a in range(k):
            xtx[a][a] += 1e-8
        beta = solve_linear(xtx, xty)

        cont_hat = []
        for n in range(n_paths):
            xm = x[n][m]
            f = [1.0, xm, xm * xm]
            cont_hat.append(sum(beta[a] * f[a] for a in range(k)))

        v_curr = [L if L >= cont_hat[n] else y_target[n] for n in range(n_paths)]
        v_next = v_curr

    return sum(v_next) / n_paths


def simulate_v4_aligned_paths(params, y_bar, rng):
    """Simulate scalar X process under constant mu* and aggregated y_bar to align with closed-form V4."""
    n_paths = params["n_paths"]
    m_steps = params["m_steps"]
    dt = params["dt"]
    x0 = params["x0"]
    sigma = params["sigma"]
    tau = params["tau"]
    c = params["c"]
    mu_star = compute_mu_star(params)

    x = [[0.0] * (m_steps + 1) for _ in range(n_paths)]
    for n in range(n_paths):
        x[n][0] = x0

    for m in range(m_steps):
        for n in range(n_paths):
            z = rng.gauss(0.0, 1.0)
            x[n][m + 1] = x[n][m] * math.exp((mu_star - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)

    cashflow = [[((1.0 - tau) * y_bar * x[n][m] - c) * dt for m in range(m_steps + 1)] for n in range(n_paths)]
    return x, cashflow


def simulate_v4_paths(params, q, y, rng):
    n_paths = params["n_paths"]
    m_steps = params["m_steps"]
    dt = params["dt"]
    x0 = params["x0"]
    j0 = params["j0"]
    sigma = params["sigma"]
    tau = params["tau"]
    c = params["c"]

    mu_star = compute_mu_star(params)
    rho_star = max((params["mu"] - mu_star) / max(params["theta"], 1e-12), 0.0)

    x = [[0.0] * (m_steps + 1) for _ in range(n_paths)]
    j = [[0] * (m_steps + 1) for _ in range(n_paths)]
    rho = [[rho_star] * (m_steps + 1) for _ in range(n_paths)]

    a = [[q[i][j_] * dt for j_ in range(3)] for i in range(3)]
    p = normalize_rows(mat_exp_series(a))

    for n in range(n_paths):
        x[n][0] = x0
        j[n][0] = j0

    for m in range(m_steps):
        for n in range(n_paths):
            z = rng.gauss(0.0, 1.0)
            x[n][m + 1] = x[n][m] * math.exp((mu_star - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
            u = rng.random()
            j[n][m + 1] = sample_next_state(j[n][m], p[j[n][m]], u)

    cashflow = [[((1.0 - tau) * x[n][m] * y[j[n][m]] - c) * dt for m in range(m_steps + 1)] for n in range(n_paths)]
    return x, j, rho, cashflow


def simulate_v5_paths(params, q, y, rng):
    n_paths = params["n_paths"]
    m_steps = params["m_steps"]
    dt = params["dt"]
    sigma = params["sigma"]
    tau = params["tau"]
    c = params["c"]

    x = [[0.0] * (m_steps + 1) for _ in range(n_paths)]
    s = [[0.0] * (m_steps + 1) for _ in range(n_paths)]
    i = [[0.0] * (m_steps + 1) for _ in range(n_paths)]
    rpop = [[0.0] * (m_steps + 1) for _ in range(n_paths)]
    j = [[0] * (m_steps + 1) for _ in range(n_paths)]
    rho = [[0.0] * (m_steps + 1) for _ in range(n_paths)]

    a = [[q[row][col] * dt for col in range(3)] for row in range(3)]
    p = normalize_rows(mat_exp_series(a))

    for n in range(n_paths):
        x[n][0] = params["x0"]
        s[n][0] = params["S0"]
        i[n][0] = params["I0"]
        rpop[n][0] = params["R0"]
        j[n][0] = params["j0"]

    for m in range(m_steps):
        for n in range(n_paths):
            N = max(s[n][m] + i[n][m] + rpop[n][m], 1e-8)
            rho_nm = i[n][m] / N
            rho[n][m] = rho_nm

            mu_t = params["mu"] - params["theta"] * rho_nm
            zx = rng.gauss(0.0, 1.0)
            x[n][m + 1] = x[n][m] * math.exp((mu_t - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * zx)

            zs = rng.gauss(0.0, 1.0)
            zi = rng.gauss(0.0, 1.0)
            zr = rng.gauss(0.0, 1.0)

            b_s = params["Lambda"] - params["beta"] * s[n][m] * i[n][m] / N - params["u"] * s[n][m] + params["omega"] * rpop[n][m]
            b_i = params["beta"] * s[n][m] * i[n][m] / N - (params["gamma"] + params["u"] + params["delta"]) * i[n][m]
            b_r = params["gamma"] * i[n][m] - (params["omega"] + params["u"]) * rpop[n][m]

            s[n][m + 1] = max(0.0, s[n][m] + b_s * dt + params["sigma_S"] * s[n][m] * math.sqrt(dt) * zs)
            i[n][m + 1] = max(0.0, i[n][m] + b_i * dt + params["sigma_I"] * i[n][m] * math.sqrt(dt) * zi)
            rpop[n][m + 1] = max(0.0, rpop[n][m] + b_r * dt + params["sigma_R"] * rpop[n][m] * math.sqrt(dt) * zr)

            u = rng.random()
            j[n][m + 1] = sample_next_state(j[n][m], p[j[n][m]], u)

    for n in range(n_paths):
        N = max(s[n][-1] + i[n][-1] + rpop[n][-1], 1e-8)
        rho[n][-1] = i[n][-1] / N

    cashflow = [[((1.0 - tau) * x[n][m] * y[j[n][m]] - c) * dt for m in range(m_steps + 1)] for n in range(n_paths)]
    return x, j, rho, cashflow


def main():
    params = {
        "mu": 0.06,
        "sigma": 0.22,
        "theta": 0.20,
        "tau": 0.20,
        "r": 0.08,
        "c": 0.80,
        "L": 8.0,
        # SIRS
        "Lambda": 20.0,
        "beta": 0.65,
        "gamma": 0.25,
        "u": 0.03,
        "delta": 0.02,
        "omega": 0.10,
        "sigma_S": 0.03,
        "sigma_I": 0.04,
        "sigma_R": 0.03,
        # simulation
        "x0": 10.0,
        "S0": 200.0,
        "I0": 30.0,
        "R0": 20.0,
        "j0": 0,
        "dt": 1.0 / 12.0,
        "m_steps": 96,
        "n_paths": 4000,
    }

    y = [1.0, 0.75, 0.05]
    q = [
        [-0.16, 0.10, 0.06],
        [0.08, -0.20, 0.12],
        [0.0, 0.0, 0.0],
    ]

    rng = random.Random(2026)

    # Keep old reference formula (CTMC-aggregated V4 closed form)
    v4_closed_ref = v4_closed_form_single_threshold(params["x0"], params, q, y)

    # Aligned V4 comparison: closed-form and LSM share the same single-state y_bar approximation
    pi = stationary_distribution_3state(q)
    y_bar = sum(pi[k] * y[k] for k in range(3))
    v4_closed = v4_closed_form_from_ybar(params["x0"], params, y_bar)
    x4a, cf4a = simulate_v4_aligned_paths(params, y_bar, rng)
    v4_lsm = lsm_backward_1d(cf4a, x4a, params)

    x5, j5, rho5, cf5 = simulate_v5_paths(params, q, y, rng)
    v5_lsm = lsm_backward(cf5, x5, j5, rho5, params)

    delta4 = v4_closed - v4_lsm
    v5_adjusted = v5_lsm + delta4

    print(f"V4 analytic value          : {v4_closed:.6f}")
    print(f"V4 LSM value               : {v4_lsm:.6f}")
    print(f"V5 LSM value               : {v5_lsm:.6f}")
    print(f"V5 corrected value         : {v5_adjusted:.6f}")
    print(f"(Reference old V4 closed)  : {v4_closed_ref:.6f}")


if __name__ == "__main__":
    main()
