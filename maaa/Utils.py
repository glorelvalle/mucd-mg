#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carlos M. Alaíz
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from matplotlib import pyplot as plt, colors, lines
from sklearn.linear_model import LinearRegression, Perceptron, Ridge

def plot_dataset(x, y):
    if (len(x.shape) == 1):
        plt.plot(x, y, "*")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("Data")
    else:
        n_plot = x.shape[1]
        fig, axs = plt.subplots(ncols=n_plot, sharey=True)
        for i in range(n_plot):
            ax = axs[i]
            ax.plot(x[:, i], y, "*")
            ax.set_xlabel("$x_{%d}$" % (i + 1))
            if (i == 0):
                ax.set_ylabel("$y$")
        plt.suptitle("Data")

def plot_linear_model(x, y_r, w_e, b_e, w_r=None, b_r=None):
    if (np.isscalar(w_e) or (len(w_e) == 1)):
        y_p = w_e * x + b_e

        plt.plot(x, y_r, "*", label="Obs.")
        plt.plot(x, y_p, "-", label="Pred")

        for i in range(len(x)):
            plt.plot([x[i].item(), x[i].item()], [y_p[i].item(), y_r[i].item()], ":k")

        if (w_r is not None) and (b_r is not None):
            plt.plot(x, w_r * x + b_r, "--k", label="Real")
            plt.legend()

        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("$y = %.2f x + %.2f$ (MSE: %.2f, MAE: %.2f, R2: %.2f)" % (w_e, b_e, mean_squared_error(y_r, y_p), mean_absolute_error(y_r, y_p), r2_score(y_r, y_p)))
    else:
        y_p = x @ w_e + b_e
        pos = np.arange(len(w_e) + 1)
        plt.bar(pos, np.append(w_e, b_e), alpha=0.5, label="Est.")

        if (w_r is not None) and (b_r is not None):
            plt.bar(pos, np.append(w_r, b_r), alpha=0.5, label="Real")
            plt.legend()

        plt.grid()
        labels = []
        for i in range(len(w_e)):
            labels.append("$w_%d$" % (i + 1))
        labels.append("$b$")
        plt.xticks(pos, labels)
        plt.title("MSE: %.2f, MAE: %.2f, R2: %.2f" % (mean_squared_error(y_r, y_p), mean_absolute_error(y_r, y_p), r2_score(y_r, y_p)))

def evaluate_linear_model(x_tr, y_tr_r, x_te, y_te_r, w, b, plot=False):
    if (np.isscalar(w)):
        y_tr_p = w * x_tr + b
        y_te_p = w * x_te + b
    else:
        y_tr_p = x_tr @ w.ravel() + b
        y_te_p = x_te @ w.ravel() + b

    er_tr = [mean_squared_error(y_tr_r, y_tr_p), mean_absolute_error(y_tr_r, y_tr_p), r2_score(y_tr_r, y_tr_p)]
    er_te = [mean_squared_error(y_te_r, y_te_p), mean_absolute_error(y_te_r, y_te_p), r2_score(y_te_r, y_te_p)]

    ers = [er_tr, er_te]
    headers=["MSE", "MAE", "R2"]

    print("%10s" % "", end="")
    for h in headers:
        print("%10s" % h, end="")
    print("")

    headersc = ["Train", "Test"]

    cnt = 0
    for er in ers:
        hc = headersc[cnt]
        cnt = cnt + 1
        print("%10s" % hc, end="")

        for e in er:
            print("%10.2f" % e, end="")
        print("")

    if plot:
        plot_linear_model(x_te, y_te_r, w.ravel(), b)

def plot_dataset_clas(x, y):
    if (len(x.shape) == 1):
        plt.plot(x, y, "*")
        plt.xlabel("$x$")
        clas = np.unique(y)
        plt.yticks(clas)
    else:
        if (len(np.unique(y) == 2)):
            ind = y == 1
            plt.scatter(x[ind, 0], x[ind, 1], c="b", zorder=100)
            ind = y != 1
            plt.scatter(x[ind, 0], x[ind, 1], c="r", zorder=100)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

    plt.axis("equal")
    plt.title("Data")

def order_points(points):
    if len(points) == 0:
        return []
    centre = points.mean(axis=0);
    angles = np.arctan2(points[:, 0] - centre[0], points[:, 1] - centre[1])
    o_points = points[np.argsort(angles), :]

    return np.vstack((o_points, o_points[0, :]))

def plot_linear_model_clas(x, y_r, w, b):
    if (len(x.shape) == 1) or (x.shape[1] != 2):
        raise ValueError("only_r two-dimensional problems can be represented")

    y_p = np.sign(x @ w + b)

    plot_dataset_clas(x, y_r)
    ax = plt.axis("equal")
    lims = np.array([ax[0] - 100, ax[1] + 100, ax[2] - 100, ax[3] + 100])

    if (w[1] != 0):
        x1 = lims[0:2]
        x2 = - (w[0] * x1 + b) / w[1]
    else:
        x2 = lims[2:]
        x1 = - (w[1] * x2 + b) / w[0]

    points = np.column_stack((np.append(x1, [lims[0], lims[1], lims[0], lims[1]]), np.append(x2, [lims[2], lims[3], lims[3], lims[2]])))

    points_p = order_points(points[points @ w + b >= - 1e-2])
    if (len(points_p) > 0):
        plt.fill(points_p[:, 0], points_p[:, 1], "b", alpha=0.3)

    points_n = order_points(points[points @ w + b <= + 1e-2])
    if (len(points_n) > 0):
        plt.fill(points_n[:, 0], points_n[:, 1], "r", alpha=0.3)

    plt.plot(x1, x2, "-k")
    plot_dataset_clas(x, y_r)
    plt.axis(ax)

    plt.title("$y = %.2f x_1 + %.2f x_2 + %.2f$ (Acc: %.2f%%)" % (w[0], w[1], b, 100 * accuracy_score(y_r, y_p)))

def fun_cross_entropy(X, y, w):
    y_b = y.copy()
    y_b[y_b == -1] = 0

    y_p = 1 / (1 + np.exp(- X @ w))

    return (- (1 - y_b) * np.log(1 - y_p) - y_b * np.log(y_p)).sum()

def grad_cross_entropy(X, y, w):
    y_b = y.copy()
    y_b[y_b == -1] = 0

    y_p = 1 / (1 + np.exp(- X @ w))

    return X.T @ (y_p - y_b)

def fit_polylinear_regression(x, y, deg=1):
    X = np.power(np.reshape(x, (len(x), 1)), np.arange(1, deg + 1))
    model = LinearRegression()
    model.fit(X, y)

    return model

def pred_polylinear_regression(model, x):
    X = np.power(np.reshape(x, (len(x), 1)), np.arange(1, len(model.coef_) + 1))
    return model.predict(X)

def plot_polylinear_model(x, y_r, model):
    xv = np.linspace(x.min(), x.max())
    plt.plot(x, y_r, "*", label="Obs.")
    plt.plot(xv, pred_polylinear_regression(model, xv), "-", label="Pred")

    y_p = pred_polylinear_regression(model, x)

    for i in range(len(x)):
        plt.plot([x[i].item(), x[i].item()], [y_p[i].item(), y_r[i].item()], ":k")

        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title("Degree: %d (MSE: %.2f, MAE: %.2f, R2: %.2f)" % (len(model.coef_), mean_squared_error(y_r, y_p), mean_absolute_error(y_r, y_p), r2_score(y_r, y_p)))

def norm_p(w, p):
    if (p == 0):
        return np.count_nonzero(w)

    if (p == np.inf):
        return np.max(np.abs(w))

    nw = np.sum(np.power(np.abs(w), p))
    if (p > 1):
        nw = np.power(nw, 1 / p)
    return nw

def plot_contour_lp(p, mini=-3, maxi=3, npoi = 21):
    x = np.linspace(mini, maxi, npoi)
    y = np.linspace(mini, maxi, npoi)
    x, y = np.meshgrid(x, y)

    z = np.apply_along_axis(norm_p, 2, np.stack([x, y], axis = 2), p)
    plt.contour(x, y, z)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal", "box")
    plt.title("Norm $\ell_{%g}$" % p)

    plt.grid()
    plt.show()

def plot_contour_l1_l2(l1_ratio=0.5, mini=-3, maxi=3, npoi = 21):
    x = np.linspace(mini, maxi, npoi)
    y = np.linspace(mini, maxi, npoi)
    x, y = np.meshgrid(x, y)

    z = l1_ratio * np.apply_along_axis(norm_p, 2, np.stack([x, y], axis = 2), 1) + (1 - l1_ratio) * np.apply_along_axis(norm_p, 2, np.stack([x, y], axis = 2), 2)
    plt.contour(x, y, z)

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal", "box")
    plt.title("%g * Norm $\ell_1$ + %g * Norm $\ell_2$" % (l1_ratio, 1 - l1_ratio))

    plt.grid()
    plt.show()

def plot_contour_linear_lp(X, y, p=None, mini=-3, maxi=3, npoi=51):
    def mse_linear(w):
        return mean_squared_error(y, X @ w)

    x1 = np.linspace(mini, maxi, npoi)
    x2 = np.linspace(mini, maxi, npoi)
    x1, x2 = np.meshgrid(x1, x2)

    z = np.apply_along_axis(mse_linear, 2, np.stack([x1, x2], axis = 2))
    plt.contour(x1, x2, z, 30)

    if p is not None:
        x = np.linspace(-1, 1, 101)
        if (p == 0):
            plt.plot([-1, 1], [0, 0], "-k")
            plt.plot([0, 0], [-1, 1], "-k")
            ball = np.abs(x1) + np.abs(x2) <= 1
        elif (p == np.inf):
            plt.plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1, ], "-k")
            plt.fill([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1, ], "k")
            ball = np.maximum(x1, x2) <= 1
        else:
            y = np.power(1 - np.power(np.abs(x), p), 1 / p)
            plt.plot(np.concatenate((x, np.flip(x))), np.concatenate((y, np.flip(-y))), "-k")
            plt.fill(np.concatenate((x, np.flip(x))), np.concatenate((y, np.flip(-y))), "k")
            ball = np.power(np.abs(x1), p) + np.power(np.abs(x2), p) <= 1

        obj = z
        obj[ball == False] = np.inf
    else:
        obj = z

    ind = np.unravel_index(np.argmin(obj), obj.shape)
    plt.plot(x1[ind[0], ind[1]], x2[ind[0], ind[1]], "r*")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.gca().set_aspect("equal", "box")
    if (p is not None):
        plt.title("Norm $\ell_{%g}$ + MSE" % p)
    else:
        plt.title("MSE")

    plt.grid()
    plt.show()

def generate_bv_example(n_rep=250, n_mod=15, n_dim=10, noise=3e-1, seed=1234):
    n_pat = n_dim
    alpha_v = np.logspace(-4, 4, n_mod)

    np.random.seed(seed)

    w = np.random.randn(n_dim)
    x_te = np.random.randn(n_pat, n_dim)
    y_te = x_te @ w + noise * np.random.randn(n_pat)

    distances = np.zeros((n_mod, n_rep))
    predictions = np.zeros((n_mod, n_rep, n_pat))
    for i, alpha in enumerate(alpha_v):
        for j in range(n_rep):
            x_tr = np.random.randn(n_pat, n_dim)
            y_tr = x_tr @ w + noise * np.random.randn(n_pat)
            y_te_p = Ridge(alpha=alpha, fit_intercept=False).fit(x_tr, y_tr).predict(x_te)
            predictions[i, j, :] = y_te_p
            distances[i, j] = mean_squared_error(y_te, y_te_p)

    return distances, predictions, y_te

def plot_perceptron_evo_epochs(x, y, max_epochs=5):
    import warnings
    warnings.filterwarnings("ignore", category=Warning)

    fig, ax = plt.subplots(nrows=1, ncols=max_iters)
    for i in range(max_epochs):
        model = Perceptron(tol=-1, max_iter=i + 1)
        model.fit(x, y)

        plt.sca(ax[i])
        plot_linear_model_clas(x, y, model.coef_[0], model.intercept_)
        if (i > 0):
            ax[i].set_yticklabels("")
            ax[i].set_ylabel("")
        plt.title("Epoch %d (Acc: %.2f%%)" % (i + 1, 100 * accuracy_score(y, model.predict(x))))
    plt.tight_layout()

def plot_perceptron_evo_iter(x, y, max_iters=5):
    import warnings
    warnings.filterwarnings("ignore", category=Warning)

    n_pat = x.shape[0]
    n_dim = x.shape[1]
    w = np.zeros(n_dim + 1)
    w[0] = 1e-1
    x_b = np.column_stack((np.ones(n_pat), x))

    nrows = int(np.ceil(max_iters / 5))
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    ax = ax.ravel()
    for i in range(max_iters):
        x_i = x_b[i % n_pat, :]
        y_i = y[i % n_pat]

        pred = np.sign(w @ x_i)

        plt.sca(ax[i])
        plot_linear_model_clas(x, y, w[1:], w[0])
        plt.scatter(x_i[1], x_i[2], s=200, linewidth=4, facecolors="none", edgecolors="k", zorder=100)
        if (i % ncols > 0):
            ax[i].set_yticklabels("")
            ax[i].set_ylabel("")
        if (i < (nrows - 1) * ncols):
            ax[i].set_xticklabels("")
            ax[i].set_xlabel("")
        plt.title("Iter. %d (Acc: %.2f%%)" % (i + 1, 100 * accuracy_score(y, np.sign(x_b @ w))))

        w += (y_i - pred) / 2 * x_i

    plt.tight_layout()

def plot_nonlinear_model(x, y_r, model, phi=None):
    if phi is None:
        phi = lambda x: np.reshape(x, (-1, 1))
    y_p = model.predict(phi(x))

    plt.plot(x, y_r, "*", label="Obs.")
    plt.plot(x, y_p, "-", label="Pred")

    for i in range(len(x)):
        plt.plot([x[i].item(), x[i].item()], [y_p[i].item(), y_r[i].item()], ":k")

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("(MSE: %.2f, MAE: %.2f, R2: %.2f)" % (mean_squared_error(y_r, y_p), mean_absolute_error(y_r, y_p), r2_score(y_r, y_p)))

def plot_nonlinear_model_clas(x, y_r, model, phi=None, n_points=31):
    if phi is None:
        phi = lambda x: np.reshape(x, (-1, 1))

    alpha = 0.3
    col_1 = np.array([31, 119, 180]) / 255
    col_2 = np.array([214, 39, 40]) / 255

    y_p = model.predict(phi(x))

    ind = y_r < 0
    plt.scatter(x[ind, 0], x[ind, 1], c=[col_1], zorder=100)
    ind = y_r >= 0
    plt.scatter(x[ind, 0], x[ind, 1], c=[col_2], zorder=100)
    ax = plt.axis("equal")

    x_1 = np.linspace(plt.xlim()[0], plt.xlim()[1], n_points)
    x_1 = np.hstack((x_1[0] - 100, x_1, x_1[-1] + 100))
    x_2 = np.linspace(plt.ylim()[0], plt.ylim()[1], n_points)
    x_2 = np.hstack((x_2[0] - 100, x_2, x_2[-1] + 100))

    x_1, x_2 = np.meshgrid(x_1, x_2, indexing="ij")

    plt.pcolormesh(x_1, x_2, np.reshape(model.predict(phi(np.column_stack((x_1.ravel(), x_2.ravel())))), x_1.shape), shading="auto", cmap=colors.ListedColormap([alpha * col_1 + 1 - alpha, alpha * col_2 + 1 - alpha]))

    plt.axis(ax)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.title("(Acc: %.2f%%)" % (100 * accuracy_score(y_r, y_p)))

def polynomial_basis(X, deg):
    def phi(x):
        return np.power(x, np.arange(0, deg + 1))
    return np.array([phi(x) for x in X])

def gaussian_basis(X, mu, sigma):
    sigma_2 = np.power(sigma, 2)
    def phi(x):
        return np.exp(- np.power(x - mu, 2) / sigma_2)
    return np.array([phi(x) for x in X])

def sigmoidal_basis(X, a, b):
    def phi(x):
        return 1 / (1 + np.exp(- (a * x - b)))
    return np.array([phi(x) for x in X])

def plot_krr_coefficients(model, label_gap=5):
    coef = model.dual_coef_
    pos = np.arange(len(coef))
    plt.bar(pos, coef, alpha=0.5)

    plt.grid()
    labels = []
    for i in range(len(coef)):
        labels.append("$\\alpha_{%d}$" % (i + 1))
    plt.xticks(pos[::label_gap], labels[::label_gap])
    plt.title("Dual Coefficients")

def plot_svc(x, y, model, n_points=151, plot_slack=False):
    alpha = 0.3
    col_1 = np.array([31, 119, 180]) / 255
    col_2 = np.array([214, 39, 40]) / 255

    ind = y != 1
    plt.scatter(x[ind, 0], x[ind, 1], c="r", s=30, zorder=100)
    ind = y == 1
    plt.scatter(x[ind, 0], x[ind, 1], c="b", s=30, zorder=100)

    lims = plt.axis("equal")

    xx = np.linspace(lims[0] - 1.1 * (lims[1] - lims[0]), lims[1] + 1.1 * (lims[1] - lims[0]), n_points)
    yy = np.linspace(lims[2], lims[3], n_points)
    yy, xx = np.meshgrid(yy, xx)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    zz = model.decision_function(xy).reshape(xx.shape)

    plt.pcolormesh(xx, yy, np.sign(zz), shading="auto", cmap=colors.ListedColormap([alpha * col_2 + 1 - alpha, alpha * col_1 + 1 - alpha]))
    plt.contour(xx, yy, zz, colors=["r", "k", "b"], levels=[-1, 0, 1], linestyles=["--", "-", "--"], linewidths=[2, 4, 2])
    plt.legend(handles=[
        lines.Line2D([], [], color="r", linestyle="--", label="Support Hyp. $-1$"),
        lines.Line2D([], [], color="k", linestyle="-", label="Sepparating Hyp."),
        lines.Line2D([], [], color="b", linestyle="--", label="Support Hyp. $+1$")
    ])

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=3, facecolors="none", edgecolors="k")

    if (plot_slack):
        w = model.coef_[0]
        b = model.intercept_
        nws = np.linalg.norm(w)**2
        for i in model.support_:
            p = x[i, :] - (w @ x[i, :] + b - y[i]) / nws * w
            style = "b:" if y[i] == 1 else "r:"
            plt.plot([p[0], x[i, 0]], [p[1], x[i, 1]], style)


    plt.axis(lims)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("SVM (%s, C=%.2g)" % (model.kernel, model.C))

def plot_all_linear_separators(x, y, plot_best=False, n_points=51):
    ang_vec = np.linspace(0, 2 * np.pi, n_points)
    b_vec = np.linspace(-5, 5, n_points)

    ang_mat, b_mat = np.meshgrid(ang_vec, b_vec, indexing="ij")
    ws = []
    bs = []
    ms = []
    svs = []

    for i_ang in range(len(ang_vec)):
        ang = ang_vec[i_ang]
        for i_b in range(len(b_vec)):
            b = b_vec[i_b]
            w = np.array([np.sin(ang), np.cos(ang)])
            d = (np.abs(x @ w + b) / np.linalg.norm(w))
            m = d.min()
            sv = np.argsort(d)[:3]
            y_p = np.sign(x @ w + b)
            if (accuracy_score(y, y_p) == 1):
                ws.append(w)
                bs.append(b)
                ms.append(m)
                svs.append(sv)

    plot_dataset_clas(x, y)
    lims = plt.axis()

    max_m = np.array(ms).max()
    for w, b, m, sv in zip(ws, bs, ms, svs):
        if (w[1] != 0):
            x1 = np.asarray(lims[0:2])
            x1[0] -= 1.1 * (lims[1] - lims[0])
            x1[1] += 1.1 * (lims[1] - lims[0])
            x2 = - (w[0] * x1 + b) / w[1]
        else:
            x2 = lims[2:]
            x1 = - (w[1] * x2 + b) / w[0]

        if (plot_best):
            if (m == max_m):
                plt.plot(x1, x2, "-k", alpha=1.0)
                plt.scatter(x[sv, 0], x[sv, 1], s=100, linewidth=3, facecolors="none", edgecolors="k")
            else:
                plt.plot(x1, x2, "-k", alpha=0.3)
        else:
            plt.plot(x1, x2, "-k", alpha=0.3)

    plt.axis(lims)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Linear Classifiers")

def plot_svr(x, y, model, n_points=151, plot_slack=False):
    x_e = np.linspace(x.min(), x.max(), n_points)
    np.append(x_e, x)
    x_e.sort()

    y_p = model.predict(x_e.reshape(-1, 1))
    y_pi = model.predict(x.reshape(-1, 1))

    plt.plot(x, y, "*", label="Obs.")
    plt.plot(x_e, y_p, "-", label="Model")
    plt.plot(x_e, y_p + model.epsilon, "--k")
    plt.plot(x_e, y_p - model.epsilon, "--k")

    plt.scatter(x[model.support_], y[model.support_], s=100, linewidth=3, facecolors="none", edgecolors="k")

    if (plot_slack):
        for i in range(len(x)):
            if (y_pi[i] > y[i] + model.epsilon):
                plt.plot([x[i].item(), x[i].item()], [y_pi[i].item() - model.epsilon, y[i].item()], ":k")
            if (y_pi[i] < y[i] - model.epsilon):
                plt.plot([x[i].item(), x[i].item()], [y_pi[i].item() + model.epsilon, y[i].item()], ":k")

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("SVM (%s, C=%.2g)" % (model.kernel, model.C))

def evaluate_nonlinear_model(x_tr, y_tr_r, x_te, y_te_r, model):
    y_tr_p = model.predict(x_tr)
    y_te_p = model.predict(x_te)

    er_tr = [mean_squared_error(y_tr_r, y_tr_p), mean_absolute_error(y_tr_r, y_tr_p), r2_score(y_tr_r, y_tr_p)]
    er_te = [mean_squared_error(y_te_r, y_te_p), mean_absolute_error(y_te_r, y_te_p), r2_score(y_te_r, y_te_p)]

    ers = [er_tr, er_te]
    headers=["MSE", "MAE", "R2"]

    print("%10s" % "", end="")
    for h in headers:
        print("%10s" % h, end="")
    print("")

    headersc = ["Train", "Test"]

    cnt = 0
    for er in ers:
        hc = headersc[cnt]
        cnt = cnt + 1
        print("%10s" % hc, end="")

        for e in er:
            print("%10.2f" % e, end="")
        print("")

def generate_sequence(env, early_stop=True, model=None, n_steps=500):
    state = env.reset()
    for i in range(n_steps):
        env.render()
        if model is None:
            action = env.action_space.sample()
        else:
            action_probs, _ = model.predict(tf.expand_dims(tf.convert_to_tensor(state), 0))
            p = np.squeeze(action_probs)
            action = np.random.choice(len(p), p=p)
        state, _, done, _ = env.step(action)
        if early_stop and done:
            print("Finished after %d steps" % i)
            break
