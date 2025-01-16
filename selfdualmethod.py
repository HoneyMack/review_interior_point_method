import numpy as np
import math


# REVIEW: この処理はわざわざクラスにする必要がないと思う．MainMethodクラスのメソッドとして実装してもよい
class FirstMethod:  # REVIEW: FirstMethodは何を表すのかわからないので，役割・処理を表す名称に変更する ex. FeasibleSolutionCalculator/ FeasibleSolutionGenerator
    # Find feasible interior solution and feasible self-dual problem
    def generate_initial_sol(self, M):  # REVIEW: Mは何を表すのかわからないので，役割・処理を表す名称に
        M_dim1 = M.shape[0]
        e = np.ones(M_dim1).reshape(-1, 1)
        r = e - M @ e
        zero = np.zeros((1, 1))
        M1 = np.concatenate((M, r), axis=1)
        M2 = np.concatenate((-r.T, zero), axis=1)
        M_processed = np.concatenate((M1, M2), axis=0)  # 実行可能内点を持つ歪行列
        x_processed = np.concatenate((e, np.ones((1, 1))), axis=0)  # 実行可能内点初期解
        s_processed = np.concatenate((e, np.ones((1, 1))), axis=0)  # 実行可能内点初期解
        return (M_processed, x_processed, s_processed)


# REVIEW: この処理はわざわざクラスにする必要がないと思う．MainMethodクラスのメソッドとして実装してもよい.
class newton:  # REVIEW: PEP8に従ってクラス名を大文字にする  i.e. Newton
    def __init__(self, M, x, s, mu):  # REVIEW: docstringでクラスやメソッドの入出力を説明
        self.M = M
        self.x = x
        self.s = s
        self.mu = mu

    def newtonMethod(self):
        X = np.diag(self.x.flatten())
        S = np.diag(self.s.flatten())
        e = np.ones(len(self.x)).reshape(-1, 1)
        # 更新方向を決定
        delta_x = np.linalg.inv(S + X @ self.M) @ (self.mu * e - X @ self.s)
        delta_s = self.M @ delta_x
        return (delta_x, delta_s)


class MainMethod:
    # self-dual interior-point method
    def __init__(self, M):
        self.M = M

    def interior_point_method(self):
        error_tol = (
            10**-6
        )  # REVIEW: ハイパラをハードコードしないのは偉い．だが，引数でデフォルト引数の形で与えたほうが汎用的でなおよい．ex. def interior_point_method(self, error_tol=10**-6):
        # initial feasible interior-point
        initials = FirstMethod()
        (skewMat, init_x, init_s) = initials.generate_initial_sol(self.M)

        gamma = 1.0 / math.sqrt(2.0 * skewMat.shape[0])  # 双対ギャップの更新パラメータ
        mu = 1.0

        # initialize
        x = init_x
        s = init_s

        while True:
            if skewMat.shape[0] * mu < error_tol:
                return (x, s)
            else:
                mu = (1.0 - gamma) * mu
                # newton方向を求める
                delta = newton(skewMat, x, s, mu)
                (delta_x, delta_s) = delta.newtonMethod()
                # 解の更新
                x = x + delta_x
                s = s + delta_s


# REVIEW: このクラスの関数はどれもインスタンスに依存しない処理なので，関数をクラスメソッドにする. i.e. 「@classmethod」デコレータを関数につける.
# REVIEW: このクラスは「solve」という名前だが，もっと具体化できる ex. LPSolver
# REVIEW: 関数間で引数の順番が違う．統一するとよい．cf.「scale_problem(A, b, c): <-> solveLP(c, A, b)」
class solve:
    def scale_problem(A, b, c):  # REVIEW: docstringでクラスやメソッドの入出力を説明
        # 行スケーリング
        row_max = np.max(np.abs(A), axis=1).reshape(-1, 1)
        A_scaled = A / row_max
        b_scaled = b / row_max

        # 列スケーリング
        col_max = np.max(np.abs(A_scaled), axis=0).reshape(1, -1)
        A_scaled = A_scaled / col_max
        c_scaled = c / col_max.T

        return A_scaled, b_scaled, c_scaled, col_max, row_max

    def solveLP(self, c, A, b):
        A, b, c, colscale, rowscale = solve.scale_problem(A, b, c)
        m, n = A.shape
        zero_m = np.zeros((m, m))
        zero_n = np.zeros((n, n))
        zero = np.zeros((1, 1))
        M1 = np.concatenate((zero_m, A, -b), axis=1)
        M2 = np.concatenate((-A.T, zero_n, c), axis=1)
        M3 = np.concatenate((b.T, -c.T, zero), axis=1)
        M = np.concatenate((M1, M2, M3), axis=0)

        method = MainMethod(M)
        x_sol_SD = method.interior_point_method()[0]
        dual_sol = (x_sol_SD[:m, :] / x_sol_SD[m + n, :]) * colscale
        main_sol = (x_sol_SD[m : m + n, :] / x_sol_SD[m + n, :]) * colscale

        dual_value = np.dot(b * rowscale.T, dual_sol)[0, 0]
        main_value = np.dot(c.T, main_sol)[0, 0]

        return dual_sol, main_sol, dual_value, main_value


# REVIEW: 以下の処理はテストコードなのでmain関数にまとめるとよい．i.e.「if __name__ == "__main__": 以下に記述」

# ある企業が、限られた広告予算を次の3つの広告チャネルに配分するとします。

# テレビ広告 (x1)
# オンライン広告 (x2)
# 新聞広告 (x3)
# それぞれの広告チャネルには異なる影響力（利益）とコストがあり、
# 広告予算の制約内で利益を最大化したいと考えています。

# 制約条件
# 広告予算は最大で 100万円。
# テレビ広告は最低でも 10万円以上使う必要がある。
# オンライン広告は最低でも 5万円以上使う必要がある。
# 広告の総支出は予算内でなければならない。

# 利益の係数
# テレビ広告 (x1) の1万円当たりの利益: 15万円。
# オンライン広告 (x2) の1万円当たりの利益: 20万円。
# 新聞広告 (x3) の1万円当たりの利益: 10万円。

# 目的関数の係数
c = np.array([[15], [20], [10]])

# 不等式制約の係数行列 (A @ x <= b)
A = np.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0]])  # 広告予算の合計  # x_1 >= 10 の変形  # x_2 >= 5 の変形

# 不等式制約の右辺
b = np.array([[100], [-10], [-5]])
import scipy

result1 = scipy.optimize.linprog(-c, A_ub=A, b_ub=b)

r = solve()
result = r.solveLP(c, A, b)
print(result)
print(result1.message)
print(result1.success)
print(result1.fun)
print(result1.x)
print(result1.nit)
