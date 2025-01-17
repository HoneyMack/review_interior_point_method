import numpy as np
import math

import scipy.optimize

# NOTE: ここで実装されている内点法は，Mehrotra predictor–corrector method？


# REVIEW: この処理はわざわざクラスにする必要がないと思う．MainMethodクラスのメソッドとして実装してもよい
class FirstMethod:  # REVIEW: FirstMethodは何を表すのかわからないので，役割・処理を表す名称に変更する ex. FeasibleSolutionCalculator/ FeasibleSolutionGenerator
    # 実行可能内点初期解を持つ自己双対線形計画問題を生成
    def generate_initial_sol(self, M):  # REVIEW: Mは何を表すのかわからないので，役割・処理を表す名称に変更する
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


# REVIEW: この処理はわざわざクラスにする必要がないと思う．MainMethodクラスのメソッドとして実装してもよい
class newton:  # REVIEW: PEP8に従ってクラス名を大文字にする  i.e. Newton
    # newton法で解の更新方向を定める。以下は予測子・修正子法を採用
    # REVIEW: docstringでクラスやメソッドの入出力を説明
    def __init__(self, M, x, s, mu, step):
        self.M = M
        self.x = x
        self.s = s
        self.mu = mu
        self.step = step

    def newtonMethod(self):
        # 各ステップごとに計算
        X = np.diag(self.x.flatten())
        S = np.diag(self.s.flatten())
        X_inv = np.linalg.inv(X)
        XinvS = X_inv @ S
        mat = np.linalg.inv(self.M + XinvS)

        if self.step:
            # predictor step, delta = 0
            delta_x = mat @ (-self.s)
            delta_s = -self.s - XinvS @ delta_x
        else:
            # correltor step, delta = 1
            diagXinv = X_inv @ np.ones(len(X)).reshape(-1, 1)
            delta_x = mat @ (self.mu * diagXinv - self.s)
            delta_s = -self.s - XinvS @ delta_x + self.mu * diagXinv
        return (delta_x, delta_s)


class MainMethod:
    def __init__(self, M):
        self.M = M

    def interior_point_method(self):
        error_tol = 10**-6  # 許容誤差

        # 実行可能初期解を持つ自己双対問題を作成
        initials = FirstMethod()
        (skewMat, init_x, init_s) = initials.generate_initial_sol(self.M)
        # 初期解
        x = init_x
        s = init_s

        while True:  # REVIEW: 無限ループになりかねないので，最大繰り返し回数を指定するようにしてもよいかも
            if np.dot(x.T, s)[0, 0] < error_tol:
                return (x, s)
            else:

                # 予測子
                mu_pre = np.dot(x.T, s)[0, 0] / len(x)
                predictor = newton(skewMat, x, s, mu_pre, True)
                (delta_x_pre, delta_s_pre) = predictor.newtonMethod()
                step_pre = 1 / (2 * math.sqrt(len(x)))  # ステップサイズは下限を使用(４次方程式の近似が文献に記載なし)

                x = x + step_pre * delta_x_pre
                s = s + step_pre * delta_s_pre

                # 修正子
                mu_cor = np.dot(x.T, s)[0, 0] / len(x)
                corrector = newton(skewMat, x, s, mu_cor, False)
                (delta_x_cor, delta_s_cor) = corrector.newtonMethod()
                step_cor = 1  # 修正子ではステップサイズ１

                x = x + step_cor * delta_x_cor
                s = s + step_cor * delta_s_cor


class solve:  # REVIEW: solveよりもLPSolverといった内容が具体的にわかる名称のほうがよい
    def solveLP(self, c, A, b):  # REVIEW: docstringでクラスやメソッドの入出力を説明
        # 自己双対問題を作成
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
        print(x_sol_SD)
        dual_sol = x_sol_SD[:m, :] / x_sol_SD[m + n, :]
        main_sol = x_sol_SD[m : m + n, :] / x_sol_SD[m + n, :]
        dual_value = np.dot(b.T, dual_sol)[0, 0]
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
c = -np.array([[-15], [-20], [-10]])

# 不等式制約の係数行列 (A @ x <= b)
A = np.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0]])  # 広告予算の合計  # x_1 >= 10 の変形  # x_2 >= 5 の変形

# 不等式制約の右辺
b = np.array([[100], [-10], [-5]])


import scipy

result1 = scipy.optimize.linprog(-c, A_ub=A, b_ub=b)

r = solve()
result = r.solveLP(c, A, b)
print(*result, sep="\n")
print(result1.message)
print(result1.success)
print(result1.fun)
print(result1.x)
print(result1.nit)
