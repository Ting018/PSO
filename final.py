import numpy as np
import math
import matplotlib.pyplot as plt

# ========== 單一粒子 ==========
class Partical(object):   # 正確拼字應該是 Particle，但保持一致
    def __init__(self, dim, lb, ub):
        self.dim = dim
        self.position = np.random.uniform(lb, ub, dim)  # 隨機位置
        self.velocity = np.random.uniform(-1, 1, dim)   # 隨機速度
        self.best_position = self.position.copy()       # 個體最佳位置
        self.best_value = float("inf")                  # 個體最佳值


# ========== PSO 演算法 ==========
class PSO_Algorithm(object):

    def __init__(self, dim=2, lb=-5.12, ub=5.12, c1=2, c2=2, w_max=0.9, w_min=0.4):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.particles = []                # 儲存所有粒子
        self.global_best_position = None   # 全域最佳位置
        self.global_best_value = float("inf") # 全域最佳值
        self.iter = 0                      # 紀錄迭代次數
        self.max_iter = 1000               # 預設迭代次數
        self.record = []                   # 收斂曲線

    def sphere(self, x):
        """Sphere function: f(x,y) = x^2 + y^2"""
        return np.sum(x**2)

    def createParticals(self, n_particles):
        """初始化粒子群"""
        self.particles = [Partical(self.dim, self.lb, self.ub) for _ in range(n_particles)]
        # 初始化全域最佳解
        for p in self.particles:
            value = self.sphere(p.position)
            p.best_value = value
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = p.position.copy()

    def search(self):
        """更新粒子速度與位置"""
        self.iter += 1
        w = self.w_max - (self.iter / self.max_iter) * (self.w_max - self.w_min)

        for p in self.particles:
            # 計算當前值
            value = self.sphere(p.position)
            # 更新個體最佳
            if value < p.best_value:
                p.best_value = value
                p.best_position = p.position.copy()
            # 更新全域最佳
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = p.position.copy()

        # 更新速度與位置
        for p in self.particles:
            r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
            cognitive = self.c1 * r1 * (p.best_position - p.position)
            social = self.c2 * r2 * (self.global_best_position - p.position)
            p.velocity = w * p.velocity + cognitive + social
            p.position = p.position + p.velocity

            # 邊界處理
            p.position = np.clip(p.position, self.lb, self.ub)

        # 收斂紀錄
        self.record.append(self.global_best_value)

    def getBestParameter(self):
        return self.global_best_position, self.global_best_value


# ========== 測試用 ==========
def PSO_test_run():
    pso = PSO_Algorithm()
    pso.createParticals(10)

    total_epoch = 1000
    pso.max_iter = total_epoch
    for epoch in range(total_epoch):
        pso.search()

    best_pos, best_val = pso.getBestParameter()
    print("最佳解:", best_pos)
    print("最小值:", best_val)

    # 繪製收斂曲線
    plt.plot(range(total_epoch), pso.record)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("PSO on sphere Function")
    plt.show()


if __name__ == "__main__":
    PSO_test_run()
