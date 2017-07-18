from numpy import *


def batch_grad_desc(data, k):
    r = mat(data)
    print(r)
    print()
    m, n = shape(r)
    p = mat(random.random((m, k)))
    q = mat(random.random((n, k)))
    gamma = 0.0002
    lambd = 0.04
    epsilon = 0.001
    max_cycle = 10000

    for step in range(max_cycle):
        for u in range(m):
            for i in range(n):
                if r[u, i] > 0:
                    # error = r_ui - q_i*p_u
                    e_ui = r[u, i] - (q[i] * p[u].T)[0, 0]
                    # update q_i and p_u
                    q[i] += gamma * (e_ui * p[u] - lambd * q[i])
                    p[u] += gamma * (e_ui * q[i] - lambd * p[u])

        loss = 0
        for u in range(m):
            for i in range(n):
                if r[u, i] > 0:
                    loss += (r[u, i] - (q[i] * p[u].T)[0, 0]) ** 2 \
                            + lambd * (linalg.norm(p[u], 2) ** 2 + linalg.norm(q[i], 2) ** 2)

        if loss < epsilon:
            break

        if step % 1000 == 0:
            print(loss)
    return q, p


if __name__ == "__main__":
    data = [[5, 3, 0, 1, 1],
            [4, 0, 0, 1, 2],
            [1, 1, 0, 5, 3],
            [0, 1, 5, 4, 0]]
    qq, pp = batch_grad_desc(data, 5)
    result = pp * qq.T

    print()
    print("p:")
    print(pp)
    print()
    print("q:")
    print(qq)
    print()
    print("result")
    print(result)
