import math
import time

import matplotlib.pyplot as plt
import numpy as np

start_time = time.time()


def linear_regression_func(x, w):
    # return sum([(x ** i) * w[i] for i in range(len(w))])
    return w[0] ** 2 - w[0] * w[1] + w[1] ** 2 + 9 * w[0] - 6 * w[1] + 20


def generate_func(x):
    # return math.sqrt(x) + 2 * math.sin(x)
    return 2 + 5 * math.sin(2 * x)


def generate_sample(_start, _count, _step):
    x = _start
    i = 0
    while i < _count:
        i += 1
        yield (generate_func(x) + get_rand(-0.4, 0.4) * get_rand(0, 0.8))
        x += _step


def get_rand(a, b):
    return np.random.uniform(a, b)


start = 0.2
count = 100
step = 0.1
X = np.arange(start, count * step + start, step)
Y = np.array([round(y, 2) for y in generate_sample(start, count, step)])

Y_real = np.array([generate_func(x) for x in X])


def get_value(x, y, w=None):
    # return [(y[i] - sum([(x[i] ** j) * w[j]
    #                  for j in range(len(w))])) ** 2 for i in range(len(x))]
    # return w[0] ** 2 - w[0] * w[1] + w[1] ** 2 + 9 * w[0] - 6 * w[1] + 20
    return x**2 - x * y + y**2 + 9 * x - 6 * y + 20


def get_function_points_value(x, y):
    res = []
    for i in range(len(x)):
        res.append(get_value(x[i], y[i]))
    return res


def get_sum(x, y, w):
    return sum(get_value(x, y, w))


def get_grad(x, y, w, size):
    # powers = np.array([x ** i for i in range(size)])
    # return np.array([-2 * (y - np.sum(w * powers)) * powers[i]
    #                  for i in range(size)])
    return np.array([2 * w[0] - w[1] + 9, -w[0] + 2 * w[1] - 6])


def shuffle(permutation):
    return list(np.random.permutation(permutation))


def sth_gradient(xs, ys, w, lr, _batch_size, max_iters):
    cur_w = w
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])
    values = [cur_w]
    iters = 0
    pos = 0
    grad_calc = 0
    while iters < max_iters:
        grad = get_nparray(w)
        if pos + _batch_size - 1 >= cnt_dots:
            pos = 0
            perm = shuffle(perm)
        for j in range(pos, pos + _batch_size):
            grad = grad + get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
            grad_calc += 1
        grad = grad / _batch_size
        pos += _batch_size
        cur_w = cur_w - lr(iters) * grad
        values += [cur_w]
        iters += 1
    return [cur_w, iters, grad_calc, values]


def momentum(cur_w, cur_momentum, grad, gamma, lr):
    next_momentum = gamma * cur_momentum + (1 - gamma) * grad
    next_w = cur_w - lr * cur_momentum
    return [next_w, next_momentum]


def momentum_gradient(xs, ys, w, lr, _batch_size, max_iters):
    cur_w = w
    cur_momentum = get_nparray(w)
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])
    values = []
    iters = 0
    pos = 0
    grad_calc = 0
    while iters < max_iters:
        grad = np.array([0.0 for i in range(len(w))])
        if pos + _batch_size - 1 >= cnt_dots:
            pos = 0
            perm = shuffle(perm)
        for j in range(pos, pos + _batch_size):
            grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
            grad_calc += 1
        grad /= _batch_size
        pos += _batch_size
        cur_w, cur_momentum = momentum(cur_w, cur_momentum, grad, 0.9, lr(iters))
        values += [cur_w]
        iters += 1
    return [cur_w, iters, grad_calc, values]


def nest(cur_w, cur_momentum, grad, gamma, lr):
    next_momentum = gamma * cur_momentum + (1 - gamma) * grad
    next_w = cur_w - lr * cur_momentum
    return [next_w, next_momentum]


def nest_gradient(xs, ys, w, lr, _batch_size, max_iters):
    cur_w = w
    cur_nest = get_nparray(w)
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])
    values = []
    iters = 0
    pos = 0
    grad_calc = 0
    while iters < max_iters:
        grad = get_nparray(w)
        if pos + _batch_size - 1 >= cnt_dots:
            pos = 0
            perm = shuffle(perm)
        for j in range(pos, pos + _batch_size):
            grad += get_grad(
                xs[perm[j]],
                ys[perm[j]],
                cur_w - lr(iters) * 0.9 * cur_nest,
                len(cur_w),
            )
            grad_calc += 1
        grad /= _batch_size
        pos += _batch_size
        cur_w, cur_nest = nest(cur_w, cur_nest, grad, 0.9, lr(iters))
        values += [cur_w]
        iters += 1
    return [cur_w, iters, grad_calc, values]


def get_nparray(w):
    return np.array([0.0 for _ in range(len(w))])


def ada_grad(cur_w, cur_grad, grad, lr):
    new_grad = cur_grad + np.square(-grad)
    new_w = cur_w + lr * np.divide(grad, np.sqrt(new_grad))
    return [new_w, new_grad]


def ada_grad_gradient(xs, ys, w, lr, _batch_size, max_iters):
    cur_w = w
    cur_ada_grad = get_nparray(w)
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])
    values = [cur_w]
    iters = 0
    pos = 0
    grad_calc = 0
    while iters < max_iters:
        for i in range(cnt_dots // _batch_size):
            grad = get_nparray(w)
            if pos + _batch_size - 1 >= cnt_dots:
                pos = 0
                perm = shuffle(perm)
            for j in range(pos, pos + _batch_size):
                grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
                grad_calc += 1
            grad /= _batch_size
            pos += _batch_size
            cur_w, cur_ada_grad = ada_grad(cur_w, cur_ada_grad, -grad, lr(iters))
        values += [cur_w]
        iters += 1
    return [cur_w, iters, grad_calc, values]


def rms_prop(cur_w, cur_grad, grad, gamma, lr):
    new_grad = gamma * cur_grad + (1 - gamma) * np.square(-grad)
    new_w = cur_w + lr * np.divide(grad, np.sqrt(new_grad) + 0.0000001)
    return [new_w, new_grad]


def rms_prop_gradient(xs, ys, w, lr, _batch_size, max_iters):
    cur_w = w
    cur_prop_grad = get_nparray(w)
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])
    values = [cur_w]
    iters = 0
    pos = 0
    grad_calc = 0
    while iters < max_iters:
        for i in range(cnt_dots // _batch_size):
            grad = get_nparray(w)
            if pos + _batch_size - 1 >= cnt_dots:
                pos = 0
                perm = shuffle(perm)
            for j in range(pos, pos + _batch_size):
                grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
                grad_calc += 1
            grad /= _batch_size
            pos += _batch_size
            cur_w, cur_prop_grad = rms_prop(cur_w, cur_prop_grad, -grad, 0.9, lr(iters))
        values += [cur_w]
        iters += 1
    return [cur_w, iters, grad_calc, values]


def adam(cur_w, cur_momentum, cur_grad, grad, beta, gamma, lr):
    new_momentum = beta * cur_momentum - (1 - beta) * grad
    new_grad = gamma * cur_grad + (1 - gamma) * np.square(-grad)
    new_w = cur_w - lr * np.divide(new_momentum, np.sqrt(new_grad) + 0.0000001)
    return [new_w, new_momentum, new_grad]


def adam_gradient(xs, ys, w, lr, _batch_size, max_iters):
    cur_w = w
    cur_prop_grad = get_nparray(w)
    cur_momentum = get_nparray(w)
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])
    values = [cur_w]
    iters = 0
    pos = 0
    grad_calc = 0
    while iters < max_iters:
        for i in range(cnt_dots // _batch_size):
            grad = get_nparray(w)
            if pos + _batch_size - 1 >= cnt_dots:
                pos = 0
                perm = shuffle(perm)
            for j in range(pos, pos + _batch_size):
                grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
                grad_calc += 1
            grad /= _batch_size
            pos += _batch_size
            cur_w, cur_momentum, cur_prop_grad = adam(
                cur_w, cur_momentum, cur_prop_grad, -grad, 0.9, 0.9, lr(iters)
            )
        values += [cur_w]
        iters += 1
    return [cur_w, iters, grad_calc, values]


def adam_gradient_with_regularization(
    xs, ys, w, lr, _batch_size, max_iters, l1_mode, coeff1, l2_mode, coeff2
):
    cur_w = w
    cur_prop_grad = get_nparray(w)
    cur_momentum = get_nparray(w)
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])
    values = [cur_w]
    iters = 0
    pos = 0
    grad_calc = 0
    while iters < max_iters:
        for i in range(cnt_dots // _batch_size):
            grad = get_nparray(w)
            if pos + _batch_size - 1 >= cnt_dots:
                pos = 0
                perm = shuffle(perm)
            for j in range(pos, pos + _batch_size):
                grad += get_grad_regularization(
                    xs[perm[j]],
                    ys[perm[j]],
                    cur_w,
                    len(cur_w),
                    l1_mode,
                    coeff1,
                    l2_mode,
                    coeff2,
                )
                grad_calc += 1
            grad /= _batch_size
            pos += _batch_size
            cur_w, cur_momentum, cur_prop_grad = adam(
                cur_w, cur_momentum, cur_prop_grad, -grad, 0.9, 0.9, lr(iters)
            )
        values += [cur_w]
        iters += 1
    return [cur_w, iters, grad_calc, values]


def regularization_func(_x, _y, w, l1_mode, alpha1, l2_mode, alpha2):
    _l = get_sum(_x, _y, w)
    weights = 0
    for i in range(len(w)):
        if l1_mode:
            weights += abs(w[i])
        if l2_mode:
            weights += w[i] * w[i]
    if l1_mode:
        _l += alpha2 * weights
    if l2_mode:
        _l += alpha1 * weights
    return _l


def get_grad_regularization(x, y, w, size, l1_mode, alpha1, l2_mode, alpha2):
    grad = get_grad(x, y, w, size)
    if l1_mode:
        loss1 = 0
        for i in range(size):
            if w[i] > 0:
                loss1 += 1
            else:
                loss1 += -1
        grad += alpha1 * loss1
    if l2_mode:
        loss2 = 0
        for i in range(size):
            loss2 += 2 * w[i]
        grad += alpha2 * loss2
    return grad


def cur_method(method):
    _test_base = method(
        list(X), list(Y), w_start, learning_rate, batch_size, epoch_base
    )
    print(
        "function:",
        _test_base[0][0],
        "+",
        str(_test_base[0][1]),
        " * x",
        "\nepoch:",
        _test_base[1],
        "\ngradient_calculations:",
        _test_base[2],
    )
    return _test_base


def stage_lr(iteration):
    return 0.002 * (0.9 ** (21 * iteration / (iteration + 1)))


def exp_lr(iteration):
    return math.exp(-0.05 * (iteration + 100))


def const_lr(iteration=None):
    return 0.1


def draw():
    cur_test = test_adam_grad
    res = cur_test[3]
    points = np.array(res)
    # print(points)
    t = np.linspace(-20, 20, 100)
    X_draw, Y_draw = np.meshgrid(t, t)
    # Z_draw = np.array(get_value(X_draw, Y_draw, cur_test[0]))
    Z_draw = np.array(get_function_points_value(X_draw, Y_draw))
    ax = plt.subplot()
    ax.grid()
    # ax.contour(X_draw,
    #            Y_draw,
    #            Z_draw,
    #            levels=sorted(
    #                [get_value([p[0]], [p[1]], cur_test[0])[0]
    #                 for p in points
    #                 ]))
    ax.contour(
        X_draw, Y_draw, Z_draw, levels=sorted([get_value(p[0], p[1]) for p in points])
    )
    ax.plot(points[:, 0], points[:, 1], "o-")
    plt.xlabel("Коэффициент: w[0]")
    plt.ylabel("Коэффициент: w[1]")
    plt.show()


w_start = np.array([5, 5])
learning_rate = const_lr
batch_size = 1
epoch_base = 12000
epoch_momentum = 10000
epoch_nesterov = 2000
epoch_ada_grad = 1000
epoch_prop_grad = 8
epoch_adam = 8
l1_coef = 20
l2_coef = 20

test_base = cur_method(sth_gradient)

# test_momentum = test_base(momentum)
#
# test_nesterov = test_method(nest_gradient)
#
# test_ada_grad = test_method(ada_grad_gradient)
#
# test_prop_grad = test_method(rms_prop_gradient)
#
# test_adam_grad_with_regularization_L1 = adam_gradient_with_regularization(
#     list(X), list(Y), w_start, learning_rate,
#     batch_size, epoch_adam,
#     True, l1_coef, False, l2_coef)
#
# test_adam_grad_with_regularization_L2 = adam_gradient_with_regularization(
#     list(X), list(Y), w_start, learning_rate,
#     batch_size, epoch_adam,
#     False, l1_coef, True, l2_coef)
#
# test_adam_grad_with_regularization_Elastic = adam_gradient_with_regularization(
#     list(X), list(Y), w_start, learning_rate,
#     batch_size, epoch_adam,
#     True, l1_coef, True, l2_coef)
#
test_adam_grad = cur_method(adam_gradient)

# plt.scatter(X, Y, alpha=0.4)
# plt.plot(X, Y_real, 'g', linewidth=2.0)
# Y_current1 = np.array([
#     linear_regression_func(x, test_adam_grad[0])
#     for x in X
# ])
# plt.plot(X, Y_current1, 'b', linewidth=2.0)
#
# Y_current1 = np.array([linear_regression_func(x, test_momentum[0]) for x in X])
# plt.plot(X, Y_current1, 'b', linewidth=2.0)
# Y_current2 = np.array([
#     linear_regression_func(x, test_adam_grad_with_regularization_L1[0])
#     for x in X
# ])
# plt.plot(X, Y_current2, 'r', linewidth=2.0)
# Y_current3 = np.array([
#     linear_regression_func(x, test_adam_grad_with_regularization_L2[0])
#     for x in X
# ])
# plt.plot(X, Y_current3, 'g', linewidth=2.0)
# Y_current4 = np.array([
#     linear_regression_func(x, test_adam_grad_with_regularization_Elastic[0])
#     for x in X
# ])
# plt.plot(X, Y_current4, 'y', linewidth=2.0)
# print("Base ADAM Loss(blue):",
#       get_sum(X, Y, test_adam_grad[0]))
# print("L1 ADAM Loss(red):",
#       regularization_func(X, Y, test_adam_grad_with_regularization_L1[0],
#                           True, l1_coef, False, l2_coef))
# print("L2 ADAM Loss(green):",
#       regularization_func(X, Y, test_adam_grad_with_regularization_L2[0],
#                           False, l1_coef, True, l2_coef))
# print("Elastic ADAM Loss(yellow):",
#       regularization_func(X, Y, test_adam_grad_with_regularization_Elastic[0],
#                           True, l1_coef, True, l2_coef))
#
draw()

# print(psutil.virtual_memory())
# end = time.time()
# print("time:", end - start)
