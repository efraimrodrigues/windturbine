import csv
import numpy as np
import random

def load_data_set():
    x = []
    y = []

    with open('data/aerogerador2.dat', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    x = np.array(x)
    y = np.array(y)

    """for i in range(0, len(x[0])):
        x_max = max(np.array(x)[:,i])
        x_min = min(np.array(x)[:,i])

        x[:,i] = 2 * (np.array(np.array(x)[:,i]) - x_min)/(x_max - x_min) - 1

    y_max = max(y)
    y_min = min(y)

    y = 2 * (y - y_min)/(y_max - y_min) - 1
    """
    return x, y

x, y = load_data_set()

def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)

def r_squared_adj(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - ((ss_res/(len(y)-2))/(ss_tot/(len(y)-1)))

def aic(y, y_hat):
    ss_res = ((y-y_hat)**2).sum()
    return 2 + 2*np.log(ss_res)

def random_solution(d):
    a = []
    for i in range(0, d):
        if i != d - 1:
            a.append(random.uniform(-0.00005, 0.00005))
        else:
            a.append(0)

    return np.array(a)

def global_random_search(x, y, d, M, p):
    best = random_solution(d)
    f_best = aic(y, np.poly1d(best)(x))

    best_1 = None
    best_2 = None
    best_3 = None
    
    c = 0
    for i in range(0, M):
        candidate = random_solution(d)
        f_candidate = aic(y, np.poly1d(candidate)(x))

        if  f_candidate < f_best:
            best = candidate
            f_best = aic(y, np.poly1d(best)(x))

            
            best_3 = best_2
            best_2 = best_1
            best_1 = best

        else:
            c = c + 1

        if c == p:
            break
            
    return best_1, best_2, best_3

def error_squared(f):
    y_hat = np.poly1d(f)(x)
    ss_res = ((y-y_hat)**2).sum()
    return ss_res

def error(f, error_squared):
    y_hat = np.poly1d(f)(x)

    if error_squared:
        ss_res = ((y-y_hat)**2).sum()
    else:
        ss_res = (np.abs(y-y_hat)).sum()
    return ss_res

def select(population):
    return population[random.randint(0, len(population)-1)]

def selection(population, error_squared):
    selected = []

    for i in range(len(population)):
        first = select(population)
        second = select(population)

        if error(first, error_squared) < error(second, error_squared):
            selected.append(first)
        else:
            selected.append(second)

    return selected

def crossover(selection, error_squared):
    offspring = []

    for i in range(0, len(selection), 2):
        first = selection[i]
        second = selection[i+1]

        if random.random() <= 0.86:
            x_1 = first + second
            x_2 = 1.005*first - 0.005*second
            x_3 = -0.005*first + 1.005*second

            e_x_1 = error(x_1, error_squared)
            e_x_2 = error(x_2, error_squared)
            e_x_3 = error(x_3, error_squared)

            best_1 = None
            best_2 = None
            if e_x_1 > e_x_2 and e_x_1 > e_x_3:
                best_1 = x_2
                best_2 = x_3

            if e_x_2 > e_x_1 and e_x_2 > e_x_3:
                best_1 = x_1
                best_2 = x_3

            if e_x_3 > e_x_1 and e_x_3 > e_x_2:
                best_1 = x_1
                best_2 = x_2

            offspring.append(best_1)
            offspring.append(best_2)
        else:
            offspring.append(first)
            offspring.append(second)

    return offspring

def mutation(offspring, error_squared):
    for i in range(len(offspring)):
        if random.random() <= 0.05:
            offspring[i] = offspring[i] + 1.0005 * random_solution(10)

    return offspring