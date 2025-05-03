# author: Bas Laurent
# last edited: 30/04/2025
# Description: This file containes functions that generate the parton model.

from numpy import sin, cos, arccos, log, pi, ndarray, array, zeros
# numeric python, for array manipulation
import scipy.optimize as opt # minimizing a function
from tqdm.contrib.concurrent import process_map # for parallel processing
from tqdm import tqdm # for for-loops with pretty progressbar.
# numeric python for array manipulation


# 1. Finding the probability distribution of theta via the SVA.
def f(theta: float) -> float:
    """The probability distribution of theta."""
    return (1 + cos(theta)**2)

def f_over(theta: float) -> float:
    """The probability distribution overestimate of theta."""
    return 2

def theta(rands: ndarray) -> tuple[float, bool]:
    """Finds theta using f_over(theta) and the SVA"""
    l = len(rands)
    if l < 2:
        raise ValueError("rands must contain two random numbers.")
    if l/2 % 1 != 0:
        raise ValueError("the length of rands must be even.")
    theta = array([arccos(1 - 2*rands[2*i]) for i in range(l//2)])
    gen = array([False if f(theta[i])/f_over(theta[i]) < rands[2*i+1] \
            else True for i in range(l//2)])
    return theta, gen


# 2. Finding the probability distribution of phi.
def phi(rand: float) -> float:
    """Finds phi using the probability distribution of phi."""
    return 2*pi*rand


# 3. Finding physical quantities.
def pT(s, theta: float) -> tuple[float,float]:
    """The norm of the transvers momenta of the quark and antiquark"""
    pT = s**0.5 * sin(theta)
    return pT, pT

def eta(s, theta: float, phi: float) -> tuple[float, float]:
    """The rapidity of the quark and antiquark"""
    kp = (1 + cos(theta))/2**0.5
    km = (1 - cos(theta))/2**0.5
    return log(kp/km)/2, log(km/kp)/2

def Tfun(theta: float, phi: float, rep: float) -> float:
    """The trustarray of the parton"""
    T = lambda t, p: sin(t)*cos(p)*sin(theta)*cos(phi) + \
         sin(t)*sin(p)*sin(theta)*sin(phi) + \
         cos(t)*cos(theta)
    resq = opt.dual_annealing(lambda arg: -T(*arg), [(0, pi), (0, pi)],
                             maxiter= rep)
    resaq = opt.dual_annealing(lambda arg: T(*arg), [(0, pi), (0, pi)], 
                             maxiter= rep)
    return -resq.fun, resq.success, -resaq.fun, resaq.success

def thrust(theta: ndarray, phi: ndarray, rep: int =1000,
           workers: int =0, chunk: int =1):
    """ The thrust of the parton"""
    theta, phi = array(theta).flatten(), array(phi).flatten()
    l = len(theta)
    Tq, genq, Taq, genaq = zeros(l), zeros(l), zeros(l), zeros(l)
    if l != len(phi):
        raise ValueError("theta and phi must have the same length.")
    if workers == 0:
        for i in tqdm(list(range(l))):
            tq, gq, taq, gaq = Tfun(theta[i], phi[i], rep)
            Tq[i], genq[i], Taq[i], genaq[i] = tq, gq, taq, gaq
    elif workers > 0 and workers % 1 == 0.:
        sol = process_map(Tfun,
                          *(theta, phi, [rep]*l),
                          max_workers= int(workers),
                          chunksize = int(chunk),
                          )
        Tq, genq, Taq, genaq = zip(*sol)
    else: raise ValueError('The amount of workers must be an integer.')
    return array(Tq), array(genq), array(Taq), array(genaq)
