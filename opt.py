import numpy as np
from scipy.optimize import minimize
from operator import itemgetter
from _pickle import dump, load
from time import time

def S(n,k):
	r"""Returns the n \times n circular matrix: (n-k) \delta_{i,j} + \sum_s \delta_{i+s,j}"""
	r = [n-k]+k*[1] + (n-k-1)*[0]
	return np.array([np.roll(r,i) for i in range(n)])

def mapp(n,k,phi,theta):
	"""mapp(n,k,phi,theta)(lambda)  = min_{X \in C^n} det (tau_{n,k}(X@X.H) - lambda P(phi,theta) .* X@X.H)"""
	w = np.exp(1j*2*np.pi/3)
	ker1 = np.array([w**i for i in range(n)])
	ker2 = ker1 ** 2
	ker1 /= n**.5
	ker2 /= n**.5
	Phi = np.cos(theta/2) * ker1 + np.sin(theta/2)*np.exp(1j*phi) * ker2 # Phi from Bloch coordinates of Phi @ Phi.H
	def wew1(l: 'optimisation parameter lamdba'):
		def wew2(x: 'n-dim complex vector'):
            # x - vector in the Hilbert space
			y = np.linalg.det(np.diag(S(n,k) @ x**2) - x.reshape(n,1) @ x.reshape(n,1).T - l * (Phi*x).reshape(n,1) @ (Phi.conj()*x).reshape(n,1).T).real
			return y
		return wew2
	return wew1

def force_success(fun: 'function of 1d numpy.array', n) -> 'minimisation result':
	"""force_success(fun) - returns first succesful result of minimisation of fun statring from a random starting point"""
	while True:
		x0 = np.random.randn(n)
		x0 /= np.sum(x0**2)**.5
		res = minimize(fun,x0,constraints={'type':'eq','fun': lambda x: np.sum(x**2) - 1})
		if res.success:
			return res

def robust_min(fun: 'function of 1d numpy.array', n, M: 'number of minimisations') -> 'minimisation result':
	"""robust_min(fun) - minimise fun over the set of M succesful minimisations starting from random points."""
	return min((force_success(fun,n) for _ in range(M)),key=itemgetter('fun'))

def lmax_new(n,k,M,phi,theta):
	"""Search via the Newton method the maximal value of lambda, for which min_{X \in B_+(C^n)} det (tau_{n,k}(X) - lambda P(phi,theta) .* X) becomes 0."""
	left_lambda = n-1
	right_lambda = n
	lv = robust_min(mapp(n,k,phi,theta)(right_lambda),n,M).fun
	for _ in range(3):
		lv,pv = robust_min(mapp(n,k,phi,theta)(left_lambda),n,M).fun, lv
		left_lambda, right_lambda = left_lambda - lv*(right_lambda - left_lambda)/(pv-lv),left_lambda
	return left_lambda

def lmax(n,k,M,phi,theta):
	r"""lmax(phi,theta) - calculates the maximal amount of map X: P(phi,theta).*X which can be subtracted from a map \tau_{n,k} without destroing its positivity."""
	print("phi = %f, theta = %f" % (phi,theta), end=' ')
	t=time()
	y = lmax_new(n,k,M,phi,theta)
	print('calculation time = %f, value = %f' % (time()-t, y))
	return y

def prepare_grid(N: 'density of the grid', n,k: 'parameters of the map, GCD(n,k)=3', M: 'number of starting points in minimisation'):
	"""create the grid in spherical coordinates, where step in both angles is pi/N and the array L of results of optimisation of lmax. The values of lmax are calculated for poles, between poles the array L is initialised with values (n-k). The tuple of 2d arrays: phi, theta, L is pickled to the file: data_n_k.pi"""
	phi = np.linspace(-np.pi,np.pi,2*N+1)[:-1]
	theta = np.linspace(0,np.pi,N+1)
	phi,theta = np.meshgrid(phi,theta[1:-1])    # without poles
	
	v0 = lmax(n,k,M,0,0)                        # values of maximal lambda
	vpi = lmax(n,k,M,0,np.pi)                   # calculated for both poles of the Bloch sphere 
	
	Lint = (n-k)*np.ones(phi.shape)             # initialise the array or miniminisation results with n-k 

	L = np.vstack((v0*np.ones((1,2*N)), Lint, vpi*np.ones((1,2*N))))        # add results on poles to array
	theta = np.vstack((0*np.ones((1,2*N)),theta,np.pi*np.ones((1,2*N))))    # add poles coordinates to array
	phi = np.vstack((phi[0,:],phi,phi[-1,:]))                               # add poles coordinates to array
	
	with open('data_%d_%d.pi' % (n,k),'wb') as f:
		dump((phi,theta,L),f)

def calculate_point(n,k,M,i,j,v=None):
	"""Calculates the value of lmax for the (i,j)th - point of the grid from the file data_n_k.pi.
The result is writted to a respective position in the L array in the file.
optional argument v: value to be written without calculation"""
	print('i=%d, j=%d' % (i,j),end=' ')
	with open('data_%d_%d.pi' % (n,k),'rb') as f:
		phi,theta,L = load(f)
	L[i,j] = lmax(n,k,M,phi[i,j],theta[i,j]) if v == None else v
	with open('data_%d_%d.pi' % (n,k),'wb') as f:
		dump((phi,theta,L),f)

if __name__ == '__main__':
	
	import warnings
	warnings.filterwarnings('ignore')
	
	N = 24
	
	n = 6; k = 3; M = 20
	prepare_grid(N,n,k,M)
	for i in range(1,N):
		for j in range(2*N):
			calculate_point(n,k,M,i,j)
	
	n = 9; k = 3; M = 100
	prepare_grid(N,n,k,M)
	for i in range(1,N):
		for j in range(2*N):
			calculate_point(n,k,M,i,j)
	
	n = 9; k = 6; M = 50
	prepare_grid(N,n,k,M)
	for i in range(1,N):
		for j in range(2*N):
			calculate_point(n,k,M,i,j)

