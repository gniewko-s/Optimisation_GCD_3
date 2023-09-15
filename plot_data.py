import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from _pickle import load
from scipy.interpolate import RectBivariateSpline

def rect():
	plt.contourf(phi,theta,lamba,np.linspace(np.min(lamba),np.max(lamba),41),cmap='jet')
	plt.colorbar()

def add_col():
	global phi,theta,lamba
	N = phi.shape[0]
	phi = np.hstack((phi,np.pi*np.ones((N,1))))
	theta = np.hstack((theta,theta[:,:1]))
	lamba = np.hstack((lamba,lamba[:,:1]))

def interp(X,Y,Z,factor):
	f = RectBivariateSpline(Y[:,0],X[0,:],Z)
	X = X[0,:]
	X = np.linspace(X[0],X[-1],factor*X.shape[0])
	Y = Y[:,0]
	Y = np.linspace(Y[0],Y[-1],factor*Y.shape[0])
	Z = f(Y,X)
	X,Y = np.meshgrid(X,Y)
	return X,Y,Z

def sphere(elev):
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	ax._axis3don = False

	norm=colors.Normalize(vmin = np.min(lamba), vmax = np.max(lamba), clip = False)

	ax.plot_surface(np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta),
					rstride=1, cstride=1, cmap=cm.jet,
					linewidth=0, antialiased=False,
					facecolors=cm.jet(norm(lamba)))

	m = cm.ScalarMappable(cmap=cm.jet)
	m.set_array(lamba)
	ax.view_init(elev=elev, azim=0)

n = 6; k = 3

with open('data_%d_%d.pi' % (n,k),'rb') as f:
	phi,theta,lamba=load(f)

add_col()
rect()
plt.savefig('rect_%d_%d.png' % (n,k))
phi, theta, lamba = interp(phi,theta,lamba,4)
sphere(0)
plt.savefig('sphere0_%d_%d.png' % (n,k))
sphere(90)
plt.savefig('sphere90_%d_%d.png' % (n,k))
plt.show()
