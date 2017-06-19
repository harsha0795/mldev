import numpy as np
import matplotlib.pyplot as plt,mpld3


	
def logistic_regression(x, y,alpha=0.05,lamda=0):
	'''
	Logistic regression for datasets
	'''
	m,n=np.shape(x)
	theta=np.ones(n)
	xTrans = x.transpose()
	oldcost=0.0
	value=True
	while(value):
		hypothesis = np.dot(x, theta)
		logistic=hypothesis/(np.exp(-hypothesis)+1)
		reg = (lamda/2*m)*np.sum(np.power(theta,2))
		loss = logistic - y
		cost = np.sum(loss ** 2)
		#print(cost)
		# avg cost per example (the 2 in 2*m doesn't really matter here.
		# But to be consistent with the gradient, I include it)
		# avg gradient per example
		gradient = np.dot(xTrans, loss)/m
		# update
		if(reg):
			cost=cost+reg
			theta = (theta - (alpha) * (gradient+reg))
		else:
			theta=theta -(alpha/m) * gradient
		if(oldcost==cost):
			value=False
		else:
			oldcost=cost
	print(accuracy(theta,m,y,x))
	return theta,accuracy(theta,m,y,x)

def accuracy(theta,m,y,x):
	print(y)
	yacc=[int(i) for i in y]
	y=np.array(yacc)
	hypothesis = np.dot(x, theta) 
	hyp=[int(h) for h in hypothesis]
	hypothesis=np.array(hyp)
	return np.mean(hypothesis == y)*100

def plot_logistic(table,X,y,theta):
	"""Plot the training data in X array along with decision boundary
	"""
	x1 = np.linspace(table.min(), table.max(), 100)
	print(x1)
	#reverse self.theta as it requires coeffs from highest degree to constant term
	x2 = np.polyval(np.poly1d(theta[::-1]),x1)
	fig=plt.figure()
	plt.plot(x1, x2, color='r', label='decision boundary');
	plt.scatter(X[:, 0], X[:, 1], s=10, c=y, cmap=plt.cm.Spectral)
	plt.legend()
	plot=mpld3.fig_to_html(fig)
	return plot
