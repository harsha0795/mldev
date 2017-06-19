import statistics as s
import numpy as np
import matplotlib.pyplot as plt
import random

def slope(xs,ys):
	'''
	This function calculates the slope using the mean square error formula.
	It is very much necessary for prediction using the linear regression.

	Parameters
	-----------
	xs,ys : Dataset parameters of the form nd array

	Return Type 
	----------- 
	m : Slope of the dataset of the form float
	b : the value of y = mx + b which is of the form float.
	'''
	xys=xs*ys
	x_bar=s.mean(xs) 
	# mean squared operation of xs
	mseofx=(xs-x_bar)**2 
	y_bar=s.mean(ys)
	# mean squared operation of ys
	mseofy=(ys-y_bar)**2
	m=(x_bar*y_bar-s.mean(xs*ys))/((x_bar**2)-(s.mean(xs**2)))
	b=y_bar-m*x_bar
	return m,b

def rsquared_coeff_determination(xs,ys):
	xys=xs*ys
	x_bar=s.mean(xs) 
	# mean squared operation of xs
	msofx=(xs-x_bar)**2 
	# Squared Sum(xs)
	sqsumofx=np.sum(msofx)
	y_bar=s.mean(ys)
	# mean squared operation of ys
	msofy=(ys-y_bar)**2
	# Squared Sum(ys) or total sum of squares of y axis
	sqsumofy=np.sum(msofy)
	m,b=slope(xs,ys)
	linereg=[(m*x+b) for x in xs]
	#regression mean square Sum
	sumofregms=np.sum((linereg-y_bar)**2)
	#Residual squared sum
	ssres=np.sum((ys-linereg)**2)
	#R Squared Theory R2 = 1-SSres/SStot
	finalR2=1-(ssres/sqsumofy)
	print (finalR2)
	return finalR2

def predict(xs,ys,features=1):
	''' 
	This function predicts the random test value using the linear 
	regression

	Parameters
	----------
	array_x,array_y,x_label,y_label

	Return Type 
	----------- 
	Predicted value of the form integer.
	'''
	xs=xs[:,features-1]
	print("Using R-squared-theory")
	m,b=slope(xs,ys)
	linereg=[(m*x+b) for x in xs]
	#finalval=m*newx+b
	rsquared_coeff_determination(xs,ys)
	return linereg

def predict_gradient_descent(xs,ys):
	'''
	Applies Gradient Descent to analyse the given dataset. This 
	'''
	print("Using Gradient Descent for linear Regression algorithm")
	m,n=np.shape(xs)
	theta=np.ones(n)
	x=gradientDescent(xs,ys,theta,0.0005,m)
	return x

def gradientDescent(x, y, theta, alpha, m):
	xTrans = x.transpose()
	oldcost=0.0
	loopval=True
	while(True):
		hypothesis = np.dot(x, theta)
		loss = hypothesis - y
		# avg cost per example (the 2 in 2*m doesn't really matter here.
		# But to be consistent with the gradient, I include it)
		cost = np.sum(loss ** 2) / (2 * m)
		#print (cost)
		# avg gradient per example
		gradient = np.dot(xTrans, loss) / m
		# update
		theta = theta - alpha * gradient
		#plt.scatter(theta)
		if(oldcost==cost):
			break
		else:
			oldcost=cost
			continue
	return theta

def genData(numPoints, bias, variance):
	x = np.zeros(shape=(numPoints, 2))
	y = np.zeros(shape=numPoints)
	# basically a straight line
	for i in range(0, numPoints):
		# bias feature
		x[i][0] = 1
		x[i][1] = i
		# our target variable
		y[i] = (i + bias) + random.uniform(0, 1) * variance
	plt.scatter(x,y)
	plt.show()
	return x, y

def plotLinReg(xs,ys,reg,features=1,heading_str="",label_x="",label_y=""):
	'''
	Parameters
	----------
	xs,ys : Dataset of the form ndarray
	newx : New training value x to predict y.
	heading_str : Heading String of the type str
	label_x,label_y : String for the x axis label and y axis label of the type str
	reg : 

	Return Type 
	----------- 
	.
	'''
	if(features==1 and heading_str=="R-Squared Theory"):
		plt.scatter(xs,ys)
		plt.plot(xs,reg,color='r')
		plt.xlabel(label_x)
		plt.ylabel(label_y)
	elif(features>1 and heading_str=="R-Squared Theory"):
		#plt.scatter(xs[:,features-1],ys)
		xi = np.arange(xs[:, features-1].min() - 1, xs[:, features-1].max() + 1)
		plt.plot(xs[:,features-1],reg,color='r')
		plt.xlabel(label_x)
		plt.ylabel(label_y)  	
	elif(features==1 and heading_str=="Gradient descent"):
		feature=features-1
		xi = np.arange(xs[:, feature].min() - 1, xs[:, feature].max() + 1)
		print (reg)
		line = 2+(reg[0] * xi)
		plt.scatter(xs,ys)
		plt.plot(line,color='r')	
	else:
		feature=features-1
		xi = np.arange(xs[:, feature].min() - 1, xs[:, feature].max() + 1)
		line = reg[0] + (reg[1] * xi)
		plt.scatter(xs[:,feature],ys)
		plt.plot(line,color='r')
	plt.show()

