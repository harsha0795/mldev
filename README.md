# mldev v1.0
Python Implementation of Basic Machine Learning Algorithms.

# Type of Algorithms
The following algorithms have been implemented.
1. Linear Regression - Here Both Statistical Method as well as Gradient (Calculus) Method is implemented.
2. Logistic Regression - Applying Sigmoid Function to the Current Linear Hypothesis and performing Gradient Descent.

# Installation
1. The following dependencies are required inorder to install this package:
Numpy
SciPy 
scikit-learn (Only for the dataset)
Matplotlib.

For Version Information, please follow scikit-learn documentation.

## Linear Regression
Here we fit a mathematical Linear model for a given dataset. We typically call this linear model as Hypothesis. For more

1. R Squared Co-efficient of Determination Theory : This method of Linear Regression involves the computation of Variance for the Residual sum of squares. Here we calculate the static slope. 

2. Gradient Descent : This method of Linear Regression involves the computation of gradient until complete convergence. However, the loss is similar of R Squared Co-efficient of Determination. This is the calculation of mean sum if squares.

### Snippet for R-Squared Coefficient Determination
```python
def predict(xs,ys,features=1):
	''' 
	This function predicts the random test value using the linear 
	regression method of R-Squared Theory
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
```

### Output
![alt text](https://raw.githubusercontent.com/harsha0795/mldev/master/img/CaptureLinReg.PNG)
### Snippet For Gradient Descent
```python
def gradientDescent(x, y, theta, alpha, m):
	xTrans = x.transpose()
	oldcost=0.0
	loopval=True
	while(True):
		hypothesis = np.dot(x, theta)
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
```
### Output
![alt text](https://raw.githubusercontent.com/harsha0795/mldev/master/img/FigureGD.png)
## Logistic Regression
Here we fit a mathematical model Similar to that of a Linear Regression except for that the hypothesis undergoes a logistic function also called as Sigmoid Function. This method of regression actually provides solution for Classification problem. The name regression because the algorithm involves the same gradient descent algorithm but with Sigmoid function Hypothesis.
### Snippet
```python
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
```
### Output
![alt text](https://raw.githubusercontent.com/harsha0795/mldev/master/img/LogReg.png)
