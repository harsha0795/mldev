import numpy as np

def converter(fp):
	''' This converts dataset file  into numpy array variables of elements
	
	Parameters: filepointer fp
	Return Value : X_dataset,Y_dataset in ndarrays
	'''
	xlist=[]
	ylist=[]
	arraylist=[]
	for line in fp:
		x,y=line.split(',')
		x=x.strip()
		y=y.strip()
		xlist.append(int(x))
		ylist.append(int(y))
	xs=np.array(xlist)
	ys=np.array(ylist)
	print(xs,ys)
	return xs,ys

