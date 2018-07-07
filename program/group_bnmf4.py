# coding:utf-8

import time
from numpy import *


reload(sys)
sys.setdefaultencoding('utf-8')

def FirstNMFofX(X,P,D,S,W,H,alpha,beta,gamma):
	#'Step 1. First NMF of X'
	#初始化W,Hh
	n,m = X.shape
	Hh = mat(random.rand(D,m))
	#print 'Hh',Hh.shape

	#update Hh
	AH = W.H * multiply(P,W*Hh) + beta * Hh + gamma * Hh + 1e-8
	BH = W.H * multiply(P,X) + gamma * H * S
	Hh = multiply(Hh,BH/AH)

	#Optimaize W and Hh
	i = 0
	while i<2000:
		AW = multiply(P,W*Hh) * Hh.H + alpha * W + 1e-8
		BW = multiply(P,X) * Hh.H
		W = multiply(W,BW/AW)

		AH = W.H * multiply(P,W*Hh) + beta * Hh + gamma * Hh + 1e-8
		BH = W.H * multiply(P,X) + gamma * H * S
		Hh = multiply(Hh,BH/AH)

		i += 1

	#print W
	#print Hh	
	return W,Hh

def NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma):
	#'Step 2. NMF of Y'
	#初始化Wm,H
	Wm = R * W
	#print 'Wm',Wm.shape
	
	#update H
	AH = Wm.H * multiply(Q,Wm*H) + beta * H + 1e-8
	BH = Wm.H * multiply(Q,Y)
	H = multiply(H,BH/AH)

	#Optimatize Wm and H
	i = 0
	while i<2000:
		AW = multiply(Q,Wm*H) * H.H + alpha * Wm + gamma * Wm + 1e-8
		BW = multiply(Q,Y) * H.H + gamma * R * W
		Wm = multiply(Wm,BW/AW)

		AH = Wm.H * multiply(Q,Wm*H) + beta * H + 1e-8
		BH = Wm.H * multiply(Q,Y)
		H = multiply(H,BH/AH)

		i += 1

	#print Wm
	#print H
	return Wm,H

def NMFofX(X,P,D,S,W,H,alpha,beta,gamma):
	#'Step 3. Subsequent NMF of X'
	#初始化W,Hh
	Hh = H * S
	
	#update W
	AW = multiply(P,W*Hh) * Hh.H + alpha * W + 1e-8
	BW = multiply(P,X) * Hh.H
	W = multiply(W,BW/AW)

	#Optimaize W and Hh
	i = 0
	while i<2000:
		AH = W.H * multiply(P,W*Hh) + beta * Hh + gamma * Hh + 1e-8
		BH = W.H * multiply(P,X) + gamma * H * S
		Hh = multiply(Hh,BH/AH)

		AW = multiply(P,W*Hh) * Hh.H + alpha * W + 1e-8
		BW = multiply(P,X) * Hh.H
		W = multiply(W,BW/AW)

		i += 1

	#print W
	#print Hh	
	return W,Hh

def CCNMF_Z1(G,E,x,y,alpha,beta,gamma) :
	#'CCNMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,:x]
	Y = G[y:,:]
	#将E矩阵分解为P,Q
	P = E[:,:x]
	Q = E[y:,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.1 * min(x,y*9))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R1,R))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S,S1))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
	#print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
		W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)

		i += 1
		#print i
	#print "Step 4. Over..."

	#Z = WH
	Z = W * H
	# filename = r"data\Z1.txt"
	# savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def CCNMF_Z2(G,E,x,y,alpha,beta,gamma) :
	#'CCNMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,x:]
	Y = G[y:,:]
	#将E矩阵分解为P,Q
	P = E[:,x:]
	Q = E[y:,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.9 * min(x,y))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R1,R))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S1,S))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
	#print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
		W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)

		i += 1
		#print i
	#print "Step 4. Over..."

	#Z = WH
	Z = W * H
	filename = r"data\Z2.txt"
	savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def CCNMF_Z3(G,E,x,y,alpha,beta,gamma) :
	#'CCNMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,:x]
	Y = G[:y,:]
	#将E矩阵分解为P,Q
	P = E[:,:x]
	Q = E[:y,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.1 * min(x,y))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R,R1))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S,S1))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
	#print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
		W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)

		i += 1
		#print i
	#print "Step 4. Over..."

	#Z = WH
	Z = W * H
	# filename = r"data\Z3.txt"
	# savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def CCNMF_Z4(G,E,x,y,alpha,beta,gamma) :
	#'CCNMF解决稀疏矩阵问题'
	#将G矩阵分解为X,Y
	X = G[:,x:]
	Y = G[:y,:]
	#将E矩阵分解为P,Q
	P = E[:,x:]
	Q = E[:y,:]
	nx,mx = X.shape
	ny,my = Y.shape
	#D = int(0.1 * min(x*9,y))
	D = 5
	W = mat(random.rand(nx,D))
	H = mat(random.rand(D,my))

	R = mat(ones((ny,ny)))
	R = diag(diag(R))
	R1 = mat(zeros((ny,nx-ny)))
	R = hstack((R,R1))

	S = mat(ones((mx,mx)))
	S = diag(diag(S))
	S1 = mat(zeros((my-mx,mx)))
	S = vstack((S1,S))

	#Step 1. First NMF of X
	W,Hh = FirstNMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 1. Over..."

	#Step 2. NMF of Y
	Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
	#print "Step 2. Over..."

	#Step 3. Subsequent NMF of X
	W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)
	#print "Step 3. Over..."

	#Step 4. Repeat Steps 2 and 3.
	i = 0
	while i<10:
		Wm,H = NMFofY(Y,Q,D,R,W,H,alpha,beta,gamma)
		W,Hh = NMFofX(X,P,D,S,W,H,alpha,beta,gamma)

		i += 1
		#print i
	#print "Step 4. Over..."

	#Z = WH
	Z = W * H
	# filename = r"data\Z4.txt"
	# savetxt(filename, around(Z, decimals=2), fmt="%.2f")
	return Z

def bnmf(G):
	E = power(G,2)

	#xl,yh，矩阵规模
	n,m = G.shape
	#print n,m

	#alpha,beta 参数
	alpha = 0.01
	beta = 0.01
	gamma = 10

	x = int(m * 0.9)
	y = int(n * 0.1)
	Z1 = CCNMF_Z1(G,E,x,y,alpha,beta,gamma)
	print "Z1 over..."
	print time.strftime('%H:%M:%S', time.localtime(time.time()))

	x = int(m * 0.1)
	y = int(n * 0.1)
	Z2 = CCNMF_Z2(G,E,x,y,alpha,beta,gamma)
	print "Z2 over..."
	print time.strftime('%H:%M:%S', time.localtime(time.time()))
	
	x = int(m * 0.9)
	y = int(n * 0.9)
	Z3 = CCNMF_Z3(G,E,x,y,alpha,beta,gamma)
	print "Z3 over..."
	print time.strftime('%H:%M:%S', time.localtime(time.time()))
	
	x = int(m * 0.1)
	y = int(n * 0.9)
	Z4 = CCNMF_Z4(G,E,x,y,alpha,beta,gamma)
	print "Z4 over..."
	print time.strftime('%H:%M:%S', time.localtime(time.time()))

	result = (Z1+Z2+Z3+Z4)/4

	return result


def main():
    for i in range(40, 49):
        print time.strftime('%H:%M:%S', time.localtime(time.time()))
        filename = "../b-nmf/group_40/%d.txt" % i
        G = mat(loadtxt(filename))

        n,m = G.shape
        v_max = G.max()
        E = ceil(G*1.0/v_max)
        sp = E.sum()/(n*m)
        sp = (1 - sp)*100
        print "%d sparse: %.2f%%" % (i, sp)

        g_mat = bnmf(G*1.0)

        filename = "../b-nmf/group_40/result/%d.txt" % i
        savetxt(filename, g_mat, fmt="%.2f")

        G = g_mat
        n,m = G.shape
        v_max = G.max()
        E = ceil(G*1.0/v_max)
        sp = E.sum()/(n*m)
        sp = (1 - sp)*100
        print "%d sparse: %.2f%%" % (i, sp)
        print i, "over..."

if __name__ == '__main__':
    main()



# 查询分组数量为5时，有383组，为10时，有192组，为20时，有96组，为30时，有64组，为40时有48组