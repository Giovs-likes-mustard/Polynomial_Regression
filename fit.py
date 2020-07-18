import numpy as np
from matplotlib import pyplot

f = open('data.txt','r')
data = f.read()
linelist = data.split('\n') #separating the lines
pointlist=[]

for e in linelist:
    if linelist.index(e)==0:
        try:
            #n = the number of coefficients
            #(the polynomial degree + 1)
            n = int(linelist[0])+1
            linelist.pop(0)
        except ValueError:
            print("the first line must be the"\
                  +" polynomial degree (int)")
            exit()    
    elif e.count(',')==1:
        dot = e.split(',') #separating x and y
        pointlist.append(dot)
    else:
        print('please, check again line',linelist.index(e)\
              ,':\n',e)
        exit()

#vX and vY will be lists of the x and y coordinates
vX = []
vY = []

for e in pointlist:
    vX.append(float(e[0]))
    vY.append(float(e[1]))

#now we start the linear algebra
#here we build the A and the b for Ax=b
A = np.ones(n*len(vX))
A.shape=(len(vX),n)
for i in range(len(A)):
    for j in range(len(A[1])-1):
        A[i][j+1] = vX[i]**(j+1)

b = np.ones(len(vY))
for i in range(len(vY)):
    b[i]=vY[i]

#Ax=b have no exact solution (probabily)
#so we use the least squares method to find xHat
#where xHat is our best aproximation for the coefficients
At = A.transpose()
AtA1 = np.linalg.inv(np.dot(At,A))
xHat = np.dot(np.dot(AtA1,At),b)

#defining the function to use in the plotting
#(just the function image)
def p_image(x):
    image = 0
    for e in range(len(xHat)):
        image += (x**e)*xHat[e]
    return image

fig , ax = pyplot.subplots()

#we put the linspace interval 40% larger than the
#distance between the more left and more right points
distX = max(vX) - min(vX)

x = np.linspace(min(vX) - 0.2*distX, max(vX) + 0.2*distX, distX*50)
y = p_image(x)

ax.grid(True,which='both')

ax.axhline(y=0, color='k',linewidth=0.6)
ax.axvline(x=0, color='k',linewidth=0.6)

ax.plot(x,y,'b-',linewidth=1)
ax.plot(vX,vY,'r.')

pyplot.gca().set_aspect('equal')
pyplot.axis('square')

print(xHat)

pyplot.show()
