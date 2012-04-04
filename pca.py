import cv
import numpy
NUM_TEST=3
NUM_TRAIN=7
NUM_CLASS=10
NUM_PC=60

def loadimage(x):
   img=cv.LoadImageM(x,cv.CV_LOAD_IMAGE_GRAYSCALE)
   return img

def readname(fp):
   str=fp.readline()
   x=str[:-1]
   return x

def cvto2d(img):
   im=numpy.asarray(img)
   return im 

def twodto1d(im):
   return numpy.ravel(im)

def transpose1d(d):
   temp=numpy.matrix(d)
   return temp.T

def average(d):
   return numpy.mean(d,axis=0)

def classical_data(n):
   d=numpy.zeros((n,10304))
   for j in range(0,n):
      x=readname(fp)
      img=loadimage(x)
      im=cvto2d(img)
      d[j]=twodto1d(im)
   return d

def classical_av_data(d):
   avg=average(d)
   newd=d-avg
   return newd

def classical_pca(newd):
   cov=(numpy.matrix(newd.T)*numpy.matrix(newd))/NUM_TRAIN
   w,v=numpy.linalg.eig(cov)
   perm=numpy.argsort(-w)
   pc=v[perm][:NUM_PC]
   return pc

def two_d_data(fp):
   x=readname(fp)
   img=loadimage(x)
   im=cvto2d(img)
   return im

def two_d_dataset(n,fp):
   d=numpy.ndarray((n,112,92))
   for i in range(0,n):
      im=two_d_data(fp)
      d[i]=im
   return d

def two_d_pca(d,dim):
   cov=numpy.zeros((dim,dim))
   for i in range(0,NUM_TRAIN):
      cov=cov+numpy.matrix(d[i])*numpy.matrix(d[i].T)
   cov=cov/NUM_TRAIN  
   w,v=numpy.linalg.eig(cov)
   perm=numpy.argsort(-w)
   pc=v[perm][:NUM_PC]
   return pc

def simpca_train(SIM_NUM):
   fp=open("train.txt","r")
   if 92%SIM_NUM !=0:
      for i in range(SIM_NUM+1,92):
         if 92%i==0:
            SIM_NUM=i
   sim_dat=numpy.ndarray((1,112,92/SIM_NUM))
   for i in range(0,NUM_TRAIN):
      im=two_d_data(fp)
      newim=numpy.hsplit(im,SIM_NUM)
      new=numpy.asarray(newim)
      sim_dat=numpy.vstack((sim_dat,new))
   count=1
   pc=numpy.ndarray((SIM_NUM,NUM_PC,112))
   out=numpy.ndarray((SIM_NUM,NUM_TRAIN,NUM_PC,92/SIM_NUM))
   for i in range(0,SIM_NUM):
      x=count
      m=numpy.ndarray((1,112,92/SIM_NUM))
      for j in range(0,NUM_TRAIN):
         m=numpy.vstack((m,[sim_dat[x]]))
         x=count+SIM_NUM+1
      count=count+1
      #print m[1:]  
      pc[i]=two_d_pca(m[1:],112)
      for k in range(1,NUM_TRAIN+1):
         out[i][k-1]=numpy.matrix(pc[i])*numpy.matrix(m[k])
   output=numpy.ndarray((1,NUM_PC,92))
   for i in range(0,NUM_TRAIN):
       m=numpy.ndarray((NUM_PC,1))
       for j in range(0,SIM_NUM):
          m=numpy.hstack((m,out[j][i]))
       output=numpy.vstack((output,[m[:,1:]]))
   return output[1:]
   fp.close()

def flpca_train(SIM_NUM):
     out=simpca_train(SIM_NUM)
     fl_out=two_d_pca(out,NUM_PC)
     #print fl_out

def mean_vector_class(out):
   return numpy.mean(out,0)

def norm(x):
   return numpy.linalg.norm(x)
    
def two_d_train():
   fp=open("train.txt","r")
   out=numpy.ndarray((NUM_TRAIN,NUM_PC,92))
   d=two_d_dataset(NUM_TRAIN,fp)
   pc=two_d_pca(d,112)
   print pc
   for i in range(0,NUM_TRAIN):
      out[i]=pc.T*d[i]
#  classifier
   fp.close()
   return out 

def two_d_test(pc):
   #fp=open("test.txt","r")
   d=two_d_dataset(NUM_TEST)
   for i in range(0,NUM_TEST):
      out[i]=pc*d[i]
   return out

def classical_train():
   #fp=open("train.txt","r")
   d=classical_data(NUM_TRAIN)
   newd=classical_av_data(d)		 
   pc=classical_pca(newd)
   out=pc*d
#  classifier(out)
   return out
  # fp.close()

def classical_test(pc):
   #fp=open("test.txt","r")
   d=classical_data(NUM_TEST)
   newd=classical_av_data(d)
   out=pc*newd
#  classifier(out)
   return out
#  fp.close()
flpca_train(4)
#two_d_train()
