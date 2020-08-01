import numpy as np
import math
import random
import cv2

#utility functions
def get_image(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image.astype(np.uint8)

def test(win_name,image,coords):
    cv2.destroyAllWindows()
    image=np.array(image,dtype=np.uint8)
    coords=np.array(coords,dtype=np.int32).reshape([1,-1,2])
    image=cv2.polylines(cv2.UMat(image),[coords],True,(255,0,0),2)
    #image=cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.imshow(win_name,image)
    cv2.waitKey(0)

def get_label(path):
    with open(path,"r") as file:
        label=list(file.read().split(' '))
        label=list(map(np.float32,label))
    #return label
    return np.array(label,dtype=np.float32)



#functions user can interact with
def compute_transform_mat(coords=None,params=None):

    if(coords is not None):

        if(coords.shape==(2,6)):
            x1,y1,x2,y2,x3,y3=list(coords[0])
            x1_,y1_,x2_,y2_,x3_,y3_=list(coords[1])
            a=np.array([[x1,x2,x3],[y1,y2,y3],[1,1,1]])
            b=np.array([[x1_,x2_,x3_],[y1_,y2_,y3_],[1,1,1]])
            A=np.linalg.solve(a.T,b.T).T
            return A
        elif(coords.shape==(2,8)):
            x1,y1,x2,y2,x3,y3,x4,y4=list(coords[0])
            x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_=list(coords[1])
            a=np.concatenate([ np.array([x1,y1,1,0,0,0,-x1*x1_,-y1*x1_]) , np.array([0,0,0,x1,y1,1,-x1*y1_,-y1*y1_]) ,
                               np.array([x2,y2,1,0,0,0,-x2*x2_,-y2*x2_]) , np.array([0,0,0,x2,y2,1,-x2*y2_,-y2*y2_]) ,
                               np.array([x3,y3,1,0,0,0,-x3*x3_,-y3*x3_]) , np.array([0,0,0,x3,y3,1,-x3*y3_,-y3*y3_]) ,
                               np.array([x4,y4,1,0,0,0,-x4*x4_,-y4*x4_]) , np.array([0,0,0,x4,y4,1,-x4*y4_,-y4*y4_]) , ]).reshape(8,8)
            b=np.array([x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_])
            A=np.linalg.solve(a,b)
            A=np.append(A,1).reshape(3,3)
            return A
        else:
            return -1

    elif(params is not None):

        if(params.shape==(6,)):
            #theta,phi,scale_x,scale_y,translation_x,translation_y
            rotation=params[0:2]
            scaling=params[2:4]
            translation=params[4:6]
            
            theta=rotation[0]
            phi=rotation[1]
            sx=scaling[0]
            sy=scaling[1]
            tx=translation[0]
            ty=translation[1]

            R1=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
            R2=np.array([[math.cos(-phi),-math.sin(-phi)],[math.sin(-phi),math.cos(-phi)]])
            S=np.array([[sx,0],[0,sy]])
            R3=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])

            temp=(((R1@R2)@S)@R3).ravel()
            A=np.array([[temp[0],temp[1],tx],[temp[2],temp[3],ty],[0,0,1]])
            return A

        elif(params.shape==(8,)):
            #isotropic scaling factor,theta,translation_x,translation_y,shear_factor,scaling_factor,elation_x,elation_y
            similarity=params[0:4]
            shear=params[4:5]
            scaling=params[5:6]
            elation=params[6:8]

            s,theta,tx,ty=similarity[0],similarity[1],similarity[2],similarity[3]
            sh=shear[0]
            sc=scaling[0]
            e1,e2=elation[0],elation[1]

            S=np.array([[s*math.cos(theta),-math.sin(theta),tx],[s*math.sin(theta),s*math.cos(theta),ty],[0,0,1]])
            Sh=np.array([[1,sh,0],[0,1,0],[0,0,1]])
            Sc=np.array([[sc,0,0],[0,1/sc,0],[0,0,1]])
            El=np.array([[1,0,0],[0,1,0],[e1,e2,1]])
            A=((S@Sh)@Sc)@El
            return A
        else:
            return -1
        
    else:
        return -1

def apply_transform(mat,label,transform_mat,cut=True):

    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed
    n1,n2,d=mat.shape

    A=transform_mat
    X=np.concatenate([ np.tile(np.arange(n1),n2)[None,:] , np.repeat(np.arange(n2),n1)[None,:] ,  np.ones([n1*n2])[None,:] ],axis=0).astype(np.int32)
    X_=(A @ X)
    X_=np.round( np.concatenate([ (X_[0]/X_[2])[None,:] , (X_[1]/X_[2])[None,:] ],axis=0) ).astype(np.int32)

    padding_x1,padding_x2,padding_y1,padding_y2 = max(0,-np.min(X_[0])),max(0,np.max(X_[0])-n2+1),max(0,-np.min(X_[1])),max(0,np.max(X_[1])-n1+1)
    n1_,n2_=n1+padding_y1+padding_y2,n2+padding_x1+padding_x2

    new_mat=np.zeros([d*n1_*n2_],dtype=mat.dtype)
    vals=mat.reshape(n1*n2,d).T.ravel()
    X_=X_+np.array([padding_x1,padding_y1])[:,None]
    X_=X_[1]*n2_+X_[0]
    X_=np.concatenate([X_+i*n1_*n2_ for i in range(d)]).astype(np.int32)

    np.put(new_mat,X_,vals,mode='raise')
    new_mat=new_mat.reshape(d,n1_*n2_).T.reshape(n1_,n2_,d)
    if(cut) : new_mat=new_mat[padding_y1:+n1+padding_y1,padding_x1:n2+padding_x1,:]


    label=label.reshape(4,2).T
    label=np.concatenate([label.ravel(),np.ones([4,])]).reshape(3,4).astype(label.dtype)
    new_label=(A @ label)
    new_label=np.round( np.concatenate([ new_label[0]/new_label[2][None,:] , new_label[1]/new_label[2][None,:] ],axis=0) ).astype(label.dtype)
    if(not cut) : new_label=new_label+np.array([padding_x1,padding_y1])[:,None]
    new_label=new_label.T.ravel().astype(label.dtype)

    return new_mat,new_label



def tilt(mat,label,params_maxdev,cut=True):
    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed

    if(params_maxdev.shape==(8,)):
        #isotropic scaling factor,theta,translation_x,translation_y,shear_factor,scaling_factor,elation_x,elation_y
        params=np.array([1.0,0,0,0,0,1,0.000,0.000])
        params=params+np.array([random.uniform(-params_maxdev[i],params_maxdev[i]) for i in range(len(params_maxdev))])

        transform_mat=compute_transform_mat(params=params)
        new_mat,new_label=apply_transform(mat,label,transform_mat,cut)
        return new_mat,new_label
    elif(params_maxdev.shape==(6,)):
        #theta,phi,scale_x,scale_y,translation_x,translation_y
        params=np.array([0,0,1,1,0,0])
        params=params+np.array([random.uniform(-params_maxdev[i],params_maxdev[i]) for i in range(len(params_maxdev))])

        transform_mat=compute_transform_mat(params=params)
        new_mat,new_label=apply_transform(mat,label,transform_mat,cut)
        return new_mat,new_label


def rotation(mat,label,angle,axis=(0,0),cut=True):

    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed
    n1,n2,d=mat.shape
    theta=math.radians(angle)

    coords=[[],[]]
    for _ in range(3):
        x=random.randrange(0,n2)
        y=random.randrange(0,n1)
        y_=round(axis[1]+math.sin(theta)*(x-axis[0])+math.cos(theta)*(y-axis[1]))
        x_=round(axis[0]+math.cos(theta)*(x-axis[0])-math.sin(theta)*(y-axis[1]))
        coords[0].append(x)
        coords[0].append(y)
        coords[1].append(x_)
        coords[1].append(y_)
    transform_mat=compute_transform_mat(coords=np.array(coords))
    new_mat,new_label=apply_transform(mat,label,transform_mat,cut)
    return new_mat,new_label

def translation(mat,label,x_ratio=0,y_ratio=0,cut=True):

    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed
    n1,n2,d=mat.shape

    coords=[[],[]]
    for _ in range(3):
        x=random.randrange(0,n2)
        y=random.randrange(0,n1)
        y_=y+y_ratio*n1
        x_=x+x_ratio*n2
        coords[0].append(x)
        coords[0].append(y)
        coords[1].append(x_)
        coords[1].append(y_)
    transform_mat=compute_transform_mat(coords=np.array(coords))
    new_mat,new_label=apply_transform(mat,label,transform_mat,cut)
    return new_mat,new_label

def scaling(mat,label,x_ratio=0,y_ratio=0,cut=True):

    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed
    n1,n2,d=mat.shape

    coords=[[],[]]
    for _ in range(3):
        x=random.randrange(0,n2)
        y=random.randrange(0,n1)
        y_=y*y_ratio
        x_=x*x_ratio
        coords[0].append(x)
        coords[0].append(y)
        coords[1].append(x_)
        coords[1].append(y_)
    transform_mat=compute_transform_mat(coords=np.array(coords))
    new_mat,new_label=apply_transform(mat,label,transform_mat,cut)
    return new_mat,new_label

def shearing(mat,label,x_ratio=0,y_ratio=0,cut=True):

    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed
    n1,n2,d=mat.shape

    coords=[[],[]]
    for _ in range(3):
        x=random.randrange(0,n2)
        y=random.randrange(0,n1)
        y_=y+x*y_ratio
        x_=x+y*x_ratio
        coords[0].append(x)
        coords[0].append(y)
        coords[1].append(x_)
        coords[1].append(y_)
    transform_mat=compute_transform_mat(coords=np.array(coords))
    new_mat,new_label=apply_transform(mat,label,transform_mat,cut)
    return new_mat,new_label

def reflection(mat,label,axis=0,type=0,cut=True):

    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed
    n1,n2,d=mat.shape

    coords=[[],[]]
    for _ in range(3):
        x=random.randrange(0,n2)
        y=random.randrange(0,n1)
        if(type==0):     #horizontal
            y_=y
            x_=-x+2*axis
        elif(type==1):   #vertical
            x_=-x
            y_=-y+2*axis
        else:
            return -1 
        coords[0].append(x)
        coords[0].append(y)
        coords[1].append(x_)
        coords[1].append(y_)
    transform_mat=compute_transform_mat(coords=np.array(coords))
    new_mat,new_label=apply_transform(mat,label,transform_mat,cut)
    return new_mat,new_label

def zoom(mat,label,zoom_ratio=0,cut=True):
    if(mat.shape[2]>3) : return -1               #only depth last matrices allowed
    n1,n2,d=mat.shape
    n1_,n2_=round((1-zoom_ratio)*n1),round((1-zoom_ratio)*n2)
    start_y,start_x=round((zoom_ratio/2)*n1),round((zoom_ratio/2)*n2)
    if(zoom_ratio>0.0):
        new_mat=mat[start_y:start_y+n1_][start_x:start_x+n2_][:]
    else:
        new_mat=np.zeros(n1_,n2_,d)
        new_mat[start_y:start_y+n1,start_x:start_x+n2,:]=mat
    #new_mat=cv2.resize(new_mat,(n1,n2))
         

    new_label=label
    new_label=(label.reshape(4,2)-np.array([start_x,start_y])).ravel()
    return new_mat,new_label

def random_crop():
    return NotImplementedError



def main():

    image_read_path=""
    label_read_path=""

    image_name=image_read_path+""
    label_name=label_read_path+""

    image=get_image(image_name)
    label=get_label(label_name)

    #testing the original image and label
    test("orig",image,label.copy())





    #commented code to demonstrate the syntax to be used


    #for simple augmentations
    
    #new_image,new_label=rotation(image,label,30,(416,416),cut=False)
    #new_image,new_label=translation(image,label,0.2,0.3,cut=False)
    #new_image,new_label=scaling(image,label,1.0,2.0,cut=False)
    #new_image,new_label=shearing(image,label,0.3,0.7,cut=False)
    #new_image,new_label=reflection(image,label,208,type=0,cut=False)
    #new_image,new_label=zoom(image,label,0.15)

    #params_maxdev=np.array([0.15,0.15,0.1,0.1,60,100])                #sample maxdeviation for affine
    #params_maxdev=np.array([0.1,0.15,100,75,0.15,0.20,0.001,0.001])   #sample maxdeviation for projective
    #new_image,new_label=tilt(image,label,params_maxdev,False)

    #test("new",new_image,new_label)





    #For advanced or custom techniques (optional)

    #coords=np.array([[0,0,416,0,416,416],[0,0,416,138,374,416]])              #sample coords for affine
    #coords=np.array([[0,0,0,416,416,0,416,416],[0,0,0,416,616,100,616,316]])  #sample coords for projective
    #transform_mat=compute_transform_mat(coords)
    #new_image,new_label=apply_transform(image,label,transform_mat,cut=True)


    #params=np.array([1.0,math.radians(0),0,0,0,1,0.000,0.000])                #sample parameters for projective , similarly can be provided for affine as well
    #transform_mat=compute_transform_mat(params=params)
    #new_image,new_label=apply_transform(image,label,transform_mat,cut=False)

    #test("new",new_image,new_label)


if(__name__=="__main__") : main()