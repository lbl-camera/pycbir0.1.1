import numpy as np
from scipy.ndimage.filters import convolve

def HOG(Im,name,label, cells, blocks):
    nwin_x=blocks;#set here the number of HOG windows per bound box
    nwin_y=blocks;
    B=cells;#set here the number of histogram bins
    L,C = Im.shape; #L num of lines ; C num of columns
    H=np.zeros((nwin_x*nwin_y*B)); # column vector with zeros
    m=(L/2)**0.5;
    
    step_x=np.floor(C/(nwin_x+1));
    step_y=np.floor(L/(nwin_y+1));
    cont=0;
    hx = np.matrix('-1 0 1;-2 0 2;-1 0 1');
    hy = -hx.T;
    
    grad_xr = convolve(Im,hx);
    grad_yu = convolve(Im,hy);
    angles=np.arctan2(grad_yu,grad_xr);
    magnit=((grad_yu**2)+(grad_xr**2))**5;
    for n in range(nwin_y):
        for m in range(nwin_x):
            cont=cont+1;
            angles2=angles[np.int(n*step_y+1):np.int((n+2)*step_y),np.int(m*step_x+1):np.int((m+2)*step_x)] 
            magnit2=magnit[np.int(n*step_y+1):np.int((n+2)*step_y),np.int(m*step_x+1):np.int((m+2)*step_x)]
            v_angles=angles2.reshape(1,angles2.shape[0]*angles2.shape[1])[0];    
            v_magnit=magnit2.reshape(1,magnit2.shape[0]*magnit2.shape[1])[0];
            K=np.max(v_angles.shape);
            #assembling the histogram with 9 bins (range of 20 degrees per bin)
            bin_=-1;
            H2=np.zeros((B));
            
            for ang_lim in np.arange(-np.pi+2*np.pi/B,np.pi,2*np.pi/B):
                bin_=bin_+1;
                for k in range(K):
                    if v_angles[k]<ang_lim:
                        try:
                            v_angles[k]=100;
                            H2[bin_]=H2[bin_]+v_magnit[k];
                            
                        except Exception as e:
                            print("ERRO") 
                            #print(e)
                            print("erro: ")
                            print(H2.shape)
                            print(bin_)
                            print(v_magnit.shape)
                            print(k)   
                            a
                                           
            H2=H2/(np.linalg.norm(H2)+0.01);        
            H[(cont-1)*B:cont*B]=H2;
            
    return H,name,label

#imagem = np.random.random((50,50))
#print(hog_rom(imagem))