from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import os
import pandas as pd

labels_df = pd.read_csv('labels.csv')
labels = list(set(labels_df['breed']))
labels = sorted(labels)
cla=-1
ii=[]
jj=[]
num_of_classes=len(labels)
for i,row in labels_df.iterrows():
    print (i)
    img = load_img('train/'+row['id']+'.jpg',target_size=(64,64),grayscale=False)  # this is a PIL image
    x = img_to_array(img)
    ii.append(x)
    z=np.zeros((num_of_classes,))
    z[labels.index(row['breed'])]=1
    jj.append(z)
f=open('train_data1.npz','wb')
np.savez(f,np.array(ii),np.array(jj))
f.close()

