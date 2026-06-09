cd /path/to/project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
numpy
pandas
scipy
SimpleITK
matplotlib
opencv-python-headless
scikit-image
python f.py \
  --annotations /content/nifti_annotations.csv \
  --data_dir /content \
  --output_dir /content/output

import os, argparse, cv2, numpy as np, pandas as pd, SimpleITK as sitk, scipy.ndimage as nd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as patches
from skimage.transform import resize

def flip_to_RAS(image):
    RAS=(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    if image.GetDirection()==RAS: return image
    flip_axes=[image.GetDirection()[i]<0 for i in (0,4,8)]
    return sitk.Flip(image,flip_axes)

def denoise_if_needed(arr,name):
    return nd.gaussian_filter(arr,sigma=0.6) if "STANDARD" in name.upper() else arr

def plot_views(img,seed_x,seed_y,bbox_x,bbox_y,bbox_w,bbox_h,slice_number,image_name,output_dir):
    arr=sitk.GetArrayFromImage(img)
    arr=denoise_if_needed(arr,image_name)
    z_idx=arr.shape[0]-slice_number
    axial=arr[z_idx,:,:]
    sagittal=np.rot90(arr[:,:,int(seed_x)],2)
    coronal=np.rot90(arr[:,arr.shape[1]//2,:],2)
    ref_shape=axial.shape
    sag_resized=resize(sagittal,ref_shape)
    cor_resized=resize(coronal,ref_shape)
    sag_scale_y=ref_shape[0]/sagittal.shape[0]; sag_scale_x=ref_shape[1]/sagittal.shape[1]
    cor_scale_y=ref_shape[0]/coronal.shape[0]; cor_scale_x=ref_shape[1]/coronal.shape[1]
    seed_sag_y=(arr.shape[0]-z_idx)*sag_scale_y; seed_sag_x=(arr.shape[1]-seed_y)*sag_scale_x
    seed_cor_y=(arr.shape[0]-z_idx)*cor_scale_y; seed_cor_x=(arr.shape[2]-seed_x)*cor_scale_x
    coords={"seed_sag_x":seed_sag_x,"seed_sag_y":seed_sag_y,"seed_cor_x":seed_cor_x,"seed_cor_y":seed_cor_y}
    views={"Axial":(axial,(bbox_x,bbox_y,bbox_w,bbox_h),(seed_x,seed_y)),
           "Sagittal":(sag_resized,(seed_sag_x-bbox_w/2,seed_sag_y-bbox_h/2,bbox_w,bbox_h),(seed_sag_x,seed_sag_y)),
           "Coronal":(cor_resized,(seed_cor_x-bbox_w/2,seed_cor_y-bbox_h/2,bbox_w,bbox_h),(seed_cor_x,seed_cor_y))}
    for title,(slc,rect,seed) in views.items():
        plt.figure(figsize=(6,6))
        plt.imshow(slc,cmap="gray"); plt.axis("off")
        plt.gca().add_patch(patches.Rectangle(rect[:2],rect[2],rect[3],linewidth=2,edgecolor="lime",facecolor="none"))
        plt.scatter(seed[0],seed[1],color="red",s=50,marker="*")
        plt.gcf().patch.set_facecolor("black")
        plt.savefig(os.path.join(output_dir,f"{image_name}_{title}_RAS_cleaned.png"),bbox_inches="tight",dpi=200,facecolor="black")
        plt.close()
    return coords

def segment_lungs(img):
    arr=sitk.GetArrayFromImage(img).astype(np.int16)
    arr=np.clip(arr,-1000,400)
    arr=(arr-arr.min())/(arr.max()-arr.min())
    mask=arr<0.55
    mask=nd.binary_opening(mask,np.ones((3,3,3)))
    mask=nd.binary_closing(mask,np.ones((5,5,5)))
    mask=nd.binary_fill_holes(mask)
    return mask.astype(np.uint8)

def plot_lung_views(img,mask_3d,coords,seed_x,seed_y,bbox_x,bbox_y,bbox_w,bbox_h,slice_number,image_name,output_dir):
    arr=sitk.GetArrayFromImage(img)
    z_idx=arr.shape[0]-slice_number
    masked=arr*mask_3d
    axial=masked[z_idx,:,:]
    sagittal=np.rot90(masked[:,:,int(seed_x)],2)
    coronal=np.rot90(masked[:,arr.shape[1]//2,:],2)
    ref_shape=axial.shape
    sag_resized=resize(sagittal,ref_shape)
    cor_resized=resize(coronal,ref_shape)
    views={"Axial_Lungs":(axial,(bbox_x,bbox_y,bbox_w,bbox_h),(seed_x,seed_y)),
           "Sagittal_Lungs":(sag_resized,(coords["seed_sag_x"]-bbox_w/2,coords["seed_sag_y"]-bbox_h/2,bbox_w,bbox_h),(coords["seed_sag_x"],coords["seed_sag_y"])),
           "Coronal_Lungs":(cor_resized,(coords["seed_cor_x"]-bbox_w/2,coords["seed_cor_y"]-bbox_h/2,bbox_w,bbox_h),(coords["seed_cor_x"],coords["seed_cor_y"]))}
    for title,(slc,rect,seed) in views.items():
        plt.figure(figsize=(6,6))
        plt.imshow(slc,cmap="gray"); plt.axis("off")
        plt.gca().add_patch(patches.Rectangle(rect[:2],rect[2],rect[3],linewidth=2,edgecolor="lime",facecolor="none"))
        plt.scatter(seed[0],seed[1],color="red",s=50,marker="*")
        plt.gcf().patch.set_facecolor("black")
        plt.savefig(os.path.join(output_dir,f"{image_name}_{title}.png"),bbox_inches="tight",dpi=200,facecolor="black")
        plt.close()

def map_phase(view,output_dir):
    files=[f for f in os.listdir(output_dir) if f.endswith(f"_{view}_Lungs.png")]
    if not files: return
    template=os.path.join(output_dir,sorted(files)[0])
    t_bgr=cv2.imread(template)
    t_gray=cv2.cvtColor(t_bgr,cv2.COLOR_BGR2GRAY)
    _,t_mask=cv2.threshold(t_gray,30,255,cv2.THRESH_BINARY_INV)
    t_mask=cv2.morphologyEx(t_mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
    t_mask=cv2.morphologyEx(t_mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    num,lbl,st,_=cv2.connectedComponentsWithStats(t_mask)
    idx=np.argsort(st[:,4])[::-1]; clean=np.zeros_like(t_mask)
    for i in idx[:3]:
        if st[i,4]>500: clean[lbl==i]=255
    canvas=t_bgr.copy()
    def rel(x,y,m):
        ys,xs=np.where(m>0); return (x-xs.min())/(xs.max()-xs.min()), (y-ys.min())/(ys.max()-ys.min())
    def mp(rx,ry,m):
        ys,xs=np.where(m>0)
        x=int(xs.min()+rx*(xs.max()-xs.min())); y=int(ys.min()+ry*(ys.max()-ys.min()))
        if m[y,x]==0:
            dist=cv2.distanceTransform(255-m,cv2.DIST_L2,3)
            iy,ix=np.unravel_index(np.argmax(dist),dist.shape)
            x,y=ix,iy
        return x,y
    for f in files:
        img=cv2.imread(os.path.join(output_dir,f))
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        m=cv2.inRange(hsv,(0,70,50),(15,255,255))|cv2.inRange(hsv,(165,70,50),(180,255,255))
        c=cv2.findNonZero(m)
        if c is None: continue
        mean=np.mean(c,axis=0)[0]; x,y=int(mean[0]),int(mean[1])
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,lm=cv2.threshold(g,30,255,cv2.THRESH_BINARY_INV)
        lm=cv2.morphologyEx(lm,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
        lm=cv2.morphologyEx(lm,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        rx,ry=rel(x,y,lm); xn,yn=mp(rx,ry,clean)
        cv2.circle(canvas,(xn,yn),8,(0,0,255),-1)
    cv2.imwrite(os.path.join(output_dir,f"{view}_Final_Template_with_Red_Nodules.png"),canvas)

def compose_all(output_dir):
    axial=cv2.cvtColor(cv2.imread(os.path.join(output_dir,"Axial_Final_Template_with_Red_Nodules.png")),cv2.COLOR_BGR2RGB)
    sagittal=cv2.cvtColor(cv2.imread(os.path.join(output_dir,"Sagittal_Final_Template_with_Red_Nodules.png")),cv2.COLOR_BGR2RGB)
    coronal=cv2.cvtColor(cv2.imread(os.path.join(output_dir,"Coronal_Final_Template_with_Red_Nodules.png")),cv2.COLOR_BGR2RGB)
    fig,axes=plt.subplots(1,3,figsize=(18,6),facecolor='black')
    for ax,img,title in zip(axes,[axial,sagittal,coronal],["Axial View","Sagittal View","Coronal View"]):
        ax.imshow(img); ax.set_title(title,fontsize=14,color='white',pad=10); ax.axis("off")
    fig.text(0.5,-0.02,"Location Distribution Map",ha='center',va='top',fontsize=20,color='white',fontweight='bold')
    plt.tight_layout(pad=2.0)
    out=os.path.join(output_dir,"AllViews_Location_Distribution_Map.png")
    plt.savefig(out,bbox_inches='tight',dpi=300,facecolor='black'); plt.close()
    return out

def heatmap_on_composite(composite_path,output_dir):
    img=cv2.imread(composite_path); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,lung=cv2.threshold(gray,30,255,cv2.THRESH_BINARY_INV)
    lung=cv2.morphologyEx(lung,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    lung=cv2.morphologyEx(lung,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    num,lbl,st,_=cv2.connectedComponentsWithStats(lung)
    idx=np.argsort(st[:,4])[::-1]; mask=np.zeros_like(lung)
    for i in idx[:7]:
        if st[i,4]>500: mask[lbl==i]=255
    mask_red=(img[:,:,2]>150)&(img[:,:,1]<100)&(img[:,:,0]<100)
    y,x=np.where(mask_red); inside_x,inside_y=[],[]
    for (xx,yy) in zip(x,y):
        if mask[yy,xx]>0: inside_x.append(xx); inside_y.append(yy)
    heat=np.zeros_like(gray,dtype=np.float32)
    for (xx,yy) in zip(inside_x,inside_y): cv2.circle(heat,(xx,yy),25,1,-1)
    heat=cv2.GaussianBlur(heat,(0,0),sigmaX=30,sigmaY=30)
    heat_norm=cv2.normalize(heat,None,0,1,cv2.NORM_MINMAX)
    overlay=cv2.addWeighted(cv2.applyColorMap((heat_norm*255).astype(np.uint8),cv2.COLORMAP_JET),0.7,img,0.4,0)
    overlay[mask==0]=(0,0,0)
    fig=plt.figure(figsize=(13,5),facecolor='black')
    ax=fig.add_axes([0.05,0.1,0.8,0.8]); ax.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)); ax.axis("off")
    cax=fig.add_axes([0.88,0.25,0.03,0.5])
    cb=plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0,vmax=3),cmap=plt.cm.jet),cax=cax)
    cb.set_label('%',color='white',fontsize=12); cb.ax.yaxis.set_tick_params(color='white'); cb.outline.set_edgecolor('white')
    plt.setp(plt.getp(cb.ax.axes,'yticklabels'),color='white')
    fig.text(0.5,0.02,"Location Distribution Map",ha='center',va='bottom',fontsize=20,color='white',fontweight='bold')
    out=os.path.join(output_dir,"AllViews_Location_Distribution_Heatmap.png")
    plt.savefig(out,bbox_inches='tight',dpi=300,facecolor='black'); plt.close()
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annotations",required=True)
    p.add_argument("--data_dir",required=True)
    p.add_argument("--output_dir",required=True)
    args=p.parse_args()
    annotations=pd.read_csv(args.annotations)
    os.makedirs(args.output_dir,exist_ok=True)
    for image_name in annotations["bidsname"].tolist():
        try:
            row=annotations[annotations["bidsname"]==image_name].iloc[0]
            img=flip_to_RAS(sitk.ReadImage(os.path.join(args.data_dir,image_name)))
            coords=plot_views(img,row["seed_x"],row["seed_y"],row["bbox_x"],row["bbox_y"],row["bbox_w"],row["bbox_h"],row["slice_number"],image_name,args.output_dir)
            mask=segment_lungs(img)
            plot_lung_views(img,mask,coords,row["seed_x"],row["seed_y"],row["bbox_x"],row["bbox_y"],row["bbox_w"],row["bbox_h"],row["slice_number"],image_name,args.output_dir)
        except Exception as e:
            print(f"Skip {image_name}: {e}")
    for v in ["Axial","Sagittal","Coronal"]:
        map_phase(v,args.output_dir)
    composite=compose_all(args.output_dir)
    heatmap_on_composite(composite,args.output_dir)

if __name__=="__main__":
    main()
