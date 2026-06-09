{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww22620\viewh13220\viewkind0
\deftab560
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0

\f0\fs28 \cf0 #!/bin/bash\
ANTSPATH="/usr/local/ants-2.5.4/bin/"\
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=100 # controls multi-threading\
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS\
\
fixed= $1\
moving= $2\
\pard\pardeftab560\slleading20\partightenfactor0
\cf0 Mask = $3\
\
If [[ ! -s $mask ]] ; then echo \'93Error: Mask file $mask not found or empty\'94; exit 1; fi\
\
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 base_path=$(echo "$moving" | sed 's/^\\(.*_ct\\).*$/\\1/')\
reg_o="$\{base_path\}_last-"\
#echo "reg_o is $reg_o"\
app_warped="$\{reg_o\}WarpedToTemplate.nii.gz"\
app_warp="$\{reg_o\}1Warp.nii.gz"\
app_affine="$\{reg_o\}0GenericAffine.mat"\
\
\pard\pardeftab560\slleading20\partightenfactor0
\cf0 app_mask = \'94$\{base_path\}_mask_registerd.nii.gz\'94\
\
if [[ ! -s $fixed ]] ; then echo \'93Error: Fixed image $fixed not found or empty\'94; exit 1; fi \
If [[ ! -s $moving ]] ; then echo \'93Error: Moving image $moving not found or empty\'94; exit 1; fi\
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 \
reg=$\{ANTSPATH\}antsRegistration # path to antsRegistration\
\
echo affine $m $f outname is $nm\
\
 antsRegistration -d 3 \\\
 --float 1 \\\
 --verbose 1 \\\
 -u 1 \\\
 -w [ 0.01,0.99 ] \\\
 -z 1 \\\
 -r [ $fixed,$moving,1 ] \\\
 -t Rigid[ 0.1 ] \\\
 -m MI[ $fixed,$moving,1,32,Regular,0.25 ] \\\
 -c [ 1000x500x250x0,1e-6,10 ] \\\
 -f 6x4x2x1 \\\
 -s 4x2x1x0 \\\
 -t Affine[ 0.1 ] \\\
 -m MI[ $fixed,$moving,1,32,Regular,0.25 ] \\\
 -c [ 1000x500x250x0,1e-6,10 ] \\\
 -f 6x4x2x1 \\\
 -s 4x2x1x0 \\\
 -t SyN[ 0.1,3,0 ]  \\\
 -m CC[ $fixed,$moving,1,4 ] \\\
 -c [ 100x70x50x10,1e-9,10 ] \\\
 -f 8x4x2x1 \\\
 -s 3x2x1x0 \\\
 -o $reg_o\
\
   antsApplyTransforms -d 3 \\\
   --float 1 \\\
   --verbose 1 \\\
   -i $moving \\\
   -o $app_warped \\\
   -r $fixed \\\
   --interpolation BSpline \\\
   -t $app_warp \\\
   -t $app_affine\
\
    antsApplyTransforms -d 3 \\\
    --float 1 \\\
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0    --verbose 1 \\\
   -i $mask \\\
   -o $app_mask \\\
   -r $fixed \\\
   --interpolation BSpline \\\
   -t $app_warp \\\
   -t $app_affine\
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0
\cf0 \
\
\pard\pardeftab560\sl480\slmult1\pardirnatural\partightenfactor0
\cf0 exit 0}