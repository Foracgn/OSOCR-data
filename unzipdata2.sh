echo $1
#CAC1=$1
#SRC=$2
# data downloaded
SRC=/home/sunhongyi/桌面/sunhongyi/ocr-data/

# cache dir 1 (100GiB~)
CAC1=/home/sunhongyi/桌面/sunhongyi/CAC1/

# cache dir 2 (GiB~)
CAC2=/home/sunhongyi/桌面/sunhongyi/CAC2/

# generated dataset dir (GiB~)
EXP=/home/sunhongyi/桌面/sunhongyi/EXP/

CODE_ROOT=${PWD}/code
#
#rm ${EXP}/* -r
#rm ${CAC1}/* -r
#rm ${CAC2}/* -r


#
cd ${SRC}/art
rm -r ${CAC1}/art
mkdir -p ${CAC1}/art
tar -xvf train_task2_images.tar.gz  --directory ${CAC1}/art
cp *json* ${CAC1}/art/;
cd ../


cd ${SRC}/mlt
rm -r ${CAC1}/mlt
mkdir -p ${CAC1}/mlt/real
unzip Chinese.zip -d ${CAC1}/mlt/synth
unzip ImagesPart1.zip -d ${CAC1}/mlt/real
unzip ImagesPart2.zip -d ${CAC1}/mlt/real
cd ${CAC1}/mlt/real;
mv */* .
rmdir *
echo "Those mates have pngs, gifs, and etc etc in the images, converting em."
for x in *; do case $x in *.[Jj][Pp][Gg]) :;; *) convert -- "$x" "${x%.*}.jpg";rm $x;; esac; done
cd ${SRC}/mlt
unzip train_gt_t13.zip -d ${CAC1}/mlt/real


cd ${SRC}/lsvt
    mkdir -p ${CAC1}/lsvt
    tar -xvf train_full_images_0.tar.gz  --directory ${CAC1}/lsvt/;
    tar -xvf train_full_images_1.tar.gz --directory ${CAC1}/lsvt/;
    cp *json ${CAC1}/lsvt/;
    cd ${CAC1}/lsvt/;
    mkdir imgs;
    mv train_full_images_0/* imgs;
    mv train_full_images_1/* imgs;
    rmdir *
cd ${SRC}


#
cd ${SRC}/hwdb
mkdir -p ${CAC1}/hwdb/train
mkdir -p ${CAC1}/hwdb/test
mkdir -p ${CAC1}/hwdb/comp

for i in $(ls | grep Train)
do
    unzip $i -d ${CAC1}/hwdb/train;
done;

for i in $(ls | grep Test)
do
    unzip $i -d ${CAC1}/hwdb/test;
done;

unzip competition-gnt.zip -d ${CAC1}/hwdb/comp

cp -r ${CODE_ROOT}/../fonts ${CAC1}/

cd ${CODE_ROOT}

export PYTHONPATH=${CODE_ROOT}

python osocr_tasks/tasksg1/ch_jap_osocr/make_dataset_2.py ${CAC1} ${CAC2} ${EXP}
#rm ${CAC2}/* -r
python osocr_tasks/tasksg1/hwdb_fslchr/make_dataset.py ${CAC1} ${CAC2} ${EXP}
python osocr_tasks/tasksg1/ctw_fslchr/make_dataset.py ${CAC1} ${CAC2} ${EXP}
python osocr_tasks/tasksg1/char_rej.py ${EXP}
