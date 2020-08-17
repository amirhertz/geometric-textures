mkdir -p ./dataset/raw
cd dataset
wget https://www.dropbox.com/s/99pffhj90yv85ho/gt_opt_demo.tar.gz
tar -xvf gt_opt_demo.tar.gz && rm gt_opt_demo.tar.gz
mv _raw/* raw/
rm -rf _raw
