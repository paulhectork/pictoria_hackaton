# generate a random dataset by creating 5 directories each containing 100 random files in `./data/dataset_dummy/`

src_dir=../data/bnf-Mission_Centenaire_Lot-vert-20241108105105/original;
dst_dir=./data/dataset_dummy/;
zipfile=./data/dataset_dummy.zip;

# remove previous dataset directory
if [ ! -d ./data ]; then mkdir ./data; fi;
if [ ! -d "$dst_dir" ]; 
  then mkdir "$dst_dir"; 
  else rm -r "$dst_dir"; mkdir "$dst_dir";
fi;

# move files to directory
for subdir in a b c d e ; do
  mkdir "${dst_dir}${subdir}";
  for infile in $(ls $src_dir | shuf -n 100); do 
    cp "${src_dir}/${infile}" "${dst_dir}${subdir}" ; 
  done ; 
done;

echo "dataset dir created ! now, extract to an archive";
