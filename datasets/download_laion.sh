

laion_root="allava_laion"

mkdir $laion_root
cd $laion_root


# 1. download annotation files
## 1.1 caption
wget -c -O ALLaVA-Caption-LAION-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/ALLaVA-Caption-LAION-4V.json?download=true

## 1.2 instruction
wget -c -O ALLaVA-Instruct-LAION-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/ALLaVA-Instruct-LAION-4V.json?download=true


# 2. download and upzip images
mkdir image_chunks

## 2.1 download
for ((i=0; i<10; i++))
do
    wget -c -O image_chunks/images_$i.zip https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/image_chunks/images_$i.zip?download=true &
done

mkdir -p images/
wait

## 2.2 unzip
for ((i=0; i<10; i++))
do
    unzip -j -o image_chunks/images_$i.zip -d images/ & # wait patiently, it takes a while...
done

wait
echo "All done!"
