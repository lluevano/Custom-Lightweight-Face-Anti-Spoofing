#Download face detection model
gdown htpps://drive.google.com/uc?id=1vIfd21qGaOZv0ALVsSOTZadU6uKaHQuy
unzip retinaface-R50.zip -d models/

#Assuming the dataset is in datasets/FAS-CVPR2024/UniAttackData 
cd datasets/FAS-CVPR2024/

python detect_norm_crop.py
cp UniAttackData/phase1/p1/*.txt norm_crop/UniAttackData/phase1/p1/
cp UniAttackData/phase1/p2.1/*.txt norm_crop/UniAttackData/phase1/p2.1/
cp UniAttackData/phase1/p2.2/*.txt norm_crop/UniAttackData/phase1/p2.2/
cp UniAttackData/phase2/p1/*.txt norm_crop/UniAttackData/phase2/p1/
cp UniAttackData/phase2/p2.1/*.txt norm_crop/UniAttackData/phase2/p2.1/
cp UniAttackData/phase2/p2.2/*.txt norm_crop/UniAttackData/phase2/p2.2/
cp UniAttackData/phase2/p1/*.txt norm_crop/UniAttackData/phase1/p1/
cp UniAttackData/phase2/p2.1/*.txt norm_crop/UniAttackData/phase1/p2.1/
cp UniAttackData/phase2/p2.2/*.txt norm_crop/UniAttackData/phase1/p2.2/

cd ../../

python train.py --config configs/config_cvpr2024_p1.py
python train.py --config configs/config_cvpr2024_p2.1.py
python train.py --config configs/config_cvpr2024_p2.2.py

python save_predictions_FASCVPR2024.py --config configs/config_cvpr2024_p2.2.py

