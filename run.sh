python preprocess.py
python get_vec.py
python sketch_vae.py
cd style_transfer
sh models/get_weights.sh
python main.py --content_dir=contents --style_dir=styles
