# Anomaly Detection on MVTecAD

# Streamlit Demonstration

1. Clone this repo
```sh
git clone https://github.com/Ly-Lynn/Anomaly-Detection-on-MVTecAD
cd Anomaly-Detection-on-MVTecAD
```
2. Create conda environment
```sh
conda create --name anomaly-detection-env python=3.8
conda activate anomaly-detection-env
```
3. Install necessary packages
```sh
pip install -r requirements.txt
```
4. Download dataset
```sh
python data_download.py
```
5. Build the database
```sh
cd database
python add_imgs.py
```
6. Run streamlit server
```sh
cd ..
streamlit run app.py
```

