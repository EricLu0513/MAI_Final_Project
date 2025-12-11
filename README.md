# MAI_Final_Project

## Dataset 
### Moment Retrieval
1. [Charades_STA evaluation set](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip)
2. [ActivityNet evaluation set (Not available)]()

### Download Step (Recommended)
```bash
apt-get update -y && apt-get install -y aria2 

mkdir -p ./Charades_v1_480_test

aria2c -x 16 -s 16 -k 1M \
  "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip" \
  -d ./Charades_v1_480_test \
  -o Charades_v1_480.zip

unzip -oq Charades_v1_480_test/Charades_v1_480.zip -d Charades_v1_480_test/
```
### Charades_STA_mini
* See Charades_test_mini.json
* Randomly select 100 data from the original test.json

### MovieNet Gemini 500 Dataset 
* Download Link: https://drive.google.com/file/d/1xaDlurAYUgIJYtkQG1NA_3FaIC4LmsnF/view?usp=drive_link 
* Structure
  * MovieNet_Gemini_500
    * MovieNet_Gemini_500_videos
    * MovieNet_Gemini_500_dataset.json
* Brief Description:
  * Randomly Select 100 Movies, and select 5 short clips among each movie, totally 500 clips
  * The Clip Length are between 15~30 frames
  * The Movie Length are between 00:00:39 ~ 01:07:23, 3 fps
  * The Movies are splits into shots(鏡頭), each shot contains 3 frames, which means the shot_id equals to seconds.
* How the dataset are made
  * Randomly Select 100 videos, 500 clips
  * Ask the Gemini-2.5-flash to generate description as description


## Execution
### Install dependencies
```bash
pip install -r requirements.txt # or "uv sync" which is faster

apt-get install -y fonts-dejavu # Install the fonts
```

### Moment Retrieval in Charades_STA
```bash
gdown https://drive.google.com/drive/folders/11tsL9BjM3xcyaYDN2Af-n6yJfmvWuggv -O eval_gts --folder

python eval/qwen2_vl_7b_mr.py # The result will be saved in "results/qwen2_vl_7b_mr_charades.json" which is dynamically created
```


