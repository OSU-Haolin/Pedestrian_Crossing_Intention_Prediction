# Mask-PCPA: Pedestrian Crossing Prediction
## OSU-CITR

### Data process

Download the JAAD dataset, and then put the videos into `./dataset/JAAD_clips`. In order to use the data, first, the video clips should be converted into images. This can be done using script `./dataset/split_clips_to_frames.sh` or via interface as follows:
```
from jaad_data import JAAD
jaad_path = <path_to_the_root_folder>
imdb = JAAD(data_path=jaad_path)
imdb.extract_and_save_images()
```

Using either of the methods will create a folder called `images` and save the extracted images grouped by corresponding video ids in the folder.
```
./dataset/images/video_0001/
				00000.png
				00001.png
				...
./dataset/images/video_0002/
				00000.png
				00001.png
				...		
...
```
Using `./dataset/data_process.py` to extract annotations and save them as TXT format in `./dataset/sequences/`  
`./dataset/sequences/xx/xxx/` (xx indicates the video_id, xxx indicates the pedestrian_id)  
`./dataset/sequences/xx/xxx/img` save the cropped pedestrian images  
`./dataset/sequences/xx/xxx/label.txt` save the labels (with contextual factors) of the pedestrian  



## TODO Lists

- [ ] Readme Completion
- [ ] Pretrained Model
- [ ] Support PIE Dataset
