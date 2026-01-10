# DGCNN++

An improved version of DGCNN for PKU SIST GNN course in 2025Fall

You have to download `Stanford3dDataset_v1.2_Aligned_Version.zip` manually from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under `data/`

### Our new model training script on area5:

The main code contribution of our work compared to base repository concentrate in `model.py` and `main_semseg_s3dis.py`, you can also track the git log for change details.

To run the training code of our proposed model, please follow the commands in this section.

#### Standard new model:
Notice that due to the edge cutting, the k need to be twice of base 20 fro a better performance.
``` 
python main_semseg_s3dis.py --test_area=5 --use_sgd=1 --model=dgcnnpp_soft_pe --k=40
```

We also apply some ablation modes that can be switched to:
#### Only with attention:
``` 
python main_semseg_s3dis.py --test_area=5 --use_sgd=1 --model=dgcnnpp_soft_pe --k=40 --ablation_mode=atten_only
```
#### Only with gating:
``` 
python main_semseg_s3dis.py --test_area=5 --use_sgd=1 --model=dgcnnpp_soft_pe --k=40 --ablation_mode=gating_only
```



### Run the evaluation script after training finished:

- Evaluate in area 6 after the model is trained in area 1-5

``` 
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval_6 --test_area=5 --model=dgcnnpp_soft_pe --eval=True --model_root=outputs/*YOUR_MODEL_PATH*
```

- Evaluate in all areas after 6 models are trained

``` 
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval --test_area=all --eval=True --model_root=outputs/*YOUR_MODEL_PATH* --model=dgcnnpp_soft_pe
```

### Visualization: 
#### Usage:

Use `--visu` to control visualization file. 

- To visualize a single room, for example the office room 1 in area 6 (the room index starts from 1), use `--visu=area_6_office_1`. 
- To visualize all rooms in an area, for example area 6, use `--visu=area_6`. 
- To visualize all rooms in all areas, use `--visu=all`. 

Use `--visu_format` to control visualization file format. 

- To output .txt file, use `--visu_format=txt`. 
- To output .ply format, use `--visu_format=ply`. 

Both .txt and .ply file can be loaded into [MeshLab](https://www.meshlab.net) for visualization. For the usage of MeshLab on .txt file, see issue [#8](https://github.com/AnTao97/dgcnn.pytorch/issues/8) for details. The .ply file can be directly loaded into MeshLab by dragging.

The visualization file name follows the format `roomname_pred_<miou>.FILE_TYPE` for prediction output or `roomname_gt.FILE_TYPE` for ground-truth, where `<miou>` shows the mIoU prediction for this room.

**Note:** In semantic segmentation, you need to first run a training or evaluation command without visualization in the above sections to preprocess dataset. With the dataset well prepared, you can run a command with visualization in the following sections. 

#### Evaluate in area 6 after the model is trained in area 1-5:

- Output the visualization file of office room 1 in area 6 with .ply format

```
# Use trained model
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval_6 --test_area=6 --eval=True --model_root=outputs/*YOUR_MODEL_PATH* --model=dgcnnpp_soft_pe --visu=area_6_office_1 --visu_format=ply
```

- Output the visualization files of all rooms in area 6 with .ply format

```
# Use trained model
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval_6 --test_area=6 --eval=True --model_root=outputs/*YOUR_MODEL_PATH* --model=dgcnnpp_soft_pe --visu=area_6 --visu_format=ply

```

#### Evaluate in all areas after 6 models are trained:

- Output the visualization file of office room 1 in area 6 with .ply format


```
# Use trained model
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval --test_area=all --eval=True --model_root=outputs/*YOUR_MODEL_PATH* --model=dgcnnpp_soft_pe --visu=area_6_office_1 --visu_format=ply

```

- Output the visualization files of all rooms in area 6 with .ply format

```
# Use trained model
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval --test_area=all --eval=True --model_root=outputs/*YOUR_MODEL_PATH* --model=dgcnnpp_soft_pe --visu=area_6 --visu_format=ply

```

- Output the visualization files of all rooms in all areas with .ply format

```
# Use trained model
python main_semseg_s3dis.py --exp_name=semseg_s3dis_eval --test_area=all --eval=True --model_root=outputs/*YOUR_MODEL_PATH* --model=dgcnnpp_soft_pe --visu=all --visu_format=ply
```
