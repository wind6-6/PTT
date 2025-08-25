
python train_tracking.py --cfg_file D:\1study\PTT\tools\cfgs\kitti_models\ptt.yaml --extra_tag car_p2b
python test_tracking.py --cfg_file cfgs\kitti_models\ptt.yaml --ckpt D:\1study\PTT\output\kitti_models\ptt\car\best_model.pth

python train_tracking.py --cfg_file /home/lishengjie/study/fhao/PTT/tools/cfgs/kitti_models/ptt.yaml --extra_tag car_p2b
python test_tracking.py --cfg_file /home/lishengjie/study/fhao/PTT/tools/cfgs/kitti_models/ptt.yaml --ckpt /home/lishengjie/study/fhao/PTT/output/best_model.pth


python test_tracking.py --cfg_file cfgs\kitti_models\ptt.yaml --ckpt D:\1study\PTT\output\kitti_models\ptt\car\ckpt\checkpoint_epoch_60.pth



python test_tracking.py --cfg_file /home/lishengjie/study/fhao/PTT/tools/cfgs/kitti_models/ptt.yaml --ckpt /home/lishengjie/study/fhao/PTT/output/home/lishengjie/study/fhao/PTT/tools/cfgs/kitti_models/ptt/car_p2b/ckpt/checkpoint_epoch_60.pth
