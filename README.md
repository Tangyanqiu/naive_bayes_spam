# naive_bayes_spam
reference:  
https://github.com/udacity/machine-learning/blob/master/projects/practice_projects/naive_bayes_tutorial/Bayesian_Inference_solution.ipynb
![mahua](mahua-logo.jpg)

## 依赖库
Python 3.6
PyTorch = 0.4.1
Torchvision 0.2.1
numpy
skimage
imageio
matplotlib
tqdm
nvidia 10.1
cudnn 7.5

## 代码（连接待修改）
git clone https://github.com/thstkdgus35/EDSR-PyTorch
cd EDSR-PyTorch

## 网络的训练
### 准备训练数据
Download DIV2K training data (800 training + 100 validtion images) from DIV2K dataset or SNU_CVLab.
Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.
For more informaiton, please refer to EDSR(PyTorch)https://github.com/yulunzhang/RCAN

### 开始训练
Cd to 'SR_TrainCode/code', run the following scripts to train models.
'''
BI, scale 2, 3, 4, 8
input=48x48, output=96x96
python main.py --model RCAN --save RCAN_BIX2 --scale 2 --n_resblocks 32 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96

input=48x48, output=144x144
python main.py --model RCAN --save RCAN_BIX3 --scale 3 --n_resblocks 32 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt

input=48x48, output=192x192
python main.py --model RCAN --save RCAN_BI --scale 4 --n_resblocks 32 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
'''

## 测试
### 准备测试数据集
Place the original test sets (e.g., Set5, other test sets are available from GoogleDrive or Baidu) in 'OriginalTestData'.
Run 'Prepare_TestData_HR_LR.m' in Matlab to generate HR/LR images with different degradation models.


### 重建SR图像
Download models for our paper and place them in '/RCAN_TestCode/model'.
Cd to '/RCAN_TestCode/code', run the following scripts.
'''
RCAN_BIX2
python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
RCAN_BIX3
python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX3.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
RCAN_BIX4
python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
'''
 ### 测试指标值
 Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.




##MaHua有哪些功能？

* 方便的`导入导出`功能
    *  直接把一个markdown的文本文件拖放到当前这个页面就可以了
    *  导出为一个html格式的文件，样式一点也不会丢失
* 编辑和预览`同步滚动`，所见即所得（右上角设置）
* `VIM快捷键`支持，方便vim党们快速的操作 （右上角设置）
* 强大的`自定义CSS`功能，方便定制自己的展示
* 有数量也有质量的`主题`,编辑器和预览区域
* 完美兼容`Github`的markdown语法
* 预览区域`代码高亮`
* 所有选项自动记忆

##有问题反馈
在使用中有任何问题，欢迎反馈给我，可以用以下联系方式跟我交流

* 邮件(dev.hubo#gmail.com, 把#换成@)
* 微信:jserme
* weibo: [@草依山](http://weibo.com/ihubo)
* twitter: [@ihubo](http://twitter.com/ihubo)

##捐助开发者
在兴趣的驱动下,写一个`免费`的东西，有欣喜，也还有汗水，希望你喜欢我的作品，同时也能支持一下。
##感激
感谢以下的项目,排名不分先后

* [ace](http://ace.ajax.org/)
* [jquery](http://jquery.com)

##关于作者

```javascript
var ihubo = {
  nickName  : "草依山",
  site : "http://jser.me"
}
```
Edit By [MaHua](http://mahua.jser.me)
