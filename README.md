# docAnalysis
maskrcnn paper document analysis

### 基于maskrcnn的论文版面分析

## 环境(Requirements)
```pip install -r requirements.txt```

## 例子🌰(Demo)
- 修改inference 中main函数所需路径

```python inference.py```

## 训练(train)
- 修改train_test.py  COCO_MODEL_PATH 为模型路径
- 可以更改ShapesConfig配置进行优化
- 如果添加或减少类别需要对应修改 NUM_CLASSES 与 self.add_class， labels_form.append
```python train_test.py```

## data制作
- 参考maskrcnn的制作 其中本rep有需要用到的脚本

## 可视化实例
### 例子
![1](https://github.com/tommyMessi/docAnalysis/blob/main/assets/1.png)
![2](https://github.com/tommyMessi/docAnalysis/blob/main/assets/2.png)
![3](https://github.com/tommyMessi/docAnalysis/blob/main/assets/3.png)
![4](https://github.com/tommyMessi/docAnalysis/blob/main/assets/4.png)
![5](https://github.com/tommyMessi/docAnalysis/blob/main/assets/5.png)

## model
- 预训练模型： 链接: https://pan.baidu.com/s/13ehRSpaU-_T4diWjcRqHaA 提取码: nkmc 复制这段内容后打开百度网盘手机App，操作更方便哦
- 犹豫数据集标注数量比较少。所以与训练模型仅仅针对paper文档类型。现在的效果也不是最好的。有需要的同学可以自己多标注一些数据。

## 其他
仅仅几十张的训练数据获取方式 关注微信公众账号 hulugeAI 留言：doc
