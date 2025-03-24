DATA软件包中包含了一些常用的数据集，已经进行了标准化，可以通过通用的命令进行调用。
# 一、数据集分类和介绍
## 1.1 常规数据集
这些常用数据集中，除 USPS、Norb、CoverType、Letter 等数据集外，其余数据集被调整为28×28像素的灰度图像。另外，CIFAR-10 和 CIFAR-100 被调整为 32×32 像素的灰度图像。
### 1.1.1 手写字符数据集
- MNIST 数据集是一个手写数字图像数据集，包含 70000 张 28×28 像素的灰度图像，涵盖 10 个数字字符。
- Chinese MNIST 是一个手写中文数字识别数据集，包含 15000 张 28×28 像素的灰度图像，涵盖 15 个中文数字字符（零、一、二、三、四、五、六、七、八、九、十、百、千、万、亿）。
- Tibetan MNIST 是一个藏文手写数字数据集，包含 17768 张 28×28 像素的灰度图像，涵盖藏文数字 0 到 9。
- Hindi MNIST 数据集是用于手写印地语数字识别的数据集，包含 20000 张 28×28 像素的灰度图像，涵盖印地语数字 0 到 9 的手写样本。
- Kannada MNIST 数据集是一个用于手写数字识别的数据集，包含 70000 张 28×28 像素的灰度图像，涵盖卡纳达语数字 0 到 9。
- Kuzushiji MNIST 数据集是一个基于古日文草书字符的图像数据集，由日本国立文学研究所创建。它包含 70000 张 28×28 像素的灰度图像，涵盖 10 个类别。
- USPS 数据集是一个用于手写数字识别的经典数据集，包含 11000 张 16×16 像素的灰度图像，涵盖了 0 到 9 的手写数字。
### 1.1.2 人脸数据集
- CASIA Face v5 数据集是一个广泛用于人脸识别研究的亚洲人脸图像数据库，包含 500 个人的 2500 张彩色人脸图像，每个人有 5 张图像，图像分辨率为 640×480，涵盖了不同光照、表情和姿态的变化。
- FEI Face Database 是一个巴西人脸数据库，包含 200 名个体的 14 张人脸图像，总计 2800 张图像。这些图像拍摄于 2005 年 6 月至 2006 年 3 月，背景为白色均匀背景，拍摄角度涵盖从侧面到正面的约 180 度旋转。
- Georgia Tech 包含 50 个人的15张人脸灰度图像。
- MUCT 数据集是一个用于面部识别和特征点检测的数据库，包含 3755 张人脸的灰度图像，涵盖了多种光照条件、年龄和种族背景，具有较高的多样性。
- ORL 数据集（AT&T Face Database）是一个经典的人脸识别数据库，由剑桥大学 AT&T 实验室于 1992 年 4 月至 1994 年 4 月期间创建。它包含 40 个人的 400 张灰度人脸图像，每个人有 10 张图像，图像大小为 92×112 的灰度像素。
- UMIST 数据集是一个常用的人脸识别数据库，包含 20 个人的 1012 张灰度图像。每个人在数据集中有多个图像，覆盖从侧面到正面的不同姿态，图像分辨率约为 220×220 像素。
- Yale 数据集是一个经典的人脸识别数据集，包含 15 个人的 165 张灰度图像，每个人有 11 张图像，涵盖了不同的光照条件、表情和姿态。
### 1.1.3 目标数据集
- CIFAR-10 数据集是一个广泛用于计算机视觉和深度学习领域的经典图像分类数据集，包含 60000 张 32×32 像素的图像，分为 10 个类别，涵盖了飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车等常见物体。
- CIFAR-100 数据集是 CIFAR-10 的扩展版本，包含 100 个类别的 60000 张 32×32 图像，每个类别有 600 张图像。该数据集由多伦多大学的研究人员于 2009 年发布，广泛用于图像分类任务，是训练和评估深度学习模型的重要资源。
- COIL-20 数据集是一个用于物体识别的图像数据集，包含 20 个不同物体的图像，每个物体在水平旋转 360度 的过程中每隔5度 拍摄一张图像，因此每个物体有72张图像，总共有 1440 张灰度图像。
- COIL-100 数据集是一个用于物体识别的图像数据集，包含 100 个不同物体的图像，每个物体在 360 度旋转过程中每隔 5 度拍摄一张，共计 72 张图像，因此整个数据集包含 7200 张 28×28 像素的灰度图像。
- CUB-200 数据集是一个用于细粒度图像分类的基准数据集，包含 200 种鸟类的图像，每种鸟类有多个图像样本。
- Fashion MNIST 是一个用于图像分类的经典数据集，包含 70000 张 28×28 像素的灰度图像，分为 10 个类别，涵盖不同类型的服装和配饰（T恤/上衣、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、踝靴）。
- Norb 数据集是一个用于 3D 物体识别的数据集，包含50个玩具的图像，属于 5 个通用类别：四足动物、人形、飞机、卡车、汽车。数据集中的物体由两台摄像机在6种光照条件下、9 个仰角（30 至 70 度，每 5 度一个）和 18 个方位角（0 至 340 度，每 20 度一个）下进行成像。
- STL-10 数据集是一个用于开发无监督特征学习、深度学习和自监督学习算法的图像识别数据集，包含 10 个类别的图像，分别是飞机、鸟、汽车、猫、鹿、狗、马、猴子、船和卡车。该数据集共有 13000 张有标签的 RGB 图像和 100000 张无标签图像，每张图像被调整为 28×28 像素的灰度图像。
### 1.1.4 其他数据集
- CoverType 数据集是一个经典的分类数据集，主要用于森林覆盖类型的预测，包含美国科罗拉多州北部罗斯福国家森林的四个荒野区域的地理和环境数据。数据集共有 581012 个样本，每个样本对应一个 30m×30m 的区域，包含 54 个特征。
- KTH-TIPS 数据集是一个经典的纹理图像数据集，旨在研究不同光照、角度和尺度下的材质表面纹理。它包含多种材质（如砂纸、铝箔、海绵等）的图像，这些图像在不同的采集条件下拍摄，具有丰富的纹理类型和变化。
- Letter 数据集是一个用于手写英文字母识别的经典数据集。它包含 20000 个样本，每个样本是一个16维的特征向量，表示一个手写英文字母的形状特征。数据集分为 26 个类别，分别对应英文字母 A 到 Z。
## 1.2 故障诊断数据集
在这些故障诊断数据集中，所有数据分别被重采样为1024维（CWRU、Jiangnan）和128维（CWRU-128、Jiangnan-128）的样本。
### 1.2.1 轴承数据集
- CWRU 和 CWRU-128 数据集（凯斯西储大学轴承数据中心轴承数据集）中包含了 0.007、0.014、0.021 英寸等 3 种尺寸的滚动体故障、内圈故障、外圈故障以及正常状态下的样本。采用的原始文件名称包括：`B007_0.mat、B014_0.mat、B021_0.mat、IR007_0.mat、IR014_0.mat、IR021_0.mat、OR007@6_0.mat、OR014@6_0.mat、OR021@6_0.mat、normal_0.mat`。
- Jiangnan 和 Jiangnan-128 数据集（江南大学轴承数据集）中包含了 1000 转每分钟速度下的内圈故障、外圈故障、滚动体故障和正常样本。采用的原始文件名称包括：`ib1000_2.csv、ob1000_2.csv、tb1000_2.csv、n1000_3_2.csv`。
- KAT 和 KAT-128 数据集（帕德博恩大学轴承数据集）中包含了 1000rpm 转速、0.7Nm 负载、1000N 径向力的工况下的4种故障状态（正常样本、内圈故障、外圈故障、混合故障）下的样本。采用的原始文件名称包括：`N09_M07_F10_K001_1.mat、N09_M07_F10_KA01_1.mat、N09_M07_F10_KB23_1.mat、N09_M07_F10_KI01_1.mat`。
- Liyue 和 Liyue-128 数据集（里约热内卢大学轴承数据集）中包含了滚动体故障、保持架故障和外圈故障在 6g、20g、35g 等 3 种负载下的故障样本以及正常样本。 采用的原始文件名称包括：`underhang/ball_fault/6g/13.1072.csv、underhang/ball_fault/20g/13.1072.csv、underhang/ball_fault/35g/13.7216.csv、underhang/cage_fault/6g/12.9024.csv、underhang/cage_fault/20g/13.312.csv、underhang/cage_fault/35g/13.312.csv、underhang/outer_race/6g/13.5168.csv、underhang/outer_race/20g/12.9024.csv、underhang/outer_race/35g/13.1072.csv、normal/12.288.csv`。
- MFPT-R 和 MFPT-R-128 数据集（机械故障预防技术协会轴承数据集）中包含了 3 种不同工况下的轴承振动信号样本。采用的原始文件名称包括：`6 - Real World Examples/IntermediateSpeedBearing.mat、6 - Real World Examples/OilPumpBearing.mat、6 - Real World Examples/PlanetBearing.mat`。
- Ottawa 和 Ottawa-128 数据集（渥太华大学轴承数据集）中包含了在加速状态下的内圈故障、外圈故障和正常样本。采用的原始文件名称包括：`H-A-1.mat、I-A-1.mat、O-A-1.mat`。
- Polito 和 Polito-128 数据集（都灵理工大学轴承数据集）中包含了 100Hz 下 150、250 和 450 微米的内圈故障和滚动体故障。采用的原始文件名称包括：`C0A_100_000_1.mat、C1A_100_000_2.mat、C2A_100_000_1.mat、C3A_100_000_1.mat、C4A_100_000_1.mat、C5A_100_000_1.mat、C6A_100_000_1.mat`。
- Sebear 和 Sebear-128 数据集（东南大学轴承数据集）中包含了 20Hz-0V 的工况下的 5 种样本。采用的原始文件名称包括：`ball_20_0.csv、comb_20_0.csv、health_20_0.csv、inner_20_0.csv、outer_20_0.csv`。 
- WuHan 和 WuHan-128 数据集（武汉大学转子数据集）中包含了 4 种样本。采用的原始文件名称包括：`180data_new_select_denoised.mat`。
### 1.2.2 齿轮数据集
- Connectiect 和 Connectiect-128 数据集（康涅狄格大学齿轮数据集）中包含了 9 种不同的故障样本。采用的原始文件名称包括：`DataForClassification_TimeDomain .mat`。
- Segear 和 Segear-128 数据集（东南大学齿轮数据就）中使用了20Hz-0V工况下的故障样本。采用的原始文件名称包括：`Chipped_20_0.csv、Health_20_0.csv、Miss_20_0.csv、Root_20_0.csv、Surface_20_0.csv`。
### 1.2.3 混合数据集
- Mix-Bear 数据集由 Ottawa、Polito 和 Sebear 混合而成。
- Mix-Gear 数据集由 Connectiect 和 Segear 混合而成。
Mix-Bear 和 Mix-Gear 数据集的细节参见：
```
Li B, Zheng Y, Ran R. 2DUMAP: Two-Dimensional Uniform Manifold Approximation and Projection for Fault Diagnosis [J]. IEEE Access, 2025, 13: 12819-12831.
```
## 1.3 医学图像数据集
这些医学图像数据集均来源于MedMNIST-v2，其细节参见：
```
Yang J, Shi R, Wei D, et al. Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification[J]. Scientific Data, 2023, 10(1): 41.
```
### 1.3.1 2D医学图像数据集
- BloodMNIST
- BreastMNIST
- ChestMNIST
- DermaMNIST
- OCTMNIST
- OrganAMNIST
- OrganCMNIST
- OrganSMNIST
- PathMNIST
- PneumoniaMNIST
- RetinaMNIST
- TissueMNIST
### 1.3.2 3D医学图像数据集
- AdrenalMNIST3D
- FractureMNIST3D
- NoduleMNIST3D
- OrganMNIST3D
- SynapseMNIST3D
- VesselMNIST3D
## 1.4 格拉斯曼数据集
- Ballet 数据集（Ballet-5 ~ Ballet-15）由 3 个受试者执行的 8 种复杂运动模式组成，包含从芭蕾舞教学 DVD 中收集的 44 个视频，视频的每一帧被调整为 20×20 的灰度图像，由于部分视频包含 2-3 个动作类别，所有的视频被建模为 59 个图像集样本。
- Cam-Ges-6 数据集（剑桥大学手势识别数据集）总共含有 900 个视频片段, 分别属于 9 种手势类型, 每种类型由 100 个手部动作序列构成. 上述 9 种手势类型由 3 种自然的手部形状（水平, 张开和 V 形）和 3 种手部动作（左倾, 右倾和收缩）联合定义。
- CASIA-B 数据集（CASIA-B-5 ~ CASIA-B-15）由 124 个人（93 名男性和 31 名女性）的步态数据组成. 在 CASIA-B 数据集中, 从 11 个角度捕获了每个受试者在 3 种行走条件（普通条件、穿大衣、携带背包）下的步态。每个受试者在正常的行走条件下有 6 条运动记录, 每条记录中在 0 度拍摄的视频被用于建模图像集样本. 视频的每一帧被调整为 20×20 的灰度图像。
- ETH-80 数据集（ETH-80-5 ~ ETH-80-15）包含 8 种不同的实物类别, 分别为牛, 杯子, 马, 狗, 土豆, 汽车, 梨以及苹果. 每个类别均含有 10 种不同的子类别，并且每个图像集样本均由 41 张从不同视角下采集到的图片组成。
- Extended Yale B 数据集（EYB-5 ~ EYB-15）包含 16128 张图像，其中包括 28 个处于 9 个姿势和 64 种照明条件下的受试者，从而形成了一个大型照明数据集。每个受试者在每个姿势下的 64 张图象组成一个图像集样本。
- First-Person Hand Action（FPHA）数据集（FPHA-5 ~ FPHA-15）是一个较大规模的用于手势估计的第一人称手部动作基准库，包含了属于 45 种不同手势类别的 1175 个动作序列, 由六位演示者在三种不同的视觉场景下采集得到。
- KTH 数据集（KTH-5 ~ KTH-15）由 2391 个视频格式的动作序列组成, 包含在 4 种不同场景中记录的 6 种类型的动作, 总共记录了 25 名受试者执行这些动作。
- RGB-D 数据集（RGB-D-5 ~ RGB-D-15）是一个大型数据集，包含 300 个对象，分为 51 个类别。每个对象有数百张图像, 每个对象随机选择 40 张图像构建一个图像集样本。
- Traffic 数据集（Traffic-5 ~ Traffic-15）由 254 段高速公路交通视频序列组成, 这些视频序列是从一个固定视角的交通摄像头采集得到的。
- UCF-Sport 数据集（UCF-S-5 ~ UCF-S-15）是一个动作识别数据集, 由来自 13 个动作类别的 150 个视频序列组成。
- UT-Kinect 数据集（UT-Kinect-5 ~ UT-Kinect-15）包含 10 个动作类别的视频、深度序列和骨架数据, 由 10 个受试者在 Kinect 设备前执行。每个受试者执行每个动作两次, 总共有 200 个序列。
- UTD-MHAD 数据集（UTD-MHAD-5 ~ UTD-MHAD-15）包含 8 名受试者执行的 27 个动作，包含 RGB 视频、深度视频、骨骼关节位置和惯性信号数据。
- Virus 数据集（Virus-5 ~ Virus-15）中包含了 15 种病毒的 1500 张图像，每种病毒的图像被建模为 10 个图像集样本。
- Weizmann 数据集（Weizmann-5 ~ Weizmann-15）是人类动作识别常用的数据库，包括来自 10 个动作类别的 90 个视频，由九名受试者进行。
- YouTube Celebrities 数据集（YTC-5 ~ YTC-15）是一个非常具有挑战性且被广泛使用的人脸视频数据库, 其中包含了 47 位知名人士（演员, 歌手和政客等）的 1910 段视频序列。这些视频序列均来源于 YouTube 在线视频网站，并且其中的绝大多数都是以高压缩比，低分辨率的条件进行录制的，每个视频序列的总帧数亦不统一，变化范围在 8 到 400 之间。
# 二、文件介绍
`DATA/GRASSMANN、DATA/MEDICAL、DATA/MFD、DATA/NORMAL`中包含了上述标准化后的数据集文件；
`DATA/reorga_data`中包含了对上述数据集进行标准化的过程代码；
`DATA/Data-from.http`中是上述数据集的下载网址；
`DATA/Data-size.http`中是上述数据集的关键技术参数；
`DATA/Load.py`中是加载数据集的主函数；
`DATA/Preprocessing.py`中是一些数据预处理的基本操作；
`DATA/utils.py`中是一些必要的函数；
`data_example.ipynb`中是调用 DATA 软件包的一些基本示例；
# 三、一些参考文献
使用这些数据集的文献主要有：
```
[1] Wang J, Ran R, Fang B. GNPENet: A novel convolutional neural network with local structure for fault diagnosis[J]. IEEE Transactions on Instrumentation and Measurement, 2023, 73: 1-16.
[2] Ran R, Wang T, Zhang W, et al. Autoencoder-based Discriminant Locality Preserving Projections for Fault Diagnosis[J]. IEEE Transactions on Instrumentation and Measurement, 2025.
[3] Li B, Zheng Y, Ran R. 2DUMAP: Two-Dimensional Uniform Manifold Approximation and Projection for Fault Diagnosis [J]. IEEE Access, 2025, 13: 12819-12831.
[4] Li B, Wang T, Ran R. Discriminant locality preserving projection on Grassmann Manifold for image-set classification[J]. The Journal of Supercomputing, 2025, 81(2): 1-27.
```
# 四、重要说明
Norb、MNIST等数据集的文件较大，在上传到github的过程中对文件进行了切割，在下载后需要对这些数据集进行缝合。数据集的列表、切割和缝合的命令在`large-files.txt`中，需要在git bush环境中进行操作。
