# 三维点云配准学习与实验

本科阶段的SVD/ICP学习复现与匹配筛选对照实践，非原创算法研究。使用合成点云，尚未完成真实扫描验证。

## 运行
在兼容这些依赖版本的Python环境中：
```bash
python -m pip install -r requirements.txt
python day03_icp.py
python day04_experiments.py
python day05_robust_icp.py
python day05_batch.py
```
依赖版本记录自本地已运行环境；跨平台兼容性尚未单独验证。

## 内容
- day03_icp.py：SVD刚体配准、普通ICP及基础可视化。
- day04_experiments.py：角度与噪声条件下的80次配准。
- day05_robust_icp.py：保留较近80%匹配的筛选版本及单次对比。
- day05_batch.py：四种离群点比例、每组十个种子、两种方法共80次配准。

## 结果与限制
初始旋转10°、正常点不加噪声、20%源点被随机乱点替换时，普通ICP平均相对配准误差8.14984%，筛选版接近浮点计算精度。每个条件重复10次。
误差是在保留的干净源点上应用估计变换后，对真实对应目标点计算RMSE，再除以目标点云包围盒对角线长度。不是仅比较筛选后匹配残差。
这一结果依赖理想的合成数据与参数，不代表任意场景、遮挡、局部重叠或真实测量性能。

## 学习和协作说明
学生逐步学习原理，整理并运行代码、提出疑问、参与调试及结果解释。AI辅助提供解释和代码示例、排查错误、实现并执行批量实验、整理图表。具体贡献不声称全部独立完成。

## 学习参考
- https://github.com/ClayFlannigan/icp
- https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html

上述为学习参考，不表示本项目复现了整个Open3D系统。算法代码由协作学习过程形成；如后续直接引入第三方源码，应保留其许可证和版权声明。
