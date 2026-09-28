##### git命令

* git status --short:查看仓库状态
* git branch --show-current:查看当前所在的git分支

##### CUDA

* echo "$CONDA_DEFAULT_ENV"：查看当前激活的conda环境名称，echo:在终端输出内容，CONDA_DEFAULT_ENV是Conda 设置的环境变量，保存当前环境名称。（cgy_pytorch)
* CUDA Driver 是 NVIDIA 显卡驱动提供的底层能力，接收 CUDA 计算任务并控制 GPU，让操作系统和程序能够使用 GPU。NVIDIA 显卡驱动是完整的驱动软件包；CUDA Driver是其中负责 GPU 通用计算的底层部分；CUDA Runtime 和 PyTorch通过它把计算任务真正提交给 GPU。NVIDIA 显卡驱动是操作系统与 GPU 硬件之间的基础软件。CUDA 是 NVIDIA 为其 GPU 提供的并行计算平台和编程体系，并不是显卡驱动本身。PyTorch 等框架通过 CUDA 提供的接口，把矩阵计算、模型训练等任务交给 GPU。
* 显卡驱动负责：
  * 识别和管理显卡；
  * 分配显存；
  * 向 GPU 提交计算任务；
  * 控制 GPU 与操作系统通信；
  * 提供底层 CUDA Driver API。
* nvidia-smi 会同时显示：Driver Version: 550.54.15，CUDA Version: 12.4，表示实际安装的显卡驱动版本和该驱动最高支持的 CUDA 版本。
* CUDA Runtime 是 PyTorch 运行 GPU 运算所需的一组 CUDA 动态库。通过 pip 或 Conda 安装 CUDA 版 PyTorch 时，相应运行库通常会一同安装在 Python/Conda 环境中。它通常不要求系统另外安装完整的 CUDA Toolkit。
* 系统 CUDA Toolkit，CUDA Toolkit 是 NVIDIA 提供的 CUDA 开发工具包。
* which python：用于查找：当你在当前终端输入 python 时，系统实际会执行哪个 Python 程序。/home/gycing/.conda/envs/cgy_pytorch/bin/python
* python --version：查看python版本

##### 显存训练差异及计算方法

* 参数量与容量：7B，13B，B是billion，即十亿。7B 表示约 70 亿个模型参数，不是 7 GB。GB是容量单位，1 GB = 10⁹ 字节，10亿字节。显卡工具也常用 GiB（1 GiB = 2³⁰ 字节）；两者数值略有差别，“精度 16、8、4、2”指的是每个数值大致使用多少  **bit（位）** ，不是 GB。8 bit = 1 byte（字节）。
* 显存计算方法：**权重大小** ≈ 参数量 × 每个参数占用的字节数，**训练显存** ≈ 模型权重 + 梯度 + 优化器状态 + 激活值 + 临时开销，对于优化器状态以常见的 AdamW 为例，它会为**每个可训练参数**保存两个状态：一阶动量和二阶动量。如果两者都是 FP32，每个各占 4 字节，对于激活值，一个隐藏状态张量的大小·= 每卡 batch × 序列长度 × 隐藏层宽度 × 每个元素的字节数，还需乘个数合层数，此外模型权重可能还包括**计算权重**（前向传播+反向传播）+**主权重**（优化器更新）。

##### CPT

* 增量预训练，喂给模型通用医疗数据集，使模型具有相应领域的知识，同时建议领域训练集中加入一些通用对话数据。但注意如果用到的数据是模型预训练中已经使用了的，如维基百科、ArXiv等LLaMA模型预训练用了的，则这些数据是没有必要再喂给LLaMA增量预训练，而且预训练样本的质量如果不够高，也可能会损害原模型的生成能力。

##### 命令

* python -m pip install pytest tensorboard：用当前 python 所属环境的 pip 安装两个工具。pytest 用于运行测试，tensorboard 用于查看训练日志。安装会修改当前 Python 环境。
* python -m pip freeze > environment_day01.txt：列出当前 Python 环境中安装的包及版本，并用 > 写入当前工作目录下的文件 environment_day01.txt。如果该文件已存在，> 会覆盖它。
*
