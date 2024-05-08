# A-single-simple-patch-achieved-bymyself
这是我个人对论文“A single simple patch is all your need”这篇验证ai和generated图像的论文的部分复现。由于原论文未开源所以我的代码可能存在诸多纰漏，也希望多多包涵。另，根据我这个预处理代码然后直接使用resnet50分类的效果并没有论文中的数据那么好。
原论文：A Single Simple Patch is All You Need for AI-generated Image Detection                        https://arxiv.org/html/2402.01123v2
在我的代码中也用到了一些别人的代码，在此感谢他们。
论文的方法流程：首先提取纹理最简单的patch，然后经过srm处理器处理之后送入resnet50进行分类。
更新：论文好像有所修改，我的复现方法是根据最初的那一个版本的论文修改的，所以请谨慎参考。










This is a partial reproduction of my paper "A single simple patch is all you need" that verifies the AI and generated images. Due to the fact that the original paper was not open-source, there may be many flaws in my code, and I hope to be more tolerant. Furthermore, based on my preprocessing code and using ResNet50 classification directly, the performance is not as good as the data in the paper.
Original paper: A Single Simple Patch is All You Need for AI generated Image Detection https://arxiv.org/html/2402.01123v2
I also used some other people's code in my code, and I would like to thank them.
The method flow of the paper: First, extract the patch with the simplest texture, then process it with an SRM processor and send it to ResNet50 for classification
Update: The paper seems to have been modified. My reproduction method is based on the original version of the paper, so please refer to it carefully.
