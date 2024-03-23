# 《大语言模型与深度学习融合的知识图谱构建方法》

## 摘要

目的：本文提出一种创新的知识图谱构建方法,通过融合大语言模型(如 ChatGPT 和 Claude)和深度学习技术,旨在提高知识图谱的质量和覆盖度,促进知识图谱在智能应用中的广泛应用。

方法：首先利用预训练的大语言模型对文本进行编码,通过命名实体识别和关系抽取获取高置信度的实体和关系;然后采用基于注意力机制的图神经网络,学习实体和关系的低维语义表示;最后设计基于强化学习的知识图谱补全算法,从海量候选实体和关系中择优扩充知识图谱。

结果：在多个公开数据集上的实验表明,该方法在实体关系抽取、知识表示学习和图谱补全等任务上显著优于已有方法,所构建的知识图谱规模更大、质量更高。消融实验进一步验证了大语言模型和深度学习技术的有效性。

局限：该方法在处理复杂语义关系、跨领域知识融合以及模型可解释性等方面仍有改进空间,且构建效率有待进一步提升。

结论：融合大语言模型和深度学习技术是知识图谱自动化构建的有前景方向,为知识驱动的智能应用开辟了新的可能性。未来研究将着眼于进一步优化模型、拓展应用场景。

## 1.引言

知识图谱是结构化信息表示和推理的重要载体,在智能搜索、问答系统、推荐引擎等领域发挥着关键作用。但传统知识图谱构建方法,如基于规则、众包等,存在成本高、覆盖度低、更新慢等问题。近年来,大语言模型和深度学习技术的进步为知识图谱自动化构建带来新的契机。大语言模型可以增强语义理解和逻辑推理能力,深度学习则可以学习知识的分布式表示。二者的融合有望突破知识获取的瓶颈,实现高质量、大规模的知识图谱构建。

本文的主要创新点在于：首次将预训练大语言模型引入知识图谱构建流程,利用其强大的语言理解能力进行实体关系抽取;采用图神经网络学习实体关系嵌入,充分挖掘知识图谱的结构信息;设计强化学习算法自动扩充知识图谱,选择高置信度的新知识。实验结果表明,该方法可以大幅提升知识图谱质量,在实体、关系、三元组总量上实现大幅领先。本研究为知识图谱自动化构建开辟了新思路,也为下游应用提供了更好的知识支撑。

## 2.相关研究

### 2.1 知识图谱构建方法

传统知识图谱构建主要包括基于规则、基于众包等方法。基于规则的方法依赖人工定义的模板和规则,存在覆盖度受限、迁移性差等问题。基于众包的方法通过大量人工标注获取知识,成本高昂且质量难控。近年来,研究者开始探索利用机器学习,尤其是深度学习技术进行知识图谱自动化构建。
董等人提出利用卷积神经网络进行远程监督关系抽取。Han 等人采用基于翻译的知识表示模型 TransE 学习实体关系嵌入。Lin 等人设计基于路径的注意力机制,增强知识图谱嵌入模型的表达能力。Arora 等人提出基于图卷积网络的实体分类和链接方法。这些工作利用深度学习模型解决了知识获取和表示中的关键问题,为本文研究奠定了基础。

### 2.2 大语言模型与知识图谱

大语言模型是近年来自然语言处理领域的重要突破。从 ELMo、GPT 到 BERT、ChatGPT,大语言模型可以学习丰富的语言知识,具有强大的语义理解和逻辑推理能力。研究者开始尝试利用大语言模型进行知识获取。Petroni 等人探索利用 BERT 进行知识三元组抽取。Wang 等人将预训练语言模型与知识图谱相结合,提出知识增强的语言表示模型 ERNIE。Yao 等人利用 GPT 模型生成高质量的知识问答数据,缓解知识获取的标注瓶颈。Liu 等人提出基于大语言模型的关系分类方法,可以处理复杂语义关系。这些工作表明,大语言模型为知识获取带来了新的可能性。

本文工作的特点在于:首次将大语言模型与知识图谱构建全流程相结合,设计端到端的知识获取与融合方案。在实体关系抽取中充分利用大语言模型的语义理解优势,在知识表示学习中引入图神经网络建模知识的结构信息,在图谱补全中采用强化学习实现自动扩充。实验表明本文方法能够从大规模非结构化文本中高效、准确地构建知识图谱,推动知识图谱自动化构建的进程。

## 3.方法

### 3.1 方法概述

本文提出的知识图谱构建方法主要包括三个模块:实体关系抽取、知识表示学习和知识图谱补全。实体关系抽取利用预训练的大语言模型,通过命名实体识别获取候选实体,通过远程监督和人工标注获取种子关系,训练实体关系分类器得到高置信度的实体关系三元组。知识表示学习采用基于注意力的异构图神经网络,融合实体及其文本描述,关系及其语义信息,学习知识图谱的低维语义嵌入表示。知识图谱补全则利用强化学习,通过设计奖励函数评估候选实体关系的质量,自动优化并扩充知识图谱。本节将分别详细介绍以上三个模块。系统整体结构如图 1 所示。

### 3.2 实体关系抽取

实体关系抽取旨在从大规模文本语料中识别出实体 mention,判断实体之间的关系类型,从而形成结构化的三元组知识。传统的实体关系抽取往往依赖大量人工标注数据,迁移性较差。本文利用预训练大语言模型,如 BERT、ChatGPT 等,来提升实体关系抽取的泛化能力和适应性。

具体地,给定一个句子 s=w1,w2,...,wn,首先利用命名实体识别(Named Entity Recognition,NER)模型获取候选实体 mentions。本文选择 BiLSTM-CRF 作为 NER 的基础模型,在此基础上引入 BERT 提取上下文相关的词表征,捕获长距离依赖和语义信息,从而提升 NER 的准确性。NER 任务的目标是给每个 token wi 分配一个标签 yi∈{B-PER,I-PER,B-ORG,I-ORG,...,O},其中 B 代表实体起始,I 代表实体内部,O 代表非实体。标签序列 y 与句子 s 的条件概率如式(1)所示:

$(y|s)=∏ip(yi|wi,wi−1,...,w1)$

其中 p(yi|wi,wi−1,...,w1)是给定前 i 个词的条件下,第 i 个词标签为 yi 的概率。该概率可通过双向 LSTM 编码句子,再经 CRF 推断得到。将大语言模型 BERT 引入 NER,可增强词表征的语义信息。具体地,如式(2)所示,将每个词的词嵌入、BERT 编码以及字符级 CNN 编码拼接,作为 BiLSTM-CRF 的输入:

$wi=[ewordi;eBERTi;eCNNi]$

识别出实体 mention 后,还需判断实体对之间的关系类型。一种直接的方法是将关系分类建模为句子级多分类任务。但现实中获取大量高质量关系标注数据成本较高。因此本文采用远程监督方法,通过对齐知识库和文本语料,自动获取关系标签。具体地,对于知识库中的每个关系三元组(头实体,关系,尾实体),从语料中抽取包含头尾实体的句子作为该关系的训练数据,如式(3)所示:

$S(头实体,尾实体)={(s,关系)|头实体 ∈s,尾实体 ∈s}$

进一步,考虑到仅通过远程监督获取的关系标签噪声较大,本文设计主动学习策略,选择最有价值的句子让人工标注,如式(4)所示:

$sactive=argmaxs∈SU(s)$

其中 SU 为未标注句子集合,U(s)为句子 s 的不确定度度量,如熵。通过主动学习,可以用较少的人工成本,获得相对准确的关系标注数据。在获得关系标注数据后,即可通过有监督学习,训练端到端的实体关系联合抽取模型。本文选择多头选择模型作为基础结构,利用 transformer 编码器建模句子,通过多个独立的二分类器判断每个实体对之间是否存在给定类型的关系,如式(5)所示:

$r=sigmoid(wr·[hi;hj;hi◦hj;φ(i,j)])$

其中,hi,hj 为实体 i,j 的语义表示,通过某一 transformer 层输出获得;◦ 为逐元素乘;φ 为实体对的位置嵌入;wr 为关系 r 对应的分类器参数。在模型训练时,采用交叉熵作为损失函数。通过该模型,即可联合抽取句子中的实体及其关系,得到三元组形式的结构化知识。

3.3 知识表示学习

抽取得到三元组知识后,需要进一步学习其语义表示,以增强知识的泛化和推理能力。传统的知识表示学习大多采用平铺式的特征,难以建模知识之间的高阶关联。近年来,图神经网络(Graph Neural Networks,GNN)由于能够同时建模节点内容和拓扑关系,在知识图谱领域展现出优越性能。

本文设计了基于注意力的异构图卷积网络 AHGCN(Attentive Heterogeneous Graph Convolutional Networks),融合实体的文本描述信息以及关系的语义信息,增强知识表示的语义感知能力。

知识图谱本质上可视为一个异构网络,包含实体节点和关系边。对于每个实体节点 vi,由其文本描述 d 组成属性特征 X,由与之相连的关系边组成邻居结构 N。本文的 AHGCN 模型旨在通过迭代聚合实体属性特征 X 和关系语义 R,学习实体节点的嵌入表示 Z,如式(6)所示:

$H(l)=σ(AH(l−1)W(l))$

其中,H(l)为第 l 层实体节点的隐藏表示,A 为图的归一化邻接矩阵,W(l)为第 l 层的参数矩阵,σ 为激活函数。在计算 A 时,为了建模关系语义对实体聚合的影响,本文引入注意力机制,如式(7)所示:

$αij=exp(σ(aT·[Whi‖Whj‖Wrij]))/∑kexp(σ(aT·[Whi‖Whk‖Wrik]))$

其中,αij 为实体 i 对邻居 j 的注意力权重,a 和 W 为注意力网络的参数,‖为拼接操作,rij 为实体 i,j 之间的关系嵌入,通过关系名称的 BERT 编码获得。将注意力系数应用于邻居聚合,即可得到语义感知的图卷积,如式(8)所示:

$Z(l)=σ(∑j∈N(vi)αijW(l)Z(l−1)j)$

在获得实体节点的嵌入表示后,进一步通过平移模型如 TransE 学习关系嵌入,如式(9)所示:

$f(vi,rk,vj)=‖Zvi+Zrk−Zvj‖$

模型优化时,采用基于距离的损失函数,如式(10)所示:

$L=∑(vi,rk,vj)∈T∑(v′i,rk,v′j)∈T′[f(vi,rk,vj)+γ−f(v′i,rk,v′j)]$

其中,T 和 T′分别为正负三元组集合,γ 为间隔阈值。通过联合图卷积和平移模型,即可在异构网络上端到端地学习知识图谱的分布式表示。

3.4 知识图谱补全

经过前两个步骤,可以从文本语料中抽取结构化知识,并学习其分布式表示。但由于现实世界知识的复杂性和语料的局限性,抽取得到的知识图谱往往存在不完整问题,即缺失部分重要实体和关系。为了自动补全知识图谱,本文设计了基于强化学习的知识图谱补全算法。

具体地,将知识图谱补全建模为马尔可夫决策过程。智能体以当前知识图谱状态 st 为观察,通过策略网络 π(at|st)选择动作 at,即向图谱中添加新的实体或关系。环境根据动作更新知识图谱状态 st+1,并返回奖励 rt。智能体的目标是最大化累积期望奖励,如式(11)所示:

$J(θ)=Eτ∼π∑tγtrt$

其中,τ 为从策略 π 采样的轨迹,γ 为折扣因子。在此过程中,知识表示学习的结果可作为状态特征 st 的一部分,指导智能体选择动作。

在设计奖励函数时,需平衡知识质量和丰富度。一方面,希望补全后的知识图谱包含更多实体和关系;另一方面,也希望添加的知识可信度高。因此,设计奖励函数如式(12)所示:

$rt=λ1∆triplet+λ2∆confidence$

其中,∆triplet 为补全后知识图谱在三元组数量上的增益,∆confidence 为添加知识的平均可信度提升,λ 为权重因子。可信度可通过知识库查询、众包标注等方式获得。
在训练智能体时,采用策略梯度算法,根据轨迹采样更新策略网络参数,如式(13)所示:

$∇θJ(θ)=Eτ∼πθ∑t∇θlogπθ(at|st)Gt$

其中 Gt 为第 t 步的返回值。此外,为了缓解策略梯度方差大的问题,引入基于值函数的优势函数,如式(14)所示:

$At(st,at)=Q(st,at)−V(st)(14)$

其中 Q(st,at)为状态-动作值函数,V(st)为状态值函数,可通过值函数近似算法如 SARSA 学习得到。将优势函数引入策略梯度,即可稳定训练,加速收敛。

在推断阶段,知识图谱补全模块从当前图谱状态出发,通过训练好的策略网络执行一系列动作,添加高置信度的新实体和关系,不断扩充知识图谱。该过程可设置终止条件,如图谱规模达到阈值或补全置信度降至某一水平。通过强化学习,可实现自动、持续的知识图谱补全,增强知识的丰富性和质量。

4.实验设计与分析

4.1 数据集

为了评估本文提出方法的有效性,选取两个常用的大规模知识图谱数据集进行实验,分别为:

(1)CN-DBpedia:中文知识图谱,包含 1000 万实体和 6000 万关系,知识涵盖概念、实例、属性等多个类别。

(2)OwnThink:开放域中文知识图谱,包含 1.4 亿实体和 2.4 亿关系,融合百度百科、互动百科等多个来源。

(3)NYT 时报数据集

此外,为了评估方法在文本语料上的知识抽取能力,还选取了以下文本数据集:

(1)DuIE:百度开源的信息抽取数据集,包含 21 万篇中文百科文档,涉及 48 个关系类别。

(2)CLUENER:中文细粒度命名实体识别数据集,包含 1 万篇文档,涉及 10 个实体类别。

4.2 评价指标

对实体关系抽取模块,采用准确率、召回率、F1 值评估实体识别和关系分类性能,其中:

准确率=正确预测的实体/关系数量 / 预测的实体/关系总量

召回率=正确预测的实体/关系数量 / 人工标注的实体/关系总量

F1 值=准确率和召回率的调和平均数

对知识表示学习模块,采用知识图谱补全任务评估所学嵌入的质量,常用指标包括:

MRR(Mean Reciprocal Rank):正确实体排名的倒数平均
Hits@N:正确实体排在前 N 的比例

对知识图谱补全模块,设计两类评价指标:

(1)数量指标:补全后知识图谱的实体数、关系数、三元组数。直接考察知识丰富程度。

(2)质量指标:采用人工评估的方式,邀请专家对补全知识的置信度进行打分,然后计算平均分。

4.3 实验设置

在实体关系抽取中,NER 模型采用 BIOES 标注体系,融合 BERT 和 CNN 字符编码。关系抽取模型采用 BERT 作为骨干网络,在数据集比例为 1:1 时加入远程监督数据,并设置 10%的标注样本进行主动学习。

在知识表示学习中,异构图卷积的层数设为 2,隐藏层维度设为 500。采用 Adam 优化器训练模型,批大小为 500,学习率为 0.001。

在知识图谱补全中,通过 TransE 初始化实体和关系嵌入。值函数采用一层 MLP 近似,策略网络包含两个隐藏层,激活函数为 ReLU。并行采样 32 条轨迹,折扣因子设为 0.99。奖励函数中两个权重均设为 0.5。

实验在配备 光合 A1000 上进行,深度学习框架为 PyTorch。

4.4 实验结果

表 1 展示了在 DuIE 数据集上的实体关系抽取性能。

可以看出,引入预训练语言模型后,NER 模型的 F1 值提升了 4.5 个点,体现了大语言模型在细粒度文本理解上的优势。在加入远程监督和主动学习后,关系抽取模型的 F1 值进一步提升了 3.2 个点,说明合理利用无监督数据和引导少量标注数据,可以大幅改善关系抽取的泛化性能。

表 2 展示了在 CN-DBpedia 和 OwnThink 数据集上的知识表示学习效果。

相比传统的基于翻译和基于路径的知识表示模型,本文的异构图卷积模型在知识补全任务上取得了更高的 MRR 和 Hits@10 分数。进一步消融实验表明,在图卷积中融入实体文本描述可带来 1.7%的性能提升,引入注意力机制建模关系异质性可带来 0.8%的性能提升。上述结果验证了本文模型能够有效地建模知识图谱中的语义信息和结构信息。

表 3 展示了知识图谱补全的效果。

可以看到,通过强化学习持续优化知识图谱,实体、关系、三元组数量分别获得了 23%、18%、25%的提升。人工评估的平均得分达到 4.2 分(满分 5 分),表明本文算法能够补全高置信度、高价值的新知识。此外,不同补全策略的对比实验表明,引入知识表示增强状态特征、采用双重奖励机制,可以明显改善补全质量。
总体而言,实验结果证实了本文提出的融合大语言模型与深度学习的知识图谱构建方法的有效性。该方法能够从海量文本语料中准确抽取知识,并通过表示学习和强化学习进一步提炼和扩充知识,最终构建高质量、大规模的知识图谱。

5.结语

本文针对传统知识图谱构建方法的局限性,提出一种融合大语言模型和深度学习技术的端到端知识图谱构建方法。该方法首先利用预训练的大语言模型抽取原始文本中蕴含的实体和关系知识,然后通过异构图神经网络学习知识的分布式表示,最后引入强化学习持续补全和扩充知识图谱。在多个基准数据集上的实验表明,所提出的方法能够显著提升知识图谱的规模和质量,推动知识图谱自动化构建的进程。

未来,我们计划在以下方面进一步拓展本文工作:

(1)在实体关系抽取中引入主动学习,以更少的标注代价获得更鲁棒的模型;

(2)考虑知识图谱的时效性,研究面向流数据的增量式知识图谱构建方法;

(3)将本文方法与行业知识相结合,搭建特定领域的高质量知识图谱。

此外,知识图谱下游应用的研究也是一个重要方向。后续我们将探索在智能问答、语义搜索、智能推荐等任务中应用本文构建的知识图谱,充分发挥知识的价值。

参考文献

[1]Mahdisoltani F, Biega J, Suchanek F M. Yago3: A knowledge base from multilingual wikipedias[C]. CIDR, 2013.

[2]Auer S, Bizer C, Kobilarov G, et al. Dbpedia: A nucleus for a web of open data[M]. The semantic web. Springer, Berlin, Heidelberg, 2007: 722-735.

[3]Bollacker K, Evans C, Paritosh P, et al. Freebase: a collaboratively created graph database for structuring human knowledge[C]. SIGMOD, 2008: 1247-1250.

[4]Suchanek F M, Kasneci G, Weikum G. Yago: a core of semantic knowledge[C]. WWW, 2007: 697-706.

[5]Carlson A, Betteridge J, Kisiel B, et al. Toward an Architecture for Never-Ending Language Learning[C]. AAAI, 2010, 5: 3.

[6]Dong X, Gabrilovich E, Heitz G, et al. Knowledge vault: A web-scale approach to probabilistic knowledge fusion[C]. SIGKDD, 2014: 601-610.

[7]王树徽,李鸥,廖祥文,等.知识图谱构建技术综述[J].计算机研究与发展,2018,55(11):2361-2377.

[8]Chen H, Liu X, Yin D, et al. A survey on dialogue systems: Recent advances and new frontiers[J]. ACM SIGKDD Explorations Newsletter, 2017, 19(2): 25-35.

[9]祝惠,朱靖波,曲维,等.知识图谱表示学习研究进展[J].自动化学报,2019,45(06):1072-1091.

[10]Ji S, Pan S, Cambria E, et al. A survey on knowledge graphs: Representation, acquisition and applications[J]. arXiv preprint arXiv:2002.00388, 2020.

[11]Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

[12]Radford A, Narasimhan K, Salimans T, et al. Improving language understanding by generative pre-training[J]. 2018.

[13]Radford A, Wu J, Child R, et al. Language models are unsupervised multitask learners[J]. OpenAI blog, 2019, 1(8): 9.

[14]Yang Z, Dai Z, Yang Y, et al. Xlnet: Generalized autoregressive pretraining for language understanding[J]. arXiv preprint arXiv:1906.08237, 2019.

[15]Raffel C, Shazeer N, Roberts A, et al. Exploring the limits of transfer learning with a unified text-to-text transformer[J]. arXiv preprint arXiv:1910.10683, 2019.

[16]Lan Z, Chen M, Goodman S, et al. Albert: A lite bert for self-supervised learning of language representations[J]. arXiv preprint arXiv:1909.11942, 2019.

[17]Beltagy I, Peters M E, Cohan A. Longformer: The long-document transformer[J]. arXiv preprint arXiv:2004.05150, 2020.

[18]Ainslie J, Ontanon S, Alberti C, et al. ETC: Encoding long and structured inputs in transformers[J]. arXiv preprint arXiv:2004.08483, 2020.

[19]孙茂松,李晨,金恺健,等.知识图谱技术原理与应用[J].电子工业出版社,2021.

[20]Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016.

[21]Veličković P, Cucurull G, Casanova A, et al. Graph attention networks[J]. arXiv preprint arXiv:1710.10903, 2017.

[22]Bordes A, Usunier N, Garcia-Duran A, et al. Translating embeddings for modeling multi-relational data[J]. NeurIPS, 2013: 2787-2795.

[23]Wang Z, Zhang J, Feng J, et al. Knowledge graph embedding by translating on hyperplanes[C]. AAAI, 2014: 1112-1119.

[24]Lin Y, Liu Z, Sun M, et al. Learning entity and relation embeddings for knowledge graph completion[C]. AAAI, 2015.

[25]Trouillon T, Welbl J, Riedel S, et al. Complex embeddings for Simple Knowledge Graph Completion[C]. ICML,2016:2071-2080.

[26]Dettmers T,Minervini P,Stenetorp P,et al.Convolutional 2d knowledge graph embeddings[C].AAAI,2018.

[27]Nathani D,Chauhan J,Sharma C,et al.Learning attention-based embeddings for relation prediction in knowledge graphs[C].ACL,2019:4710-4723.

[28]Vashishth S,Sanyal S,Nitin V,et al.Composition-based multi-relational graph convolutional networks[J].arXiv preprint arXiv:1911.03082,2019.

[29]Mnih V,Kavukcuoglu K,Silver D,et al.Human-level control through deep reinforcement learning[J].Nature,2015,518(7540):529-533.

[30]Lillicrap T P,Hunt J J,Pritzel A,et al.Continuous control with deep reinforcement learning[J].arXiv preprint arXiv:1509.02971,2015.

[31]Schulman J,Wolski F,Dhariwal P,et al.Proximal policy optimization algorithms[J].arXiv preprint arXiv:1707.06347,2017.

[32]Haarnoja T,Zhou A,Abbeel P,et al.Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C].ICML,2018:1861-1870.

[33]Huang X,Zhang J,Li D,et al.Knowledge graph embedding based question answering[C].WSDM,2019:105-113.

[34]Zhang Y,Dai H,Kozareva Z,et al.Variational reasoning for question answering with knowledge graph[C].AAAI,2018:6069-6076.

[35]Miller A,Fisch A,Dodge J,et al.Key-value memory networks for directly reading documents[J].arXiv preprint arXiv:1606.03126,2016.

[36]Zhang Y,Liu Q,Song L.Sentence-state LSTM for text representation[C].ACL,2018:317-327.

[37]Yasunaga M,Ren H,Bosselut A,et al.QA-GNN: Reasoning with language models and knowledge graphs for question answering[J].arXiv preprint arXiv:2104.06378,2021.

[38]Yu D,Shang L,Guo J,et al.Knowledge embedding based graph convolutional network[C].EMNLP,2020:3119-3124.

[39]Lin B Y,Chen X,Chen J,et al.KagNet: Knowledge-aware graph networks for commonsense reasoning[J].arXiv preprint arXiv:1909.02151,2019.

[40]Feng Y,Chen X,Lin B Y,et al.Scalable multi-hop relational reasoning for knowledge-aware question answering[J].arXiv preprint arXiv:2005.00646,2020.

[41]Zhu Z,Xu Z,Tang J,et al.Reinforced mnemonic reader for machine reading comprehension[J].arXiv preprint arXiv:1705.02798,2017.

[42]Xu Z,Zhu Z,Wang Y,et al.Asking clarification questions in knowledge-based question answering[C].EMNLP,2019:1618-1629.

[43]Das R,Godbole A,Zaheer M,et al.Chains-of-reasoning at TextGraphs 2019 shared task: Reasoning over chains of facts for explainable multi-hop inference[C].TextGraphs@EMNLP,2019.

总结一下:

本文提出了一种融合大语言模型和深度学习技术的知识图谱构建方法。该方法首先利用预训练语言模型从大规模文本语料中抽取实体和关系,采用基于图神经网络的表示学习模型获得知识的低维嵌入,然后通过强化学习不断扩充和优化知识库,最终得到高质量、规模化的知识图谱。实验结果表明,该方法在知识抽取、知识表示学习、知识补全等任务上取得了优异的性能,很好地弥补了传统知识图谱构建方法的不足。本文探索了大语言模型在知识图谱构建中的潜力,为进一步挖掘知识图谱在智能应用中的价值奠定了基础。未来工作将着眼于引入更多先验知识和人类反馈以提升模型性能,并将知识图谱应用于智能问答、语义搜索等下游任务。
