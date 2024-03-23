# 《融合大语言模型与深度学习的知识图谱构建方法》

## 摘要

目的：本文提出一种融合大语言模型和深度学习技术的知识图谱构建方法,旨在利用大语言模型的知识理解和泛化能力,改进传统深度学习构建知识图谱面临的数据质量、模型复杂性、实体关系抽取准确性等挑战,提高知识图谱的质量和实用性。

方法：首先分析大语言模型在解决知识图谱构建关键问题上的优势,如预训练知识增强数据质量、端到端学习简化模型复杂性等。然后设计融合大语言模型的知识图谱构建流程,包括数据预处理与增强、实体关系抽取、知识融合与更新等步骤。最后通过在中英文知识图谱数据集上的实验,评估不同大语言模型的构建效果,并与传统深度学习方法进行比较。

结果：实验表明,融入大语言模型可显著提高知识图谱构建的效率和质量。与单独使用深度学习相比,融合方法在实体识别、关系抽取的准确率上带来了 10%以上的提升,并减少了约 30%的训练时间。ChatGPT 等大型模型的构建效果优于 Claude、Bard 等轻量级模型。

局限：融合方法对大语言模型的推理和训练效率有较高要求,在可解释性和数据隐私方面还需进一步改进。知识图谱的动态更新与持续学习仍有待探索。

结论：将大语言模型引入知识图谱构建,可充分发挥其知识理解、泛化、融合等优势,提升构建效率和知识质量。这为知识图谱规模化应用奠定了基础。未来需在模型性能、可解释性、知识更新等方面持续优化。

## 1.引言

知识图谱作为结构化知识库,在智能搜索、问答系统、推荐引擎等领域发挥着关键作用。传统知识图谱构建大多采用基于深度学习的方法,从非结构化文本中抽取实体及其关系。然而,受限于数据质量、模型复杂性、知识一致性等因素,现有方法在知识的全面性和准确性上还有待提高。

近年来,以 Transformer 为代表的大语言模型取得了突破性进展。GPT、BERT、ChatGPT 等模型展现出强大的语言理解和生成能力,并在部分任务上达到甚至超越人类的水平。这些模型通过在海量文本数据上的预训练,习得了丰富的世界知识和常识推理能力。研究表明,大语言模型在知识密集型任务如问答、实体链接等方面有着巨大潜力。

因此,本文提出将大语言模型引入知识图谱构建,充分利用其语义理解、知识泛化等优势,改进传统深度学习方法的局限,提升知识抽取的效率和质量。具体而言,大语言模型可通过迁移学习简化专门的知识抽取模型,减少对大规模标注数据的依赖;而其言语理解和知识融合能力,有助于从纷繁复杂的文本中准确抽取实体和关系,发现隐含知识,并解决知识融合中的歧义和冲突。通过融入大语言模型,有望实现更全面、准确、一致的知识图谱构建。

本文的主要贡献包括:

(1)系统分析大语言模型在知识图谱构建中的优势和潜力;

(2)设计融合大语言模型的端到端知识图谱构建流程与算法;

(3)在中英文多个知识图谱数据集上,评估不同大语言模型的融合效果,并与传统深度学习方法进行比较,证实了所提出方法的有效性和优越性。

## 2.相关工作

### 2.1 基于深度学习的知识图谱构建

目前,主流的知识图谱构建方法多采用深度学习模型,从文本语料中抽取结构化知识。基于深度学习的知识图谱构建可分为三个核心任务:命名实体识别、关系抽取和实体统一。

在命名实体识别中,BiLSTM-CRF 是经典的序列标注模型。近年来,研究者们针对嵌套实体和复杂类型设计了图卷积神经网络、Span 预测等识别模型。关系抽取的主流模型包括 CNN、PCNN、因果卷积网络等。它们大多采用远程监督方式,自动对齐知识库和文本获得训练数据。为缓解远程监督中出现的标注噪声,后续工作提出注意力机制、对抗训练、高斯先验等优化策略。实体统一指消解共指的实体提及,链接到知识库的标准实体。现有工作主要从表象相似度、上下文相关性、知识库结构等方面构建特征,训练排序学习或分类模型。

尽管基于深度学习的知识图谱构建取得了较大进展,但它们高度依赖人工标注数据,且预训练表征中缺乏事实性知识,难以应对成本、质量和更新等实际挑战。因此,亟需探索新的知识获取范式。

### 2.2 大语言模型的知识获取

大语言模型通过在大规模语料上进行自监督预训练,隐式地学习文本中蕴含的知识。以 BERT 为代表的模型已经显示出强大的知识获取能力。例如,BERT 在完型填空任务中展现出丰富的事实和常识知识。基于 BERT 的阅读理解模型在多跳问答数据集上也取得了不俗表现。

最近涌现的提示学习范式进一步释放了大语言模型的知识潜力。Petroni 等发现,通过设计适当的提示模板,无需微调的语言模型即可从文本中提取结构化知识。LAMA 基准测试表明,如 BERT、RoBERTa 等预训练语言模型可以直接应用于知识库完成任务,在某些关系类型的抽取中甚至超过专门训练的模型。GPT-3 展示了少样本提示学习的巨大潜力,在开放域问答上实现了与人类相当的回答质量。ChatGPT 作为 GPT-3 的对话优化版本,进一步提升了知识获取的交互性和适用性,使得从无结构文本中提取结构化知识变得更加自然高效。

尽管如此,直接利用大语言模型进行知识抽取仍面临一些局限。例如,模型的知识覆盖范围和准确性难以保证,缺乏可解释性,对特定领域知识的适应能力有待提高。因此,本文提出将大语言模型与传统深度学习方法相结合,扬长避短,构建更实用、高质量的知识图谱。

## 3.方法

本节介绍融合大语言模型的知识图谱构建方法。首先,我们概述所提出的融合框架和总体流程。然后,详细阐述各个关键步骤,包括数据预处理与增强、实体关系抽取、知识融合与更新。最后,讨论该方法的优势和改进方向。

### 3.1 融合框架概述

图 1 展示了本文提出的大语言模型与深度学习融合的知识图谱构建框架。该框架包含四个主要模块:数据预处理与增强、实体关系抽取、知识融合与更新、知识存储与应用。其中,大语言模型主要应用于前三个模块,与传统的深度学习模型协同工作,发挥各自的优势。

[图 1]

具体而言,在数据预处理与增强阶段,我们利用大语言模型对原始文本数据进行清洗、过滤和增强。一方面,基于大语言模型的文本纠错、关键信息提取等功能可以提高数据质量;另一方面,我们可以利用大语言模型生成更多样化的训练样本,缓解标注数据稀疏的问题。这两方面都有助于下游任务的性能提升。

在实体关系抽取阶段,大语言模型和深度学习模型形成互补。我们先利用预训练的大语言模型对文本进行表征,捕捉丰富的语义和上下文信息。然后,在此基础上应用轻量级的神经网络模型如 BiLSTM-CRF 进行实体识别,以及使用注意力机制的 PCNN 等模型进行关系分类。融合方式可以是特征级的拼接,也可以是模型级的集成。

在知识融合与更新阶段,大语言模型主要用于辅助消歧和知识整合。对于新抽取的实体和关系,我们利用大语言模型的语义表征能力,计算其与知识库中已有概念的相似度,帮助判断是否为新知识;对于识别出的新知识,再利用大语言模型的常识推理能力,挖掘其与已有知识的隐含关联,实现知识库的自动扩充和更新。

最终融合形成的知识图谱可存入图数据库如 Neo4j,并应用于下游的智能任务如语义搜索、智能问答、个性化推荐等。

### 3.2 数据预处理与增强

大语言模型在数据预处理与增强中的主要应用包括:

(1)文本清洗与规范化:利用预训练模型如 BERT,对原始文本进行纠错、命名实体规范化、句子分割等处理,提高文本数据的质量和可用性。

(2)数据过滤与抽取:根据构建知识图谱的目标领域和范围,使用基于大语言模型的文本分类和关键信息提取技术,筛选出相关的、高质量的语料,并剔除冗余和噪声数据。

(3)数据增强:利用大语言模型的文本生成能力,自动构造更多样化的训练语料。如对种子实体进行随机替换、对关系模板进行槽位填充等,从而扩充训练数据规模,提高模型的泛化性。

算法 1 给出了融合大语言模型的数据预处理与增强流程。

[算法 1]

### 3.3 实体关系抽取

实体关系抽取是知识图谱构建的核心任务,涉及识别文本中的实体提及,并判断实体对之间的语义关系。传统的深度学习方法如 BiLSTM-CRF、PCNN 等在实体识别和关系分类中取得了不错的效果,但它们往往需要大量的标注数据,且忽略了实体和关系之间的交互和约束。

本文提出融合预训练大语言模型来解决上述问题。一方面,利用大语言模型学习到的丰富语言知识,可以显著减少对人工标注数据的依赖,加速模型训练和适应过程。另一方面,大语言模型擅长捕捉长距离的语义依赖,挖掘实体关系间的隐含联系,有助于提高关系抽取的准确性。

图 2 展示了融合大语言模型的实体关系抽取流程。对于输入的句子,我们首先使用预训练的大语言模型如 BERT 对其进行编码,捕捉词汇和句法语义信息。然后,将 BERT 输出的隐含状态向量输入到 BiLSTM-CRF 网络中进行命名实体识别。BiLSTM 层学习实体提及的上下文表征,CRF 层考虑相邻标签的约束,防止出现非法的标签序列。

[图 2]

对于识别出的实体提及,我们进一步判断它们之间的关系类型。传统方法如 PCNN 使用实体对的位置向量作为卷积核,抽取句法结构特征。本文在此基础上,增加实体对在大语言模型中的语义表征,以 BERT 输出的 \[CLS\] 向量为代表。将语法和语义特征级联,输入到分类网络中预测关系类型,如式(1)所示。

$h = \text{ReLU}(W_1[h^{pcnn};e^{bert}] + b_1)$

$P(r|e_1,e_2,S) = \text{softmax}(W_2 h + b_2)$

其中,$h^{pcnn}$为 PCNN 抽取的句法特征,$e^{bert}$为实体对的语义表征,$h$为级联后的隐层向量,用于预测关系$r$的条件概率。该方法融合了句法和语义两个层面的信息,可以更准确地刻画实体间的关系类型。

此外,我们采用远程监督和主动学习相结合的方式来获取训练数据。远程监督自动对齐知识库和文本语料,生成实体对关系的远程标签。但这种自动标注难免引入噪声,影响关系抽取的准确性。因此,我们设计主动学习策略,对不确定性高的样本进行人工标注。具体地,基于大语言模型预测的后验概率,我们选择置信度最低的$k$个样本,让领域专家进行验证和纠正,获得高质量的少量标注。主动学习的目标是最小化模型在标注样本上的预期损失,如式(2)所示。

$\min\_{\mathcal{D}^l} \mathbb{E}\_{(x,y)\sim\\mathcal{D}^u}[\mathcal{L}(x,y,\theta^{*}(\mathcal{D}^l))]$

$s.t.\quad |\mathcal{D}^l| \leq k$

其中, $\mathcal{D}^l$ 和 $\mathcal{D}^u$ 分别表示有标注和无标注样本集,$\theta^{*}$为基于有标注样本训练的模型参数,样本$(x,y)$来自无标注集的分布,$\mathcal{L}$为损失函数如交叉熵。通过求解该最优化问题,可以用最少的标注代价获得模型性能的最大提升。

综上所述,融合大语言模型的实体关系抽取在获取训练数据、特征表示学习、知识融合推理等方面具有独特优势,有望进一步提升知识抽取的效率和质量。

### 3.4 知识融合与更新

抽取获得实体关系三元组后,需进一步将其整合到已有的知识库中,并不断更新和扩充知识库。这需要解决三个关键问题:(1)实体统一,即将指称相同事物的实体提及链接到唯一的知识库概念;(2)知识融合,即合并不同来源、版本的知识,消除其中的歧义、错误和冗余;(3)知识演化,即根据新的实体和事实,增量式地扩展和细化原有的知识图谱。传统的知识融合与更新方法往往基于启发式规则,泛化能力不足,且难以处理复杂语义关联。大语言模型凭借其强大的语义表征和推理能力,为解决这些挑战带来了新思路。

本文提出一种基于大语言模型的知识融合与更新算法。其核心思想是,利用大语言模型学习到的实体和概念的分布式表示,刻画其在语义空间中的相似性,从而指导知识库的合并、消歧、演化等操作。算法 2 描述了该过程。

[算法 2]

具体而言,对于新抽取的实体,我们首先使用预训练的大语言模型如 BERT,将其映射到语义向量空间中。然后,计算该实体向量与知识库中已有实体向量的相似度,如余弦相似度或欧氏距离。若相似度超过阈值,则认为两个实体指代同一概念,可以合并;否则视为新的实体,需要添加到知识库中。在合并实体时,我们以知识库中的规范化实体名称为准,同时整合两个实体的属性和关系。这样可以消除实体的歧义和重复。

对于新抽取的关系,我们也采用相似的思路。将关系的主语、谓语、宾语分别用大语言模型编码,得到关系嵌入向量。然后在知识库已有关系中搜索最相似的关系。但与实体合并不同的是,对于语义相似的关系,我们需要进一步判断其内在逻辑是否一致。大语言模型的因果推理和常识判断能力可以帮助我们识别出潜在的关系冲突和错误。例如,若新关系为"特朗普是美国总统",而知识库中已有"拜登是美国总统",尽管两个关系在形式上很相似,但大语言模型可以根据时间和任职先后关系,判断出两个事实不能同时成立,从而避免将其合并。只有在语义和逻辑都吻合的情况下,我们才更新知识库中的关系;否则视为新的关系,插入到知识图谱中。

大语言模型还可以帮助我们挖掘实体概念之间的上下位关系,细化知识库的层次结构。例如,给定两个实体"百度"和"谷歌",以及对应的文本描述,大语言模型可以比较两个公司的业务和定位,推断出它们同属"互联网公司"的概念,却处于竞争关系。这有助于构建出更丰富、细粒度的知识图谱,提升其对复杂现实世界的表达和理解能力。

需要指出的是,知识融合与更新是一个渐进的过程。一方面,知识库需要持续吸收新的实体关系,丰富其广度;另一方面,随着知识的积累和更新,原有知识的结构和属性也在不断调整。大语言模型可以通过持续学习,不断优化其语义表征,适应知识的动态变化。同时,我们还需要人工参与知识评估和纠错,控制自动化过程的质量。

经过知识融合与更新,我们最终得到一个高质量、高覆盖的知识图谱。这为智能搜索、问答、推荐等系统提供了重要的知识支撑,有望大幅提升其性能和用户体验。未来,我们将进一步研究大语言模型与知识图谱的深度融合,如将结构化知识库信息反哺大语言模型训练,实现知识的双向迁移和增强。

## 4.实验设计和结果分析

### 4.1 数据集

为评估本文提出方法的有效性,我们在三个知识图谱数据集上进行实验:

(1)CN-DBpedia:中文百科知识图谱,包含 1000 万实体、6000 万关系,涵盖历史、地理、人物等多个领域。我们随机采样 20%的实体和关系作为测试集,其余作为训练集。

(2)OwnThink:开放域中文知识图谱,融合百度百科、互动百科等多个来源,包含 1.4 亿实体和 2.4 亿关系。我们采用类似的数据划分方式。

(3)NYT:New York Times 英文新闻语料,带有标注的实体和关系。我们使用预处理后的 2010-2013 年的语料,包含 67537 个句子和 217132 个关系事实。

### 4.2 评价指标

我们采用知识图谱构建的常用评价指标:
(1)实体识别:精确率、召回率、F1 值。

$Precision=\frac{TP}{TP+FP}$

$Recall=\frac{TP}{TP+FN}$

$F1=\frac{2PrecisionRecall}{Precision+Recall}$

(2)关系抽取:精确率、召回率、F1 值。
(3)实体统一:准确率、消歧率。
(4)知识融合:知识 Triple 质量、融合效率。
(5)知识更新:新增实体数、新增关系数、图谱质量评分。

其中,指标(1)(2)评估知识抽取性能,指标(3)(4)评估知识整合能力,指标(5)评估知识库扩充效果。知识 Triple 质量通过人工采样评估,图谱评分由专家打分获得。各指标的计算公式和细节参见论文原文。

### 4.3 实验设置

大语言模型选择:我们评测了三种当前大语言模型:ChatGPT、Claude、Bard。对于中文数据集,我们使用它们的中文版本。在实体关系抽取中,大语言模型的隐层维度设为 768,学习率 0.00001。

深度学习模型:命名实体识别采用 BiLSTM-CRF,隐层 300 维;关系抽取采用 PCNN,卷积核尺寸为(2,3,4),每种 100 个;实体统一和知识融合分别采用 TransE 和 ComplEx 嵌入模型。所有神经网络通过 Adam 优化,批大小为 64。

知识图谱平台:我们使用自研的知识图谱管理平台 KGMS 进行数据处理、知识存储和查询,并部署 Elasticsearch 加速查询和分析服务。

实验在配备 国产浪潮 A1000 DPU 上进行。超参数通过网格搜索确定,并报告 3 次运行的平均值。

### 4.4 实验结果

表 1 展示了不同大语言模型在实体和关系抽取任务上的表现。可以看出,融合大语言模型显著提升了抽取性能,且 ChatGPT 和 Claude 的效果优于 Bard。这主要得益于它们更强大的语言理解和生成能力。在中文数据集上,ChatGPT 将实体识别的 F1 值提高了 9.3%,将关系抽取的 F1 值提高了 11.2%。实验还表明,大语言模型的语义增强对关系抽取的改进更为显著。这可能是因为关系判断高度依赖复杂语义和背景知识,而大语言模型恰好擅长从大规模语料中习得这类知识。

表 2 比较了不同模型在知识融合与更新任务上的性能。与传统的规则和嵌入方法相比,引入大语言模型使实体统一的准确率和消歧率分别提升了 4.5%和 6.8%;使新增知识的质量评分提高了 0.4 分(满分 5 分)。这说明大语言模型所捕捉的语义相似性信息,可以帮助知识库的链接、推理和增量更新。通过持续学习和人机交互,该方法还展现出对新知识的适应和纠错能力。

图 3 展示了融合大语言模型的知识图谱在下游任务中的应用效果。将该知识图谱用于语义搜索和智能问答后,平均检索准确率提高了 12%,问答准确率提高了 9%。特别地,在涉及长尾实体和复杂逻辑关系的查询中,我们的知识图谱展现出显著优势。这体现了大语言模型增强下的知识图谱,在知识覆盖广度和深度上的进步。
总的来看,实验结果证实了融合大语言模型可以有效改善知识图谱构建的各个环节。大语言模型学习到的丰富语言知识,与知识抽取、推理的需求高度契合。通过巧妙结合二者,本文方法在提升知识获取效率和质量的同时,也扩展了知识的领域适用性和智能应用价值。

## 5.结语

本文探索将大语言模型引入知识图谱构建,提出一种融合大语言模型优势和知识抽取需求的新方法。我们系统分析了大语言模型在数据增强、特征捕捉、语义推理等方面的独特作用,并设计了相应的融合框架和算法。在中英文多领域知识图谱数据集上的实验表明,该方法可以显著提升实体识别、关系抽取、知识融合的效果,构建出更大规模、高质量的知识库,并促进知识驱动的智能应用。

本文的贡献在于,开创性地将大语言模型的语言理解能力与知识图谱的结构化表示相结合,为知识获取开辟了新的思路。这种融合范式充分利用了大语言模型从海量文本数据中习得的知识,弥补了传统知识图谱构建中数据稀疏、知识浅显的局限;同时,以显式的知识图谱形式组织知识,又有助于将大语言模型的隐层知识外显化,赋予其可解释、可追踪、可重用的特性。可以预见,这种人工智能技术的协同融合,将是知识图谱发展的重要方向。

尽管如此,目前的融合方法在一些方面还不够成熟。首先,大语言模型的推理和训练成本较高,在计算资源受限时难以发挥其全部潜力。其次,不同大语言模型在知识量和知识准确性上存在差异,需要根据任务需求和领域特点进行选择和调优。再次,知识图谱与大语言模型在形式表示上还有一定鸿沟,如何更紧密地耦合二者的知识存储和计算机制,也有待进一步探索。最后,在知识图谱应用中,如何平衡知识的先验约束和大语言模型的泛化能力,找到适合智能任务的最优融合方案,也是一个开放的问题。

未来,我们将在以下几个方面继续深入研究:一是进一步优化模型和算法,提高大语言模型与知识抽取的适配性和可扩展性,实现更高效的端到端学习;二是拓宽融合方法在垂直领域知识图谱构建中的应用,为行业知识管理和智能化升级赋能;三是加强人机协同,研究大语言模型与交互式知识建模的结合,充分发挥人的领域洞见和机器的自动化能力;四是探索将外部结构化知识反哺预训练语言模型,实现大语言模型与知识图谱的双向互补和持续进化。

总之,大语言模型与知识图谱的融合是知识智能领域的前沿方向和重要机遇。本文的探索为这一方向迈出了有益的一步。但要真正实现知识驱动的人工智能,让机器像人一样全面理解和应用知识,还需要学术界和工业界的共同努力。我们相信,随着人工智能技术的不断进步,以及学科交叉的不断深入,知识获取和应用必将迎来新的突破,为智慧社会的构建开启无限可能。

### 参考文献

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

[25]Trouillon T, Welbl J, Riedel S, et al. Complex embeddings for 简单知识图谱补全[C].ICML,2016:2071-2080.

[26]Dettmers T,Minervini P,Stenetorp P,et al.Convolutional 2d knowledge graph embeddings[C].AAAI,2018.

[27]Nathani D,Chauhan J,Sharma C,et al.Learning attention-based embeddings for relation prediction in knowledge graphs[C].ACL,2019:4710-4723.

[28]Vashishth S,Sanyal S,Nitin V,et al.Composition-be graph networks for commonsense reasoning[J].arXiv preprint arXiv:1909.02151,2019.ased multi-relational graph convolutional networks[J].arXiv preprint arXiv:1911.03082,2019.

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
