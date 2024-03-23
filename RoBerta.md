# RoBERTa (Robustly Optimized BERT Pretraining Approach) 是由 Facebook AI 研究团队提出的一种优化 BERT 预训练方法。它在 BERT 的基础上进行了一系列改进,主要包括:

(1) 动态 Masking:与 BERT 采用静态 Mask 不同,RoBERTa 在每个序列的每次训练时动态生成不同的 Mask。这种动态 Masking 策略可以让模型看到更多的数据,从而提高泛化能力。
(2) 去除 Next Sentence Prediction(NSP)任务:实验表明,去除 NSP 任务后,模型在下游任务上的表现反而有所提升。这可能是因为 NSP 任务与很多下游任务关联度不高。
(3) 更大的 batch size:batch size 是指一次训练在等于多少样本数据上计算损失函数,参数更新一次。RoBERTa 采用更大的 batch size(8k),这有助于提升模型训练的稳定性和精度。
(4) 更多的训练数据:RoBERTa 在更大规模数据集上进行训练,包括 16G 的 BookCorpus 数据、38G 的 CC-NEWS 数据等,这进一步扩充了模型学习的知识。
(5) 更长的训练时间:RoBERTa 训练了更长的时间(100 万步),使模型充分学习到训练数据中的知识。
RoBERTa 编码器的结构与 BERT 类似,采用双向 Transformer 编码器。给定输入序列 X={x1,…,xn},首先将其转换为嵌入表示:

H0=[e1,…,en]

其中 ei∈Rdh 为第 i 个 token 的嵌入,由词嵌入、位置嵌入和段嵌入三部分求和得到,dh 为隐藏层维度。
然后,将 H0 输入 L 层堆叠的 Transformer 编码块,第 l 层的计算过程为:

Hl′=LayerNorm(Hl−1+MHAtt(Hl−1))
Hl=LayerNorm(Hl′+FFN(Hl′))

其中 LayerNorm 为层归一化,MHAtt 为多头自注意力模块,FFN 为前馈神经网络。
最后,取最后一层输出 HL 作为整个序列 X 的上下文表示,用于下游的各种 NLP 任务。
为了适应特定任务,RoBERTa 编码器通常还会引入任务相关的提示(如[CLS]、[SEP]等特殊符号),并在编码后接入特定的输出层(如用于序列标注的 CRF 层、用于文本分类的 Softmax 层等)。
需要注意的是,RoBERTa 虽然在多个任务上取得了比 BERT 更好的效果,但其参数量和计算开销也相对更大。在实际应用中,需要根据任务需求和资源限制,权衡模型的性能和效率。
