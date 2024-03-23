# math

\begin{equation}
\mathbf{H} = \mathrm{RoBERTa}(\mathbf{X})
\end{equation}

\begin{equation}
\mathbf{H}^\prime = \mathrm{RoBERTa}(\mathbf{X}^\prime)
\end{equation}

\begin{equation}
P(\mathbf{Y}|\mathbf{X}) = \mathrm{CRF}(\mathbf{H}^\prime)
\end{equation}

\begin{equation}
\mathrm{loss} = -\log P(\mathbf{Y}|\mathbf{X})
\end{equation}

\begin{equation}
P(r|\mathbf{X},e_h,e_t) = \mathrm{Softmax}(\mathrm{MLP}([\mathbf{H}_h^\prime;\mathbf{q}_h;\mathbf{H}_t^\prime;\mathbf{q}_t]))
\end{equation}

\begin{equation}
P(r|\mathbf{X}\_p,e_h,e_t) = \mathrm{Softmax}(\mathrm{MLP}(\mathbf{h}\_p))
\end{equation}

\begin{equation}
P(e*1 \equiv e_2|\mathbf{X}\_1,\mathbf{X}\_2) = \mathrm{Sigmoid}(\mathrm{MLP}([\mathbf{h}*{e*1};\mathbf{h}*{e_2}]))
\end{equation}

\begin{equation}
P((e_h,r,e_t)|\mathbf{X}\_s) = \mathrm{Sigmoid}(\mathrm{MLP}(\mathbf{h}\_s))
\end{equation}

以上是文章中出现的所有数学公式,使用 LaTeX 格式列出。公式(1)和(2)表示利用 RoBERTa 对输入文本 X 进行编码得到语义表示 H 和 H′。公式(3)和(4)表示在语义表示基础上,通过 CRF 序列标注模型识别实体并计算损失。公式(5)和(6)表示实体关系抽取,分别基于句子级特征和语言模型表示预测关系概率。公式(7)表示实体统一,通过实体嵌入相似度预测两个实体指称是否一致。公式(8)表示知识库更新,利用支持句子 Xs 的语义表示 hs 预测待插入三元组的合理性。
