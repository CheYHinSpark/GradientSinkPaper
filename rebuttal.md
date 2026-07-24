# Review and Rebuttal

## Official Review of Submission17495 by Reviewer gq1u

**Summary:**

This paper studies the phenomena of attention sinks and large activations in large pretrained transformer-based LLMs, through the lens of backpropagation. Specifically the authors identify large gradient concentrations induced by attention sinks (gradient sinks) and their relationship to massive activations.

The paper presents an empirical and theoretical study of the relationship between these three phenomena, by demonstrating that massive activations act as local adaptive regulators of the gradient sinks, thus preserving training stability.

This theoretical analysis leads to the derivation of V-scale, a reparameterization of attention values that adaptively scales attention values, by preserving attention sinks, while attenuating large gradients that correspond to small values.

Evaluation shows comparable performance on downstream tasks for 1B LLMs trained with V-scale (no quantization), while demonstrating superior performance in the Needle-in-the-haystack retrieval setting under quantization.

**Strengths**

- The analysis of training dynamics and their relationship to the phenomena of attention sinks and massive activations can provide insights into LLM operation.
- The theoretical and empirical setting is realistic, taking into account modern transformer architecture design (Prenorm, RoPE, Billion-scale models).
- The paper is well-written and intuition is clearly presented.

**Weaknesses**

- The proposed V-scale reparameterization leads to comparable performance on downstream LM tasks, with some benchmarks improving and some benchmarks showing reduced performance. Since in Section 3.2 the papers states that gradient sinks do not impact training stability, and the baseline transformer architecture already regulates them, is this intervention necessary, or is this a solution asking for a problem?
    - This point is very relevant in the reviewer's opinion, since the proposed V-scale modification is applied at the pre-training stage, thus verification is costly and requires strong evidence for potential benefits.
- The most promising benefit of this modification is the quantization aspect. Massive activations can hurt quantization performance, thus their mitigation may help quantized model performance. This is a direct claim made in the paper (L214). However this claim is not verified in Table 4, appendix C.4, and direct comparison for this claim is pushed to the Appendix.
- Regarding the evaluation for the Needle-in-the-haystack problem improvements are more clear, however looking again in the appendix C.4 there are some settings where V-scale is catastrophic (e.g., AWQ / GPTQ quantization, C.4 Table 5 and 6)

Overall, the analysis of gradient sink phenomena in this paper is a valuable contribution, however the proposed V-scale approach and subsequent evaluation brings into question the benefits of this intervention. The reviewer is hesitant to raise concerns about results presented in the Appendix, however I believe they are directly related to claims made in the main paper text.

**Rating:** 3.

**Confidence**: 3.

**Paper Formatting Concerns:** Please format the best results in each table in bold. This will significantly improve readability.

## Reply to Reviewer gq1u

首先是感谢的套话

**Reply to Weakness 1:**

回复策略：
明确文章定位。论文的核心是机制和原理发现与分析，而不是提出新的SOTA。我们不声称V-scale是普遍更优的架构。事实上我们也没有和其他新兴架构对比。

指出逻辑漏洞。审稿人的核心关注是：如果标准架构中MA已经可以消除梯度sink，不影响训练稳定，那么没有必须消除MA。回复：是的，您说得对。然而您的质疑恰恰建立在本文的核心贡献之上：即sink token巨大激活在反向传播中的作用。To our knowledge，这在此前并不是周知的。我们并没有宣称巨大激活影响了训练稳定。

讨论潜在价值。虽然，我们的结果可能对未来研究产生积极影响。我们揭示了梯度的内在情况，为后续架构设计提供了参照。

**Reply to Weakness 2:**

您说得对。巨大激活，或者更准确的说，激活的不均衡，是量化的头号敌人。但是具体量化方法涉及到许多细节。

首先，我们在论文中其实没有直接宣称V-scale模型相对标准结构更加能够抵抗量化。

量化这个事情没有看起来那么简单直接，很多量化是W4A16，即只是量化了权重而没有量化激活值，因此这些量化方法下不必然有什么优势

其次，即便是SmoothQuant这些W8A8量化，包括AWQ这些，都针对标准模型的激活值不均衡问题做了专门处理，已经处理得很好了。我们的模型反而不一定适应这种处理。 最最重磅的来了，即便是W8A8量化了激活值，事实上，这里的“激活值”只是矩阵乘法前的激活值，例如计算QKV、这些激活值已经经过了RMSNorm，没有那么危险了。计算结果加回残差流的时候已经是16bit。

那么V-scale真的就没有量化收益了吗，其实是有的。我们可以采取更加粗暴的量化配置

参见下面3种

```python
elif method == "w8a8":
    int8_weight_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    }

    int8_act_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    }

    recipe = [
        QuantizationModifier(
            config_groups={
                "linear_w8a8": {
                    "targets": ["Linear"],
                    "weights": int8_weight_args,
                    "input_activations": int8_act_args,
                }
            },
            ignore=["lm_head"],
        )
    ]
elif method == "w8a8_act":
    weight_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    }

    activation_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }

    recipe = [
        QuantizationModifier(
            config_groups={
                "linear_w8a8": {
                    "targets": ["Linear"],
                    "weights": weight_args,
                    "input_activations": activation_args,
                }
            },
            ignore=["lm_head"],
        )
    ]
elif method == "w8a8_stress":
    int8_args = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }
    recipe = [
        QuantizationModifier(
            config_groups={
                "linear_w8a8": {
                    "targets": ["Linear"],
                    "weights": int8_args,
                    "input_activations": int8_args,
                }
            },
            ignore=["lm_head"],
        )
    ]
```

在后两种配置下，baseline会被直接干报废，但是V-scale仍然可以维持一定性能。

但是这是非常粗暴不合理的量化配方，因此我们没有在论文中给出。

**Reply to Weakness 3:**

与前面类似

**Reply to Paper Formatting Concerns:**

感谢建议，如果接受在Camera-Ready版本会改的。

## Official Review of Submission17495 by Reviewer pqQy

**Summary:**

The authors present an empirical and theoretical analysis of the coupling between massive activations and attention sinks from a new perspective, i.e., the backward pass. Specifically, the study shows how attention sinks under causal attention masking lead to gradient concentrations that they term gradient sinks. Then, they provide an interpretation of massive activations as regulators of gradient sinks and verify this via their bespoke intervention V-scale, towards strong reduction in massive activations. While performance improvement is not the goal of the paper, V-scale does add notable improvements in recall performance.

**Strengths**

Notation is clear and easy to understand
The analysis for how massive activations align with changes in gradient is clearly explained and motivates their theoretical investigation for massive activations as gradient regulators.

Theorems are presented concisely yet explained clearly.

Practical takeaway towards V-scale is motivated well and overall, the empirical evidence is consistent with the paper’s claims and the present theory motivates their proposed intervention well. The authors do not overclaim their intervention to provide superior language modelling per-se, yet the efficacy of more control over massive activations in V-scale is verified via NIAH tasks.

**Weaknesses**

I did not identify any major methodological or experimental weaknesses. Evidence is correctly inferred and supports the paper's claims.

**Questions:**

One minor suggestion for strengthening the work would be to include a discussion on the strong improvement in ARC-C and BoolQ, both of which require skills of reasoning-based inference from context. Could there be a coupling of these skills with NIAH that could explain the benefits of V-scale more? Note that this suggestion does not impact my score.

**Rating:** 5.

**Confidence:** 4.

## Reply to Reviewer pqQy

Thank you for this insightful suggestion. We agree that the concurrent improvements on ARC-C, BoolQ, and (multi-key) NIAH merit a joint discussion.

A plausible commonality is query-conditioned selection and routing of relevant information under competing cues. BoolQ requires identifying supporting evidence in a passage; multi-key NIAH requires retrieving the queried association among multiple candidates; and ARC-C, although less directly a contextual retrieval task, requires selecting and combining relevant scientific knowledge while discriminating among competing answer choices. Notably, the results do not show a uniform improvement across all benchmarks. This makes a shared demand for selective information use a plausible, although not yet established, explanation for the stronger gains on these tasks.

Appendix Figure 13 provides a tentative mechanistic connection. It shows the learned V-scale parameters across layers and value heads. Beyond the clear layer-wise pattern, different heads within the same layer learn different V-scale strengths. This heterogeneity is potentially relevant because prior work has identified linguistically specialized attention heads [1], a small set of heads that transport compact task representations (“function vectors”) [2], and sparse retrieval heads central to long-context factuality [3]. Together, these observations motivate the hypothesis that head-dependent value-path gradient modulation may interact with functionally heterogeneous pathways for information selection and routing.

We emphasize that this interpretation remains a hypothesis. Figure 13 does not identify the functions of individual heads, and our experiments do not establish a causal coupling among the benchmark gains. A detailed analysis is an interesting direction for future work. We will add this discussion while clearly presenting it as a hypothesis rather than a conclusion.

[1] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov. _Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned_. ACL 2019.

[2] Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, David Bau. _Function Vectors in Large Language Models_. ICLR 2024.

[3] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu. _Retrieval Head Mechanistically Explains Long-Context Factuality_. ICLR 2025.

## Official Review of Submission17495 by Reviewer 6L5L

**Summary:**

This paper provides a mechanistic account for the co-existence of attention sinks (AS) and massive activations (MA). The author establishes a rather intriguing mental image: the attention sink results in a large gradient norm on the $v$ activation at the sink token position, which entails a large pre-norm residual activation to alleviate the gradient magnitude. Such an account is corroborated with empirical measurements, mathematical modeling, and causal intervention. The V-scaling serves both as a causal intervention to test their hypothesis and as a model add-on with practical value, particularly for quantization.

**Strengths**

(1) The story is quite compelling. The authors provide comprehensive empirical observations and mathematical explanations. Overall, the logic is sound, and it complements the previous understanding of attention sink from the perspective of the forward pass.

(2) The presentation is well-organized and neat. Although the story's complexity makes reading a bit involved, I think I get the gist of it with no problem.

**Weakness**

(1) The paper could use more pictorial illustrations to clarify the main points, which I will get to in the questions.

(2) The causal intervention of V-scale is slightly flawed in logic. I will also discuss this in the questions. Nonetheless, I believe its contribution as a practical model gadget to facilitate quantization still holds up well.

**Questions:**

(1) Regarding the intervention experiments. I think it is not completely rigorous. It is also admitted in the paper. The $\phi$ function modulates not only the backward gradient but also the forward value for small activation tokens. Thus, one might suspect whether it is this effect on the forward pass that mitigates the MA. I would suggest the following experiments. You may choose either one to conduct:

i) Write a hook function to modify only the backward gradient norm of v to see if it is really this factor that affects MA. Do it for either one layer or all the layers. In this case, the training loss might not decrease as quickly, or it might simply result in divergence. But I think it is a good way to see how the input residual norm reacts to changing only the backward gradient norm of v.

ii) Run a PC algorithm (for a conditional independence test) to see if AS and $\|h^l\|$ are independent conditioned on $\|\nabla_{v_s}\mathcal{L}\|$ at the sink token position. This is a non-interventional way to check the causal link you established.

Overall, I am still impressed by the practical value of your current intervention.

(2) I found it a bit hard to follow when I read the definition of Bloat/Change/Compress. I get the ideas behind the naming in the subsequent visualization section, but I was completely lost when I first read these names. Maybe it is better to mention that the intention behind these names will be clarified later. More importantly, I think it will greatly improve the clarity of the presentation if you can have a pictorial illustration of Bloat/Change/Compress. You simply need to create a plot showing the computation flow you defined under line 71, and the gradients of the two points being compared in that flow. For example, Compress, I believe, is the comparison before and after the RMS norm.

(3) I think it is better, again, to use a pictorial graph to summarize the logic you are presenting in Sections 4 and 5. To me, it is AS -> large gradient on v -> calls for large residual activation (MA) to reduce the gradient norm. This will make it much easier to follow.

At this point, I am leaning toward accepting this paper. Though I believe addressing my first question may greatly improve the rigor of the script.

**Rating:** 4.

**Confidence:** 4.

## Reply to Reviewer 6L5L

先说一些感谢的套话

**Reply to Question(1):**

我们接受建议并完成第一个建议的实验。

具体而言我们设置了一个Backward-only 版本的 V-scale：
在计算完v_proj得到V之后

$$
\hat v = \mathrm{stopgrad}(v) + (\phi(v) - \mathrm{stopgrad}(\phi(v))).
$$

其中 stopgrad 通过 PyTorch 的 detach 方法实现。$\phi$表示的是论文中的V-scale公式。这样得到了前向保持V，反向却具有V-scale效应的算子。虽然我们认为使用了detach不是正常的大模型预训练，但这的确可以有效补充因果关系的严谨性。

等待结果（必须要用文本表达）

**Reply to Question(2):**

首先要感谢审稿人，您的理解完全正确。

承诺如果被接受，在Camera-ready版本中我们会补充正式的示意图（注意rebuttal也不允许用链接方式绕过媒介限制），现在rebuttal中我们用markdown文本画一个示意图

**Reply to Question(3):**

再次感谢审稿人的建议。同样承诺如果接受会补充图片或者公式说明。
