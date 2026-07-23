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
