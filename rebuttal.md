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

We thank the reviewer for recognizing the value of our backward-pass analysis, the realism of our setting, and the clarity of the presentation. We also appreciate the concerns regarding the role and evaluation of V-scale, which we address point by point below.

**Reply to Weakness 1:**

We agree that V-scale is not necessary for the stable training of Transformers, and we do not propose it as a universally superior architecture. Its primary role in this work is as a mechanistic test that introduces a transparent gradient valve in the value path.

Section 3.2 indeed shows that, in the trained baseline, severe branch-local gradient amplification produces only mild changes in the residual-stream gradient. Crucially, it also identifies the reason. Namely, the same massive-activation sites exhibit strong RMSNorm-mediated compression. The stable transport of residual-stream gradients in the baseline is therefore the phenomenon explained by our central finding: *massive activations act as learned local regulators of sink-induced gradient pressure*. In other words, the observation that the baseline already regulates this pressure is precisely our main mechanistic claim.

V-scale tests a direct prediction of this account. If massive activations emerge in response to localized gradient pressure, then providing an alternative value-path gradient valve should reduce reliance on massive activations while leaving the attention-sink structure largely intact. This is the predicted pattern we observe. The comparable downstream results serve as a basic capability check, showing that suppressing massive activations does not simply damage ordinary model behavior, rather than as evidence of universal performance superiority.

We also agree that V-scale must be introduced during pretraining and is therefore costly to verify. This training-time intervention is nevertheless necessary for the specific test above because our hypothesis concerns why massive activations emerge during optimization. A post-hoc modification of an already trained checkpoint cannot test whether relieving sink-induced gradient pressure during learning reduces the emergence of massive activations. Thus, the matched from-scratch runs should not be interpreted as a recommendation that practitioners replace existing pretrained models. We will clarify this distinction and state explicitly that the practical observations are secondary to V-scale’s evidential role.

**Reply to Weakness 2:**

Thank you for raising this distinction. We acknowledge that the current presentation (e.g., L214) could be read as making a stronger claim than intended. Our original evidence establishes that V-scale attains higher multi-key retrieval accuracy under the evaluated PTQ settings. It does not establish smaller quantization degradation. We will revise this wording to describe quantization as a potential, conditional implication rather than a universal advantage of V-scale.

The PTQ methods in the submission also probe different effects. BNB, GPTQ, and AWQ are W4A16 settings and therefore do not quantize runtime activations, so reducing massive activations is not expected to automatically improve them. SmoothQuant is a W8A8 method, but it explicitly compensates for activation outliers. Moreover, its INT8 activation quantization is applied to Linear inputs rather than directly to the residual-stream activations where MA is measured. Therefore, the submitted experiments showing downstream behavior under these PTQ methods are not a direct test of sensitivity to the residual-stream activation range.

To test activation-range sensitivity more directly, we added INT8 experiments with three configurations:

- W8A8 with static per-channel weights and dynamic per-token activations over all Linear layers;
- the same per-channel weights with calibrated static per-tensor activations over all Linear layers;
- a localized setting that applies the second W8A8 scheme only to `mlp.down_proj`, leaving all other modules unquantized.

The first two settings use the same weight format and differ only in the activation-scaling policy. Dynamic per-token scaling provides a less range-sensitive control, whereas a calibrated per-tensor scale is shared across tokens and therefore deliberately exposes sensitivity to token-wise activation outliers. The last localized setting quantizes the SwiGLU intermediate passed to the down projection, whose output is the MLP site where we observe the strongest reduction of massive activations.

The results below report Baseline/V-scale pairs on standard LM-eval tasks. Lower perplexity and higher accuracy are better; mean accuracy is the unweighted average over the ten accuracy metrics reported in Table 4, Appendix C.4.

| Setting | WikiText PPL(↓) | LAMBADA PPL(↓) | Mean accuracy(↑) |
| --- | ---: | ---: | ---: |
| BF16 | 22.84/**22.83** | **15.27**/15.91 | 51.62/**52.23** |
| All Linear, dynamic-token W8A8 | 23.16/**23.04** | **16.35**/16.90 | 51.77/**52.21** |
| All Linear, static-tensor W8A8 | 33.55/**25.46** | 46.85/**23.65** | 48.54/**50.55** |
| `down_proj` only, static-tensor W8A8 | 32.94/**25.21** | 43.63/**23.07** | 48.37/**50.67** |

Dynamic per-token W8A8 leaves both models close to BF16. Under all-Linear static-tensor W8A8, however, V-scale exhibits substantial advantages. Quantizing only `down_proj` nearly reproduces the all-Linear static result, showing that this single pathway is sufficient to induce most of the observed sensitivity. The corresponding NIAH results are reported in our reply to Weakness 3.

We emphasize the scope of this evidence. The localized static scheme is a controlled range-sensitivity diagnostic, not a proposed deployment recipe, and it quantizes the input to `down_proj` rather than the residual stream itself. It supports the conditional conclusion that V-scale is less sensitive when coarse activation quantization is applied to the MLP pathway associated with its strongest MA reduction, but it does not establish universal quantization robustness.

**Reply to Weakness 3:**

We agree that the degradations highlighted by the reviewer are substantial. We do not regard these differences as noise, nor do we claim that V-scale is uniformly compatible with every quantization and long-context setting.

These severe cases are nevertheless localized rather than evidence that the quantized V-scale model fails across NIAH. V-scale improves every reported multi-key setting. Thus, these mixed results reveal a task- and context-dependent interaction, not a uniform benefit or failure. We do not currently have evidence that identifies its cause.

We used the same diagnostic W8A8 configurations described in Reply 2 and evaluated all three NIAH variants at the native 2048-token context length (Baseline/V-scale):

| Setting | Single-2 | Single-3 | Multi-key |
| --- | ---: | ---: | ---: |
| BF16 | **98.00**/97.80 | 85.48/**95.60** | 65.04/**71.64** |
| All Linear, dynamic-token W8A8 | 96.04/**96.28** | 77.00/**91.40** | 63.44/**66.88** |
| All Linear, static-tensor W8A8 | 1.52/**79.40** | 0.00/**75.32** | 2.76/**56.04** |
| `down_proj` only, static-tensor W8A8 | 2.16/**79.28** | 0.04/**79.08** | 3.40/**56.84** |

Dynamic per-token W8A8 does not cause either model to collapse. Under calibrated static-tensor activation quantization, however, the baseline collapses to near-zero accuracy on all three tasks, whereas V-scale remains functional. Applying the same static quantization only to `down_proj` reproduces this sharp contrast almost completely. Thus, this is a qualitative difference in fragility to a fixed activation range at the mechanistically implicated MLP pathway.

These activation-quantization results do not erase or explain the unfavorable W4A16/YaRN cases; the two evaluations probe different effects. We agree that presenting only the uniformly positive multi-key result in the main text can make the practical claim appear broader than intended, even though the appendix describes the single-needle results as mixed and reports the negative cases. We will make those limitations visible in the main text and replace the broad statement that V-scale “improves long-context retrieval robustness” with more precise observations that it consistently improves multi-key retrieval in the evaluated settings.

Most importantly, these mixed downstream interactions constrain the generality of a secondary practical observation. They do not contradict the paper's central mechanistic evidence linking attention sinks, gradient sinks, and massive activations, or the role of V-scale as an intervention that tests that mechanism.

**Reply to Paper Formatting Concerns:**

Thank you for the concrete suggestion. We agree that the current tables can be made easier to read. We will bold the best result in each comparison throughout the revised manuscript.

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

We thank the reviewer for the careful reading and encouraging assessment. We especially appreciate the concrete suggestions for isolating the causal mechanism and improving the presentation. We address the questions below.

**Reply to Question(1):**

We appreciate this suggestion and implemented the first proposed intervention: a backward-only version of V-scale. After computing the value projection $v$, we define

$$
\hat v = \mathrm{stopgrad}(v) + (\phi(v) - \mathrm{stopgrad}(\phi(v))).
$$

Here, $\mathrm{stopgrad}$ is implemented with PyTorch's `detach`, and $\phi$ is the V-scale transformation defined in the paper. In the forward pass, the two $\phi(v)$ terms cancel exactly, so $\hat v=v$. In the backward pass, however, $\partial\hat v/\partial v=J_\phi(v)$, yielding the V-scale gradient modulation without directly changing the forward values. This construction therefore isolates the backward intervention from the forward-pass contraction identified by the reviewer. We agree that a `detach`-based operator is a causal diagnostic rather than a conventional architecture for deployment, which is precisely its purpose here.

| condition | Valid loss | AS mean | AS max | x_out t0 mean | x_out t0 max | x_out t0/early | MLP t0 mean | MLP t0 max | MLP t0/early |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2.9168 | 0.6172 | 0.9844 | 1915.9839 | 2852.1907 | 11.1214 | 346.3468 | 2332.6074 | 6.9438 |
| Full V-scale | 2.9068 | 0.5018 | 0.9976 | 345.5052 | 617.2366 | 3.0300 | 75.4235 | 452.0810 | 2.2218 |
| Backward-only | 2.9307 | 0.4583 | 0.8247 | 606.8524 | 951.6826 | 3.6308 | 112.7720 | 433.3082 | 2.3357 |

**Reply to Question(2):**

Thank you for pointing this out. We agree and will add more explanation where we introduce these names. We will also add the suggested schematic to illustrate the computation. For the attention branch, the relevant forward computation is

```text
h^l --RMSNorm--> tilde{h}^l --Attention--> r_attn^l
 |                                             |
 +---------------- residual addition ----------+--> h^{l+1/2} = h^l + r_attn^l
```

Let

- $g_1=\nabla_{h^{\ell+1/2}}\mathcal L=\nabla_{r^{\mathrm{attn},\ell}}\mathcal L$ be the residual-stream gradient after the attention branch,
- $g_2=\nabla_{\widetilde h^\ell}\mathcal L$ be the gradient at the RMSNorm output,
- $g_3=\nabla_{h^\ell}\mathcal L-\nabla_{h^{\ell+1/2}}\mathcal L=J_{\mathrm{RMSNorm}}(h^\ell)^\top g_2$ be the contribution transmitted back through the normalized branch.

Accordingly,

- $\mathrm{Bloat}=\|g_2\|/\|g_1\|$ measures branch-local amplification from $g_1$ to $g_2$,
- $\mathrm{Change}=\|g_1+g_3\|/\|g_1\|$ measures the total change in residual-stream gradient norm across the branch,
- $\mathrm{Compress}=\|g_3\|/\|g_2\|$ measures the norm gain when the branch gradient is backpropagated through RMSNorm.

The MLP definitions are exactly analogous. We agree that presenting the names before this computational intuition makes them unnecessarily difficult to follow. We will introduce a formal schematic highlighting the compared gradient sites.

**Reply to Question(3):**

Thanks again for your kind suggestion, and the summary captures the logic. We agree and will add an overview schematic connecting Sections 3, 4 and 5 along the following chain:

```text
attention sink
    |
    |   (<-- alternative gradient valve)
    |
gradient aggregation
    |
    |<-- RMSNorm compression <-- massive activation
    |
mild change in residual-stream gradient norm
```

The same schematic will also show how V-scale works as an alternative value-path gradient valve. By weakening sink-induced value-path gradient pressure, V-scale produces the predicted reduction in reliance on massive activations. We will place a formal overview before the detailed analysis and intervention so that the reader has the complete conceptual map before encountering the individual derivations and measurements.
