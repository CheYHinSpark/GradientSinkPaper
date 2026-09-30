# Attention Sinks Induce Gradient Sinks: Massive Activations as Gradient Regulators in Transformers

Yihong Chen, Zhouchen Lin, Quanming Yao

NeurIPS 2026 poster

**Abstract:**

Attention sinks and massive activations are recurring and closely related phenomena in Transformer models. Existing explanations have largely focused on the forward pass, yet in pre-norm Transformers, large residual-stream norms play only an indirect forward role because sublayers operate on normalized inputs. We study this relationship from the perspective of backpropagation. Empirically and theoretically, we show that under causal masking, attention sinks can induce pronounced gradient concentration, which we term gradient sinks. Since the RMSNorm Jacobian attenuates gradients roughly in inverse proportion to input norm, massive activations can be understood as adaptive regulators of this localized gradient pressure during training. This interpretation predicts that attenuating sink-induced gradients should weaken massive activations. We test this prediction with V-scale, a modification that adjusts backpropagated gradients on the value path. In V-scale models, attention sinks are preserved, whereas massive activations are suppressed. These results identify gradient sinks as a backward-pass counterpart of attention sinks, and massive activations as an adaptive RMSNorm-mediated response that attenuates the resulting localized training pressure. Our code is available at https://anonymous.4open.science/r/GradientSinkCode-B309.

## Paper Decision

**Decision:** Accept (poster)

**Metareview:**

This paper studies the attention-sink and massive-activation phenomena of pretrained Transformer-based LLMs through the lens of backpropagation. The paper shows that under causal masking, attention sinks can induce pronounced gradient concentration, which the authors term gradient sinks, and interpret massive activations as adaptive regulators of that localized gradient pressure. This interpretation predicts that attenuating sink-induced gradients should weaken massive activations, and the authors test that prediction with V-scale, a modification that adjusts the gradients backpropagated along the value path.

- Strengths

The analysis of the relationship between attention sinks and massive activations provides insight into how LLMs train and operate. The claims are well supported by empirical evidence.

- Weaknesses

The reviewers raised several questions regarding the clarification of the V-scale, quantization evidence, and revisions to the figures.

Overall, the initial meta-review is leaning to accept this paper with the anticipation that the authors will successfully address all the questions raised by the reviewers.

**Final Justification:**

The reviewers raised several questions regarding the clarification of the V-scale and the quantization evidence, and requested that the figures be revised. During the author-reviewer discussion phase, all reviewers were satisfied with the rebuttal and maintained their stance in favor of accepting the paper. After reading the reviews, rebuttal, and discussions, the AC recommends accepting this paper.

---

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

**Rating:** 4.

**Confidence**: 3.

**Paper Formatting Concerns:** Please format the best results in each table in bold. This will significantly improve readability.

**Final Justification:**

The authors have sufficiently supported that the main contribution of this paper is the theoretical analysis and agreed to qualify some weakly supported claims about the practical application. I am raising my score.

### Rebuttal by Authors

We thank the reviewer for recognizing the value of the training-dynamics analysis, the realism of the empirical and theoretical setting, and the clarity of the presentation.

The reviewer’s concerns primarily concern the usefulness and evaluation of V-scale in Section 5. Before addressing them individually, we would like to clarify the central novelty and the intended role of that section.

The main contribution of the paper lies in Sections 3–4: we identify gradient sinks as a backward-pass counterpart of attention sinks and provide empirical and theoretical evidence that massive activations act as RMSNorm-mediated regulators of localized gradient pressure. Section 5 is designed primarily as a counterfactual intervention that tests a prediction derived from this mechanism, rather than as a proposal that V-scale is a universally superior practical architecture.

We recognize that the title “Practical Application” and the emphasis on quantization (e.g., L214) may have made the secondary practical role appear to be the primary contribution. We will revise this positioning.

#### Question 1: Given the mixed downstream results and the cost of modifying pretraining, what is the usefulness of V-scale?

We agree that V-scale is not necessary for the stable training of standard Transformers, and we do not recommend it as a general-purpose replacement for existing architectures.

Its primary usefulness in this paper is evidential.

Sections 3–4 establish the following mechanism:

1. attention sinks aggregate gradient signals from later tokens toward the sink position;
2. this creates localized gradient pressure, especially on the value pathway;
3. RMSNorm makes token-wise activation scale an effective local attenuation factor;
4. massive activations can therefore regulate this localized pressure.

This mechanism produces a counterfactual prediction: introducing an alternative value-path gradient valve during optimization should reduce the model’s reliance on massive activations while retaining substantial attention-sink behavior. V-scale is introduced to test this prediction.

The downstream evaluations mainly serve as behavioral controls. Without them, a reduction in massive activations could be explained trivially by failed training, undertraining, or a general loss of language-model capability. The standard LM-eval results show that the intervention remains a normally functioning language model, while NIAH probes whether contextual retrieval remains functional.

We agree that these results do not establish uniform practical superiority. We will therefore revise Section 5 to:

1. rename it “Mechanistic Intervention and Validation”;
2. present V-scale first as a counterfactual test of the mechanism;
3. describe standard downstream tasks as capability controls;
4. separate the mechanistic conclusion from secondary practical observations;
5. explicitly acknowledge mixed downstream behavior.

The training-time intervention is necessary for the scientific question being tested. Our hypothesis concerns why massive activations emerge during optimization. A post-hoc change to an already trained checkpoint cannot test whether relieving gradient pressure during learning changes their emergence. Thus, the matched pretraining runs are necessary for the intervention study, but they should not be interpreted as a recommendation that practitioners retrain existing models with V-scale.

We also conducted a backward-only control in response to another reviewer (see our response to Reviewer 6L5L, Q1). It leaves the forward values unchanged while applying the V-scale gradient rule. This model remains well trained and strongly suppresses massive activations, further strengthening the interpretation of Section 5 as mechanistic validation.

#### Question 2: Does the paper adequately verify a quantization benefit?

Thank you for raising this distinction. We agree that the original wording could be read as making a broader claim than the evidence supports.

The submitted PTQ methods probe different effects. BNB, GPTQ, and AWQ are W4A16 settings and therefore do not quantize runtime activations. Reducing residual-stream massive activations is consequently not expected to yield an automatic improvement under these methods. SmoothQuant is W8A8, but it explicitly compensates for activation outliers and applies activation quantization to Linear inputs rather than directly to the residual-stream sites at which massive activations are measured. These results therefore do not directly isolate sensitivity to the residual-stream activation range.

To test this narrower implication, we conducted a controlled activation-range diagnostic using three symmetric INT8 configurations:

1. **All-Linear dynamic-token W8A8.** All `Linear` modules except lm_head use static per-channel INT8 weights and dynamic per-token INT8 input activations.
2. **All-Linear static-tensor W8A8.** The target modules and weight format are unchanged, while activations use a calibrated static per-tensor INT8 scale with MinMax observers.
3. **`down_proj`-only static-tensor W8A8.** The same static per-channel weight and static per-tensor activation formats are applied only to modules matching `mlp.down_proj`.

Dynamic per-token scaling is a less range-sensitive control. Static per-tensor scaling shares one calibrated range across tokens and therefore deliberately exposes sensitivity to token-wise activation outliers. The localized `down_proj` setting targets the MLP pathway at which we observe the strongest reduction of massive activations.

The results below report Baseline/V-scale pairs. Lower perplexity and higher accuracy are better. Mean accuracy is the unweighted average over the ten accuracy metrics reported in Table 4.

| Setting | WikiText PPL $\downarrow$ | LAMBADA PPL $\downarrow$ | Mean accuracy $\uparrow$ |
| --- | ---: | ---: | ---: |
| BF16 | 22.84/**22.83** | **15.27**/15.91 | 51.62/**52.23** |
| All Linear, dynamic-token W8A8 | 23.16/**23.04** | **16.35**/16.90 | 51.77/**52.21** |
| All Linear, static-tensor W8A8 | 33.55/**25.46** | 46.85/**23.65** | 48.54/**50.55** |
| `down_proj` only, static-tensor W8A8 | 32.94/**25.21** | 43.63/**23.07** | 48.37/**50.67** |

Dynamic per-token W8A8 leaves both models close to BF16. When only the activation-scaling policy is changed to calibrated static per-tensor scaling, V-scale exhibits a substantial advantage. Quantizing only `down_proj` nearly reproduces the all-Linear static result, showing that this pathway is sufficient to induce most of the observed sensitivity.

We emphasize the scope of this evidence. Static per-tensor quantization is intentionally a controlled range-sensitivity diagnostic, not a competitive deployment recipe. These results support the conditional conclusion that V-scale is less sensitive when a coarse activation range is applied to the MLP pathway associated with its strongest reduction of massive activations. They do not establish universal quantization robustness.

We will revise the wording in Section 5 to describe quantization as a potential and conditional implication rather than a general benefit.

#### Question 3: How should the severe degradations in some AWQ/GPTQ and NIAH settings be interpreted?

We agree that the degradations highlighted by the reviewer are substantial. We do not regard them as noise, and we do not claim that V-scale is uniformly compatible with every quantization method, retrieval task, and context-extension setting.

V-scale improves every reported multi-key setting. In contrast, the harder single-needle settings are mixed and include several unfavorable AWQ/GPTQ/YaRN cases. We currently do not have sufficient evidence to identify the cause of these interactions.

To test activation-range sensitivity more directly, we evaluated the same W8A8 configurations on all three NIAH variants at the native 2048-token context length (baseline/V-scale pair, larger is better):

| Setting | Single-2 | Single-3 | Multi-key |
| --- | ---: | ---: | ---: |
| BF16 | **98.00**/97.80 | 85.48/**95.60** | 65.04/**71.64** |
| All Linear, dynamic-token W8A8 | 96.04/**96.28** | 77.00/**91.40** | 63.44/**66.88** |
| All Linear, static-tensor W8A8 | 1.52/**79.40** | 0.00/**75.32** | 2.76/**56.04** |
| `down_proj` only, static-tensor W8A8 | 2.16/**79.28** | 0.04/**79.08** | 3.40/**56.84** |

Dynamic per-token W8A8 does not cause either model to collapse. Under static per-tensor activation quantization, however, the baseline collapses to near-zero accuracy on all three tasks, whereas V-scale remains functional. Applying the same static quantization only to `down_proj` nearly reproduces this contrast. This demonstrates a qualitative difference in fragility to a fixed activation range at the mechanistically implicated MLP pathway.

These new activation-quantization results do not erase or explain the unfavorable W4A16/YaRN cases. We agree that highlighting only the uniformly positive multi-key result in the main text may have made the practical observation appear broader than intended, even though the appendix reports the mixed single-needle outcomes.

We will therefore:

1. admit the mixed single-needle results in the main text;
2. avoid claiming uniform improvement across quantization methods or long-context tasks;
3. distinguish activation-range sensitivity from general PTQ performance;
4. state explicitly that the W4A16/YaRN interaction remains unresolved.

These mixed downstream findings constrain the generality of a secondary practical observation, but do not challenge the core empirical and theoretical contribution in Sections 3–4 or the evidential role of Section 5 as a counterfactual intervention.

#### Formatting concern

We agree with the reviewer’s suggestion and will bold the best result in each table.

We thank the reviewer again for prompting us to clarify the contribution hierarchy, reposition Section 5, add a more direct activation-range diagnostic, and narrow the practical claims.

### Official Comment by Reviewer gq1u

I thank the authors for their clarifications and agree about the main contribution of this paper is the theoretical analysis. I hope the qualifications about the practical claims are included in the camera ready. I am inclined to raise my score.

### Replying to Official Comment by Reviewer gq1u

Thank you very much for considering our response and updating your score. We will ensure that the practical claims are appropriately qualified in the revision.

---

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

**Final Justification:**

I maintain my score of 5, following the authors thorough answer to the question I had raised.

### Rebuttal by Authors

We thank the reviewer for the positive assessment and for the insightful suggestion concerning ARC-C, BoolQ, and multi-key NIAH.

#### Question 1: Could the gains on ARC-C and BoolQ be connected to the improvements on multi-key NIAH?

A plausible commonality is query-conditioned selection and routing of relevant information under competing cues. Indeed, V-scale exhibit stronger gains on tasks where selective routing or retrieval may be particularly important, rather than improve all benchmarks uniformly.

BoolQ requires locating evidence relevant to a question within a passage. Multi-key NIAH requires retrieving the queried association while ignoring several competing key-value pairs. ARC-C is less directly a contextual-retrieval task, but it similarly requires selecting and combining relevant scientific information while discriminating among plausible answer choices.

Selective information use under competing cues may therefore be a shared demand across these tasks.

The learned V-scale parameters also vary across layers and heads (see Figure 13 in Appendix), suggesting that the intervention does not act uniformly on all attention pathways. This heterogeneity may interact with functionally specialized pathways for information selection, transport, or retrieval.

However, we emphasize that this remains a hypothesis rather than an established conclusion. Our current experiments do not identify the functions of individual heads or establish a causal relationship between the benchmark gains.

We will add a concise discussion of this possible connection while clearly presenting it as a hypothesis and a direction for future mechanistic analysis.

We thank the reviewer again for suggesting this useful interpretation.

### Official Comment by Reviewer pqQy

I thank the authors for their detailed answer and their careful consideration of my question while remaining fixed in the scope of the paper's results.

### Replying to Official Comment by Reviewer pqQy

Thank you very much for engaging in the discussion. We will include this possible interpretation in the revision.

---

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

**Rating:** 5.

**Confidence:** 4.

**Final Justification:**

The authors adequately addressed my concerns. I particularly value their new causal experiments to eliminate the forward-pass confounder. I suggest Accept for this paper.

### Rebuttal by Authors

We thank the reviewer for the careful reading, encouraging assessment, and concrete suggestions for strengthening both the causal evidence and the presentation. We answer each question explicitly below.

#### Question 1: Since V-scale changes both the backward gradient and the forward value states, how can the paper establish that backward gradient modulation is responsible for suppressing massive activations?

We agree that the original full V-scale intervention contains a forward-pass confound. To directly address it, we followed the reviewer’s suggested option (i) and trained a model with a backward-only version of V-scale.

After computing the value projection $v$, we define

$$
\hat v
=
\operatorname{stopgrad}(v)
+
\left[
\phi(\|v\|_2^2)v
-
\operatorname{stopgrad}\!\left(\phi(\|v\|_2^2)v\right)
\right].
$$

Here, `stopgrad` is implemented using PyTorch’s `detach`, and $\phi$ is the V-scale function defined in the paper.

In the forward pass, the two $\phi(\|v\|_2^2)v$ terms cancel exactly, so

$$
\hat v=v.
$$

The attention computation therefore receives exactly the original value state, with no forward contraction.

In the backward pass, the stop-gradient terms contribute no derivative, so the V-scale Jacobian is applied to the value-path gradient. We froze $\theta_{\ell,h}=0$ throughout training. This is a deliberate diagnostic choice because, without freezing, the surrogate construction could produce gradients for $\theta_{\ell,h}$ even though the forward value is independent of it.

This construction imposes a surrogate backward rule by using `detach` rather than the ordinary derivative of its forward map. We therefore do not propose it as a practical model architecture. It is used solely as a diagnostic intervention that removes the direct forward modification.

We trained the backward-only model under the same 0.3B configuration and evaluated all three models at checkpoint 20,000. We use the same AS and MA definitions and evaluation protocol as in Figure 7 of the paper. The table reports the mean and maximum of each layer-wise statistic. `t0/early` is the token-0 norm divided by the mean over positions 1–15.

| Model | Valid loss | AS mean (max) | $x_{\mathrm{out}}$ t0 mean (max) | $x_{\mathrm{out}}$ t0/early | MLP t0 mean (max) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2.9168 | 0.6172 (0.9844) | 1915.9839 (2852.1907) | 11.1214 | 346.3468 (2332.6074) |
| Full V-scale | 2.9068 | 0.5018 (0.9976) | 345.5052 (617.2366) | 3.0300 | 75.4235 (452.0810) |
| Backward-only | 2.9307 | 0.4583 (0.8247) | 606.8524 (951.6826) | 3.6308 | 112.7720 (433.3082) |

The backward-only model remains well trained. Its validation loss is 2.9307, compared with 2.9168 for the baseline, corresponding to a 0.48% relative increase.

Substantial attention-sink behavior also remains, although its mean strength is reduced. At the same time, massive activations are strongly suppressed. Relative to the baseline:

- the mean token-0 residual-stream output norm decreases by 68.3%;
- the maximum token-0 residual-stream output norm decreases by 66.6%;
- the token-0-to-early-token ratio decreases from 11.12 to 3.63;
- the mean token-0 MLP output norm decreases by 67.4%;
- the maximum token-0 MLP output norm decreases by 81.4%.

The qualitative effect closely follows that of full V-scale despite the complete removal of its direct forward transformation.

Therefore, changing only the value-path backward rule is sufficient to produce the predicted suppression of massive activations without causing training failure or eliminating attention-sink behavior. This directly addresses the forward-pass confound identified by the reviewer.

This result does not imply that the forward component of full V-scale has no additional effect. It establishes the necessary conclusion that backward gradient modulation alone is sufficient.

We will add the backward-only construction, its exact forward/backward identities, the comparison table or figure, and the full layer-wise measurements.

#### Question 2: Can Bloat, Change, and Compress be explained more clearly with a pictorial illustration?

We agree. Presenting the names before explaining the computational intuition makes the definitions unnecessarily difficult to follow.

For the attention branch, the forward computation can be summarized as:

```text
h^l --RMSNorm--> \tilde{h}^l --Attention--> r_attn^l
 |                                              |
 +--------------- residual addition -----------+--> h^{l+1/2}
```

Define

- $g_1 = \nabla_{h^{\ell+1/2}}\mathcal L = \nabla_{r^{\mathrm{attn},\ell}}\mathcal L,$ which is the residual-stream gradient after the attention branch;
- $g_2 = \nabla_{\widetilde h^\ell}\mathcal L,$ which is the gradient at the RMSNorm output;
- $g_3 = \nabla_{h^\ell}\mathcal L - \nabla_{h^{\ell+1/2}}\mathcal L = J_{\mathrm{RMSNorm}}(h^\ell)^\top g_2,$ which is the additional gradient contribution transmitted backward through the normalized attention branch.

The three measurements then have the following interpretations:

- $\mathrm{Bloat} = \frac{\|g_2\|}{\|g_1\|},$ which measures branch-local amplification from the branch-output gradient $g_1$ to the normalized-input gradient $g_2$;
- $\mathrm{Change} = \frac{\|g_1+g_3\|}{\|g_1\|},$ which measures the total change in residual-stream gradient norm across the branch;
- $\mathrm{Compress} = \frac{\|g_3\|}{\|g_2\|},$ which measures the norm gain—typically an attenuation factor—when the branch gradient is transmitted backward through RMSNorm.

The definitions for the MLP branch are directly analogous.

We will add a formal schematic that marks $g_1$, $g_2$, and $g_3$ on the computational graph and explain these interpretations before presenting the equations.

#### Question 3: Can the paper include an overview figure summarizing the complete logic of Sections 3–5?

We agree, and the reviewer’s summary captures the central logic well. We will add an overview schematic before the detailed theory and intervention:

```text
attention sink
      |
      v
value-path gradient aggregation
      |
      v
localized gradient pressure at the normalized branch input
      |
      |<---- RMSNorm-mediated compression <---- massive activation
      v
mild change in the residual-stream gradient norm
```

The same figure will show V-scale as an alternative value-path gradient valve:

```text
attention sink
      |
      v
value-path gradient aggregation
      |
      +---- V-scale gradient valve
      |
      v
reduced localized gradient pressure
      |
      v
reduced reliance on massive activations
```

This figure will connect the empirical observations, theoretical results, and intervention in one conceptual map:

1. attention sinks route many later-token gradients toward an early sink token;
2. the value pathway carries the strongest localized gradient concentration;
3. RMSNorm makes activation scale an effective local gradient-compression mechanism;
4. massive activations can therefore emerge as learned regulators of this pressure;
5. an alternative value-path gradient valve reduces the pressure and correspondingly suppresses massive activations.

We will place this overview before the detailed derivations so that readers have the complete conceptual structure before encountering the individual measurements and theorems.

We thank the reviewer again. The suggested backward-only experiment provides a substantially stronger causal control, and the proposed figures will materially improve the accessibility of the paper.

### Official Comment by Reviewer 6L5L

I thank the author for their rather detailed answer. I am quite convinced by the added intervention experiment and believe that it substantially improves the logic soundness of the paper. I will raise my score to 5.

### Replying to Official Comment by Reviewer 6L5L

Thank you very much for your valuable suggestions and for updating your score. We will incorporate the changes into the revision.
