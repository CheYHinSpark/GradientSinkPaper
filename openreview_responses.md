# OpenReview Responses

## 1. Overall response to the meta-review

We thank the Area Chair and all reviewers for their careful evaluation and constructive feedback.

We are encouraged by the broad agreement on the paper’s central contribution: identifying gradient sinks as a backward-pass counterpart of attention sinks and providing an empirical and theoretical account of massive activations as RMSNorm-mediated regulators of localized gradient pressure. The remaining questions primarily concern Section 5—namely, the positioning and causal rigor of V-scale, the scope of its quantization evidence, the interpretation of mixed downstream results, and the clarity of the presentation.

We address these questions below.

### Question 1: What is the core novelty of the paper, and what is the role of Section 5?

We would first like to clarify the contribution hierarchy.

The core novelty lies in Sections 3–4:

1. Section 3 empirically identifies gradient sinks and shows that the excess gradient at the sink token is concentrated mainly on the key and, especially, value pathways.
2. Section 3 further shows that massive-activation sites align with strong RMSNorm-mediated gradient compression, allowing severe branch-local amplification to coexist with relatively mild changes in residual-stream gradient norms.
3. Section 4 formalizes this mechanism through exact backward identities and theoretical analysis. Attention weights aggregate value-path gradients toward sink tokens, while the RMSNorm Jacobian makes activation scale an effective local gradient-attenuation factor.

Section 5 is not intended to introduce V-scale primarily as a universally superior practical architecture. Its primary role is to test a counterfactual prediction derived from the mechanism:

> If massive activations emerge partly in response to sink-induced gradient pressure, then providing an alternative value-path gradient valve during training should reduce the model’s reliance on massive activations while retaining substantial attention-sink behavior.

We recognize that the title “Practical Application” and the emphasis on downstream and quantization results may have caused V-scale to be interpreted mainly as a practical method contribution. We do not claim that V-scale is a general-purpose replacement for standard Transformers. We will revise the title and organization of Section 5 to present it as “Mechanistic Intervention and Validation” and explicitly separate:

- the primary evidential role of V-scale;
- the capability-control experiments;
- the secondary and conditional practical observations.

### Question 2: Does V-scale isolate the proposed backward-pass mechanism?

The original V-scale transformation changes both the forward value state and its backward gradient. To remove this forward-pass confound, we trained an additional model using a backward-only version of V-scale.

The construction leaves the forward value state exactly unchanged, $\hat v=v$, while applying the V-scale Jacobian during backpropagation. The backward-only model remains well trained. Substantial attention-sink behavior remains, although its mean strength is reduced. At the same time, massive activations are strongly suppressed relative to the baseline.

This experiment directly removes the forward-contraction explanation. It shows that changing only the value-path backward rule is sufficient to produce the predicted suppression of massive activations.

We will add the backward-only construction, its exact forward/backward identities, the comparison results, and the full layer-wise measurements.

### Question 3: What do the results establish about quantization?

We agree that the original wording could be interpreted as making a broader quantization claim than the submitted evidence supports.

BNB, GPTQ, and AWQ are W4A16 settings and therefore do not quantize runtime activations. They are not direct tests of whether reducing residual-stream massive activations improves activation quantization. SmoothQuant uses activation quantization but also explicitly compensates for activation outliers and applies quantization at Linear inputs rather than directly at the residual-stream sites where massive activations are measured.

To test the narrower activation-range implication, we conducted a controlled W8A8 diagnostic:

1. dynamic per-token activation scaling;
2. calibrated static per-tensor activation scaling;
3. static per-tensor activation scaling localized to `mlp.down_proj`.

Dynamic per-token W8A8 leaves both models close to BF16. Under static per-tensor scaling, however, the baseline degrades sharply while V-scale remains substantially more functional. Applying the same static quantization only to `down_proj` nearly reproduces the all-Linear result, connecting the sensitivity to the MLP pathway where V-scale produces its strongest reduction of massive activations.

We emphasize that static per-tensor quantization is a deliberately range-sensitive diagnostic, not a proposed mainstream deployment recipe. The evidence supports the conditional conclusion that V-scale is less sensitive to coarse activation ranges at the implicated MLP pathway. It does not establish universal quantization robustness.

### Question 4: How should the mixed NIAH and PTQ results be interpreted?

We agree that the unfavorable AWQ/GPTQ/YaRN cases in the single-needle experiments are substantial. We do not regard them as noise, and we do not claim that V-scale uniformly improves every quantization method, retrieval task, or context-extension setting.

V-scale improves every reported multi-key setting, but the single-needle results are mixed and include severe degradations in several W4A16/YaRN combinations. We currently do not have sufficient evidence to identify the cause of these interactions. We will therefore avoid claiming universal long-context or quantization improvement and explicitly state that the W4A16/YaRN interaction remains unresolved.

These mixed downstream results constrain a secondary practical observation. They do not contradict the empirical and theoretical mechanism in Sections 3–4 or the role of Section 5 as an intervention that tests its counterfactual prediction.

### Question 5: How will the presentation be improved?

We will make the following revisions:

1. Rename and reposition Section 5 as “Mechanistic Intervention and Validation.”
2. Add a computational schematic showing the gradient sites compared by Bloat, Change, and Compress.
3. Explain the intuitive meanings of these quantities before presenting their formal definitions.
4. Add an overview figure summarizing the complete mechanism:
   attention sink
   $\rightarrow$ value-path gradient aggregation
   $\rightarrow$ localized gradient pressure
   $\rightarrow$ RMSNorm-mediated compression through massive activations
   $\rightarrow$ mild residual-stream gradient change.
5. Show V-scale in the same figure as an alternative gradient valve.
6. Bold the best results in all comparison tables.
7. Explicitly discuss the limitations and mixed long-context results.

We thank the reviewers again for helping us clarify the contribution hierarchy, strengthen the causal evidence, narrow the practical claims, and improve the presentation.

---

## 2. Response to Reviewer gq1u

We thank the reviewer for recognizing the value of the training-dynamics analysis, the realism of the empirical and theoretical setting, and the clarity of the presentation.

The reviewer’s concerns primarily concern the usefulness and evaluation of V-scale in Section 5. Before addressing them individually, we would like to clarify the central novelty and the intended role of that section.

The main contribution of the paper lies in Sections 3–4: we identify gradient sinks as a backward-pass counterpart of attention sinks and provide empirical and theoretical evidence that massive activations act as RMSNorm-mediated regulators of localized gradient pressure. Section 5 is designed primarily as a counterfactual intervention that tests a prediction derived from this mechanism, rather than as a proposal that V-scale is a universally superior practical architecture.

We recognize that the title “Practical Application” and the emphasis on quantization (e.g., L214) may have made the secondary practical role appear to be the primary contribution. We will revise this positioning.

### Question 1: Given the mixed downstream results and the cost of modifying pretraining, what is the usefulness of V-scale?

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

### Question 2: Does the paper adequately verify a quantization benefit?

Thank you for raising this distinction. We agree that the original wording could be read as making a broader claim than the evidence supports.

The submitted PTQ methods probe different effects. BNB, GPTQ, and AWQ are W4A16 settings and therefore do not quantize runtime activations. Reducing residual-stream massive activations is consequently not expected to yield an automatic improvement under these methods. SmoothQuant is W8A8, but it explicitly compensates for activation outliers and applies activation quantization to Linear inputs rather than directly to the residual-stream sites at which massive activations are measured. These results therefore do not directly isolate sensitivity to the residual-stream activation range.

To test this narrower implication, we conducted a controlled activation-range diagnostic using three symmetric INT8 configurations:

1. **All-Linear dynamic-token W8A8.** All `Linear` modules except `lm_head` use static per-channel INT8 weights and dynamic per-token INT8 input activations.
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

### Question 3: How should the severe degradations in some AWQ/GPTQ and NIAH settings be interpreted?

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

### Formatting concern

We agree with the reviewer’s suggestion and will bold the best result in each table.

We thank the reviewer again for prompting us to clarify the contribution hierarchy, reposition Section 5, add a more direct activation-range diagnostic, and narrow the practical claims.

---

## 3. Response to Reviewer pqQy

We thank the reviewer for the positive assessment and for the insightful suggestion concerning ARC-C, BoolQ, and multi-key NIAH.

### Question 1: Could the gains on ARC-C and BoolQ be connected to the improvements on multi-key NIAH?

A plausible commonality is query-conditioned selection and routing of relevant information under competing cues. Indeed, V-scale exhibit stronger gains on tasks where selective routing or retrieval may be particularly important, rather than improve all benchmarks uniformly.

BoolQ requires locating evidence relevant to a question within a passage. Multi-key NIAH requires retrieving the queried association while ignoring several competing key-value pairs. ARC-C is less directly a contextual-retrieval task, but it similarly requires selecting and combining relevant scientific information while discriminating among plausible answer choices.

Selective information use under competing cues may therefore be a shared demand across these tasks.

The learned V-scale parameters also vary across layers and heads (see Figure 13 in Appendix), suggesting that the intervention does not act uniformly on all attention pathways. This heterogeneity may interact with functionally specialized pathways for information selection, transport, or retrieval.

However, we emphasize that this remains a hypothesis rather than an established conclusion. Our current experiments do not identify the functions of individual heads or establish a causal relationship between the benchmark gains.

We will add a concise discussion of this possible connection while clearly presenting it as a hypothesis and a direction for future mechanistic analysis.

We thank the reviewer again for suggesting this useful interpretation.

---

## 4. Response to Reviewer 6L5L

We thank the reviewer for the careful reading, encouraging assessment, and concrete suggestions for strengthening both the causal evidence and the presentation. We answer each question explicitly below.

### Question 1: Since V-scale changes both the backward gradient and the forward value states, how can the paper establish that backward gradient modulation is responsible for suppressing massive activations?

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

### Question 2: Can Bloat, Change, and Compress be explained more clearly with a pictorial illustration?

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

### Question 3: Can the paper include an overview figure summarizing the complete logic of Sections 3–5?

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
