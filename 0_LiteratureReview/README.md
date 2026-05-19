# Literature Review
## 1. Introduction and Scope
Banks are under intensifying supervisory and market pressure to identify and manage Environmental, Social, and Governance (ESG) risks. These risks propagate through credit, market, and operational channels, and increasingly materialize via controversies and regulatory actions surfaced first in unstructured text—news, disclosures, and earnings calls. Recent advances in Natural Language Processing (NLP), especially transformer-based models pre-trained on financial text (e.g., FinBERT), enable automated extraction of ESG-relevant signals at scale. This review synthesizes (i) regulatory expectations for ESG risk in banking, (ii) machine learning applications in financial risk management, (iii) NLP methods for financial text, (iv) transformer architectures and domain-adapted models, and (v) empirical findings in ESG controversy and sentiment detection. It motivates a PyTorch-based framework that fine-tunes FinBERT for ESG risk classification, using general financial sentiment data for warm-start and ESG-focused data for task alignment.

## 2. ESG Risk in Banking
Supervisory bodies (ECB, EBA, BCBS) increasingly require banks to embed ESG risks across governance, strategy, risk appetite, ICAAP/ILAAP, and disclosures. Core themes include:

ESG risk taxonomy and materiality: physical climate hazards, transition risk from decarbonization policies, social conduct and labor practices, and governance failures. These risks influence credit worthiness, collateral values, operational resilience, and reputational standing.
Scenario analysis and stress testing: forward-looking assessment of counterparty vulnerabilities to climate/ESG scenarios, including sectoral transition pathways and acute/chronic physical events.
Data and disclosure expectations: robust risk data aggregation, internal controls, and transparent reporting under sustainable finance regulations.
A key gap identified by supervisors and practitioners is timely, high-frequency indicators of ESG controversies. Unstructured text is central here: news and filings often surface allegations, fines, investigations, or stakeholder conflicts before structured metrics do. Consequently, scalable text-mining with auditability and controls becomes a differentiator for early risk detection.

## 3. Machine Learning in Financial Risk Management
Machine learning has transformed tasks such as credit scoring, fraud detection, and market surveillance by modeling non-linear patterns and ingesting alternative data. For risk monitoring:

Classification pipelines produce alerts from heterogeneous signals, requiring calibrated probabilities and controlled false-positive rates.
Model Risk Management (MRM) is essential: documentation, validation, stability checks, and explainability are supervisory priorities for AI in banking.
Data governance is critical with alternative data (e.g., web/news): provenance, labeling quality, bias, and reproducibility affect downstream fairness and compliance.
Despite strong performance, deep models can suffer from domain drift, label scarcity, and opacity—salient issues for ESG, where definitions evolve and adverse events are context-dependent. Methods that support continual learning, robust calibration, and interpretable attributions are favored.

## 4. NLP for Financial Text
Traditional NLP (tokenization, TF–IDF, n-grams, topic models) established baselines for sentiment and document classification but struggles with:

Domain-specific semantics, hedging, and modality in financial writing.
Long-range dependencies and cross-sentence cues (e.g., negation, legal qualifiers).
Entity-centric narratives where risk attaches to specific issuers, sectors, or geographies.
Contextual embeddings and pre-trained language models address these limitations by encoding word meaning conditional on context and capturing relationships across tokens. For finance, domain adaptation is crucial to handle jargon (e.g., downgrade, covenant breach, class action), idioms, and reporting styles.

## 5. Transformers and Financial-Domain Adaptation
The transformer architecture, with self-attention and pre-training objectives (e.g., masked language modeling), has set state-of-the-art results in text classification. Transfer learning—pre-training on large corpora, followed by task-specific fine-tuning—improves data efficiency and generalization.

Financial domain models: FinBERT variants are pre-trained on financial corpora and often fine-tuned for sentiment, achieving superior performance over general BERT on financial tasks due to better coverage of domain terms and patterns.
Fine-tuning strategies: learning-rate schedules, early stopping on macro-F1, class-weighted losses or focal loss for imbalance, and layer-wise learning-rate decay improve stability on small datasets.
Multi-task/continual learning: jointly training on sentiment and ESG labels or sequentially fine-tuning helps retain general financial semantics while specializing for ESG risk cues.

## 6. ESG Sentiment and Controversy Detection

## Literature on ESG NLP highlights:

ESG controversy mining: identifying news indicative of environmental violations, labor disputes, human rights issues, anti-competitive behavior, or governance failures (e.g., fraud, bribery, board conflicts).
Pillar-aware classification: separating Environmental, Social, and Governance signals, often as multi-label tasks due to co-occurrence.
Data challenges: scarcity of high-quality labeled ESG datasets, label noise in weakly supervised corpora, and class imbalance (severe events are rarer but critical).
Explainability: token-level attributions (e.g., Integrated Gradients, SHAP) support analyst review, trust, and audit trails; negation/modality tests guard against spurious cues.
Integration with risk systems: calibrated outputs, cost-sensitive thresholds, human-in-the-loop review of high-impact alerts, and drift monitoring are standard in deployments.

## 7. Data Foundations and Their Implications
For this project, the Kaggle “Sentiment Analysis for Financial News” dataset provides headline-level Positive/Negative/Neutral labels from a retail-investor perspective. While not ESG-specific, it is valuable for:

Warm-starting FinBERT to capture finance-specific sentiment and syntax on short headlines.
Establishing baselines and sharpening the model’s handling of polarity—useful because ESG controversies often present as negative or neutrally worded but risk-relevant.

## Gaps with respect to ESG:

* No explicit ESG category labels or issuer/entity spans.
* Headline brevity limits context; many ESG events require longer passages or multi-sentence reasoning.
* Potential class imbalance and domain shift between generic financial sentiment and ESG controversies.
* These gaps motivate a two-stage approach: pre-fine-tune on financial sentiment, then adapt to ESG via curated or weakly supervised ESG-labeled corpora. * For longer documents (e.g., sustainability reports), chunk-and-vote or long-context transformers are appropriate.

## 8. Methods, Evaluation, and Controls
Modeling: FinBERT in PyTorch/HF with 3-way sentiment objective for Stage 1; ESG binary or multi-label (E/S/G, severity) for Stage 2. Use stratified splits, max length ~64–96 for headlines, and learning rates in the 2e-5 to 5e-5 range with warmup.
Metrics: macro-F1 and per-class F1 (neutral is typically hardest), MCC, confusion matrices. For risk thresholds, evaluate calibration (ECE/Brier) and cost-weighted utility emphasizing false negatives for controversies.
Robustness: adversarial synonyms, negation/modality checks, OOD generalization to disclosures and call transcripts.
Explainability and MRM: token attributions, global feature summaries, sector/region error analysis, model cards documenting data lineage, limitations, and monitoring plans.

## 9. Research Gap and Contribution
Despite rapid progress, there is limited end-to-end research that combines:

* Banking-relevant ESG taxonomies and supervisory expectations,
* Financial-domain transformers,
* ESG-specific labeling strategies at scale,
* And banking-grade MRM (explainability, calibration, drift monitoring).

# This project addresses the gap by:

* Building a two-stage FinBERT pipeline: financial sentiment pre-fine-tuning followed by ESG task adaptation.
* Comparing classical baselines to transformer models for transparent performance attribution.
* Incorporating explainability, calibration, and robustness checks aligned with banking deployment.
* Demonstrating practical workflows—data governance, human-in-the-loop review, and integration patterns for news-driven ESG risk alerts.

## 10. Future Directions
* Multi-language ESG monitoring using multilingual or aligned encoders.
* Real-time retrieval-augmented classification with entity linking for issuer-specific risk.
* Explainable AI at scale (faithful attributions, counterfactual explanations).
* Graph methods to model ESG risk propagation across supply chains and counterparties.
* Time-aware transformers to forecast ESG risk trajectories and persistence of controversies.
