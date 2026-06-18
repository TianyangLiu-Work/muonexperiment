# E11 CIFAR-100-LT Tuned Benchmark Fairness Audit

This generated audit checks whether the tuned CIFAR-100-LT ResNet18 benchmark is reviewer-fair before any final-performance claim. It is a validation-budget and comparison-parity audit: it permits budget disclosure and progress accounting, but it does not convert partial validation rows into benchmark evidence.

## Family Budget Matrix

| recipe_family          | claim_role                                           | optimizer   |   setting_count |   setting_fraction |   lr_count |   weight_decay_count |   warmup_count |   newton_schulz_count | class_reweighting                                      | sampler                   | validation_seed_set   | budget_interpretation                                |
|:-----------------------|:-----------------------------------------------------|:------------|----------------:|-------------------:|-----------:|---------------------:|---------------:|----------------------:|:-------------------------------------------------------|:--------------------------|:----------------------|:-----------------------------------------------------|
| adamw_cb_loss_tuned    | class-balanced AdamW baseline                        | adamw       |              24 |            0.1463  |          3 |                    2 |              2 |                     1 | effective-number class-balanced loss beta=0.999;0.9999 | long-tail natural sampler | 10..14                | tuned_baseline_surface                               |
| adamw_cb_sampler_tuned | sampler baseline required before any practical claim | adamw       |               8 |            0.04878 |          2 |                    2 |              2 |                     1 | cross entropy                                          | class-balanced sampler    | 10..14                | tuned_baseline_surface                               |
| adamw_ce_tuned         | tuned AdamW baseline                                 | adamw       |              12 |            0.07317 |          3 |                    2 |              2 |                     1 | none                                                   | long-tail natural sampler | 10..14                | tuned_baseline_surface                               |
| ns_muon_cb_tuned       | candidate practical Muon class-balanced recipe       | ns_muon     |              36 |            0.2195  |          3 |                    2 |              2 |                     3 | effective-number class-balanced loss beta=0.9999       | long-tail natural sampler | 10..14                | candidate_search_budget_disclose_not_final_advantage |
| ns_muon_matrix_tuned   | candidate practical Muon recipe                      | ns_muon     |              72 |            0.439   |          4 |                    2 |              3 |                     3 | none                                                   | long-tail natural sampler | 10..14                | candidate_search_budget_disclose_not_final_advantage |
| sgd_momentum_ce_tuned  | tuned SGD baseline                                   | sgd         |              12 |            0.07317 |          3 |                    2 |              2 |                     1 | none                                                   | long-tail natural sampler | 10..14                | tuned_baseline_surface                               |

## Seed And Metric Parity

| parity_id                       | status   | evidence                                                                    | forbidden_asymmetry                                           |
|:--------------------------------|:---------|:----------------------------------------------------------------------------|:--------------------------------------------------------------|
| TBF-P1-validation-seed-parity   | pass     | all validation rows use seed_set 10..14                                     | changing validation seeds by optimizer family                 |
| TBF-P2-final-seed-parity        | pass     | final_claim seed_set 20..29 tuning_allowed=no                               | letting one optimizer family tune on final seeds              |
| TBF-P3-primary-metric-parity    | pass     | selection_rules.csv fixes few balanced accuracy for every family            | using a different validation objective for Muon and baselines |
| TBF-P4-multiplicity-parity      | pass     | final comparisons to tuned AdamW and tuned SGD are one Holm-adjusted family | claiming only the best-looking unadjusted comparison          |
| TBF-P5-reporting-surface-parity | pass     | acceptance gates require many/medium/few/all reporting                      | reporting tail-only gains while hiding all-class collapse     |

## Fairness Gate Matrix

| gate_id                            | status               | evidence                                                                                     | claim_effect                                                                        |
|:-----------------------------------|:---------------------|:---------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
| TBF-1-registered-family-coverage   | pass                 | 6 recipe families; 164 registered settings                                                   | benchmark scope names every tuned family rather than a hand-picked subset           |
| TBF-2-baseline-tuning-surface      | pass                 | AdamW CE=12; SGD CE=12; class-balanced loss and sampler baselines present                    | Muon cannot be compared only to undertuned vanilla baselines                        |
| TBF-3-candidate-budget-disclosure  | pass_with_disclosure | Muon candidate settings=108; baseline settings=56                                            | larger Muon validation budget must be disclosed and cannot count as final evidence  |
| TBF-4-seed-metric-reporting-parity | pass                 | validation seeds, final seeds, primary metric, Holm family, and reporting surface are shared | final paired comparison uses one rule set for every optimizer family                |
| TBF-5-final-gates-still-blocked    | pass                 | TVS-1=not_ready; TVS-3=pass                                                                  | fairness audit does not authorize final-performance wording from partial validation |

## Budget Disclosure Contract

| contract_id                         | status   | required_wording                                                                                | forbidden_wording                                                                      |
|:------------------------------------|:---------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| TBF-C1-validation-budget-disclosure | active   | report baseline validation budget=56 settings and Muon candidate validation budget=108 settings | describing validation leaderboard rank as final benchmark performance                  |
| TBF-C2-final-evidence-parity        | active   | final claims require paired seed set 20..29 for every selected family                           | using a larger validation search budget as evidence of a final optimizer advantage     |
| TBF-C3-baseline-strength-boundary   | active   | compare against tuned AdamW, tuned SGD, and class-balanced AdamW baselines                      | claiming practical Muon superiority against only the weak reporting baseline or pilots |

## Operating Rule

A larger Muon validation search budget is allowed only as a disclosed candidate-search budget. It is not final evidence. Any practical-performance wording must wait for complete validation selection, shared final seed set `20..29`, Holm-adjusted final comparisons against tuned AdamW and tuned SGD, all-class guardrails, and complete many/medium/few/all reporting for every selected recipe.
