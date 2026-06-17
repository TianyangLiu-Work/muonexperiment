# E11 Matrix-Block Tightness Audit

This generated artifact is a deterministic sanity audit for the matrix-block
theorem proof contract. It does not add new empirical results. It checks exact
diagonal singular-spectrum witnesses for the Frobenius numerator, the spectral
sandwich numerator, the ratio identity
`I_spectral / I_frobenius = ssrank(B_T,A_T) / nrank(G_H)`, and the degeneracy
cases where the rank boundary must not be interpreted.

## Rank-Boundary Cases

| case_id                  | g_singular_values   | b_singular_values   | a_singular_values   |   nrank_G_H | ssrank_B_T_A_T   | ratio_I_spectral_over_I_frobenius   | rank_boundary_applicable   | predicted_relation                      | paper_use                                                                             |
|:-------------------------|:--------------------|:--------------------|:--------------------|------------:|:-----------------|:------------------------------------|:---------------------------|:----------------------------------------|:--------------------------------------------------------------------------------------|
| MBTA-1-spectral-favored  | [1, 1, 1, 1]        | [1, 0, 0, 0]        | [1, 0, 0, 0]        |           4 | 1                | 0.25                                | yes                        | spectral_smaller_worst_case_bound       | clean rank-boundary witness for nrank(G_H)>ssrank(B_T,A_T)                            |
| MBTA-2-equality-boundary | [1, 1]              | [1, 1]              | [1, 1]              |           2 | 2                | 1                                   | yes                        | equal_worst_case_bound                  | shows strict inequality is needed for a strict spectral bound advantage               |
| MBTA-3-frobenius-favored | [1]                 | [1, 1]              | [1, 1]              |           1 | 2                | 2                                   | yes                        | frobenius_smaller_worst_case_bound      | claim-boundary witness where tail sensitivity rank exceeds head numerical rank        |
| MBTA-4-degenerate-tail   | [1, 1, 1]           | [0, 0]              | [1, 1]              |           3 | n/a              | n/a                                 | no                         | degenerate_tail_boundary_not_applicable | nondegenerate-tail assumption witness; rank boundary is intentionally not interpreted |

## Formula And Tightness Checks

| case_id                  |   frobenius_tail_numerator_formula |   frobenius_unit_witness_value |   spectral_tail_numerator_formula |   spectral_unit_witness_value | direct_ratio_I_spectral_over_I_frobenius   | rank_ratio_ssrank_over_nrank   | ratio_absolute_error   | tightness_status                    |
|:-------------------------|-----------------------------------:|-------------------------------:|----------------------------------:|------------------------------:|:-------------------------------------------|:-------------------------------|:-----------------------|:------------------------------------|
| MBTA-1-spectral-favored  |                                  1 |                              1 |                                 1 |                             1 | 0.25                                       | 0.25                           | 0                      | exact_for_diagonal_singular_witness |
| MBTA-2-equality-boundary |                                  1 |                              1 |                                 2 |                             2 | 1                                          | 1                              | 0                      | exact_for_diagonal_singular_witness |
| MBTA-3-frobenius-favored |                                  1 |                              1 |                                 2 |                             2 | 2                                          | 2                              | 0                      | exact_for_diagonal_singular_witness |
| MBTA-4-degenerate-tail   |                                  0 |                              0 |                                 0 |                             0 | n/a                                        | n/a                            | n/a                    | not_applicable_degenerate_tail      |

## Caveats

| caveat_id                          | condition                                                                        | mathematical_effect                                                                          | paper_action                                                                       |
|:-----------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| MBTC-1-nondegenerate-tail-required | sigma_1(B_T) sigma_1(A_T) = 0                                                    | both tail numerator formulas vanish and ssrank(B_T,A_T) is undefined                         | do not interpret the nrank-vs-ssrank rank boundary for a zero tail-sensitive block |
| MBTC-2-strict-inequality-required  | nrank(G_H) = ssrank(B_T,A_T)                                                     | I_spectral/I_frobenius = 1, so the theorem gives no strict advantage                         | state strict spectral advantage only under nrank(G_H)>ssrank(B_T,A_T)              |
| MBTC-3-worst-case-not-realized     | the optimizer update is not aligned with the worst-case tail singular directions | the bound can be loose for a realized direction even when the coefficient ordering is strict | separate theorem-level bound ordering from empirical realized-drift prediction     |

Artifacts:
- [rank_boundary_cases.csv](../results/e11_matrix_block_tightness_audit/rank_boundary_cases.csv)
- [formula_checks.csv](../results/e11_matrix_block_tightness_audit/formula_checks.csv)
- [caveat_checks.csv](../results/e11_matrix_block_tightness_audit/caveat_checks.csv)
- [config.json](../results/e11_matrix_block_tightness_audit/config.json)
