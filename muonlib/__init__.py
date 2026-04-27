from .algorithms import (MuonOptimizer, SGDOptimizer, AdamOptimizer, 
                         RMSpropOptimizer, MomentumSGDOptimizer, ALGORITHMS)
from .data import (generate_target_matrix, generate_matrix_sensing,
                   generate_matrix_factorization, generate_measurement_matrices,
                   generate_rectangular_target, compute_gradient_matrix_sensing,
                   compute_loss_matrix_sensing, compute_loss_mf, compute_gradient_mf)
from .metrics import (compute_flops_matrix_sensing, compute_flops_matrix_sensing_rect,
                      compute_flops_mf, enrich_result_row,
                      compute_statistics, welch_ttest, paired_ttest,
                      wilcoxon_test, bonferroni_correction, benjamini_hochberg,
                      compute_effect_size, compute_power, convergence_rate)
from .visualization import (plot_convergence, plot_heatmap, plot_boxplot,
                            plot_pareto_frontier, plot_convergence_with_ci)
