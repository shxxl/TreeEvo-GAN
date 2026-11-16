# Final trainer: experiment sampling settings are defined in search_cfg.py.
# Paper alignment: distance-first pairing, structural mutation, fair/equal subnet updates.
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import random
import heapq

from utils.Single_objective_sort import single_sort
from utils.nsga import NSGA_2
from utils.utils import count_parameters_in_MB
from archs.fully_super_network import Generator
from pytorch_gan_metrics import get_inception_score_and_fid, get_fid, get_inception_score
from torchvision.utils import make_grid, save_image

logger = logging.getLogger(__name__)

TRAINER_GENERATOR3_VARIANT = 'three_objective_elite_pools_v1'
TRAINER_MEMORY_FIX = 'sequential_backward_freeze_discriminator_v1'
TRAINER_TREE_CROSSOVER = 'ordered_three_stage_same_position_subtree_v1'
TRAINER_TREE_MUTATION = 'tree_mutation_and_distance_diverse_parent_v1'
TRAINER_EXTERNAL_ARCHIVE = 'history_backed_external_elite_archive_param_gate_fid_forest_size13_v1'
TRAINER_FOREST_VARIANT = 'objective_conditioned_three_forests_v1'
TRAINER_FID_EXPLOITATION = 'constrained_fid_first_params_gate_stage_soft_v1'
TRAINER_TREE_PRUNING = 'fid_safe_tree_branch_pruning_after_search_v1'
TRAINER_THREE_FOREST_PRUNING = 'objective_conditioned_three_forest_pruning_v1'
TRAINER_THREE_FOREST_PRUNING_EFFECTIVE = 'effective_objective_conditioned_tree_pruning_v1'
TRAINER_THREE_FOREST_PRUNING_RECOVERY = 'three_forest_prune_recover_validate_v1'
TRAINER_CORE_SIMPLIFY = 'remove_kd_guarded_rank_guarded_local_v1'
TRAINER_PURE_METHOD = 'pure_tree_evolution_three_forests_pruning_v1'
TRAINER_PARAM_FID_DIVERSITY = 'effective_no_recovery_fid_gated_smaller_params_diversity_v1'
TRAINER_STABLE_NO_FAR_DISTANCE = 'effective_no_recovery_fid_local_no_far_distance_v1'
TRAINER_DYNAMIC_PARAM_PCS = 'dynamic_param_cap_pcs_v1'
TRAINER_OBJECTIVE_PRUNE_POLICY = 'objective_conditioned_pruning_policy_v1'
TRAINER_COMPACT_SURVIVAL = 'fid_window_compact_survival_quota_and_memory_pool_v1'
TRAINER_BASELINE_STABILITY = 'pcs108_freeze_history_fidsafe_baseline_v1'
TRAINER_COMPACT_VALID_PRUNE = 'compact_valid_final_pruning_target10_v1'

try_balance = False
Extra_subnet = 2
num_normal = 7






class GenTrainer():
    """
    Original three-objective elite-pool architecture search; no performance predictors, training forest feedback, or KD.

    Core search flow:
        current population + dynamic IS/FID/Params elite pools
        -> same-pool / cross-pool / population parent allocation
        -> crossover / mutation generates the exact-evaluation offspring budget
        -> exact results are cached in historical_population
        -> constraint-first IS/FID survivor selection directly forms next population
        -> selected architectures and historical elites update operation statistics.

    Key design:
    1) historical_population stores all exact-evaluated architectures.
    2) three elite pools are rebuilt from real IS/FID/Params results, not predictor output.
    3) returned offspring are evaluated directly, without surrogate prescreening.
    4) the population update is direct survivor selection; no FIFO queue is used in this version.
    """

    def __init__(self, args, gen_net, dis_net, gen_optimizer, dis_optimizer, train_loader, gan_alg, dis_genotype,
                 gen_fixed_genotype):
        self.args = args
        self.gen_net = gen_net
        self.dis_net = dis_net
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.train_loader = train_loader
        self.gan_alg = gan_alg
        self.dis_genotype = dis_genotype
        self.gen_fixed_genotype = gen_fixed_genotype
        logger.info('[TRAINER-G3-VARIANT] %s', TRAINER_GENERATOR3_VARIANT)
        logger.info('[TRAINER-MEMORY-FIX] %s', TRAINER_MEMORY_FIX)
        logger.info('[CORE-SIMPLIFY] %s', TRAINER_CORE_SIMPLIFY)
        logger.info('[PURE-METHOD] %s', TRAINER_PURE_METHOD)
        logger.info('[DYNAMIC-PARAM-PCS] %s', TRAINER_DYNAMIC_PARAM_PCS)
        logger.info('[BASELINE-STABILITY] %s', TRAINER_BASELINE_STABILITY)
        logger.info('[OBJECTIVE-PRUNE-POLICY] %s', TRAINER_OBJECTIVE_PRUNE_POLICY)
        logger.info('[COMPACT-SURVIVAL] %s', TRAINER_COMPACT_SURVIVAL)
        logger.info('[THREE-FOREST-PRUNE] %s', TRAINER_THREE_FOREST_PRUNING)
        logger.info('[THREE-FOREST-PRUNE-EFFECTIVE] %s', TRAINER_THREE_FOREST_PRUNING_EFFECTIVE)
        logger.info('[THREE-FOREST-PRUNE-RECOVERY] %s', TRAINER_THREE_FOREST_PRUNING_RECOVERY)

        # exact evaluation history: key -> dict(is, fid, param, exact, vector)
        self.historical_population = {}
        self.param_cache = {}
        self.arch_visit_counter = {}
        self.replay_buffer = []

        # search/training defaults (set via args when available, otherwise safe defaults)
        # Legacy attributes retained for callers; training uses direct fair sampling
        # and equal per-subnet gradient accumulation, without NSAS reweighting.
        self.nsas_beta = float(getattr(args, 'nsas_beta', 0.35))
        self.use_diverse_sampling = False
        self.diverse_candidate_multiplier = int(getattr(args, 'diverse_candidate_multiplier', 4))
        # Pure method: no replay buffer. Parents should come from the current
        # population and the three objective-conditioned tree forests.
        self.replay_buffer_size = 0
        self.parent_diversity_tournament = int(getattr(args, 'parent_diversity_tournament', 4))

        # Preserve the single-elite activation threshold independently.
        self.elite_min_history = int(getattr(args, 'elite_min_history', 24))

        # Dynamic elite pool: no extra long-term archive is introduced.
        # It is rebuilt from exact historical_population records each round.
        self.use_elite_pool = bool(getattr(args, 'use_elite_pool', True))
        self.elite_parent_ratio = float(getattr(args, 'elite_parent_ratio', 0.25))
        self.elite_operation_ratio = float(getattr(args, 'elite_operation_ratio', 0.25))

        # Three objective-specific elite archives. They are rebuilt only from
        # exact evaluations. Each pool optimizes one objective while applying
        # data-adaptive quality gates to the other objectives.
        self.use_three_elite_pools = bool(getattr(args, 'use_three_elite_pools', True))
        default_pool_size = max(4, int(getattr(args, 'num_selected', 12)) // 2)
        self.three_elite_pool_size = int(getattr(args, 'three_elite_pool_size', default_pool_size))
        default_min_history = max(12, int(getattr(args, 'num_selected', 12)))
        self.three_elite_min_history = int(
            getattr(args, 'three_elite_min_history', default_min_history)
        )
        self.three_elite_quality_keep_ratio = float(
            getattr(args, 'three_elite_quality_keep_ratio', 0.60)
        )
        self.three_elite_same_pool_ratio = float(
            getattr(args, 'three_elite_same_pool_ratio', 0.60)
        )
        self.three_elite_cross_pool_ratio = float(
            getattr(args, 'three_elite_cross_pool_ratio', 0.30)
        )
        self.three_elite_population_ratio = float(
            getattr(args, 'three_elite_population_ratio', 0.10)
        )
        # Objective weights are used only when choosing parents from the three
        # elite pools. Current experiments focus on improving FID first, so FID
        # elites dominate parent sampling, IS remains a supporting signal, and
        # Params is kept as a weak constraint signal for later pruning.
        self.three_elite_fid_parent_weight = float(args.three_elite_fid_parent_weight)
        self.three_elite_is_parent_weight = float(args.three_elite_is_parent_weight)
        self.three_elite_params_parent_weight = float(args.three_elite_params_parent_weight)
        self.three_elite_param_max_fid = float(
            getattr(args, 'three_elite_param_max_fid', 55.0)
        )
        self.three_elite_param_best_fid_margin = float(
            getattr(args, 'three_elite_param_best_fid_margin', 2.5)
        )
        self.three_elite_param_min_is = float(
            getattr(args, 'three_elite_param_min_is', 5.0)
        )
        logger.info(
            f'[THREE-ELITE] enabled={self.use_three_elite_pools}, '
            f'pool_size={self.three_elite_pool_size}, '
            f'min_history={self.three_elite_min_history}, '
            f'parent_mix={self.three_elite_same_pool_ratio:.2f}/'
            f'{self.three_elite_cross_pool_ratio:.2f}/'
            f'{self.three_elite_population_ratio:.2f}, '
            f'objective_parent_weight='
            f'FID:{self.three_elite_fid_parent_weight:.2f}/'
            f'IS:{self.three_elite_is_parent_weight:.2f}/'
            f'Params:{self.three_elite_params_parent_weight:.2f}, '
            f'params_gate=(FID<=min({self.three_elite_param_max_fid:.1f}, '
            f'bestFID+{self.three_elite_param_best_fid_margin:.1f})).'
        )

        # Three objective-conditioned forests. Each objective pool first gathers
        # a larger quality-gated candidate set, clusters it by ordered-tree
        # distance, and keeps the corresponding objective winner per cluster.
        # Parent-source weights are supplied by the experiment configuration.
        self.use_three_forest_clustering = bool(
            getattr(args, 'use_three_forest_clustering', True)
        )
        default_forest_multiplier = int(
            getattr(args, 'forest_candidate_multiplier', 4)
        )
        default_forest_clusters = int(
            getattr(args, 'forest_num_clusters', max(3, self.three_elite_pool_size))
        )
        self.use_is_forest_clustering = bool(
            getattr(args, 'use_is_forest_clustering', self.use_three_forest_clustering)
        )
        self.use_fid_forest_clustering = bool(
            getattr(args, 'use_fid_forest_clustering', self.use_three_forest_clustering)
        )
        self.use_params_forest_clustering = bool(
            getattr(args, 'use_params_forest_clustering', self.use_three_forest_clustering)
        )
        self.is_forest_candidate_multiplier = int(
            getattr(args, 'is_forest_candidate_multiplier', default_forest_multiplier)
        )
        self.fid_forest_candidate_multiplier = int(
            getattr(args, 'fid_forest_candidate_multiplier', default_forest_multiplier)
        )
        self.params_forest_candidate_multiplier = int(
            getattr(args, 'params_forest_candidate_multiplier', default_forest_multiplier)
        )
        self.is_forest_num_clusters = int(
            getattr(args, 'is_forest_num_clusters', default_forest_clusters)
        )
        self.fid_forest_num_clusters = int(
            getattr(args, 'fid_forest_num_clusters', default_forest_clusters)
        )
        self.params_forest_num_clusters = int(
            getattr(args, 'params_forest_num_clusters', default_forest_clusters)
        )
        self.is_forest_best_fid_margin = float(
            getattr(args, 'is_forest_best_fid_margin', 2.5)
        )
        self.params_forest_best_fid_margin = float(
            getattr(args, 'params_forest_best_fid_margin', 2.5)
        )
        logger.info(
            f'[THREE-FOREST] variant={TRAINER_FOREST_VARIANT}, enabled='
            f'IS:{self.use_is_forest_clustering}/'
            f'FID:{self.use_fid_forest_clustering}/'
            f'Params:{self.use_params_forest_clustering}, '
            f'candidate_multiplier='
            f'{self.is_forest_candidate_multiplier}/'
            f'{self.fid_forest_candidate_multiplier}/'
            f'{self.params_forest_candidate_multiplier}, '
            f'clusters='
            f'{self.is_forest_num_clusters}/'
            f'{self.fid_forest_num_clusters}/'
            f'{self.params_forest_num_clusters}, '
            f'select_by=best_objective_per_structure_cluster, '
            f'FID_gate_margin=IS:+{self.is_forest_best_fid_margin:.1f}/'
            f'Params:+{self.params_forest_best_fid_margin:.1f}.'
        )

        # Offspring evaluation uses observed IS/FID directly.

        # Pure method: mutation is tree/genotype random mutation, not
        # operation-success guided mutation.
        self.use_guided_mutation = False
        self.guided_mutation_eps = float(getattr(args, 'guided_mutation_eps', 0.25))
        self.success_decay = float(getattr(args, 'success_decay', 0.98))
        self.success_smoothing = float(getattr(args, 'success_smoothing', 1.0))

        # Size is now treated as a hard constraint, not as a Pareto objective.
        # This avoids keeping tiny but poor-FID architectures as non-dominated
        # solutions. NSGA-style selection below uses only IS and FID after
        # filtering by args.max_model_size.
        self.enforce_model_size_constraint = True
        # Two-level size control:
        #   target_model_size: preferred search region. This relaxed-size version
        #   uses 13.0 MB by default because recent evo-only results showed that
        #   the 12-13 MB region can give better FID than the over-constrained
        #   9-10 MB region; max_model_size remains the absolute hard ceiling.
        # Selection first tries to use architectures <= target_model_size. Only
        # when there are not enough feasible candidates does it fall back to
        # <= max_model_size. Params are still NOT a Pareto objective.
        self.target_model_size = float(getattr(args, 'target_model_size', 13.0))
        self.prefer_target_model_size = False
        self.target_size_attempt_ratio = float(getattr(args, 'target_size_attempt_ratio', 0.35))
        self.use_dynamic_params_pcs = False
        total_evo_for_pcs = max(1, int(getattr(args, 'Total_evolutionary_algebra', 30)))
        self.dynamic_pcs_start_model_size = float(
            getattr(args, 'dynamic_pcs_start_model_size', self.target_model_size)
        )
        self.dynamic_pcs_mid_model_size = float(
            getattr(args, 'dynamic_pcs_mid_model_size', min(self.dynamic_pcs_start_model_size, 11.5))
        )
        self.dynamic_pcs_end_model_size = float(
            getattr(args, 'dynamic_pcs_end_model_size', min(self.dynamic_pcs_mid_model_size, 10.8))
        )
        default_pcs_t1 = max(1, int(round(total_evo_for_pcs * 0.27)))
        default_pcs_t2 = max(default_pcs_t1 + 1, int(round(total_evo_for_pcs * 0.67)))
        self.dynamic_pcs_t1 = int(getattr(args, 'dynamic_pcs_t1', default_pcs_t1))
        self.dynamic_pcs_t2 = int(getattr(args, 'dynamic_pcs_t2', default_pcs_t2))
        self.dynamic_pcs_t1 = max(0, min(self.dynamic_pcs_t1, max(0, total_evo_for_pcs - 1)))
        self.dynamic_pcs_t2 = max(self.dynamic_pcs_t1 + 1, min(self.dynamic_pcs_t2, max(1, total_evo_for_pcs)))
        default_pcs_freeze = max(
            self.dynamic_pcs_t2,
            min(total_evo_for_pcs - 1, int(round(total_evo_for_pcs * 0.80))),
        )
        self.dynamic_pcs_freeze_generation = int(
            getattr(args, 'dynamic_pcs_freeze_generation', default_pcs_freeze)
        )
        self.dynamic_pcs_freeze_generation = max(
            self.dynamic_pcs_t2,
            min(self.dynamic_pcs_freeze_generation, max(0, total_evo_for_pcs - 1)),
        )
        self._dynamic_pcs_logged_generation = None
        logger.info(
            f'[SIZE-RELAX] target_model_size={self.target_model_size:.3f}MB, '
            f'max_model_size={float(getattr(args, "max_model_size", self.target_model_size)):.3f}MB, '
            f'prefer_target={self.prefer_target_model_size}, '
            f'target_attempt_ratio={self.target_size_attempt_ratio:.2f}.'
        )
        logger.info(
            f'[DYNAMIC-PARAM-PCS] enabled={self.use_dynamic_params_pcs}, '
            f'schedule=start/mid/end='
            f'{self.dynamic_pcs_start_model_size:.3f}/'
            f'{self.dynamic_pcs_mid_model_size:.3f}/'
            f'{self.dynamic_pcs_end_model_size:.3f}MB, '
            f't1/t2={self.dynamic_pcs_t1}/{self.dynamic_pcs_t2}, '
            f'freeze_gen={self.dynamic_pcs_freeze_generation}, '
            f'hard_max={float(getattr(args, "max_model_size", self.target_model_size)):.3f}MB.'
        )
        self.use_fid_gated_smaller_params = bool(getattr(args, 'use_fid_gated_smaller_params', True))
        self.fid_tie_param_margin = float(getattr(args, 'fid_tie_param_margin', 1.0))
        self.use_slow_fid_weight_schedule = False
        self.use_quality_guided_selection = False
        self.quality_fid_weight = float(getattr(args, 'quality_fid_weight', 0.85))
        self.quality_is_weight = float(getattr(args, 'quality_is_weight', 0.15))
        self.quality_param_weight = float(getattr(args, 'quality_param_weight', 0.0))
        self.pareto_candidate_multiplier = int(getattr(args, 'pareto_candidate_multiplier', 3))
        logger.info(
            f'[PARAM-FID-DIVERSITY] variant={TRAINER_PARAM_FID_DIVERSITY}, '
            f'fid_gated_smaller_params={self.use_fid_gated_smaller_params}, '
            f'fid_tie_margin={self.fid_tie_param_margin:.2f}, '
            f'slow_fid_weight_schedule={self.use_slow_fid_weight_schedule}.'
        )

        # Plug-in compactness pressure. This does not replace the existing
        # IS/FID selection, three forests, predictor, or final pruning. It only
        # guarantees a small quota for architectures whose FID is close to the
        # current best while using fewer parameters, and keeps a tiny exact-
        # history-backed compact elite pool for parent sampling.
        self.use_fid_window_compact_quota = False
        self.fid_window_compact_start_gen = int(
            getattr(args, 'fid_window_compact_start_gen', 3)
        )
        self.fid_window_compact_margin = float(
            getattr(args, 'fid_window_compact_margin', 1.2)
        )
        self.fid_window_compact_quota_ratio = float(
            getattr(args, 'fid_window_compact_quota_ratio', 0.28)
        )
        self.fid_window_compact_max_quota_ratio = float(
            getattr(args, 'fid_window_compact_max_quota_ratio', 0.35)
        )
        self.fid_window_compact_min_quota = int(
            getattr(args, 'fid_window_compact_min_quota', 2)
        )
        self.fid_window_compact_min_is = float(
            getattr(args, 'fid_window_compact_min_is', 5.0)
        )
        self.fid_window_compact_require_target = bool(
            getattr(args, 'fid_window_compact_require_target', False)
        )

        self.use_compact_elite_pool = False
        self.compact_elite_pool_size = int(
            getattr(args, 'compact_elite_pool_size', max(8, int(getattr(args, 'num_selected', 12))))
        )
        self.compact_elite_min_history = int(
            getattr(args, 'compact_elite_min_history', max(12, int(getattr(args, 'num_selected', 12))))
        )
        self.compact_elite_fid_margin = float(getattr(args, 'compact_elite_fid_margin', 1.8))
        self.compact_elite_min_is = float(getattr(args, 'compact_elite_min_is', 5.0))
        self.compact_elite_parent_ratio = float(getattr(args, 'compact_elite_parent_ratio', 0.25))
        self.compact_elite_parent_min = int(getattr(args, 'compact_elite_parent_min', 2))
        self.compact_elite_pool = []
        self._compact_elite_log_generation = None
        self.use_compact_prescreen_quota = bool(getattr(args, 'use_compact_prescreen_quota', True))
        self.compact_prescreen_ratio = float(getattr(args, 'compact_prescreen_ratio', 0.20))
        self.compact_prescreen_fid_margin = float(getattr(args, 'compact_prescreen_fid_margin', 1.0))
        self.compact_prescreen_min_is = float(getattr(args, 'compact_prescreen_min_is', 0.0))
        logger.info(
            f'[COMPACT-SURVIVAL] quota_enabled={self.use_fid_window_compact_quota}, '
            f'quota_start_gen={self.fid_window_compact_start_gen}, '
            f'fid_window=best+{self.fid_window_compact_margin:.2f}, '
            f'quota_ratio={self.fid_window_compact_quota_ratio:.2f}, '
            f'compact_pool_enabled={self.use_compact_elite_pool}, '
            f'pool_size={self.compact_elite_pool_size}, '
            f'pool_fid_window=best+{self.compact_elite_fid_margin:.2f}, '
            f'parent_ratio={self.compact_elite_parent_ratio:.2f}, '
            f'prescreen_quota={self.use_compact_prescreen_quota}, '
            f'prescreen_ratio={self.compact_prescreen_ratio:.2f}, '
            f'prescreen_fid_window=best+{self.compact_prescreen_fid_margin:.2f}.'
        )


        # Tree crossover changes only the parent representation and crossover
        # granularity. Tree mutation is enabled only on the tree route below;
        # predictor, exact evaluation, population selection and search space are
        # unchanged.
        self.use_tree_subtree_crossover = True
        self.use_tree_mutation = True
        self.tree_mutation_prob = 1.0
        self.tree_mutation_add_prob = float(args.tree_mutation_add_prob)
        self.tree_mutation_delete_prob = float(args.tree_mutation_delete_prob)
        self.tree_mutation_replace_prob = float(args.tree_mutation_replace_prob)
        self.tree_mutation_subtree_prob = float(args.tree_mutation_subtree_prob)
        self.use_tree_distance_parent_selection = bool(
            getattr(args, 'use_tree_distance_parent_selection', True)
        )
        self.tree_parent_diversity_tournament = int(
            getattr(args, 'tree_parent_diversity_tournament', 6)
        )
        # Legacy option: positive enables a quality tie-break; it is not added to distance.
        self.tree_parent_quality_weight = float(
            getattr(args, 'tree_parent_quality_weight', 0.20)
        )
        self.use_good_gene_crossover = bool(getattr(args, 'use_good_gene_crossover', True))
        self.good_gene_parent_bias = float(args.good_gene_parent_bias)
        self.link_mutation_prob = float(getattr(args, 'link_mutation_prob', 0.45))
        logger.info(
            f'[TREE-SUBTREE] enabled={self.use_tree_subtree_crossover}, '
            f'encoding={TRAINER_TREE_CROSSOVER}, mutation={self.use_tree_mutation}.'
        )
        logger.info(
            f'[TREE-MUTATION] enabled={self.use_tree_mutation}, '
            f'variant={TRAINER_TREE_MUTATION}, '
            f'prob={self.tree_mutation_prob:.2f}, '
            f'add/delete/replace/subtree='
            f'{self.tree_mutation_add_prob:.2f}/'
            f'{self.tree_mutation_delete_prob:.2f}/'
            f'{self.tree_mutation_replace_prob:.2f}/'
            f'{self.tree_mutation_subtree_prob:.2f}, '
            f'diverse_parent={self.use_tree_distance_parent_selection}.'
        )
        self.use_fid_forest_local_mutation = False
        self.fid_forest_local_start_gen = int(getattr(args, 'fid_forest_local_start_gen', 3))
        self.fid_forest_local_ratio = float(getattr(args, 'fid_forest_local_ratio', 0.10))
        self.fid_forest_local_elite_top_k = int(getattr(args, 'fid_forest_local_elite_top_k', 3))
        self.fid_forest_local_neighbors_per_elite = int(getattr(args, 'fid_forest_local_neighbors_per_elite', 8))
        self.fid_forest_local_mutation_steps = int(getattr(args, 'fid_forest_local_mutation_steps', 1))
        self.use_far_distance_candidates = False
        self.far_distance_start_gen = int(getattr(args, 'far_distance_start_gen', 6))
        self.far_distance_candidate_ratio = float(getattr(args, 'far_distance_candidate_ratio', 0.10))
        self.far_distance_pool_multiplier = int(getattr(args, 'far_distance_pool_multiplier', 6))
        self.far_distance_min_tree_distance = float(getattr(args, 'far_distance_min_tree_distance', 0.35))
        logger.info(
            f'[DIVERSITY-RESCUE] fid_local={self.use_fid_forest_local_mutation}, '
            f'local_start={self.fid_forest_local_start_gen}, local_ratio={self.fid_forest_local_ratio:.2f}, '
            f'far_distance={self.use_far_distance_candidates}, far_start={self.far_distance_start_gen}, '
            f'far_ratio={self.far_distance_candidate_ratio:.2f}, '
            f'far_min_distance={self.far_distance_min_tree_distance:.2f}.'
        )
        logger.info(
            f'[STABLE-NO-FAR-DISTANCE] variant={TRAINER_STABLE_NO_FAR_DISTANCE}, '
            f'far_distance_candidates=False, '
            f'fid_local_ratio={self.fid_forest_local_ratio:.2f}, '
            f'fid_local_neighbors={self.fid_forest_local_neighbors_per_elite}.'
        )

        # External elite archive is a lightweight index over exact historical
        # evaluations. It does not replace the three elite pools; it only keeps
        # global champions available for parent mixing and final saving.
        # Pure method: use exact historical_population to rebuild three forests.
        # Do not add a separate external archive parent/return mechanism.
        self.use_external_elite_archive = False
        self.external_archive_size = int(
            getattr(args, 'external_archive_size', max(24, int(getattr(args, 'num_selected', 12)) * 3))
        )
        self.external_archive_parent_size = int(
            getattr(args, 'external_archive_parent_size', max(6, int(getattr(args, 'num_selected', 12))))
        )
        self.external_archive_return_size = int(
            getattr(args, 'external_archive_return_size', int(getattr(args, 'num_selected', 12)))
        )
        self.external_archive_parent_ratio = 0.0
        self.external_archive_param_max_fid = float(
            getattr(args, 'external_archive_param_max_fid', 55.0)
        )
        self.external_archive_param_min_is = float(
            getattr(args, 'external_archive_param_min_is', 5.0)
        )
        self.external_elite_archive = []
        logger.info(
            f'[EXTERNAL-ARCHIVE] enabled={self.use_external_elite_archive}, '
            f'variant={TRAINER_EXTERNAL_ARCHIVE}, '
            f'size={self.external_archive_size}, '
            f'parent_size={self.external_archive_parent_size}, '
            f'return_size={self.external_archive_return_size}, '
            f'parent_ratio={self.external_archive_parent_ratio:.2f}, '
            f'param_gate=(FID<=min({self.external_archive_param_max_fid:.1f}, '
            f'bestFID+{self.three_elite_param_best_fid_margin:.1f})).'
        )

        # Best-FID archive feedback strengthens exploitation: historical best-FID
        # genotypes are preserved in the next population and are forced to act as
        # parents for a small fraction of offspring in the next generation.
        # Pure method: do not force historical best-FID back into every
        # generation. FID exploitation should come from FID forest selection.
        self.use_archive_best_fid_injection = False
        self.archive_best_fid_inject_top_k = int(getattr(args, 'archive_best_fid_inject_top_k', 2))
        self.archive_best_fid_parent_top_k = int(getattr(args, 'archive_best_fid_parent_top_k', 3))
        self.archive_best_fid_forced_parent_ratio = 0.0
        logger.info(
            f'[ARCHIVE-BEST-FID-INJECT] enabled={self.use_archive_best_fid_injection}, '
            f'inject_top_k={self.archive_best_fid_inject_top_k}, '
            f'parent_top_k={self.archive_best_fid_parent_top_k}, '
            f'forced_parent_ratio={self.archive_best_fid_forced_parent_ratio:.2f}.'
        )

        # Lightweight FID-stage protection. It uses only exact archive records:
        # low-FID architectures vote for promising stage rows, and at most one
        # stage is restored after the normal crossover/mutation pipeline. The
        # other stages remain free to evolve, avoiding a full architecture lock.
        # Pure method: remove stage protection. Branch preservation/pruning
        # should be handled by the objective-conditioned forests and final tree pruning.
        self.use_fid_stage_protection = False
        self.fid_stage_protection_start_gen = int(
            getattr(args, 'fid_stage_protection_start_gen', 18)
        )
        self.fid_stage_protection_min_history = int(
            getattr(args, 'fid_stage_protection_min_history', 300)
        )
        self.fid_stage_protection_min_best_fid = float(
            getattr(args, 'fid_stage_protection_min_best_fid', 55.0)
        )
        self.fid_stage_archive_top_k = int(
            getattr(args, 'fid_stage_archive_top_k', 8)
        )
        self.fid_stage_archive_fid_margin = float(
            getattr(args, 'fid_stage_archive_fid_margin', 3.0)
        )
        self.fid_stage_min_support = int(
            getattr(args, 'fid_stage_min_support', 2)
        )
        self.fid_stage_protection_prob = float(
            getattr(args, 'fid_stage_protection_prob', 0.30)
        )
        self.fid_stage_max_protected = int(
            getattr(args, 'fid_stage_max_protected', 1)
        )
        protected_layers = getattr(args, 'fid_stage_protected_layers', (1, 2))
        if isinstance(protected_layers, str):
            protected_layers = [
                value.strip()
                for value in protected_layers.split(',')
                if value.strip()
            ]
        self.fid_stage_protected_layers = tuple(
            int(layer) for layer in protected_layers
        )
        self._fid_stage_bank_cache_generation = None
        self._fid_stage_bank_cache_history_size = None
        self._fid_stage_bank_cache = {}
        self._fid_stage_protection_log_generation = None
        logger.info(
            f'[FID-STAGE-PROTECT] enabled={self.use_fid_stage_protection}, '
            f'variant={TRAINER_FID_EXPLOITATION}, '
            f'start_gen={self.fid_stage_protection_start_gen}, '
            f'min_history={self.fid_stage_protection_min_history}, '
            f'min_best_fid={self.fid_stage_protection_min_best_fid:.1f}, '
            f'archive_top_k={self.fid_stage_archive_top_k}, '
            f'archive_margin=+{self.fid_stage_archive_fid_margin:.1f}, '
            f'min_support={self.fid_stage_min_support}, '
            f'prob={self.fid_stage_protection_prob:.2f}, '
            f'max_protected={self.fid_stage_max_protected}, '
            f'layers={self.fid_stage_protected_layers}.'
        )

        # fixed search space sizes inferred from the original code.
        self.num_layers = 3
        self.num_up_slots = 2
        self.num_normal_slots = 5
        self.num_up_ops = 3
        self.num_normal_ops = 7

        # Shared structural encoding remains available for NSAS distance,
        # archive features, adaptive mutation and diversity selection.
        self.use_structure_encoding = bool(getattr(args, 'use_structure_encoding', True))
        self.use_structure_distance = bool(getattr(args, 'use_structure_distance', True))

        # ===== guarded elite local refinement =====
        # This is also late-stage only. It starts only when global search already
        # found a good region; otherwise local refinement around poor elites will
        # simply exploit the wrong neighborhood.
        # Removed from the main line: archive-FID local refinement and tree
        # mutation already provide local exploitation. This late guarded local
        # refinement is redundant for the current ablation-friendly version.
        self.use_guarded_local_refinement = False
        self.local_refine_start_gen = int(getattr(args, 'local_refine_start_gen', 20))
        self.local_refine_min_best_fid = float(getattr(args, 'local_refine_min_best_fid', 65.0))
        self.local_refine_min_history = int(getattr(args, 'local_refine_min_history', 300))
        self.local_refine_elite_top_k = int(getattr(args, 'local_refine_elite_top_k', 3))
        self.local_refine_neighbors_per_elite = int(getattr(args, 'local_refine_neighbors_per_elite', 10))
        self.local_refine_eval_num = int(getattr(args, 'local_refine_eval_num', max(4, int(getattr(args, 'num_selected', 8)) // 2)))
        self.local_refine_mutate_slots = int(getattr(args, 'local_refine_mutate_slots', 1))
        self.local_refine_skip_seen = bool(getattr(args, 'local_refine_skip_seen', True))
        self.local_refine_max_attempt_multiplier = int(getattr(args, 'local_refine_max_attempt_multiplier', 20))
        logger.info('[GUARDED-LOCAL] disabled by core simplification.')

        # Archive best-FID local refinement is a late-stage exploitation module.
        # It does not change the global evolutionary flow; it only appends a few
        # exact-evaluated neighbors around the best FID architectures preserved in
        # the external archive.
        # Pure method: no extra archive-local exploitation. Keep the search
        # pressure inside three forests + tree crossover/mutation.
        self.use_archive_fid_local_refinement = False
        self.archive_fid_local_start_gen = int(getattr(args, 'archive_fid_local_start_gen', 8))
        self.archive_fid_local_min_best_fid = float(getattr(args, 'archive_fid_local_min_best_fid', 65.0))
        self.archive_fid_local_min_history = int(getattr(args, 'archive_fid_local_min_history', 128))
        self.archive_fid_local_elite_top_k = int(getattr(args, 'archive_fid_local_elite_top_k', 3))
        self.archive_fid_local_neighbors_per_elite = int(getattr(args, 'archive_fid_local_neighbors_per_elite', 14))
        self.archive_fid_local_eval_num = int(
            getattr(args, 'archive_fid_local_eval_num', max(4, int(getattr(args, 'num_selected', 8)) // 2))
        )
        self.archive_fid_local_mutate_slots = int(getattr(args, 'archive_fid_local_mutate_slots', 1))
        self.archive_fid_local_skip_seen = bool(getattr(args, 'archive_fid_local_skip_seen', True))
        self.archive_fid_local_max_attempt_multiplier = int(getattr(args, 'archive_fid_local_max_attempt_multiplier', 30))
        logger.info(
            f'[ARCHIVE-FID-LOCAL] enabled={self.use_archive_fid_local_refinement}, '
            f'start_gen={self.archive_fid_local_start_gen}, '
            f'min_best_fid={self.archive_fid_local_min_best_fid}, '
            f'elite_top_k={self.archive_fid_local_elite_top_k}, '
            f'neighbors_per_elite={self.archive_fid_local_neighbors_per_elite}, '
            f'eval_num={self.archive_fid_local_eval_num}, '
            f'mutate_slots={self.archive_fid_local_mutate_slots}.'
        )

        # Final FID-safe tree pruning. This runs only after the last offline
        # evolutionary generation. It starts from exact best-FID archive
        # genotypes and greedily closes normal branches only when the parameter
        # count drops and FID stays within a small tolerance of the seed.
        self.use_fid_safe_tree_pruning = True
        self.fid_prune_start_final_only = bool(getattr(args, 'fid_prune_start_final_only', True))
        self.fid_prune_archive_top_k = int(getattr(args, 'fid_prune_archive_top_k', 3))
        self.fid_prune_fid_margin = float(getattr(args, 'fid_prune_fid_margin', 1.0))
        self.fid_prune_accept_fid_delta = float(getattr(args, 'fid_prune_accept_fid_delta', 0.8))
        self.fid_prune_min_param_drop = float(getattr(args, 'fid_prune_min_param_drop', 0.05))
        self.fid_prune_max_steps = int(getattr(args, 'fid_prune_max_steps', 3))
        self.fid_prune_max_candidates_per_seed = int(getattr(args, 'fid_prune_max_candidates_per_seed', 12))
        self.fid_prune_return_size = int(getattr(args, 'fid_prune_return_size', int(getattr(args, 'num_selected', 12))))
        self.fid_prune_eval_epoch = getattr(args, 'fid_prune_eval_epoch', None)
        self.use_three_forest_tree_pruning = True
        self.three_forest_prune_seeds_per_forest = int(
            getattr(
                args,
                'three_forest_prune_seeds_per_forest',
                max(1, int(math.ceil(float(self.fid_prune_archive_top_k) / 3.0))),
            )
        )
        self.three_forest_prune_accept_fid_delta = float(
            getattr(args, 'three_forest_prune_accept_fid_delta', self.fid_prune_accept_fid_delta)
        )
        self.three_forest_prune_accept_is_drop = float(getattr(args, 'three_forest_prune_accept_is_drop', 0.15))
        self.three_forest_prune_params_fid_delta = float(
            getattr(args, 'three_forest_prune_params_fid_delta', self.fid_prune_accept_fid_delta)
        )
        self.three_forest_prune_params_is_drop = float(
            getattr(args, 'three_forest_prune_params_is_drop', self.three_forest_prune_accept_is_drop + 0.05)
        )
        self.three_forest_prune_keep_per_objective = int(getattr(args, 'three_forest_prune_keep_per_objective', 1))
        self.three_forest_prune_max_candidates_per_seed = int(
            getattr(args, 'three_forest_prune_max_candidates_per_seed', max(self.fid_prune_max_candidates_per_seed, 24))
        )
        self.three_forest_prune_min_param_drop = float(
            getattr(args, 'three_forest_prune_min_param_drop', min(self.fid_prune_min_param_drop, 0.02))
        )
        self.three_forest_prune_relaxed_accept = False
        self.three_forest_prune_relaxed_fid_delta = float(
            getattr(args, 'three_forest_prune_relaxed_fid_delta', 1.0)
        )
        self.three_forest_prune_relaxed_is_drop = float(
            getattr(args, 'three_forest_prune_relaxed_is_drop', 0.35)
        )
        self.three_forest_prune_params_relaxed_fid_delta = float(
            getattr(args, 'three_forest_prune_params_relaxed_fid_delta', 1.2)
        )
        self.three_forest_prune_params_relaxed_is_drop = float(
            getattr(args, 'three_forest_prune_params_relaxed_is_drop', 0.50)
        )
        self.three_forest_prune_generate_pair_delete = bool(
            getattr(args, 'three_forest_prune_generate_pair_delete', True)
        )
        self.three_forest_prune_generate_replace = bool(
            getattr(args, 'three_forest_prune_generate_replace', True)
        )
        # Objective-conditioned pruning policy:
        #   FID forest   -> conservative, keep FID almost unchanged;
        #   IS forest    -> medium, keep IS while allowing mild FID movement;
        #   Params forest-> aggressive, accept larger quality trade-off for size.
        self.use_objective_prune_policy = True
        self.fid_forest_prune_accept_fid_delta = float(
            getattr(args, 'fid_forest_prune_accept_fid_delta', 0.60)
        )
        self.fid_forest_prune_relaxed_fid_delta = float(
            getattr(args, 'fid_forest_prune_relaxed_fid_delta', 0.80)
        )
        self.fid_forest_prune_accept_is_drop = float(
            getattr(args, 'fid_forest_prune_accept_is_drop', 0.20)
        )
        self.fid_forest_prune_relaxed_is_drop = float(
            getattr(args, 'fid_forest_prune_relaxed_is_drop', 0.30)
        )
        self.fid_forest_prune_min_param_drop = float(
            getattr(args, 'fid_forest_prune_min_param_drop', 0.03)
        )
        self.fid_forest_prune_max_steps = int(getattr(args, 'fid_forest_prune_max_steps', 2))
        self.fid_forest_prune_pair_delete = bool(getattr(args, 'fid_forest_prune_pair_delete', False))

        self.is_forest_prune_accept_fid_delta = float(
            getattr(args, 'is_forest_prune_accept_fid_delta', 0.80)
        )
        self.is_forest_prune_relaxed_fid_delta = float(
            getattr(args, 'is_forest_prune_relaxed_fid_delta', 1.00)
        )
        self.is_forest_prune_accept_is_drop = float(
            getattr(args, 'is_forest_prune_accept_is_drop', 0.15)
        )
        self.is_forest_prune_relaxed_is_drop = float(
            getattr(args, 'is_forest_prune_relaxed_is_drop', 0.25)
        )
        self.is_forest_prune_min_param_drop = float(
            getattr(args, 'is_forest_prune_min_param_drop', 0.03)
        )
        self.is_forest_prune_max_steps = int(getattr(args, 'is_forest_prune_max_steps', 2))
        self.is_forest_prune_pair_delete = bool(getattr(args, 'is_forest_prune_pair_delete', False))

        self.params_forest_prune_accept_fid_delta = float(
            getattr(args, 'params_forest_prune_accept_fid_delta', 0.80)
        )
        self.params_forest_prune_relaxed_fid_delta = float(
            getattr(args, 'params_forest_prune_relaxed_fid_delta', 1.20)
        )
        self.params_forest_prune_accept_is_drop = float(
            getattr(args, 'params_forest_prune_accept_is_drop', 0.35)
        )
        self.params_forest_prune_relaxed_is_drop = float(
            getattr(args, 'params_forest_prune_relaxed_is_drop', 0.50)
        )
        self.params_forest_prune_min_param_drop = float(
            getattr(args, 'params_forest_prune_min_param_drop', 0.08)
        )
        self.params_forest_prune_max_steps = int(getattr(args, 'params_forest_prune_max_steps', 3))
        self.params_forest_prune_pair_delete = bool(getattr(args, 'params_forest_prune_pair_delete', True))

        self.three_forest_prune_recovery_enabled = False
        self.three_forest_prune_recovery_steps = int(
            getattr(args, 'three_forest_prune_recovery_steps', 20)
        )
        self.three_forest_prune_recovery_top_k = int(
            getattr(args, 'three_forest_prune_recovery_top_k', 2)
        )
        self.three_forest_prune_recovery_bs = int(
            getattr(args, 'three_forest_prune_recovery_bs', min(64, int(getattr(args, 'gen_bs', 64))))
        )
        self.three_forest_prune_recovery_lr = float(
            getattr(args, 'three_forest_prune_recovery_lr', float(getattr(args, 'g_lr', 2e-4)) * 0.5)
        )
        self.three_forest_prune_recovery_beta1 = float(
            getattr(args, 'three_forest_prune_recovery_beta1', float(getattr(args, 'beta1', 0.0)))
        )
        self.three_forest_prune_recovery_beta2 = float(
            getattr(args, 'three_forest_prune_recovery_beta2', float(getattr(args, 'beta2', 0.9)))
        )
        self.use_history_balanced_final_selection = False
        self.history_final_balanced_fid_ceiling = float(
            getattr(args, 'history_final_balanced_fid_ceiling', 52.0)
        )
        self.history_final_params_fid_ceiling = float(
            getattr(args, 'history_final_params_fid_ceiling', 53.0)
        )
        self.history_final_best_fid_margin = float(
            getattr(args, 'history_final_best_fid_margin', 1.5)
        )
        self.use_compact_valid_final_pruning = False
        self.compact_valid_target_params = float(
            getattr(args, 'compact_valid_target_params', 10.0)
        )
        self.compact_valid_relaxed_target_params = float(
            getattr(args, 'compact_valid_relaxed_target_params', 10.10)
        )
        self.compact_valid_seed_param_buffer = float(
            getattr(args, 'compact_valid_seed_param_buffer', 0.80)
        )
        self.compact_valid_seed_fid_margin = float(
            getattr(args, 'compact_valid_seed_fid_margin', 2.0)
        )
        self.compact_valid_accept_fid_delta = float(
            getattr(args, 'compact_valid_accept_fid_delta', 1.50)
        )
        self.compact_valid_fid_ceiling = float(
            getattr(args, 'compact_valid_fid_ceiling', 50.50)
        )
        self.compact_valid_max_steps = int(
            getattr(args, 'compact_valid_max_steps', 3)
        )
        self.compact_valid_max_candidates_per_step = int(
            getattr(args, 'compact_valid_max_candidates_per_step', 18)
        )
        self.compact_valid_min_param_drop = float(
            getattr(args, 'compact_valid_min_param_drop', 0.03)
        )
        self.compact_valid_eval_epoch = getattr(args, 'compact_valid_eval_epoch', None)
        logger.info(
            f'[TREE-PRUNE] enabled={self.use_fid_safe_tree_pruning}, '
            f'variant={TRAINER_TREE_PRUNING}, final_only={self.fid_prune_start_final_only}, '
            f'archive_top_k={self.fid_prune_archive_top_k}, fid_margin=+{self.fid_prune_fid_margin:.2f}, '
            f'accept_delta=+{self.fid_prune_accept_fid_delta:.2f}, '
            f'min_param_drop={self.fid_prune_min_param_drop:.3f}, '
            f'max_steps={self.fid_prune_max_steps}, '
            f'max_candidates_per_seed={self.fid_prune_max_candidates_per_seed}.'
        )
        logger.info(
            f'[THREE-FOREST-PRUNE] enabled={self.use_three_forest_tree_pruning}, '
            f'variant={TRAINER_THREE_FOREST_PRUNING}, '
            f'seeds_per_forest={self.three_forest_prune_seeds_per_forest}, '
            f'keep_per_objective={self.three_forest_prune_keep_per_objective}, '
            f'fid_delta=+{self.three_forest_prune_accept_fid_delta:.2f}, '
            f'is_drop={self.three_forest_prune_accept_is_drop:.2f}, '
            f'params_fid_delta=+{self.three_forest_prune_params_fid_delta:.2f}, '
            f'params_is_drop={self.three_forest_prune_params_is_drop:.2f}.'
        )
        logger.info(
            f'[THREE-FOREST-PRUNE-EFFECTIVE] min_param_drop={self.three_forest_prune_min_param_drop:.3f}, '
            f'max_candidates_per_seed={self.three_forest_prune_max_candidates_per_seed}, '
            f'relaxed_accept={self.three_forest_prune_relaxed_accept}, '
            f'relaxed_fid_delta=+{self.three_forest_prune_relaxed_fid_delta:.2f}, '
            f'relaxed_is_drop={self.three_forest_prune_relaxed_is_drop:.2f}, '
            f'params_relaxed_fid_delta=+{self.three_forest_prune_params_relaxed_fid_delta:.2f}, '
            f'params_relaxed_is_drop={self.three_forest_prune_params_relaxed_is_drop:.2f}, '
            f'pair_delete={self.three_forest_prune_generate_pair_delete}, '
            f'replace={self.three_forest_prune_generate_replace}.'
        )
        logger.info(
            f'[OBJECTIVE-PRUNE-POLICY] enabled={self.use_objective_prune_policy}, '
            f'FID(delta={self.fid_forest_prune_accept_fid_delta:.2f}/'
            f'{self.fid_forest_prune_relaxed_fid_delta:.2f}, '
            f'drop={self.fid_forest_prune_min_param_drop:.3f}, '
            f'pair={self.fid_forest_prune_pair_delete}); '
            f'IS(delta={self.is_forest_prune_accept_fid_delta:.2f}/'
            f'{self.is_forest_prune_relaxed_fid_delta:.2f}, '
            f'drop={self.is_forest_prune_min_param_drop:.3f}, '
            f'pair={self.is_forest_prune_pair_delete}); '
            f'Params(delta={self.params_forest_prune_accept_fid_delta:.2f}/'
            f'{self.params_forest_prune_relaxed_fid_delta:.2f}, '
            f'drop={self.params_forest_prune_min_param_drop:.3f}, '
            f'pair={self.params_forest_prune_pair_delete}).'
        )
        logger.info(
            f'[THREE-FOREST-PRUNE-RECOVERY] enabled={self.three_forest_prune_recovery_enabled}, '
            f'steps={self.three_forest_prune_recovery_steps}, '
            f'top_k={self.three_forest_prune_recovery_top_k}, '
            f'bs={self.three_forest_prune_recovery_bs}, '
            f'lr={self.three_forest_prune_recovery_lr:.6f}.'
        )
        logger.info(
            f'[HISTORY-FINAL-SELECT] enabled={self.use_history_balanced_final_selection}, '
            f'balanced_fid<=min(best+{self.history_final_best_fid_margin:.2f}, '
            f'{self.history_final_balanced_fid_ceiling:.2f}), '
            f'params_fid<={self.history_final_params_fid_ceiling:.2f}.'
        )
        logger.info(
            f'[COMPACT-VALID-PRUNE] enabled={self.use_compact_valid_final_pruning}, '
            f'variant={TRAINER_COMPACT_VALID_PRUNE}, targetP<{self.compact_valid_target_params:.2f}, '
            f'relaxedP<={self.compact_valid_relaxed_target_params:.2f}, '
            f'fid_ceiling={self.compact_valid_fid_ceiling:.2f}, '
            f'accept_delta=+{self.compact_valid_accept_fid_delta:.2f}, '
            f'max_steps={self.compact_valid_max_steps}, '
            f'max_candidates={self.compact_valid_max_candidates_per_step}.'
        )
        self.current_evolution_algebra = -1


        # operation success statistics for probability-guided mutation.
        self.up_op_success = np.ones((self.num_layers, self.num_up_slots, self.num_up_ops), dtype=np.float32)
        self.normal_op_success = np.ones((self.num_layers, self.num_normal_slots, self.num_normal_ops), dtype=np.float32)

        self.genotypes = np.stack([gan_alg.search() for _ in range(args.num_individual)], axis=0)
        self._validate_tree_round_trip(self.genotypes[0])

        # Safety fix: warm validation needs a real fixed generator genotype.
        # Some entry scripts pass gen_fixed_genotype=None, which makes
        # validate() crash in super_network.py when it indexes genotypes[0].
        # Fall back to the first initialized population member without changing
        # the normal training/search logic.
        if self.gen_fixed_genotype is None:
            self.gen_fixed_genotype = np.array(self.genotypes[0], copy=True)
            logger.info(
                '[FIXED-GENOTYPE] gen_fixed_genotype is None; '
                'use self.genotypes[0] for warm validation.'
            )

        self.best_is_evo = 0.0
        self.best_fid_evo = 999.0


    def _resize_fake_to_real(self, fake_imgs, real_imgs):
        """Match generated image size to real image size when a searched subnet
        outputs a different spatial resolution.
        """
        if fake_imgs.shape[2:] != real_imgs.shape[2:]:
            fake_imgs = torch.nn.functional.interpolate(
                fake_imgs, size=real_imgs.shape[2:], mode='bilinear', align_corners=False
            )
        return fake_imgs

    def _resize_to_target_img_size(self, imgs):
        """Resize generated images to the dataset image size for validation /
        auxiliary scoring functions when necessary.
        """
        target_size = int(getattr(self.args, 'img_size', imgs.shape[-1]))
        if imgs.shape[2:] != (target_size, target_size):
            imgs = torch.nn.functional.interpolate(
                imgs, size=(target_size, target_size), mode='bilinear', align_corners=False
            )
        return imgs

    # ---------------------------------------------------------------------
    # basic helpers
    # ---------------------------------------------------------------------
    def _encode_genotype(self, genotype_G):
        try:
            encoded = self.gan_alg.encode(genotype_G)
        except Exception:
            encoded = np.asarray(genotype_G, dtype=int).reshape(-1).tolist()

        if isinstance(encoded, np.ndarray):
            encoded = encoded.tolist()
        if isinstance(encoded, list):
            return tuple(encoded)
        return encoded

    def _flatten_genotype(self, genotype_G):
        return np.asarray(genotype_G, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # structure-aware / hierarchical architecture encoding
    # ------------------------------------------------------------------
    def _one_hot(self, value, size):
        vec = np.zeros(size, dtype=np.float32)
        idx = int(value)
        if 0 <= idx < size:
            vec[idx] = 1.0
        return vec


    def _arch_feature_vector(self, genotype_G, param_size=None, include_resource=True):
        """Hierarchical architecture encoding for the MLP predictor.

        macro-level: upsample choices in each layer;
        meso-level: active/sparse pattern inside each layer;
        micro-level: concrete normal-operation choices;
        resource-level: parameter-size statistics.
        """
        g = np.asarray(genotype_G, dtype=np.int64).copy()
        g = g.reshape(self.num_layers, self.num_up_slots + self.num_normal_slots)

        features = []

        # macro-level: upsample operation choices.
        # Keep the predictor feature dimensionality independent from
        # FID-stage protection. Protection may use only L1/L2, but the
        # architecture encoder must always describe all stages.
        for layer in range(self.num_layers):
            for slot in range(self.num_up_slots):
                features.append(self._one_hot(g[layer, slot], self.num_up_ops))

        # micro-level: normal operation choices.
        for layer in range(self.num_layers):
            for slot in range(self.num_normal_slots):
                op_id = g[layer, self.num_up_slots + slot]
                features.append(self._one_hot(op_id, self.num_normal_ops))

        # meso-level: block active pattern / sparsity statistics.
        meso = []
        for layer in range(self.num_layers):
            normal_ops = g[layer, self.num_up_slots:self.num_up_slots + self.num_normal_slots]
            active = (normal_ops != 0).astype(np.float32)
            group1 = active[:2]
            group2 = active[2:]
            active_count = float(active.sum())
            meso.extend([
                active_count / max(1, self.num_normal_slots),
                float(group1.sum()) / 2.0,
                float(group2.sum()) / 3.0,
                1.0 - active_count / max(1, self.num_normal_slots),
            ])
        features.append(np.asarray(meso, dtype=np.float32))

        # resource-level: param statistics. For structure-distance computations we can
        # disable this part to avoid constructing Generator repeatedly.
        if include_resource:
            if param_size is None:
                param_size = self._get_param_size(g)
            target = max(1e-8, float(getattr(self, 'target_model_size', 10.0)))
            max_size = max(target, float(getattr(self.args, 'max_model_size', target)))
            total_active = float(np.sum(g[:, self.num_up_slots:] != 0)) / max(1, self.num_layers * self.num_normal_slots)
            resource = np.asarray([
                total_active,
                min(float(param_size) / target, 3.0),
                max(0.0, float(param_size) - target) / target,
                min(float(param_size) / max_size, 3.0),
            ], dtype=np.float32)
        else:
            resource = np.zeros(4, dtype=np.float32)
        features.append(resource)

        return np.concatenate(features, axis=0).astype(np.float32)


    def _arch_structure_vector(self, genotype_G):
        return self._arch_feature_vector(genotype_G, param_size=0.0, include_resource=False)

    def _geno_distance(self, genotype_a, genotype_b):
        if getattr(self, 'use_structure_distance', True):
            vec_a = self._arch_structure_vector(genotype_a)
            vec_b = self._arch_structure_vector(genotype_b)
            return float(np.mean(np.abs(vec_a - vec_b)))

        vec_a = self._flatten_genotype(genotype_a)
        vec_b = self._flatten_genotype(genotype_b)
        if vec_a.shape != vec_b.shape:
            min_len = min(len(vec_a), len(vec_b))
            vec_a = vec_a[:min_len]
            vec_b = vec_b[:min_len]
        return float(np.mean(vec_a != vec_b))

    def _deduplicate_genotypes(self, genotype_list):
        unique = []
        seen = set()
        for genotype in genotype_list:
            key = self._encode_genotype(genotype)
            if key not in seen:
                unique.append(np.array(genotype, copy=True))
                seen.add(key)
        return unique

    def _record_architecture_visit(self, genotypes):
        for genotype in genotypes:
            key = self._encode_genotype(genotype)
            self.arch_visit_counter[key] = self.arch_visit_counter.get(key, 0) + 1

    def _get_param_size(self, genotype_G):
        key = self._encode_genotype(genotype_G)
        if key not in self.param_cache:
            self.param_cache[key] = count_parameters_in_MB(Generator(self.args, genotype_G))
        return self.param_cache[key]

    def _cache_exact_result(self, genotype_G, is_value, fid_value, param_size):
        key = self._encode_genotype(genotype_G)
        self.historical_population[key] = {
            'is': float(is_value),
            'fid': float(fid_value),
            'param': float(param_size),
            'exact': True,
            'vector': self._flatten_genotype(genotype_G).astype(np.float32),
            'arch_feat': self._arch_feature_vector(genotype_G, param_size=param_size).astype(np.float32),
            'genotype': np.array(genotype_G, copy=True),
        }

    def _external_archive_enabled(self):
        return False

    def _external_archive_candidate_records(self):
        max_model_size = float(getattr(self.args, 'max_model_size', 1e9))
        records = []
        for record in self._get_exact_history():
            genotype = record.get('genotype', None)
            if genotype is None:
                continue
            param = record.get('param', None)
            if param is None:
                param = self._get_param_size(genotype)
            param = float(param)
            if self.enforce_model_size_constraint and param > max_model_size:
                continue
            records.append(record)
        if len(records) == 0:
            records = [record for record in self._get_exact_history() if record.get('genotype', None) is not None]
        return records

    def _external_archive_params_quality_ok(self, record):
        fid_value = float(record.get('fid', 999.0))
        max_fid = float(getattr(self, 'external_archive_param_max_fid', 55.0))
        margin = max(0.0, float(getattr(self, 'three_elite_param_best_fid_margin', 2.5)))
        best_fid = self._best_exact_fid()
        if not np.isfinite(best_fid):
            return fid_value <= max_fid
        return fid_value <= min(max_fid, float(best_fid) + margin)

    def _external_archive_sort_key(self, record, objective):
        is_value = float(record.get('is', 0.0))
        fid_value = float(record.get('fid', 999.0))
        param = float(record.get('param', 1e9))
        if objective == 'fid':
            return (fid_value, -is_value, param)
        if objective == 'is':
            return (-is_value, fid_value, param)
        if objective == 'params':
            quality_penalty = 0 if self._external_archive_params_quality_ok(record) else 1
            return (quality_penalty, param, fid_value, -is_value)
        if objective == 'balanced':
            return (fid_value, -is_value, param)
        raise ValueError(f'Unknown external archive objective: {objective}')

    def _external_archive_add_records(self, archive, seen, records, objective, limit):
        for record in sorted(records, key=lambda item: self._external_archive_sort_key(item, objective)):
            genotype = record.get('genotype', None)
            if genotype is None:
                continue
            key = self._encode_genotype(genotype)
            if key in seen:
                continue
            seen.add(key)
            archive.append(record)
            if len(archive) >= limit:
                return True
        return False

    def _refresh_external_elite_archive(self, log_summary=False):
        if not self._external_archive_enabled():
            self.external_elite_archive = []
            return []
        records = self._external_archive_candidate_records()
        if len(records) == 0:
            self.external_elite_archive = []
            return []

        archive, seen = [], set()
        limit = max(1, int(self.external_archive_size))
        per_objective = max(1, min(limit, int(math.ceil(limit / 4.0))))
        for objective in ('fid', 'is', 'params', 'balanced'):
            objective_source = records
            if objective == 'params':
                objective_source = [
                    record for record in records
                    if self._external_archive_params_quality_ok(record)
                ]
                if len(objective_source) == 0:
                    continue
            objective_records = sorted(
                objective_source,
                key=lambda item: self._external_archive_sort_key(item, objective),
            )[:per_objective]
            self._external_archive_add_records(archive, seen, objective_records, objective, limit)
            if len(archive) >= limit:
                break

        fill_records = [
            record for record in records
            if self._external_archive_params_quality_ok(record)
        ]
        if len(fill_records) == 0:
            fill_records = records
        if len(archive) < limit:
            self._external_archive_add_records(archive, seen, fill_records, 'fid', limit)
        if len(archive) < limit:
            self._external_archive_add_records(archive, seen, fill_records, 'is', limit)

        self.external_elite_archive = list(archive[:limit])
        if log_summary and len(self.external_elite_archive) > 0:
            best_fid = min(self.external_elite_archive, key=lambda r: (float(r.get('fid', 999.0)), -float(r.get('is', 0.0))))
            best_is = max(self.external_elite_archive, key=lambda r: (float(r.get('is', 0.0)), -float(r.get('fid', 999.0))))
            param_pool = [
                record for record in self.external_elite_archive
                if self._external_archive_params_quality_ok(record)
            ]
            if len(param_pool) == 0:
                param_pool = self.external_elite_archive
            best_param = min(param_pool, key=lambda r: self._external_archive_sort_key(r, 'params'))
            logger.info(
                '[EXTERNAL-ARCHIVE] size=%d, best_fid(IS=%.3f,FID=%.3f,P=%.3f), '
                'best_is(IS=%.3f,FID=%.3f,P=%.3f), best_params(IS=%.3f,FID=%.3f,P=%.3f).',
                len(self.external_elite_archive),
                float(best_fid.get('is', 0.0)), float(best_fid.get('fid', 999.0)), float(best_fid.get('param', 1e9)),
                float(best_is.get('is', 0.0)), float(best_is.get('fid', 999.0)), float(best_is.get('param', 1e9)),
                float(best_param.get('is', 0.0)), float(best_param.get('fid', 999.0)), float(best_param.get('param', 1e9)),
            )
        return self.external_elite_archive

    def _get_external_archive_genotypes(self, top_k=None):
        return []
        if not self._external_archive_enabled():
            return []
        if top_k is None:
            top_k = self.external_archive_parent_size
        if len(getattr(self, 'external_elite_archive', [])) == 0:
            self._refresh_external_elite_archive(log_summary=False)
        genotypes = []
        for record in self.external_elite_archive[:max(0, int(top_k))]:
            genotype = record.get('genotype', None)
            if genotype is not None:
                genotypes.append(np.array(genotype, copy=True))
        return genotypes

    def _archive_best_fid_injection_enabled(self):
        return bool(getattr(self, 'use_archive_best_fid_injection', False)) and self._external_archive_enabled()

    def _get_external_archive_best_fid_genotypes(self, top_k=None):
        return []
        if not self._archive_best_fid_injection_enabled():
            return []
        if top_k is None:
            top_k = self.archive_best_fid_parent_top_k
        top_k = max(0, int(top_k))
        if top_k <= 0:
            return []
        archive = self._refresh_external_elite_archive(log_summary=False)
        if len(archive) == 0:
            return []
        max_model_size = float(getattr(self.args, 'max_model_size', 1e9))
        records = []
        for record in archive:
            genotype = record.get('genotype', None)
            if genotype is None:
                continue
            param = float(record.get('param', self._get_param_size(genotype)))
            if self.enforce_model_size_constraint and param > max_model_size:
                continue
            records.append(record)
        if len(records) == 0:
            records = [record for record in archive if record.get('genotype', None) is not None]
        records = sorted(
            records,
            key=lambda record: (
                float(record.get('fid', 999.0)),
                -float(record.get('is', 0.0)),
                float(record.get('param', 1e9)),
            ),
        )
        genotypes, seen = [], set()
        for record in records:
            genotype = record.get('genotype', None)
            if genotype is None:
                continue
            key = self._encode_genotype(genotype)
            if key in seen:
                continue
            seen.add(key)
            genotypes.append(np.array(genotype, copy=True))
            if len(genotypes) >= top_k:
                break
        return genotypes

    def _fid_stage_protection_allowed(self):
        if not bool(getattr(self, 'use_fid_stage_protection', False)):
            return False
        generation = int(getattr(self, 'current_evolution_algebra', -1))
        if generation < int(getattr(self, 'fid_stage_protection_start_gen', 12)):
            return False
        if len(self._get_exact_history()) < int(getattr(self, 'fid_stage_protection_min_history', 300)):
            return False
        if float(getattr(self, 'best_fid_evo', 999.0)) > float(
            getattr(self, 'fid_stage_protection_min_best_fid', 55.0)
        ):
            return False
        return True

    def _get_fid_stage_archive_records(self):
        if not self._fid_stage_protection_allowed():
            return []
        records = [
            record for record in self._refresh_external_elite_archive(log_summary=False)
            if record.get('genotype', None) is not None
        ]
        if len(records) == 0:
            return []
        records = sorted(
            records,
            key=lambda record: (
                float(record.get('fid', 999.0)),
                -float(record.get('is', 0.0)),
                float(record.get('param', 1e9)),
            ),
        )
        best_fid = float(records[0].get('fid', 999.0))
        margin = max(0.0, float(getattr(self, 'fid_stage_archive_fid_margin', 3.0)))
        qualified = [
            record for record in records
            if float(record.get('fid', 999.0)) <= best_fid + margin
        ]
        top_k = max(1, int(getattr(self, 'fid_stage_archive_top_k', 8)))
        return qualified[:top_k]

    def _build_fid_stage_bank(self):
        generation = int(getattr(self, 'current_evolution_algebra', -1))
        history_size = len(self._get_exact_history())
        if (
            self._fid_stage_bank_cache_generation == generation
            and self._fid_stage_bank_cache_history_size == history_size
        ):
            return self._fid_stage_bank_cache

        records = self._get_fid_stage_archive_records()
        bank = {}
        min_support = max(1, int(getattr(self, 'fid_stage_min_support', 2)))
        protected_layers = [
            int(layer)
            for layer in getattr(self, 'fid_stage_protected_layers', (1, 2))
            if 0 <= int(layer) < int(self.num_layers)
        ]
        for layer in protected_layers:
            stage_stats = {}
            for rank_idx, record in enumerate(records):
                genotype = np.asarray(record['genotype'], dtype=np.int64)
                stage_key = tuple(genotype[layer].tolist())
                stat = stage_stats.setdefault(
                    stage_key,
                    {
                        'support': 0,
                        'weighted_fid': 0.0,
                        'weight_sum': 0.0,
                        'best_fid': 999.0,
                        'best_is': 0.0,
                    },
                )
                rank_weight = 1.0 / (1.0 + float(rank_idx))
                fid_value = float(record.get('fid', 999.0))
                is_value = float(record.get('is', 0.0))
                stat['support'] += 1
                stat['weighted_fid'] += rank_weight * fid_value
                stat['weight_sum'] += rank_weight
                stat['best_fid'] = min(stat['best_fid'], fid_value)
                stat['best_is'] = max(stat['best_is'], is_value)

            supported = [
                (stage_key, stat)
                for stage_key, stat in stage_stats.items()
                if int(stat['support']) >= min_support
            ]
            if len(supported) == 0:
                supported = list(stage_stats.items())
            if len(supported) == 0:
                continue
            supported.sort(
                key=lambda item: (
                    float(item[1]['weighted_fid']) / max(float(item[1]['weight_sum']), 1e-8),
                    float(item[1]['best_fid']),
                    -int(item[1]['support']),
                    -float(item[1]['best_is']),
                )
            )
            stage_key, stat = supported[0]
            bank[layer] = {
                'stage': np.asarray(stage_key, dtype=np.int64),
                'support': int(stat['support']),
                'best_fid': float(stat['best_fid']),
            }

        self._fid_stage_bank_cache_generation = generation
        self._fid_stage_bank_cache_history_size = history_size
        self._fid_stage_bank_cache = bank
        if (
            len(bank) > 0
            and self._fid_stage_protection_log_generation != generation
        ):
            summary = ','.join(
                f'L{layer}:support={bank[layer]["support"]},bestFID={bank[layer]["best_fid"]:.3f}'
                for layer in sorted(bank)
            )
            logger.info(
                f'[FID-STAGE-BANK] gen={generation}, archive_records={len(records)}, '
                f'protected_candidates={len(bank)}, {summary}.'
            )
            self._fid_stage_protection_log_generation = generation
        return bank

    def _apply_fid_stage_protection(self, genotype_G):
        child = np.asarray(genotype_G, dtype=np.int64).copy()
        return child
        if not self._fid_stage_protection_allowed():
            return child
        probability = min(
            max(float(getattr(self, 'fid_stage_protection_prob', 0.70)), 0.0),
            1.0,
        )
        if random.random() >= probability:
            return child
        bank = self._build_fid_stage_bank()
        available_layers = sorted(bank)
        if len(available_layers) == 0:
            return child
        protect_num = min(
            len(available_layers),
            max(1, int(getattr(self, 'fid_stage_max_protected', 1))),
        )
        for layer in random.sample(available_layers, protect_num):
            child[layer] = np.asarray(bank[layer]['stage'], dtype=np.int64)
        return self._repair_genotype(child)

    def _inject_archive_best_fid_into_population(self, population, target_size=None, context=''):
        population = np.asarray(population, dtype=np.int64)
        if population.ndim != 3 or population.shape[0] == 0:
            return population
        if target_size is None:
            target_size = population.shape[0]
        target_size = max(1, int(target_size))
        return population[:target_size].astype(np.int64)
        elites = self._get_external_archive_best_fid_genotypes(self.archive_best_fid_inject_top_k)
        if len(elites) == 0:
            return population[:target_size].astype(np.int64)

        next_population, seen = [], set()
        injected = 0
        for genotype in elites:
            if len(next_population) >= target_size:
                break
            key = self._encode_genotype(genotype)
            if key in seen:
                continue
            seen.add(key)
            next_population.append(np.array(genotype, copy=True))
            injected += 1

        for genotype in population:
            if len(next_population) >= target_size:
                break
            key = self._encode_genotype(genotype)
            if key in seen:
                continue
            seen.add(key)
            next_population.append(np.array(genotype, copy=True))

        idx = 0
        while len(next_population) < target_size:
            next_population.append(np.array(population[idx % population.shape[0]], copy=True))
            idx += 1

        if injected > 0:
            logger.info(
                f'[ARCHIVE-BEST-FID-INJECT] {context}: injected={injected}, '
                f'population_size={target_size}.'
            )
        return np.stack(next_population[:target_size], axis=0).astype(np.int64)

    def _get_external_archive_return_genotypes(self, target_num=None):
        return []
        if target_num is None:
            target_num = self.external_archive_return_size
        target_num = max(1, int(target_num))
        archive = self._refresh_external_elite_archive(log_summary=True)
        if len(archive) == 0:
            return []

        selected, seen = [], set()

        def add_one(objective):
            for record in sorted(archive, key=lambda item: self._external_archive_sort_key(item, objective)):
                genotype = record.get('genotype', None)
                if genotype is None:
                    continue
                key = self._encode_genotype(genotype)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(np.array(genotype, copy=True))
                return True
            return False

        # First preserve the best exact elites for the three explicit objectives.
        for objective in ('fid', 'is', 'params'):
            add_one(objective)
            if len(selected) >= target_num:
                return selected

        # Then fill the remaining slots by cycling objectives instead of letting one metric dominate.
        while len(selected) < target_num:
            before = len(selected)
            for objective in ('balanced', 'fid', 'is', 'params'):
                add_one(objective)
                if len(selected) >= target_num:
                    return selected
            if len(selected) == before:
                break
        return selected

    def _get_exact_history(self):
        exact_entries = []
        for value in self.historical_population.values():
            if value.get('exact', False):
                exact_entries.append(value)
        return exact_entries

    def _elite_pool_ready(self):
        return (
            bool(getattr(self, 'use_elite_pool', True))
            and len(self._get_exact_history()) >= int(self.elite_min_history)
        )

    def _three_elite_pools_ready(self):
        return (
            bool(getattr(self, 'use_elite_pool', True))
            and bool(getattr(self, 'use_three_elite_pools', True))
            and len(self._get_exact_history()) >= int(self.three_elite_min_history)
        )

    def _eligible_exact_elite_records(self):
        records = []
        for item in self._get_exact_history():
            if item.get('genotype') is None:
                continue
            p = item.get('param')
            if p is None:
                p = self._get_param_size(item['genotype'])
            if np.all(np.isfinite([item.get('is', np.nan), item.get('fid', np.nan), p])) and p <= self.args.max_model_size:
                records.append(item)
        return records

    def _three_elite_params_quality_ok(self, record):
        # Eligibility is enforced by the full quantile gates in the forest builder.
        key = self._encode_genotype(record['genotype'])
        return any(self._encode_genotype(r['genotype']) == key for r in
                   self._get_objective_elite_records('params', len(self._get_exact_history())))

    def _objective_elite_sort_key(self, record, objective):
        s, f, p = float(record['is']), float(record['fid']), float(record['param'])
        if objective == 'is':
            return (-s, f, p)
        if objective == 'fid':
            return (f, -s, p)
        if objective == 'params':
            return (p, f, -s)
        raise ValueError('Unknown forest objective: %s' % objective)

    def _current_three_forest_parent_weights(self):
        """Use the configured weights unchanged throughout the run."""
        return {
            'fid': self.three_elite_fid_parent_weight,
            'is': self.three_elite_is_parent_weight,
            'params': self.three_elite_params_parent_weight,
        }

    def _weighted_choice_pool_name(self, pool_names):
        pool_names = [name for name in pool_names if name in ('fid', 'is', 'params')]
        if len(pool_names) == 0:
            return None
        weight_map = self._current_three_forest_parent_weights()
        weights = np.asarray([weight_map.get(name, 0.0) for name in pool_names], dtype=np.float64)
        if float(np.sum(weights)) <= 0.0:
            weights = np.ones(len(pool_names), dtype=np.float64)
        weights = weights / np.sum(weights)
        return pool_names[int(np.random.choice(len(pool_names), p=weights))]

    def _objective_forest_enabled(self, objective):
        if objective not in ('is', 'fid', 'params'):
            return False
        return bool(getattr(self, f'use_{objective}_forest_clustering', False))

    def _objective_forest_candidate_multiplier(self, objective):
        return max(
            1,
            int(getattr(self, f'{objective}_forest_candidate_multiplier', 4)),
        )

    def _objective_forest_num_clusters(self, objective, target_k):
        return max(
            1,
            int(getattr(self, f'{objective}_forest_num_clusters', target_k)),
        )

    def _objective_forest_record_sort_key(self, record, objective):
        return self._objective_elite_sort_key(record, objective)

    def _build_objective_forest_records(self, records, target_k, objective):
        """Keep objective elites that are strong and structurally diverse.

        Records have already passed the objective-specific quality gate in
        _get_objective_elite_records. This method deduplicates them, builds
        ordered-tree-distance prototypes, assigns each record to its nearest
        prototype, and keeps the objective winner from every structure cluster.
        """
        target_k = max(0, int(target_k))
        if target_k <= 0 or len(records) == 0:
            return []
        if objective not in ('is', 'fid', 'params'):
            raise ValueError(f'Unknown forest objective: {objective}')
        sort_key = lambda record: self._objective_forest_record_sort_key(record, objective)
        if (not self._objective_forest_enabled(objective)) or len(records) <= target_k:
            return sorted(records, key=sort_key)[:target_k]

        unique, seen = [], set()
        for record in sorted(records, key=sort_key):
            genotype = record.get('genotype', None)
            if genotype is None:
                continue
            key = self._encode_genotype(genotype)
            if key in seen:
                continue
            seen.add(key)
            unique.append(record)

        if len(unique) <= target_k:
            return unique[:target_k]

        cluster_num = min(
            len(unique),
            target_k,
            self._objective_forest_num_clusters(objective, target_k),
        )
        prototypes = [unique[0]]
        while len(prototypes) < cluster_num:
            best_record = None
            best_score = -1e9
            for rank_idx, record in enumerate(unique):
                record_key = self._encode_genotype(record['genotype'])
                if any(record_key == self._encode_genotype(proto['genotype']) for proto in prototypes):
                    continue
                min_dist = min(
                    self._tree_genotype_distance(record['genotype'], proto['genotype'])
                    for proto in prototypes
                )
                objective_rank_bonus = 1.0 / (1.0 + float(rank_idx))
                score = float(min_dist) + 0.02 * objective_rank_bonus
                if score > best_score:
                    best_score = score
                    best_record = record
            if best_record is None:
                break
            prototypes.append(best_record)

        clusters = [[] for _ in prototypes]
        for record in unique:
            nearest = 0
            nearest_dist = None
            for idx, proto in enumerate(prototypes):
                dist = self._tree_genotype_distance(record['genotype'], proto['genotype'])
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest = idx
            clusters[nearest].append(record)

        selected, selected_keys = [], set()
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            best = sorted(cluster, key=sort_key)[0]
            key = self._encode_genotype(best['genotype'])
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(best)

        for record in unique:
            if len(selected) >= target_k:
                break
            key = self._encode_genotype(record['genotype'])
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(record)

        selected = sorted(selected, key=sort_key)[:target_k]
        if len(selected) > 0:
            best = selected[0]
            logger.info(
                f'[{objective.upper()}-FOREST] candidates={len(unique)}, '
                f'clusters={len(prototypes)}, '
                f'selected={len(selected)}, best(IS={float(best.get("is", 0.0)):.3f},'
                f'FID={float(best.get("fid", 999.0)):.3f},'
                f'P={float(best.get("param", 1e9)):.3f}).'
            )
        return selected

    def _best_exact_fid(self, records=None):
        if records is None:
            records = self._eligible_exact_elite_records()
        values = [
            float(record.get('fid', 999.0))
            for record in records
            if np.isfinite(float(record.get('fid', 999.0)))
        ]
        return min(values) if len(values) > 0 else float(getattr(self, 'best_fid_evo', 999.0))

    def _objective_forest_fid_limit(self, objective, records, adaptive_ceiling):
        best_fid = self._best_exact_fid(records)
        if objective == 'is':
            margin = float(getattr(self, 'is_forest_best_fid_margin', 6.0))
        elif objective == 'params':
            margin = float(getattr(self, 'params_forest_best_fid_margin', 8.0))
        else:
            return float(adaptive_ceiling)
        return min(float(adaptive_ceiling), float(best_fid) + max(0.0, margin))

    def _get_objective_elite_records(self, objective, top_k):
        """Strict quality gates; never fill from an ineligible candidate set."""
        records = self._eligible_exact_elite_records()
        top_k = max(0, int(top_k))
        if not records or top_k == 0:
            return []
        rho = float(self.three_elite_quality_keep_ratio)
        if not 0 < rho <= 1:
            raise ValueError('Forest retention ratio must be in (0, 1].')
        tau_s = float(np.quantile([r['is'] for r in records], 1 - rho))
        tau_f = float(np.quantile([r['fid'] for r in records], rho))
        fmin = min(r['fid'] for r in records)
        if objective == 'is':
            limit = min(tau_f, fmin + self.is_forest_best_fid_margin)
            gated = [r for r in records if r['fid'] <= limit]
        elif objective == 'fid':
            gated = [r for r in records if r['is'] >= tau_s]
        elif objective == 'params':
            limit = min(tau_f, fmin + self.params_forest_best_fid_margin)
            gated = [r for r in records if r['is'] >= tau_s and r['fid'] <= limit]
        else:
            raise ValueError('Unknown forest objective: %s' % objective)
        if self._objective_forest_enabled(objective):
            return self._build_objective_forest_records(gated, top_k, objective)
        return sorted(gated, key=lambda r: self._objective_elite_sort_key(r, objective))[:top_k]

    def _get_three_elite_records(self, top_k=None):
        if top_k is None:
            top_k = self.three_elite_pool_size
        if not self._three_elite_pools_ready():
            return {'is': [], 'fid': [], 'params': []}
        return {
            objective: self._get_objective_elite_records(objective, top_k)
            for objective in ('is', 'fid', 'params')
        }

    def _build_three_elite_pools(self):
        records_by_pool = self._get_three_elite_records(self.three_elite_pool_size)
        pools = {}
        for objective, records in records_by_pool.items():
            pools[objective] = [
                np.array(record['genotype'], copy=True)
                for record in records
            ]

        if any(len(pool) > 0 for pool in pools.values()):
            summaries = []
            for objective in ('is', 'fid', 'params'):
                records = records_by_pool[objective]
                if len(records) == 0:
                    summaries.append(f'{objective}=0')
                    continue
                best = records[0]
                summaries.append(
                    f'{objective}={len(records)}'
                    f'(IS={float(best.get("is", 0.0)):.3f},'
                    f'FID={float(best.get("fid", 999.0)):.3f},'
                    f'P={float(best.get("param", 1e9)):.3f})'
                )
            logger.info('[THREE-ELITE] ' + ', '.join(summaries))
        return pools

    def _sample_three_elite_parent_pair(self, current_population, elite_pools):
        """Sample parents using same-pool, cross-pool and population routes."""
        current_population = np.asarray(current_population, dtype=np.int64)
        nonempty_names = [
            name for name in ('is', 'fid', 'params')
            if len(elite_pools.get(name, [])) > 0
        ]
        same_ready = [
            name for name in nonempty_names
            if len(elite_pools[name]) >= 2
        ]

        ratios = np.asarray([
            max(0.0, float(self.three_elite_same_pool_ratio)),
            max(0.0, float(self.three_elite_cross_pool_ratio)),
            max(0.0, float(self.three_elite_population_ratio)),
        ], dtype=np.float64)
        if float(np.sum(ratios)) <= 0.0:
            ratios = np.asarray([0.60, 0.30, 0.10], dtype=np.float64)
        ratios = ratios / np.sum(ratios)
        route = int(np.random.choice(3, p=ratios))

        if route == 0 and len(same_ready) > 0:
            pool_name = self._weighted_choice_pool_name(same_ready)
            if pool_name is None:
                pool_name = random.choice(same_ready)
            pool = elite_pools[pool_name]
            if self.use_tree_distance_parent_selection:
                return self._sample_tree_diverse_pair(pool)
            a, b = random.sample(range(len(pool)), 2)
            return np.array(pool[a], copy=True), np.array(pool[b], copy=True)

        if route == 1 and len(nonempty_names) >= 2:
            for _ in range(8):
                pool_a_name = self._weighted_choice_pool_name(nonempty_names)
                remaining_names = [name for name in nonempty_names if name != pool_a_name]
                pool_b_name = self._weighted_choice_pool_name(remaining_names)
                if pool_a_name is None or pool_b_name is None:
                    pool_a_name, pool_b_name = random.sample(nonempty_names, 2)
                if self.use_tree_distance_parent_selection:
                    parent_a, parent_b = self._sample_tree_diverse_cross_pool_pair(
                        elite_pools[pool_a_name], elite_pools[pool_b_name]
                    )
                else:
                    parent_a = random.choice(elite_pools[pool_a_name])
                    parent_b = random.choice(elite_pools[pool_b_name])
                if self._encode_genotype(parent_a) != self._encode_genotype(parent_b):
                    return np.array(parent_a, copy=True), np.array(parent_b, copy=True)

        if current_population.shape[0] >= 2:
            a, b = self._select_parent_indices(current_population)
            return (
                np.array(current_population[a], copy=True),
                np.array(current_population[b], copy=True),
            )

        combined = []
        seen = set()
        for name in nonempty_names:
            for genotype in elite_pools[name]:
                key = self._encode_genotype(genotype)
                if key not in seen:
                    seen.add(key)
                    combined.append(np.array(genotype, copy=True))
        for genotype in current_population:
            key = self._encode_genotype(genotype)
            if key not in seen:
                seen.add(key)
                combined.append(np.array(genotype, copy=True))

        if len(combined) >= 2:
            return tuple(random.sample(combined, 2))
        if len(combined) == 1:
            return np.array(combined[0], copy=True), np.array(combined[0], copy=True)
        raise RuntimeError('No parent is available for offspring generation.')

    def _get_elite_records(self, top_k, exclude_keys=None):
        """Select real-evaluated elite records from historical_population.

        historical_population stores all exact IS/FID records. The elite pool is
        rebuilt from it on demand, so predictor-only candidates can never become
        elites before exact evaluation.
        """
        top_k = max(0, int(top_k))
        if top_k <= 0:
            return []

        exclude_keys = set(exclude_keys or [])
        max_model_size = float(getattr(self.args, 'max_model_size', 1e9))
        records = []
        for item in self._get_exact_history():
            genotype = item.get('genotype', None)
            if genotype is None:
                continue
            key = self._encode_genotype(genotype)
            if key in exclude_keys:
                continue
            param = float(item.get('param', self._get_param_size(genotype)))
            if self.enforce_model_size_constraint and param > max_model_size:
                continue
            records.append(item)

        records = sorted(
            records,
            key=lambda r: (float(r.get('fid', 999.0)), -float(r.get('is', 0.0)), float(r.get('param', 1e9))),
        )
        return records[:top_k]

    def _get_elite_pool(self, top_k, exclude_keys=None):
        return [
            np.array(item['genotype'], copy=True)
            for item in self._get_elite_records(top_k, exclude_keys=exclude_keys)
        ]

    def _build_elite_parent_source(self, current_population):
        current_population = np.asarray(current_population, dtype=np.int64)
        if current_population.ndim < 3 or current_population.shape[0] == 0:
            return current_population
        if not self._elite_pool_ready():
            return current_population

        ratio = min(max(float(getattr(self, 'elite_parent_ratio', 0.25)), 0.0), 1.0)
        elite_num = max(1, int(round(current_population.shape[0] * ratio))) if ratio > 0 else 0
        if elite_num <= 0:
            return current_population

        parent_list = [np.array(g, copy=True) for g in current_population]
        seen = {self._encode_genotype(g) for g in parent_list}
        elites = self._get_elite_pool(elite_num, exclude_keys=seen)
        if len(elites) == 0:
            return current_population

        parent_list.extend(elites)
        logger.info(
            f'[ELITE-POOL] parent_source: current={current_population.shape[0]}, '
            f'elite={len(elites)}, total={len(parent_list)}.'
        )
        return np.stack(parent_list, axis=0).astype(np.int64)

    def _update_operation_statistics_with_elites(self, selected_genotypes, is_values, fid_values, params):
        return
        selected_genotypes = np.asarray(selected_genotypes, dtype=np.int64)
        if len(selected_genotypes) == 0:
            return

        geno_list = [np.array(g, copy=True) for g in selected_genotypes]
        is_list = [float(v) for v in np.asarray(is_values, dtype=np.float32)]
        fid_list = [float(v) for v in np.asarray(fid_values, dtype=np.float32)]
        param_list = [float(v) for v in np.asarray(params, dtype=np.float32)]
        seen = {self._encode_genotype(g) for g in geno_list}

        if self._elite_pool_ready():
            ratio = min(max(float(getattr(self, 'elite_operation_ratio', 0.25)), 0.0), 1.0)
            elite_num = max(1, int(round(len(geno_list) * ratio))) if ratio > 0 else 0
            elite_records = []
            if elite_num > 0 and self._three_elite_pools_ready():
                per_pool = max(1, int(math.ceil(float(elite_num) / 3.0)))
                records_by_pool = self._get_three_elite_records(per_pool)
                for rank_idx in range(per_pool):
                    for objective in ('is', 'fid', 'params'):
                        records = records_by_pool[objective]
                        if rank_idx < len(records):
                            elite_records.append(records[rank_idx])
            elif elite_num > 0:
                elite_records = self._get_elite_records(elite_num, exclude_keys=seen)

            added_elites = 0
            for item in elite_records:
                genotype = item.get('genotype', None)
                if genotype is None:
                    continue
                key = self._encode_genotype(genotype)
                if key in seen:
                    continue
                seen.add(key)
                geno_list.append(np.array(genotype, copy=True))
                is_list.append(float(item.get('is', 0.0)))
                fid_list.append(float(item.get('fid', 999.0)))
                param_list.append(float(item.get('param', self._get_param_size(genotype))))
                added_elites += 1
                if added_elites >= elite_num:
                    break

        self._update_operation_statistics(
            np.asarray(geno_list, dtype=np.int64),
            np.asarray(is_list, dtype=np.float32),
            np.asarray(fid_list, dtype=np.float32),
            np.asarray(param_list, dtype=np.float32),
        )

    def _update_replay_buffer(self, genotypes):
        self.replay_buffer = []
        return
        for genotype in genotypes:
            key = self._encode_genotype(genotype)
            kept = []
            for cached in self.replay_buffer:
                if self._encode_genotype(cached) != key:
                    kept.append(cached)
            self.replay_buffer = kept
            self.replay_buffer.insert(0, np.array(genotype, copy=True))
            if len(self.replay_buffer) > self.replay_buffer_size:
                self.replay_buffer = self.replay_buffer[:self.replay_buffer_size]

    # ---------------------------------------------------------------------
    # quality-guided selection helpers
    # ---------------------------------------------------------------------
    def _normalize_objective(self, values, higher_is_better=True):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if abs(v_max - v_min) < 1e-12:
            return np.ones_like(values, dtype=np.float32) * 0.5
        norm = (values - v_min) / (v_max - v_min)
        if not higher_is_better:
            norm = 1.0 - norm
        return norm.astype(np.float32)

    def _smaller_params_score(self, params):
        """Score parameter size as a bounded preference for smaller models.

        Dynamic PCS supplies the preferred ceiling for the current generation.
        Inside that ceiling, this version rewards smaller architectures instead
        of rewarding architectures merely because they are close to the target.
        """
        params = np.asarray(params, dtype=np.float32)
        if params.size == 0:
            return params
        target = max(1e-8, float(self._current_pcs_target_model_size()))
        score = 1.0 - np.minimum(params / target, 1.0)
        return score.astype(np.float32)

    def _quality_scores(self, is_values, fid_values, params):
        """Composite score used only for tie-breaking and final ordering.

        The actual NSGA selection uses only IS/FID after a hard model-size
        filter. This score is only used as an FID-dominant tie-breaker for
        final ordering. Params has zero weight by default.
        """
        is_values = np.asarray(is_values, dtype=np.float32)
        fid_values = np.asarray(fid_values, dtype=np.float32)
        params = np.asarray(params, dtype=np.float32)

        is_score = self._normalize_objective(is_values, higher_is_better=True)
        fid_score = self._normalize_objective(fid_values, higher_is_better=False)
        if bool(getattr(self, 'use_fid_gated_smaller_params', True)):
            param_score = self._smaller_params_score(params)
        elif self.prefer_target_model_size:
            target = max(1e-8, float(self.target_model_size))
            param_score = 1.0 - np.minimum(np.abs(params - target) / target, 1.0)
            param_score = param_score.astype(np.float32)
        else:
            param_score = self._normalize_objective(params, higher_is_better=False)

        weight_sum = max(
            1e-8,
            abs(self.quality_is_weight) + abs(self.quality_fid_weight) + abs(self.quality_param_weight),
        )
        score = (
            self.quality_is_weight * is_score
            + self.quality_fid_weight * fid_score
            + self.quality_param_weight * param_score
        ) / weight_sum
        return score.astype(np.float32)

    def _crowding_scores_for_all(self, objectives):
        fronts = self._non_dominated_sort(objectives, [True, False][:len(objectives)])
        rank = np.zeros(len(objectives[0]), dtype=np.float32)
        crowd_all = np.zeros(len(objectives[0]), dtype=np.float32)

        for front_id, front in enumerate(fronts):
            crowd = self._crowding_distance(front, objectives)
            for idx in front:
                rank[idx] = float(front_id)
                value = crowd.get(idx, 0.0)
                crowd_all[idx] = 1.0 if math.isinf(value) else float(value)

        if crowd_all.size > 0 and np.max(crowd_all) > 0:
            crowd_all = crowd_all / (np.max(crowd_all) + 1e-12)
        return rank, crowd_all

    def _raw_arch_quality(self, is_value, fid_value, param_value):
        # Absolute-scale quality for comparing two cached/predicted parents.
        # Penalize only the part above target_model_size, so we do not
        # accidentally prefer extremely tiny but low-quality generators.
        excess_param = max(0.0, float(param_value) - float(self.target_model_size))
        return float(is_value) - 0.05 * float(fid_value) - 0.05 * excess_param

    def _estimate_arch_quality(self, genotype_G):
        key = self._encode_genotype(genotype_G)
        cached = self.historical_population.get(key, None)
        if cached is not None and cached.get('exact', False):
            return self._raw_arch_quality(cached['is'], cached['fid'], cached['param'])

        # An unseen architecture has no observed quality yet.
        return None

    # ---------------------------------------------------------------------
    # training sample utilities (external seed feedback removed)
    # ---------------------------------------------------------------------













    def _fill_training_genotype_list(self, selected, target_num):
        selected = self._deduplicate_genotypes(selected)
        attempts = 0
        while len(selected) < target_num and attempts < target_num * 20:
            selected.append(np.array(self.gan_alg.search(), copy=True))
            selected = self._deduplicate_genotypes(selected)
            attempts += 1
        while len(selected) < target_num:
            selected.append(np.array(self.gan_alg.search(), copy=True))
        return selected[:target_num]


    # ---------------------------------------------------------------------
    # NSAS-style diversity-aware path sampling
    # ---------------------------------------------------------------------
    def _select_diverse_subset(self, candidate_pool, target_num):
        """
        Greedy diverse subset selection inspired by the NSAS diversity objective:
            max_M sum_{alpha_i, alpha_j in M} dis(alpha_i, alpha_j)
        """
        candidate_pool = self._deduplicate_genotypes(candidate_pool)
        if len(candidate_pool) == 0:
            candidate_pool = [self.gan_alg.search() for _ in range(target_num)]
            candidate_pool = self._deduplicate_genotypes(candidate_pool)

        if len(candidate_pool) <= target_num:
            while len(candidate_pool) < target_num:
                candidate_pool.append(np.array(self.gan_alg.search(), copy=True))
                candidate_pool = self._deduplicate_genotypes(candidate_pool)
                if len(candidate_pool) >= target_num:
                    break
            chosen = candidate_pool[:target_num]
            self._record_architecture_visit(chosen)
            return np.stack(chosen, axis=0)

        # start from the least-visited architecture to improve fairness.
        start_idx = None
        best_visit = None
        for idx, genotype in enumerate(candidate_pool):
            key = self._encode_genotype(genotype)
            visit = self.arch_visit_counter.get(key, 0)
            if best_visit is None or visit < best_visit:
                best_visit = visit
                start_idx = idx

        selected = [candidate_pool[start_idx]]
        remaining = [candidate_pool[i] for i in range(len(candidate_pool)) if i != start_idx]

        while len(selected) < target_num and remaining:
            best_idx = 0
            best_score = -1.0
            for idx, candidate in enumerate(remaining):
                min_dist = min(self._geno_distance(candidate, chosen) for chosen in selected)
                visit_penalty = 1.0 / (1.0 + self.arch_visit_counter.get(self._encode_genotype(candidate), 0))
                score = min_dist + 0.05 * visit_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        self._record_architecture_visit(selected)
        return np.stack(selected, axis=0)

    def _sample_training_genotypes(self, num_g):
        # Use the operation-balanced sampler directly, as in EWSGAN.
        num_g = max(1, int(num_g))
        return np.stack(
            [self.gan_alg.search() for _ in range(num_g)], axis=0
        ).astype(np.int64)

    # ---------------------------------------------------------------------
    # exact-objective Pareto selection helpers
    # ---------------------------------------------------------------------






    def _pareto_dominates(self, idx_a, idx_b, objectives, maximize_flags):
        better_or_equal = True
        strictly_better = False
        for values, maximize in zip(objectives, maximize_flags):
            va = values[idx_a]
            vb = values[idx_b]
            if maximize:
                if va < vb:
                    better_or_equal = False
                    break
                if va > vb:
                    strictly_better = True
            else:
                if va > vb:
                    better_or_equal = False
                    break
                if va < vb:
                    strictly_better = True
        return better_or_equal and strictly_better

    def _non_dominated_sort(self, objectives, maximize_flags):
        pop_size = len(objectives[0])
        domination_count = [0 for _ in range(pop_size)]
        dominates = [set() for _ in range(pop_size)]
        fronts = [[]]

        for i in range(pop_size):
            for j in range(pop_size):
                if i == j:
                    continue
                if self._pareto_dominates(i, j, objectives, maximize_flags):
                    dominates[i].add(j)
                elif self._pareto_dominates(j, i, objectives, maximize_flags):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        front_id = 0
        while front_id < len(fronts) and len(fronts[front_id]) > 0:
            next_front = []
            for idx in fronts[front_id]:
                for dominated_idx in dominates[idx]:
                    domination_count[dominated_idx] -= 1
                    if domination_count[dominated_idx] == 0:
                        next_front.append(dominated_idx)
            if next_front:
                fronts.append(next_front)
            front_id += 1
        return fronts

    def _crowding_distance(self, front, objectives):
        if len(front) == 0:
            return {}
        if len(front) <= 2:
            return {idx: float('inf') for idx in front}

        distance = {idx: 0.0 for idx in front}
        for values in objectives:
            ordered = sorted(front, key=lambda idx: values[idx])
            distance[ordered[0]] = float('inf')
            distance[ordered[-1]] = float('inf')
            value_min = values[ordered[0]]
            value_max = values[ordered[-1]]
            denom = value_max - value_min
            if abs(denom) < 1e-12:
                continue
            for i in range(1, len(ordered) - 1):
                if math.isinf(distance[ordered[i]]):
                    continue
                prev_v = values[ordered[i - 1]]
                next_v = values[ordered[i + 1]]
                distance[ordered[i]] += float((next_v - prev_v) / denom)
        return distance

    def _pareto_select_indices(self, objectives, maximize_flags, select_num):
        total = len(objectives[0])
        if select_num >= total:
            return np.arange(total, dtype=np.int64)
        if total == 0:
            return np.array([], dtype=np.int64)

        fronts = self._non_dominated_sort(objectives, maximize_flags)
        chosen = []
        for front in fronts:
            if len(chosen) + len(front) <= select_num:
                chosen.extend(front)
                continue
            crowd = self._crowding_distance(front, objectives)
            front = sorted(
                front,
                key=lambda idx: (
                    crowd[idx],
                    objectives[0][idx],
                    -objectives[1][idx] if len(objectives) > 1 else 0.0,
                    -objectives[2][idx] if len(objectives) > 2 else 0.0,
                ),
                reverse=True,
            )
            needed = select_num - len(chosen)
            chosen.extend(front[:needed])
            break
        return np.array(chosen, dtype=np.int64)


    # ---------------------------------------------------------------------
    # mutation and selection support
    # ---------------------------------------------------------------------






    # ---------------------------------------------------------------------
    # mutation statistics inspired by probability-guided evolutionary search
    # ---------------------------------------------------------------------
    def _update_operation_statistics(self, selected_genotypes, is_values, fid_values, params):
        return
        if len(selected_genotypes) == 0:
            return

        self.up_op_success *= self.success_decay
        self.normal_op_success *= self.success_decay

        rank_order = self._pareto_select_indices(
            [is_values, fid_values],
            [True, False],
            len(selected_genotypes),
        )

        # earlier items in the Pareto-ranked set receive larger weights.
        for position, idx in enumerate(rank_order):
            genotype = selected_genotypes[idx]
            weight = 1.0 / (1.0 + float(position))
            for layer in range(self.num_layers):
                for slot in range(self.num_up_slots):
                    op_id = int(genotype[layer][slot])
                    if 0 <= op_id < self.num_up_ops:
                        self.up_op_success[layer, slot, op_id] += weight
                for slot in range(self.num_normal_slots):
                    op_id = int(genotype[layer][slot + 2])
                    if 0 <= op_id < self.num_normal_ops:
                        self.normal_op_success[layer, slot, op_id] += weight

    def _sample_guided_op(self, layer, index, is_up):
        if is_up:
            pool = list(self.gan_alg.Up_G_fixed[2 * layer + index])
            if len(pool) == 0:
                return 0
            if (not self.use_guided_mutation) or random.random() < self.guided_mutation_eps:
                return int(random.choice(pool))
            scores = self.up_op_success[layer, index, np.array(pool, dtype=np.int64)]
        else:
            normal_index = index - 2
            pool = list(self.gan_alg.Normal_G_fixed[5 * layer + normal_index])
            if len(pool) == 0:
                return 0
            if (not self.use_guided_mutation) or random.random() < self.guided_mutation_eps:
                return int(random.choice(pool))
            scores = self.normal_op_success[layer, normal_index, np.array(pool, dtype=np.int64)]

        scores = np.maximum(scores.astype(np.float64), 0.0)
        probs = (scores + self.success_smoothing) / np.sum(scores + self.success_smoothing)
        return int(np.random.choice(np.array(pool, dtype=np.int64), p=probs))

    def _adaptive_mutation_budget(self, alphas_a, alphas_b):
        max_mute = max(1, int(getattr(self.args, 'mute_max_num', 1)))
        similarity = 1.0 - self._geno_distance(alphas_a, alphas_b)
        similarity = min(max(similarity, 0.0), 1.0)
        budget = 1 + int(round(similarity * (max_mute - 1)))
        return max(1, min(max_mute, budget))

    def _sample_guided_nonzero_op(self, layer, index):
        if index < self.num_up_slots:
            return self._sample_guided_op(layer, index, is_up=True)

        normal_index = index - self.num_up_slots
        pool = list(self.gan_alg.Normal_G_fixed[5 * layer + normal_index])
        nonzero_pool = [int(op) for op in pool if int(op) != 0]
        if len(nonzero_pool) == 0:
            return self._sample_guided_op(layer, index, is_up=False)

        if (not self.use_guided_mutation) or random.random() < self.guided_mutation_eps:
            return int(random.choice(nonzero_pool))

        scores = self.normal_op_success[layer, normal_index, np.array(nonzero_pool, dtype=np.int64)]
        scores = np.maximum(scores.astype(np.float64), 0.0)
        probs = (scores + self.success_smoothing) / np.sum(scores + self.success_smoothing)
        return int(np.random.choice(np.array(nonzero_pool, dtype=np.int64), p=probs))

    def _repair_genotype(self, genotype_G):
        genotype_G = np.asarray(genotype_G, dtype=np.int64).copy()

        # In this search space, normal slots with value 0 behave like empty /
        # inactive choices in the original validity constraints. Keep at least
        # one active choice in each normal-slot group.
        for layer in range(self.num_layers):
            for start, length in [(2, 2), (4, 3)]:
                group = genotype_G[layer, start:start + length]
                if np.sum(group != 0) == 0:
                    slot = random.randint(start, start + length - 1)
                    genotype_G[layer, slot] = self._sample_guided_nonzero_op(layer, slot)

        return genotype_G

    def _apply_operation_mutation(self, new_alphas):
        layer = random.randint(0, self.num_layers - 1)
        index = random.randint(0, self.num_up_slots + self.num_normal_slots - 1)
        original_value = int(new_alphas[layer][index])

        for _retry in range(8):
            if index < self.num_up_slots:
                new_value = self._sample_guided_op(layer, index, is_up=True)
            else:
                new_value = self._sample_guided_op(layer, index, is_up=False)

            if new_value != original_value:
                new_alphas[layer][index] = new_value
                break

        return self._repair_genotype(new_alphas)

    def _apply_link_mutation(self, new_alphas):
        # The current genotype does not expose free graph edges. We approximate
        # "link mutation" through the normal-operation slots where 0 represents
        # an inactive/empty choice in the original validity checks.
        layer = random.randint(0, self.num_layers - 1)
        start, length = random.choice([(2, 2), (4, 3)])
        index = random.randint(start, start + length - 1)
        group = new_alphas[layer, start:start + length]
        active_count = int(np.sum(group != 0))

        if int(new_alphas[layer][index]) == 0:
            new_alphas[layer][index] = self._sample_guided_nonzero_op(layer, index)
        else:
            # If the group has more than one active slot, allow deleting one
            # link. Otherwise keep the group valid and only change the operation.
            if active_count > 1 and random.random() < 0.5:
                new_alphas[layer][index] = 0
            else:
                new_alphas[layer][index] = self._sample_guided_nonzero_op(layer, index)

        return self._repair_genotype(new_alphas)

    def _apply_mutation(self, new_alphas, num_mute):
        new_alphas = np.asarray(new_alphas, dtype=np.int64).copy()
        for _ in range(num_mute):
            if random.random() < self.link_mutation_prob:
                new_alphas = self._apply_link_mutation(new_alphas)
            else:
                new_alphas = self._apply_operation_mutation(new_alphas)
        return self._repair_genotype(new_alphas)

    def _select_parent_indices(self, alphas):
        pop_size = alphas.shape[0]
        if pop_size < 2:
            return 0, 0

        a = np.random.randint(0, pop_size)
        candidate_indices = list(range(pop_size))
        candidate_indices.remove(a)
        random.shuffle(candidate_indices)
        tournament_size = self.parent_diversity_tournament
        if self.use_tree_distance_parent_selection:
            tournament_size = max(tournament_size, self.tree_parent_diversity_tournament)
        tournament = candidate_indices[:max(1, min(len(candidate_indices), tournament_size))]
        best_b = tournament[0]
        best_score = None
        for idx in tournament:
            if self.use_tree_distance_parent_selection:
                score = self._tree_parent_pair_score(alphas[a], alphas[idx])
            else:
                score = self._geno_distance(alphas[a], alphas[idx])
            if best_score is None or score > best_score:
                best_score = score
                best_b = idx
        return a, best_b

    def _tree_active_mask(self, genotype_G):
        genotype = np.asarray(genotype_G, dtype=np.int64).reshape(
            self.num_layers, self.num_up_slots + self.num_normal_slots
        )
        return (genotype[:, self.num_up_slots:] != 0).astype(np.float32)

    def _tree_stage_signature(self, genotype_G):
        genotype = np.asarray(genotype_G, dtype=np.int64).reshape(
            self.num_layers, self.num_up_slots + self.num_normal_slots
        )
        signatures = []
        for layer in range(self.num_layers):
            row = genotype[layer]
            signatures.extend([
                tuple(row.tolist()),
                (int(row[0]), int(row[2]), int(row[4])),
                (int(row[1]), int(row[3]), int(row[5])),
                (int(row[6]),),
            ])
        return signatures

    def _tree_genotype_distance(self, genotype_a, genotype_b):
        a = np.asarray(genotype_a, dtype=np.int64).reshape(
            self.num_layers, self.num_up_slots + self.num_normal_slots
        )
        b = np.asarray(genotype_b, dtype=np.int64).reshape(
            self.num_layers, self.num_up_slots + self.num_normal_slots
        )
        gene_distance = float(np.mean(a != b))
        active_distance = float(np.mean(self._tree_active_mask(a) != self._tree_active_mask(b)))
        sig_a = self._tree_stage_signature(a)
        sig_b = self._tree_stage_signature(b)
        subtree_distance = float(np.mean([sa != sb for sa, sb in zip(sig_a, sig_b)]))
        w_gene = float(getattr(self.args, 'tree_w_gene', 0.40))
        w_active = float(getattr(self.args, 'tree_w_active', 0.35))
        w_subtree = float(getattr(self.args, 'tree_w_subtree', 0.25))
        return min(1.0, max(0.0,
            w_gene * gene_distance + w_active * active_distance + w_subtree * subtree_distance))

    def _tree_parent_quality_score(self, genotype_G):
        pred = self._estimate_arch_quality(genotype_G)
        if pred is not None:
            return float(pred)
        key = self._encode_genotype(genotype_G)
        rec = self.historical_population.get(key, None)
        if rec is None:
            return 0.0
        return self._quality_scores(
            np.asarray([float(rec.get('is', 0.0))], dtype=np.float32),
            np.asarray([float(rec.get('fid', 999.0))], dtype=np.float32),
            np.asarray([float(rec.get('param', self._get_param_size(genotype_G)))], dtype=np.float32),
        )[0]

    def _tree_parent_pair_score(self, genotype_a, genotype_b):
        # Lexicographic comparison: quality can only break an exact distance tie.
        distance = self._tree_genotype_distance(genotype_a, genotype_b)
        quality = 0.0
        if float(self.tree_parent_quality_weight) > 0.0:
            quality = 0.5 * (
                self._tree_parent_quality_score(genotype_a)
                + self._tree_parent_quality_score(genotype_b)
            )
        return distance, quality

    def _sample_tree_diverse_pair(self, genotypes):
        genotypes = [np.asarray(g, dtype=np.int64) for g in genotypes]
        if len(genotypes) < 2:
            if len(genotypes) == 1:
                return np.array(genotypes[0], copy=True), np.array(genotypes[0], copy=True)
            raise RuntimeError('No genotype is available for tree-diverse parent sampling.')
        anchor_idx = random.randrange(len(genotypes))
        candidates = [idx for idx in range(len(genotypes)) if idx != anchor_idx]
        random.shuffle(candidates)
        candidates = candidates[:max(1, min(len(candidates), self.tree_parent_diversity_tournament))]
        best_idx = candidates[0]
        best_score = None
        for idx in candidates:
            score = self._tree_parent_pair_score(genotypes[anchor_idx], genotypes[idx])
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        return np.array(genotypes[anchor_idx], copy=True), np.array(genotypes[best_idx], copy=True)

    def _sample_tree_diverse_cross_pool_pair(self, pool_a, pool_b):
        pool_a = [np.asarray(g, dtype=np.int64) for g in pool_a]
        pool_b = [np.asarray(g, dtype=np.int64) for g in pool_b]
        if len(pool_a) == 0 or len(pool_b) == 0:
            raise RuntimeError('Both pools must be non-empty for cross-pool parent sampling.')
        cand_a = random.sample(pool_a, min(len(pool_a), self.tree_parent_diversity_tournament))
        cand_b = random.sample(pool_b, min(len(pool_b), self.tree_parent_diversity_tournament))
        best_pair = (cand_a[0], cand_b[0])
        best_score = None
        for genotype_a in cand_a:
            for genotype_b in cand_b:
                if self._encode_genotype(genotype_a) == self._encode_genotype(genotype_b):
                    continue
                score = self._tree_parent_pair_score(genotype_a, genotype_b)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (genotype_a, genotype_b)
        return np.array(best_pair[0], copy=True), np.array(best_pair[1], copy=True)
    # ---------------------------------------------------------------------
    # lightweight single-teacher feature distillation plug-in
    # ---------------------------------------------------------------------






    # ---------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------
    def train(self, epoch, writer_dict, fid_stat, schedulers=None):
        writer = writer_dict['writer']
        gen_step = 0

        # train mode
        gen_net = self.gen_net.train()
        dis_net = self.dis_net.train()

        # number of g
        num_g = max(1, int(len(self.gan_alg.Normal_G_fixed[0]) // self.args.num_op_g))
        genotype_G = self._sample_training_genotypes(num_g)

        for iter_idx, (imgs, _) in enumerate(tqdm(self.train_loader)):
            global_steps = writer_dict['train_global_steps']

            real_imgs = imgs.type(torch.cuda.FloatTensor)
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (imgs.shape[0], self.args.latent_dim)))

            # ============================ train D ============================
            # Keep the original hinge loss and cycle through fairly sampled subnets.
            self.dis_optimizer.zero_grad()
            real_validity = dis_net(real_imgs)
            current_idx = iter_idx % len(genotype_G)
            fake_imgs = gen_net(z, genotype_G[current_idx]).detach()
            fake_imgs = self._resize_fake_to_real(fake_imgs, real_imgs)

            assert fake_imgs.size() == real_imgs.size()
            fake_validity = dis_net(fake_imgs)
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) +                      torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            self.dis_optimizer.step()
            writer.add_scalar('d_loss', d_loss.item(), global_steps)

            # ============================ train G ============================
            if global_steps % self.args.n_critic == 0:
                # D is only a differentiable scorer during the G update. Freeze
                # its parameters while preserving gradients from D back to G.
                dis_params = list(dis_net.parameters())
                dis_requires_grad = [param.requires_grad for param in dis_params]
                for param in dis_params:
                    param.requires_grad_(False)

                self.gen_optimizer.zero_grad()
                gen_z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.gen_bs, self.args.latent_dim)))

                try:
                    # Accumulate equal, unscaled subnet losses and step once,
                    # matching EWSGAN's fair multi-subnet training update.
                    path_losses = []
                    for genotype in genotype_G:
                        gen_imgs = gen_net(gen_z, genotype)
                        gen_imgs = self._resize_fake_to_real(gen_imgs, real_imgs)
                        path_loss = -torch.mean(dis_net(gen_imgs))
                        path_loss.backward()
                        path_losses.append(path_loss.detach())
                    self.gen_optimizer.step()
                    g_loss = torch.stack(path_losses).sum()
                finally:
                    for param, requires_grad in zip(dis_params, dis_requires_grad):
                        param.requires_grad_(requires_grad)

                # Resample fairly after each G update.
                genotype_G = self._sample_training_genotypes(num_g)

                if schedulers:
                    gen_scheduler, dis_scheduler = schedulers
                    g_lr = gen_scheduler.step(global_steps)
                    d_lr = dis_scheduler.step(global_steps)
                    writer.add_scalar('LR/g_lr', g_lr, global_steps)
                    writer.add_scalar('LR/d_lr', d_lr, global_steps)

                writer.add_scalar('g_loss/total', g_loss.item(), global_steps)
                gen_step += 1

            if gen_step and iter_idx % self.args.print_freq == 0:
                tqdm.write(
                    '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' %
                    (epoch, self.args.max_epoch_D, iter_idx % len(self.train_loader), len(self.train_loader),
                     d_loss.item(), g_loss.item()))
            writer_dict['train_global_steps'] = global_steps + 1

        if epoch % 5 == 0:
            is_warm, is_std_warm, fid_warm = self.validate(self.gen_fixed_genotype, fid_stat, epoch)
            logger.info(
                f'IS_warm: {round(is_warm, 3)}, FID_warm: {round(fid_warm, 3)}, @ epoch {epoch}.')

    def get_E_DGz(self, geno):
        gen_net = self.gen_net.eval()
        dis_net = self.dis_net.eval()
        gen_z = torch.cuda.FloatTensor(np.random.normal(
            0, 1, (100, self.args.latent_dim)))
        gen_imgs = gen_net(gen_z, geno)
        gen_imgs = self._resize_to_target_img_size(gen_imgs)
        fake_validity = dis_net(gen_imgs)
        E_DGz = torch.mean(fake_validity)
        return E_DGz

    def Judge_relevance(self, fid_stat):
        Random_population = np.stack([self.gan_alg.search() for _ in range(100)], axis=0)
        is_values = np.zeros(len(Random_population))
        fid_values = np.zeros(len(Random_population))
        D_Gz = np.zeros(len(Random_population))

        for idx, genotype_G in enumerate(tqdm(Random_population)):
            is_value, is_std, fid_value = self.validate(genotype_G, fid_stat, 40)
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            D_Gz[idx] = self.get_E_DGz(genotype_G)
        logger.info(f'IS: {is_values}, FID: {fid_values}, D_G(z): {D_Gz}.')

    # ---------------------------------------------------------------------
    # population evaluation and selection
    # ---------------------------------------------------------------------
    def _evaluate_population(self, genotypes, fid_stat, epoch=None):
        is_values = np.zeros(len(genotypes), dtype=np.float32)
        fid_values = np.zeros(len(genotypes), dtype=np.float32)
        params = np.zeros(len(genotypes), dtype=np.float32)

        archive_changed = False
        for idx, genotype_G in enumerate(tqdm(genotypes)):
            key = self._encode_genotype(genotype_G)
            param_size = self._get_param_size(genotype_G)
            cached = self.historical_population.get(key, None)

            if cached is not None and cached.get('exact', False):
                is_value = cached['is']
                fid_value = cached['fid']
            else:
                is_value, is_std, fid_value = self.validate(genotype_G, fid_stat, epoch)
                self._cache_exact_result(genotype_G, is_value, fid_value, param_size)
                archive_changed = True

            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_size

            if is_value > self.best_is_evo:
                self.best_is_evo = float(is_value)
                file_path = os.path.join(self.args.path_helper['ckpt_path'], 'best_is_gen_.npy')
                np.save(file_path, genotype_G)
            if fid_value < self.best_fid_evo:
                self.best_fid_evo = float(fid_value)
                file_path = os.path.join(self.args.path_helper['ckpt_path'], 'best_fid_gen.npy')
                np.save(file_path, genotype_G)

        if archive_changed or len(getattr(self, 'external_elite_archive', [])) == 0:
            self._refresh_external_elite_archive(log_summary=False)
        return is_values, fid_values, params

    def _size_feasible_indices(self, params):
        p = np.asarray(params, dtype=np.float64)
        return np.flatnonzero(np.isfinite(p) & (p <= float(self.args.max_model_size)))

    def _current_pcs_target_model_size(self, generation=None):
        """Compatibility helper: only the fixed hard bound remains; no PCS."""
        return float(self.args.max_model_size)



    def _offspring_size_limit(self, attempts, max_attempts):
        return float(self.args.max_model_size)








    def _select_population(self, is_values, fid_values, params, select_num):
        """Hard parameter feasibility followed ONLY by IS/FID NSGA-II."""
        s, f, p = (np.asarray(v, dtype=np.float64) for v in (is_values, fid_values, params))
        count = int(select_num)
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        if s.ndim != 1 or s.shape != f.shape or s.shape != p.shape:
            raise ValueError('IS, FID and Params must be equally sized 1-D arrays.')
        valid = np.isfinite(s) & np.isfinite(f) & np.isfinite(p)
        feasible = np.flatnonzero(valid & (p <= float(self.args.max_model_size)))
        if len(feasible) < count:
            raise RuntimeError('Only %d feasible candidates for %d survivors under Pmax=%sM.'
                               % (len(feasible), count, self.args.max_model_size))
        local = self._pareto_select_indices([s[feasible], f[feasible]], [True, False], count)
        return feasible[np.asarray(local, dtype=np.int64)]

    def search_evol_arch(self, epoch, fid_stat):
        offsprings = self.gen_offspring(self.genotypes)
        genotypes = np.concatenate((self.genotypes, offsprings), axis=0)
        is_values, fid_values, params = self._evaluate_population(genotypes, fid_stat, epoch)
        keep_N, selected_N = len(offsprings), self.args.num_selected

        logger.info(
            f'mean_IS_values: {np.mean(is_values):.4f}, mean_FID_values: {np.mean(fid_values):.4f}, '
            f'mean_param_values: {np.mean(params):.4f}, @ epoch {epoch}.')

        keep = self._select_population(is_values, fid_values, params, keep_N)
        selected = self._select_population(is_values, fid_values, params, selected_N)

        for i in selected:
            logger.info(
                f'genotypes_{i}, IS_values: {is_values[i]:.4f}, FID_values: {fid_values[i]:.4f}, '
                f'param_size: {params[i]:.4f} || @ epoch {epoch}.')

        self.genotypes = np.asarray(genotypes[keep], dtype=np.int64)
        self._update_replay_buffer(genotypes[selected])
        self._update_operation_statistics_with_elites(genotypes[selected], is_values[selected], fid_values[selected], params[selected])
        return genotypes[selected]


    # ---------------------------------------------------------------------
    # guarded elite local refinement: only after global search finds a good area
    # ---------------------------------------------------------------------
    def _guarded_local_refine_allowed(self, num):
        return False

    def _get_local_elite_genotypes(self, top_k=None):
        if top_k is None:
            top_k = self.local_refine_elite_top_k
        max_model_size = float(getattr(self.args, 'max_model_size', 1e9))
        exact_history = self._get_exact_history()
        feasible = [r for r in exact_history if float(r.get('param', 1e9)) <= max_model_size and 'genotype' in r]
        if len(feasible) == 0:
            feasible = [r for r in exact_history if 'genotype' in r]
        feasible = sorted(feasible, key=lambda r: (float(r.get('fid', 999.0)), -float(r.get('is', 0.0))))
        return [np.array(r['genotype'], copy=True) for r in feasible[:max(1, int(top_k))]]

    def _generate_guarded_local_candidates(self, num):
        return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)
        if not self._guarded_local_refine_allowed(num):
            return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)

        elites = self._get_local_elite_genotypes(self.local_refine_elite_top_k)
        if len(elites) == 0:
            return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)

        target_num = max(1, int(self.local_refine_eval_num))
        raw_target = max(target_num, len(elites) * max(1, int(self.local_refine_neighbors_per_elite)))
        max_attempts = max(100, raw_target * max(1, int(self.local_refine_max_attempt_multiplier)))
        candidates, seen = [], set()
        attempts = 0
        while len(candidates) < raw_target and attempts < max_attempts:
            elite = elites[attempts % len(elites)]
            child = np.array(elite, copy=True)
            for _ in range(max(1, int(self.local_refine_mutate_slots))):
                child = self._apply_mutation(child, 1)
            key = self._encode_genotype(child)
            attempts += 1
            if key in seen:
                continue
            if self.local_refine_skip_seen and key in self.historical_population:
                continue
            if not self.judege_model_size(child, limit=float(getattr(self.args, 'max_model_size', self.target_model_size))):
                continue
            seen.add(key)
            candidates.append(child)

        if len(candidates) == 0:
            logger.info(
                f'[GUARDED-LOCAL] skipped at gen={num}: no valid unseen neighbors from top-{len(elites)} elites.'
            )
            return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)

        # Retain the local-route budget without surrogate ranking.
        selected = candidates[:min(target_num, len(candidates))]
        logger.info(
            f'[GUARDED-LOCAL] active at gen={num}: generated={len(candidates)}, '
            f'appended={len(selected)}, best_fid={self.best_fid_evo:.3f}.'
        )
        return np.stack(selected, axis=0).astype(np.int64)

    def judege_model_size(self, genotype_G, limit):
        """
        Test whether the model size constraint is met.
        Cached parameter counting avoids repeated Generator construction.
        """
        param_size = self._get_param_size(genotype_G)
        if param_size <= limit:
            return True
        return False

    def _archive_fid_local_refine_allowed(self, num):
        return False

    def _get_archive_fid_local_elite_genotypes(self, top_k=None):
        if top_k is None:
            top_k = self.archive_fid_local_elite_top_k
        archive = self._refresh_external_elite_archive(log_summary=False)
        records = [
            record for record in archive
            if record.get('genotype', None) is not None
        ]
        if len(records) == 0:
            return []
        max_model_size = float(getattr(self.args, 'max_model_size', 1e9))
        feasible = [
            record for record in records
            if float(record.get('param', 1e9)) <= max_model_size
        ]
        if len(feasible) == 0:
            feasible = records
        feasible = sorted(
            feasible,
            key=lambda record: (
                float(record.get('fid', 999.0)),
                -float(record.get('is', 0.0)),
                float(record.get('param', 1e9)),
            ),
        )
        return [
            np.array(record['genotype'], copy=True)
            for record in feasible[:max(1, int(top_k))]
        ]

    def _apply_archive_fid_local_mutation(self, genotype_G):
        """Make a one-step neighbor around an archive best-FID genotype."""
        child = np.array(genotype_G, copy=True)
        layer = np.random.randint(0, self.num_layers)
        if np.random.rand() < 0.25:
            slot = np.random.randint(0, self.num_up_slots)
            current = int(child[layer, slot])
            choices = [op for op in range(self.num_up_ops) if op != current]
        else:
            slot = self.num_up_slots + np.random.randint(0, self.num_normal_slots)
            current = int(child[layer, slot])
            choices = [op for op in range(self.num_normal_ops) if op != current]
        if len(choices) > 0:
            child[layer, slot] = int(random.choice(choices))
        return child

    def _generate_archive_fid_local_candidates(self, num):
        return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)
        if not self._archive_fid_local_refine_allowed(num):
            return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)

        elites = self._get_archive_fid_local_elite_genotypes(self.archive_fid_local_elite_top_k)
        if len(elites) == 0:
            return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)

        target_num = max(1, int(self.archive_fid_local_eval_num))
        raw_target = max(target_num, len(elites) * max(1, int(self.archive_fid_local_neighbors_per_elite)))
        max_attempts = max(100, raw_target * max(1, int(self.archive_fid_local_max_attempt_multiplier)))
        candidates, seen = [], set()
        attempts = 0
        while len(candidates) < raw_target and attempts < max_attempts:
            elite = elites[attempts % len(elites)]
            child = np.array(elite, copy=True)
            for _ in range(max(1, int(self.archive_fid_local_mutate_slots))):
                child = self._apply_archive_fid_local_mutation(child)
            key = self._encode_genotype(child)
            attempts += 1
            if key in seen:
                continue
            if self.archive_fid_local_skip_seen and key in self.historical_population:
                continue
            if not self.judege_model_size(child, limit=float(getattr(self.args, 'max_model_size', self.target_model_size))):
                continue
            seen.add(key)
            candidates.append(child)

        if len(candidates) == 0:
            logger.info(
                f'[ARCHIVE-FID-LOCAL] skipped at gen={num}: no valid unseen neighbors from archive best-FID top-{len(elites)}.'
            )
            return np.empty((0, self.num_layers, self.num_up_slots + self.num_normal_slots), dtype=np.int64)

        selected = candidates[:min(target_num, len(candidates))]
        logger.info(
            f'[ARCHIVE-FID-LOCAL] active at gen={num}: generated={len(candidates)}, '
            f'appended={len(selected)}, best_fid={self.best_fid_evo:.3f}.'
        )
        return np.stack(selected, axis=0).astype(np.int64)

    # ---------------------------------------------------------------------
    # final FID-safe tree pruning
    # ---------------------------------------------------------------------
    def _fid_safe_tree_pruning_allowed(self, num):
        if not bool(getattr(self, 'use_fid_safe_tree_pruning', False)):
            return False
        if bool(getattr(self, 'fid_prune_start_final_only', True)):
            final_gen = int(getattr(self.args, 'Total_evolutionary_algebra', 1)) - 1
            if int(num) < final_gen:
                return False
        return len(self._get_exact_history()) > 0

    def _get_fid_pruning_seed_records(self):
        records = [
            record for record in self._eligible_exact_elite_records()
            if record.get('genotype', None) is not None
        ]
        if len(records) == 0:
            return []

        top_k = max(1, int(getattr(self, 'fid_prune_archive_top_k', 3)))
        per_objective = max(1, int(math.ceil(float(top_k) / 3.0)))
        if self._three_elite_pools_ready():
            records_by_pool = self._get_three_elite_records(per_objective)
            seeds, seen = [], set()
            for objective in ('is', 'fid', 'params'):
                for record in records_by_pool.get(objective, []):
                    genotype = record.get('genotype', None)
                    if genotype is None:
                        continue
                    key = self._encode_genotype(genotype)
                    if key in seen:
                        continue
                    seen.add(key)
                    seeds.append(record)
            if len(seeds) > 0:
                logger.info(
                    '[TREE-PRUNE] objective-conditioned seeds from forests: '
                    'IS=%d, FID=%d, Params=%d, unique=%d.',
                    len(records_by_pool.get('is', [])),
                    len(records_by_pool.get('fid', [])),
                    len(records_by_pool.get('params', [])),
                    len(seeds),
                )
                return seeds

        records = sorted(
            records,
            key=lambda record: (
                float(record.get('fid', 999.0)),
                -float(record.get('is', 0.0)),
                float(record.get('param', self._get_param_size(record['genotype']))),
            ),
        )
        best_fid = float(records[0].get('fid', 999.0))
        margin = max(0.0, float(getattr(self, 'fid_prune_fid_margin', 1.0)))
        seeds = [
            record for record in records
            if float(record.get('fid', 999.0)) <= best_fid + margin
        ][:top_k]
        if len(seeds) == 0:
            seeds = records[:top_k]
        return seeds

    def _make_record_from_exact_cache(self, genotype_G):
        key = self._encode_genotype(genotype_G)
        record = self.historical_population.get(key, None)
        if record is None or not record.get('exact', False):
            return None
        return record

    def _objective_prune_limits(self, objective, relaxed=False):
        """Return objective-specific pruning limits.

        These limits make the three forests play different roles instead of
        using one shared pruning rule for all objectives.
        """
        if not bool(getattr(self, 'use_objective_prune_policy', True)):
            if objective == 'params':
                fid_delta = (
                    self.three_forest_prune_params_relaxed_fid_delta
                    if relaxed else self.three_forest_prune_params_fid_delta
                )
                is_drop = (
                    self.three_forest_prune_params_relaxed_is_drop
                    if relaxed else self.three_forest_prune_params_is_drop
                )
            else:
                fid_delta = (
                    self.three_forest_prune_relaxed_fid_delta
                    if relaxed else self.three_forest_prune_accept_fid_delta
                )
                is_drop = (
                    self.three_forest_prune_relaxed_is_drop
                    if relaxed else self.three_forest_prune_accept_is_drop
                )
            return {
                'fid_delta': max(0.0, float(fid_delta)),
                'is_drop': max(0.0, float(is_drop)),
                'min_drop': max(0.0, float(getattr(self, 'three_forest_prune_min_param_drop', 0.02))),
                'max_steps': max(0, int(getattr(self, 'fid_prune_max_steps', 3))),
                'pair_delete': bool(getattr(self, 'three_forest_prune_generate_pair_delete', True)),
            }

        if objective == 'fid':
            return {
                'fid_delta': max(0.0, float(
                    self.fid_forest_prune_relaxed_fid_delta if relaxed else self.fid_forest_prune_accept_fid_delta
                )),
                'is_drop': max(0.0, float(
                    self.fid_forest_prune_relaxed_is_drop if relaxed else self.fid_forest_prune_accept_is_drop
                )),
                'min_drop': max(0.0, float(self.fid_forest_prune_min_param_drop)),
                'max_steps': max(0, int(self.fid_forest_prune_max_steps)),
                'pair_delete': bool(self.fid_forest_prune_pair_delete),
            }
        if objective == 'is':
            return {
                'fid_delta': max(0.0, float(
                    self.is_forest_prune_relaxed_fid_delta if relaxed else self.is_forest_prune_accept_fid_delta
                )),
                'is_drop': max(0.0, float(
                    self.is_forest_prune_relaxed_is_drop if relaxed else self.is_forest_prune_accept_is_drop
                )),
                'min_drop': max(0.0, float(self.is_forest_prune_min_param_drop)),
                'max_steps': max(0, int(self.is_forest_prune_max_steps)),
                'pair_delete': bool(self.is_forest_prune_pair_delete),
            }
        return {
            'fid_delta': max(0.0, float(
                self.params_forest_prune_relaxed_fid_delta if relaxed else self.params_forest_prune_accept_fid_delta
            )),
            'is_drop': max(0.0, float(
                self.params_forest_prune_relaxed_is_drop if relaxed else self.params_forest_prune_accept_is_drop
            )),
            'min_drop': max(0.0, float(self.params_forest_prune_min_param_drop)),
            'max_steps': max(0, int(self.params_forest_prune_max_steps)),
            'pair_delete': bool(self.params_forest_prune_pair_delete),
        }

    def _generate_single_branch_prune_candidates(self, genotype_G, seen_keys=None, objective='fid'):
        """Generate lightweight pruning candidates around one tree genotype.

        The first complete three-forest pruning version only closed one normal
        branch at a time. In practice this can be too narrow: the repaired
        genotype may keep almost the same structure, or the exact FID/IS noise
        may reject every candidate. This broader generator still stays inside
        the same 3x7 search space, but it tries three pruning-like edits:
          1) close one normal branch;
          2) replace a normal op by a cheaper op when it reduces parameters;
          3) close two normal branches together.
        """
        if seen_keys is None:
            seen_keys = set()
        base = np.asarray(genotype_G, dtype=np.int64).copy()
        base = base.reshape(self.num_layers, self.num_up_slots + self.num_normal_slots)
        base_key = self._encode_genotype(base)
        base_param = float(self._get_param_size(base))
        limits = self._objective_prune_limits(objective, relaxed=False)
        min_drop = max(0.0, float(limits.get('min_drop', 0.02)))
        allow_pair_delete = bool(limits.get('pair_delete', True))

        candidates, candidate_keys = [], set()

        def add_candidate(child, edit_priority):
            child = self._repair_genotype(child)
            key = self._encode_genotype(child)
            if key == base_key:
                return
            if key in seen_keys or key in candidate_keys:
                return
            if not self.judege_model_size(
                child,
                limit=float(getattr(self.args, 'max_model_size', self.target_model_size)),
            ):
                return
            child_param = float(self._get_param_size(child))
            if child_param > base_param - min_drop:
                return
            candidate_keys.add(key)
            candidates.append((int(edit_priority), float(child_param), child))

        active_slots = []
        for layer in range(self.num_layers):
            for slot in range(self.num_up_slots, self.num_up_slots + self.num_normal_slots):
                if int(base[layer, slot]) != 0:
                    active_slots.append((layer, slot))

        # 1) Soft pruning first: replace by another normal op only when it
        # really lowers params. This is safer for FID than deleting a branch.
        if bool(getattr(self, 'three_forest_prune_generate_replace', True)):
            for layer, slot in active_slots:
                current = int(base[layer, slot])
                for op_id in range(self.num_normal_ops):
                    if op_id == current or op_id == 0:
                        continue
                    child = np.array(base, copy=True)
                    child[layer, slot] = int(op_id)
                    add_candidate(child, 0)

        # 2) Delete one active normal branch. This is stronger than replace, so
        # it is considered after soft pruning.
        for layer, slot in active_slots:
            child = np.array(base, copy=True)
            child[layer, slot] = 0
            add_candidate(child, 1)

        # 3) Delete two active normal branches together. This is useful when a
        # single deletion is repaired back or too weak to reduce parameters.
        # It is disabled by default for FID/IS forests and enabled for Params.
        if bool(getattr(self, 'three_forest_prune_generate_pair_delete', True)) and allow_pair_delete:
            for idx_a in range(len(active_slots)):
                for idx_b in range(idx_a + 1, len(active_slots)):
                    child = np.array(base, copy=True)
                    layer_a, slot_a = active_slots[idx_a]
                    layer_b, slot_b = active_slots[idx_b]
                    child[layer_a, slot_a] = 0
                    child[layer_b, slot_b] = 0
                    add_candidate(child, 2)

        if len(candidates) == 0:
            return []

        # Prefer candidates that actually reduce parameter count; keep the exact
        # FID decision for the later validation step.
        # FID/IS forests try low-risk edits first; Params forest prioritizes
        # larger parameter reduction.
        if objective == 'params':
            sort_key = lambda item: (
                item[1],
                item[0],
                self._tree_genotype_distance(base, item[2]),
            )
        else:
            sort_key = lambda item: (
                item[0],
                item[1],
                self._tree_genotype_distance(base, item[2]),
            )
        candidates = sorted(
            candidates,
            key=sort_key,
        )
        max_candidates = max(1, int(getattr(self, 'three_forest_prune_max_candidates_per_seed', 24)))
        return [np.asarray(item[2], dtype=np.int64).copy() for item in candidates[:max_candidates]]

    def _greedy_fid_safe_prune_record(self, seed_record, fid_stat):
        seed_genotype = np.asarray(seed_record['genotype'], dtype=np.int64).copy()
        seed_fid = float(seed_record.get('fid', 999.0))
        current_genotype = np.array(seed_genotype, copy=True)
        current_record = seed_record
        current_param = float(seed_record.get('param', self._get_param_size(seed_genotype)))
        accept_limit = seed_fid + max(0.0, float(getattr(self, 'fid_prune_accept_fid_delta', 1.0)))
        min_drop = max(0.0, float(getattr(self, 'fid_prune_min_param_drop', 0.05)))
        max_steps = max(0, int(getattr(self, 'fid_prune_max_steps', 3)))
        eval_epoch = getattr(self, 'fid_prune_eval_epoch', None)

        accepted = [current_record]
        seen_keys = {self._encode_genotype(current_genotype)}
        for step in range(max_steps):
            candidates = self._generate_single_branch_prune_candidates(current_genotype, seen_keys=seen_keys)
            candidates = [
                candidate for candidate in candidates
                if self._get_param_size(candidate) <= current_param - min_drop
            ]
            if len(candidates) == 0:
                break

            cand_array = np.stack(candidates, axis=0).astype(np.int64)
            is_values, fid_values, params = self._evaluate_population(cand_array, fid_stat, eval_epoch)
            feasible = [
                idx for idx in range(len(candidates))
                if float(params[idx]) <= current_param - min_drop
                and float(fid_values[idx]) <= accept_limit
            ]
            if len(feasible) == 0:
                break

            best_idx = min(
                feasible,
                key=lambda idx: (
                    float(params[idx]),
                    float(fid_values[idx]),
                    -float(is_values[idx]),
                ),
            )
            current_genotype = np.asarray(candidates[best_idx], dtype=np.int64).copy()
            current_param = float(params[best_idx])
            current_record = self._make_record_from_exact_cache(current_genotype)
            if current_record is None:
                break
            accepted.append(current_record)
            seen_keys.add(self._encode_genotype(current_genotype))
            logger.info(
                '[TREE-PRUNE] accepted seedFID=%.3f step=%d: IS=%.3f, FID=%.3f, P=%.3f.',
                seed_fid,
                step + 1,
                float(is_values[best_idx]),
                float(fid_values[best_idx]),
                float(params[best_idx]),
            )

        return accepted

    def _three_forest_record_sort_key(self, record, objective):
        is_value = float(record.get('is', 0.0))
        fid_value = float(record.get('fid', 999.0))
        param_value = float(record.get('param', self._get_param_size(record['genotype'])))
        if objective == 'is':
            return (-is_value, fid_value, param_value)
        if objective == 'fid':
            return (fid_value, -is_value, param_value)
        if objective == 'params':
            return (param_value, fid_value, -is_value)
        return (fid_value, param_value, -is_value)

    def _get_three_forest_pruning_seed_records(self):
        # Construct all final forests/seeds before pruning modifies the archive.
        forests = {o: self._get_objective_elite_records(o, self.three_elite_pool_size)
                   for o in ('is', 'fid', 'params')}
        feasible = self._eligible_exact_elite_records()
        if not feasible:
            raise RuntimeError('No feasible evaluated pruning seed exists.')
        seeds = {}
        for objective in ('is', 'fid', 'params'):
            seed = min(forests[objective] or feasible,
                       key=lambda r: self._objective_elite_sort_key(r, objective))
            seed = dict(seed)
            seed['genotype'] = np.asarray(seed['genotype'], dtype=np.int64).copy()
            seed['source_forest'] = objective
            seeds[objective] = [seed]
        return seeds

    def _three_forest_prune_accepts(self, objective, seed_record, current_param, candidate_record, relaxed=False):
        seed_is = float(seed_record.get('is', 0.0))
        seed_fid = float(seed_record.get('fid', 999.0))
        cand_is = float(candidate_record.get('is', 0.0))
        cand_fid = float(candidate_record.get('fid', 999.0))
        cand_param = float(candidate_record.get('param', 1e9))
        limits = self._objective_prune_limits(objective, relaxed=relaxed)
        min_drop = max(0.0, float(limits.get('min_drop', 0.02)))
        fid_delta = max(0.0, float(limits.get('fid_delta', 1.0)))
        is_drop = max(0.0, float(limits.get('is_drop', 0.15)))

        if cand_param > float(current_param) - min_drop:
            return False

        if objective == 'is':
            return cand_is >= seed_is - is_drop and cand_fid <= seed_fid + fid_delta

        if objective == 'fid':
            return cand_fid <= seed_fid + fid_delta and cand_is >= seed_is - is_drop

        if objective == 'params':
            return cand_fid <= seed_fid + fid_delta and cand_is >= seed_is - is_drop

        return False





    def _greedy_three_forest_prune_record(self, objective, seed_record, fid_stat):
        """Keep each accepted step, anchored to immutable seed IS/FID."""
        current = dict(seed_record)
        current.update(source_forest=objective, prune_step=0, pruned_from_seed=False,
                       accept_mode='seed', recovered=False)
        accepted = [current]
        seen = {self._encode_genotype(current['genotype'])}
        steps = self._objective_prune_limits(objective, relaxed=False)['max_steps']
        for step in range(int(steps)):
            candidates = self._generate_single_branch_prune_candidates(
                current['genotype'], seen_keys=seen, objective=objective)
            self._evaluate_population(candidates, fid_stat, getattr(self, 'fid_prune_eval_epoch', None))
            admissible = []
            for g in candidates:
                record = self._make_record_from_exact_cache(g)
                if record is not None and record['param'] <= self.args.max_model_size and self._three_forest_prune_accepts(
                        objective, seed_record, current['param'], record, relaxed=False):
                    admissible.append(record)
            if not admissible:
                break
            current = dict(min(admissible, key=lambda r: self._three_forest_record_sort_key(r, objective)))
            current.update(source_forest=objective, prune_step=step + 1, pruned_from_seed=True,
                           accept_mode='strict', recovered=False)
            accepted.append(current)
            seen.add(self._encode_genotype(current['genotype']))
            logger.info('[PAPER-PRUNE] %s step=%d seedIS=%.4f seedFID=%.4f IS=%.4f FID=%.4f P=%.4f',
                        objective, step + 1, seed_record['is'], seed_record['fid'],
                        current['is'], current['fid'], current['param'])
        return accepted









    def _run_three_forest_tree_pruning(self, fid_stat, return_size):
        """Return exactly the IS/FID/Params terminal architectures, in that order.

        return_size remains in the signature for existing entry scripts; final
        output has three source-labelled records, even if genotypes overlap.
        """
        seeds = self._get_three_forest_pruning_seed_records()
        selected = []
        for objective in ('is', 'fid', 'params'):
            path = self._greedy_three_forest_prune_record(objective, seeds[objective][0], fid_stat)
            selected.append(path[-1])
        self.final_terminal_records = selected
        directory = self.args.path_helper['ckpt_path']
        os.makedirs(directory, exist_ok=True)
        for index, record in enumerate(selected):
            objective = record['source_forest']
            np.save(os.path.join(directory, 'three_forest_pruned_%s_%d.npy' % (objective, index)), record['genotype'])
            logger.info('[PAPER-TERMINAL] source=%s step=%d IS=%.4f FID=%.4f P=%.4f',
                        objective, record['prune_step'], record['is'], record['fid'], record['param'])
        import json
        with open(os.path.join(directory, 'paper_terminal_records.json'), 'w', encoding='utf-8') as handle:
            json.dump(selected, handle, indent=2,
                      default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())
        return [np.asarray(r['genotype'], dtype=np.int64).copy() for r in selected]

    def _select_pruned_records(self, records, return_size):
        unique, seen = [], set()
        for record in records:
            genotype = record.get('genotype', None)
            if genotype is None:
                continue
            key = self._encode_genotype(genotype)
            if key in seen:
                continue
            seen.add(key)
            unique.append(record)
        if len(unique) == 0:
            return []

        best_fid = min(float(record.get('fid', 999.0)) for record in unique)
        fid_limit = best_fid + max(0.0, float(getattr(self, 'fid_prune_accept_fid_delta', 1.0)))
        safe = [
            record for record in unique
            if float(record.get('fid', 999.0)) <= fid_limit
        ]
        if len(safe) == 0:
            safe = unique

        selected, selected_keys = [], set()

        def add_sorted(pool, key_fn):
            for record in sorted(pool, key=key_fn):
                if len(selected) >= return_size:
                    return
                key = self._encode_genotype(record['genotype'])
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                selected.append(record)

        add_sorted(safe, lambda record: (float(record.get('fid', 999.0)), float(record.get('param', 1e9))))
        add_sorted(safe, lambda record: (float(record.get('param', 1e9)), float(record.get('fid', 999.0))))
        add_sorted(
            safe,
            lambda record: (
                float(record.get('fid', 999.0)) - best_fid
                + 0.15 * float(record.get('param', 1e9)),
                float(record.get('fid', 999.0)),
            ),
        )
        if len(selected) < return_size:
            add_sorted(unique, lambda record: (float(record.get('fid', 999.0)), float(record.get('param', 1e9))))
        return selected[:return_size]

    def _run_fid_safe_tree_pruning(self, fid_stat, return_size):
        seeds = self._get_fid_pruning_seed_records()
        if len(seeds) == 0:
            logger.info('[TREE-PRUNE] skipped: no exact best-FID seeds available.')
            return []

        all_records = []
        for seed in seeds:
            seed_genotype = seed.get('genotype', None)
            if seed_genotype is None:
                continue
            logger.info(
                '[TREE-PRUNE] seed: IS=%.3f, FID=%.3f, P=%.3f.',
                float(seed.get('is', 0.0)),
                float(seed.get('fid', 999.0)),
                float(seed.get('param', self._get_param_size(seed_genotype))),
            )
            all_records.extend(self._greedy_fid_safe_prune_record(seed, fid_stat))

        selected = self._select_pruned_records(all_records, max(1, int(return_size)))
        if len(selected) == 0:
            logger.info('[TREE-PRUNE] no FID-safe pruned architecture accepted.')
            return []

        for idx, record in enumerate(selected):
            genotype = np.asarray(record['genotype'], dtype=np.int64)
            file_path = os.path.join(self.args.path_helper['ckpt_path'], f'tree_pruned_gen_{idx}.npy')
            np.save(file_path, genotype)
            logger.info(
                '[TREE-PRUNE] selected_%d: IS=%.3f, FID=%.3f, P=%.3f.',
                idx,
                float(record.get('is', 0.0)),
                float(record.get('fid', 999.0)),
                float(record.get('param', self._get_param_size(genotype))),
            )
        return [np.asarray(record['genotype'], dtype=np.int64).copy() for record in selected]

    def my_search_evolv2(self, parents, fid_stat, num):
        """
        Evolutionary search with guarded late-stage refinements:
          - original offspring generation with direct exact evaluation;
          - guarded elite local refinement only after global search finds a good area;
          - target-size-filtered IS/FID Pareto survivor selection;
          - direct survivor selection forms the next population.
        """
        self.current_evolution_algebra = int(num)
        offsprings = self.gen_offspring(parents)
        local_candidates = self._generate_guarded_local_candidates(num)
        archive_local_candidates = self._generate_archive_fid_local_candidates(num)
        candidate_blocks = [parents, offsprings]
        if len(local_candidates) > 0:
            candidate_blocks.append(local_candidates)
        if len(archive_local_candidates) > 0:
            candidate_blocks.append(archive_local_candidates)
        genotypes = np.concatenate(candidate_blocks, axis=0)
        # Keep the next population size unchanged. Local candidates are only an
        # additional exact-evaluation pool, not an expansion of the population.
        keep_N, selected_N = len(offsprings), self.args.num_selected

        is_values, fid_values, params = self._evaluate_population(genotypes, fid_stat)

        keep = self._select_population(is_values, fid_values, params, keep_N)
        selected = self._select_population(is_values, fid_values, params, selected_N)

        next_population = np.asarray(genotypes[keep], dtype=np.int64)
        next_population = self._inject_archive_best_fid_into_population(
            next_population,
            target_size=keep_N,
            context=f'gen={num}',
        )
        avg_keep_is = float(np.mean(is_values[keep])) if len(keep) else 0.0
        avg_keep_fid = float(np.mean(fid_values[keep])) if len(keep) else 0.0
        avg_keep_param = float(np.mean(params[keep])) if len(keep) else 0.0
        logger.info(
            f'mean_IS_keep: {round(avg_keep_is, 3)}, mean_FID_keep: {round(avg_keep_fid, 3)}, '
            f'mean_params_keep: {round(avg_keep_param, 3)}, max_IS_all: {round(float(np.max(is_values)), 3)}, '
            f'min_FID_all: {round(float(np.min(fid_values)), 3)} @ evolutionary_algebra: {num}.')

        for i in selected:
            logger.info(
                f'genotypes_{i}, IS: {round(float(is_values[i]), 3)}, '
                f'FID: {round(float(fid_values[i]), 3)}, param_size: {round(float(params[i]), 3)}.')

        self.genotypes = next_population
        self._update_replay_buffer(genotypes[selected])
        self._update_operation_statistics_with_elites(genotypes[selected], is_values[selected], fid_values[selected], params[selected])

        selected_genotypes = genotypes[selected]
        archive_return = self._get_external_archive_return_genotypes(selected_N)
        if len(archive_return) > 0:
            selected_genotypes = np.stack(archive_return, axis=0).astype(np.int64)
        if self._fid_safe_tree_pruning_allowed(num):
            if bool(getattr(self, 'use_three_forest_tree_pruning', True)):
                pruned_return = self._run_three_forest_tree_pruning(fid_stat, selected_N)
            else:
                pruned_return = self._run_fid_safe_tree_pruning(fid_stat, selected_N)
            if len(pruned_return) > 0:
                selected_genotypes = np.stack(pruned_return, axis=0).astype(np.int64)
                logger.info(
                    '[TREE-PRUNE] return %d pruned genotypes as final selected architectures.',
                    len(selected_genotypes),
                )
        keep_is = np.array([
            self.historical_population.get(self._encode_genotype(g), {}).get('is', np.nan)
            for g in next_population
        ], dtype=np.float32)
        keep_fid = np.array([
            self.historical_population.get(self._encode_genotype(g), {}).get('fid', np.nan)
            for g in next_population
        ], dtype=np.float32)
        return next_population, selected_genotypes, avg_keep_is, avg_keep_fid, keep_is, keep_fid

    def my_search_evol(self, parents, fid_stat, num):
        self.current_evolution_algebra = int(num)
        offsprings = self.gen_offspring(parents)
        genotypes = np.concatenate((parents, offsprings), axis=0)
        keep_N, selected_N = len(offsprings), self.args.num_selected

        is_values, fid_values, params = self._evaluate_population(genotypes, fid_stat)
        keep = self._select_population(is_values, fid_values, params, keep_N)
        selected = self._select_population(is_values, fid_values, params, selected_N)

        next_population = np.asarray(genotypes[keep], dtype=np.int64)
        next_population = self._inject_archive_best_fid_into_population(
            next_population,
            target_size=keep_N,
            context=f'gen={num}',
        )

        avg_keep_is = float(np.mean(is_values[keep])) if len(keep) else 0.0
        avg_keep_fid = float(np.mean(fid_values[keep])) if len(keep) else 0.0
        avg_keep_param = float(np.mean(params[keep])) if len(keep) else 0.0
        logger.info(
            f'mean_IS_keep: {round(avg_keep_is, 3)}, mean_FID_keep: {round(avg_keep_fid, 3)}, '
            f'mean_params_keep: {round(avg_keep_param, 3)}, max_IS_all: {round(float(np.max(is_values)), 3)}, '
            f'min_FID_all: {round(float(np.min(fid_values)), 3)} @ evolutionary_algebra: {num}.')

        for i in selected:
            logger.info(
                f'genotypes_{i}, IS: {round(float(is_values[i]), 3)}, '
                f'FID: {round(float(fid_values[i]), 3)}, param_size: {round(float(params[i]), 3)}.')

        self.genotypes = next_population
        self._update_replay_buffer(genotypes[selected])
        self._update_operation_statistics_with_elites(genotypes[selected], is_values[selected], fid_values[selected], params[selected])

        next_is = np.array([
            self.historical_population.get(self._encode_genotype(g), {}).get('is', np.nan)
            for g in next_population
        ], dtype=np.float32)
        next_fid = np.array([
            self.historical_population.get(self._encode_genotype(g), {}).get('fid', np.nan)
            for g in next_population
        ], dtype=np.float32)

        valid_is = next_is[~np.isnan(next_is)]
        valid_fid = next_fid[~np.isnan(next_fid)]
        best_is = float(np.max(valid_is)) if valid_is.size else float(np.max(is_values[keep]))
        best_fid = float(np.min(valid_fid)) if valid_fid.size else float(np.min(fid_values[keep]))
        archive_return = self._get_external_archive_return_genotypes(selected_N)
        selected_genotypes = genotypes[selected]
        if len(archive_return) > 0:
            selected_genotypes = np.stack(archive_return, axis=0).astype(np.int64)
        return next_population, selected_genotypes, avg_keep_is, avg_keep_fid, best_is, best_fid

    def validate(self, genotype_G, fid_stat, epoch=None):
        """
        Calculate IS and FID.
        """
        # Safety fix: avoid None genotype during warm validation / sampling.
        # This keeps the experiment running even if the entry script did not
        # provide gen_fixed_genotype.
        if genotype_G is None:
            if hasattr(self, 'gen_fixed_genotype') and self.gen_fixed_genotype is not None:
                genotype_G = self.gen_fixed_genotype
            elif hasattr(self, 'genotypes') and self.genotypes is not None and len(self.genotypes) > 0:
                genotype_G = np.array(self.genotypes[0], copy=True)
                self.gen_fixed_genotype = genotype_G
            else:
                genotype_G = self.gan_alg.search()
                self.gen_fixed_genotype = np.array(genotype_G, copy=True)
            logger.info('[VALIDATE] genotype_G is None; fallback to a valid generator genotype.')

        gen_net = self.gen_net.eval()
        eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
        fakes = []
        with torch.no_grad():
            for iter_idx in tqdm(range(eval_iter), desc='sample images'):
                z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
                gen_imgs = (gen_net(z, genotype_G) + 1) / 2
                gen_imgs = self._resize_to_target_img_size(gen_imgs)
                fakes.append(gen_imgs.cpu())
        fakes = torch.cat(fakes, dim=0)

        (mean, std), fid_score = get_inception_score_and_fid(fakes, fid_stat, verbose=True)
        return mean, std, fid_score

    # ---------------------------------------------------------------------
    # offspring generation
    # ---------------------------------------------------------------------





    def gen_offspring(self, alphas, offspring_ratio=1.0):
        """Generate offspring with the original three objective pools and direct evaluation.

        Flow:
            current population + IS/FID/Params elite pools
            -> same-pool / cross-pool / population parent sampling
            -> candidate set M
            -> fixed-budget offspring for exact evaluation.
        Only exact-evaluated records in historical_population can become elites.
        """
        alphas = np.asarray(alphas, dtype=np.int64)
        n_offspring = max(1, int(offspring_ratio * alphas.shape[0]))
        elite_pools = self._build_three_elite_pools()
        three_elite_active = any(len(pool) > 0 for pool in elite_pools.values())

        archive_best_fid_parents = self._get_external_archive_best_fid_genotypes(
            self.archive_best_fid_parent_top_k
        )

        if three_elite_active:
            parent_list = [np.array(genotype, copy=True) for genotype in alphas]
            parent_keys = {self._encode_genotype(genotype) for genotype in parent_list}
            for objective in ('is', 'fid', 'params'):
                for genotype in elite_pools[objective]:
                    key = self._encode_genotype(genotype)
                    if key not in parent_keys:
                        parent_keys.add(key)
                        parent_list.append(np.array(genotype, copy=True))
            for genotype in archive_best_fid_parents:
                key = self._encode_genotype(genotype)
                if key not in parent_keys:
                    parent_keys.add(key)
                    parent_list.append(np.array(genotype, copy=True))
            if self._external_archive_enabled() and random.random() < min(max(float(self.external_archive_parent_ratio), 0.0), 1.0):
                for genotype in self._get_external_archive_genotypes(self.external_archive_parent_size):
                    key = self._encode_genotype(genotype)
                    if key not in parent_keys:
                        parent_keys.add(key)
                        parent_list.append(np.array(genotype, copy=True))
            parent_source = np.stack(parent_list, axis=0).astype(np.int64)
        else:
            # Preserve the original single-elite behavior during warm-up or
            # when the three-pool module is disabled.
            parent_source = self._build_elite_parent_source(alphas)
            if len(archive_best_fid_parents) > 0:
                parent_source = np.concatenate(
                    (parent_source, np.stack(archive_best_fid_parents, axis=0).astype(np.int64)),
                    axis=0,
                )
            if self._external_archive_enabled() and random.random() < min(max(float(self.external_archive_parent_ratio), 0.0), 1.0):
                archive_parents = self._get_external_archive_genotypes(self.external_archive_parent_size)
                if len(archive_parents) > 0:
                    parent_source = np.concatenate(
                        (parent_source, np.stack(archive_parents, axis=0).astype(np.int64)),
                        axis=0,
                    )


        target_candidates = n_offspring

        offsprings = []
        local_seen = set()
        attempts = 0
        max_attempts = max(100, target_candidates * 50)
        forced_parent_ratio = min(max(float(getattr(self, 'archive_best_fid_forced_parent_ratio', 0.0)), 0.0), 1.0)
        forced_archive_parent_target = 0
        if len(archive_best_fid_parents) > 0:
            forced_archive_parent_target = int(round(target_candidates * forced_parent_ratio))
            if forced_parent_ratio > 0.0:
                forced_archive_parent_target = max(1, forced_archive_parent_target)
        forced_archive_parent_used = 0
        if forced_archive_parent_target > 0:
            logger.info(
                f'[ARCHIVE-BEST-FID-PARENT] candidates={target_candidates}, '
                f'forced_children={forced_archive_parent_target}, '
                f'best_fid_parents={len(archive_best_fid_parents)}.'
            )

        while len(offsprings) < target_candidates and attempts < max_attempts:
            attempts += 1
            if forced_archive_parent_used < forced_archive_parent_target:
                parent_a = np.array(random.choice(archive_best_fid_parents), copy=True)
                parent_b = np.array(parent_source[np.random.randint(0, parent_source.shape[0])], copy=True)
                if self._encode_genotype(parent_a) == self._encode_genotype(parent_b) and parent_source.shape[0] > 1:
                    parent_b = np.array(parent_source[np.random.randint(0, parent_source.shape[0])], copy=True)
                forced_archive_parent_used += 1
            elif three_elite_active:
                parent_a, parent_b = self._sample_three_elite_parent_pair(
                    alphas, elite_pools
                )
            else:
                a, b = self._select_parent_indices(parent_source)
                if a == b:
                    continue
                parent_a = parent_source[a]
                parent_b = parent_source[b]

            child = self.uniform_crossover_mutation(parent_a, parent_b)
            child = self._apply_fid_stage_protection(child)
            key = self._encode_genotype(child)

            if key in local_seen:
                continue
            if key in self.historical_population:
                continue
            if not self.judege_model_size(child, limit=self._offspring_size_limit(attempts, max_attempts)):
                continue

            local_seen.add(key)
            offsprings.append(child)

        # Random fallback is used only to fill the exact-evaluation budget.
        fallback_attempts = 0
        while len(offsprings) < n_offspring and fallback_attempts < max_attempts * 2:
            random_child = np.array(self.gan_alg.search(), copy=True)
            key = self._encode_genotype(random_child)
            fallback_attempts += 1
            if key in local_seen:
                continue
            if key in self.historical_population:
                continue
            if not self.judege_model_size(random_child, limit=self._offspring_size_limit(fallback_attempts, max_attempts * 2)):
                continue
            local_seen.add(key)
            offsprings.append(random_child)

        # Final emergency fallback: mutate a current/elite parent to avoid crash.
        emergency_trials = 0
        while len(offsprings) < n_offspring and emergency_trials < max_attempts * 4:
            parent = np.array(parent_source[np.random.randint(0, parent_source.shape[0])], copy=True)
            child = self._apply_tree_mutation(parent, 1)
            emergency_trials += 1
            key = self._encode_genotype(child)
            if key not in local_seen and self.judege_model_size(child, limit=float(getattr(self.args, 'max_model_size', self.target_model_size))):
                local_seen.add(key)
                offsprings.append(child)
                continue
            if self.judege_model_size(parent, limit=float(getattr(self.args, 'max_model_size', self.target_model_size))):
                offsprings.append(parent)

        while len(offsprings) < n_offspring:
            parent = np.array(parent_source[np.random.randint(0, parent_source.shape[0])], copy=True)
            offsprings.append(parent)

        # Preserve the evaluation budget without performance prediction.
        offsprings = offsprings[:n_offspring]
        offsprings = np.stack(offsprings[:n_offspring], axis=0)
        return offsprings

    def judge_repeat(self, alphas, new_alphas):
        diff = np.reshape(np.absolute(
            alphas - np.expand_dims(new_alphas, axis=0)), (alphas.shape[0], -1))
        diff = np.sum(diff, axis=1)
        return np.sum(diff == 0)

    # ---------------------------------------------------------------------
    # reversible ordered-tree representation and same-position crossover
    # ---------------------------------------------------------------------
    @staticmethod
    def _stage_row_to_ordered_tree(stage_row):
        """Convert one 7-slot stage into an ordered, reversible tree.

        The two upsampling operations are ordered trunk nodes. Normal
        operations c0/c2 attach to trunk 0, c1/c3 attach to trunk 1, and c4 is
        the ordered bridge from the merged node2 feature to node3.
        """
        row = np.asarray(stage_row, dtype=np.int64).reshape(-1)
        if row.size != 7:
            raise ValueError(f'Each stage must contain 7 genes, got {row.size}.')
        return {
            'trunks': [
                {
                    'up': int(row[0]),
                    'to_node2': int(row[2]),
                    'to_node3': int(row[4]),
                },
                {
                    'up': int(row[1]),
                    'to_node2': int(row[3]),
                    'to_node3': int(row[5]),
                },
            ],
            'node2_to_node3': int(row[6]),
        }

    def _genotype_to_ordered_stage_trees(self, genotype_G):
        """Map a 3x7 genotype to three ordered stage trees without information loss."""
        genotype = np.asarray(genotype_G, dtype=np.int64)
        expected_shape = (
            self.num_layers,
            self.num_up_slots + self.num_normal_slots,
        )
        if genotype.shape != expected_shape:
            raise ValueError(
                f'Generator genotype must have shape {expected_shape}, '
                f'got {genotype.shape}.'
            )
        return [
            self._stage_row_to_ordered_tree(genotype[layer])
            for layer in range(self.num_layers)
        ]

    def _ordered_stage_trees_to_genotype(self, stage_trees):
        """Restore three ordered stage trees to the original 3x7 genotype."""
        if len(stage_trees) != self.num_layers:
            raise ValueError(
                f'Expected {self.num_layers} stage trees, got {len(stage_trees)}.'
            )

        genotype = np.zeros(
            (self.num_layers, self.num_up_slots + self.num_normal_slots),
            dtype=np.int64,
        )
        for layer, tree in enumerate(stage_trees):
            trunks = tree['trunks']
            if len(trunks) != self.num_up_slots:
                raise ValueError(
                    f'Stage {layer} must contain {self.num_up_slots} ordered trunks.'
                )
            genotype[layer] = np.asarray([
                trunks[0]['up'],
                trunks[1]['up'],
                trunks[0]['to_node2'],
                trunks[1]['to_node2'],
                trunks[0]['to_node3'],
                trunks[1]['to_node3'],
                tree['node2_to_node3'],
            ], dtype=np.int64)
        return genotype

    def _validate_tree_round_trip(self, genotype_G):
        """Fail early if the tree view cannot exactly reconstruct a genotype."""
        genotype = np.asarray(genotype_G, dtype=np.int64)
        restored = self._ordered_stage_trees_to_genotype(
            self._genotype_to_ordered_stage_trees(genotype)
        )
        if not np.array_equal(genotype, restored):
            raise RuntimeError('Ordered tree encoding is not exactly reversible.')
        return True

    @staticmethod
    def _tree_subtree_value(stage_tree, subtree_name):
        if subtree_name == 'stage':
            return stage_tree
        if subtree_name == 'trunk_0':
            return stage_tree['trunks'][0]
        if subtree_name == 'trunk_1':
            return stage_tree['trunks'][1]
        if subtree_name == 'node2_to_node3':
            return stage_tree['node2_to_node3']
        raise ValueError(f'Unknown subtree name: {subtree_name}')

    @staticmethod
    def _replace_tree_subtree(target_tree, donor_tree, subtree_name):
        if subtree_name == 'stage':
            return deepcopy(donor_tree)
        if subtree_name == 'trunk_0':
            target_tree['trunks'][0] = deepcopy(donor_tree['trunks'][0])
        elif subtree_name == 'trunk_1':
            target_tree['trunks'][1] = deepcopy(donor_tree['trunks'][1])
        elif subtree_name == 'node2_to_node3':
            target_tree['node2_to_node3'] = int(donor_tree['node2_to_node3'])
        else:
            raise ValueError(f'Unknown subtree name: {subtree_name}')
        return target_tree

    def tree_subtree_crossover_mutation(self, alphas_a, alphas_b):
        """Same-position subtree crossover followed by the original mutation.

        One parent is selected as the recipient using the existing quality
        bias. A structurally different subtree is then copied from the other
        parent at exactly the same stage and semantic position. Candidate
        subtrees are a whole stage, either ordered upsampling trunk with its
        normal branches, or the node2-to-node3 bridge branch.
        """
        alphas_a = np.asarray(alphas_a, dtype=np.int64)
        alphas_b = np.asarray(alphas_b, dtype=np.int64)
        trees_a = self._genotype_to_ordered_stage_trees(alphas_a)
        trees_b = self._genotype_to_ordered_stage_trees(alphas_b)

        quality_a = self._estimate_arch_quality(alphas_a)
        quality_b = self._estimate_arch_quality(alphas_b)
        if quality_a is None or quality_b is None:
            base_is_a = random.random() < 0.5
        else:
            better_is_a = quality_a >= quality_b
            choose_better = random.random() < self.good_gene_parent_bias
            base_is_a = better_is_a if choose_better else not better_is_a

        base_trees = deepcopy(trees_a if base_is_a else trees_b)
        donor_trees = trees_b if base_is_a else trees_a

        subtree_names = (
            'stage',
            'trunk_0',
            'trunk_1',
            'node2_to_node3',
        )
        crossover_points = []
        for layer in range(self.num_layers):
            for subtree_name in subtree_names:
                base_value = self._tree_subtree_value(
                    base_trees[layer], subtree_name
                )
                donor_value = self._tree_subtree_value(
                    donor_trees[layer], subtree_name
                )
                if base_value != donor_value:
                    crossover_points.append((layer, subtree_name))

        if crossover_points:
            layer, subtree_name = random.choice(crossover_points)
            base_trees[layer] = self._replace_tree_subtree(
                base_trees[layer],
                donor_trees[layer],
                subtree_name,
            )

        new_alphas = self._ordered_stage_trees_to_genotype(base_trees)
        self._validate_tree_round_trip(new_alphas)

        num_mute = self._adaptive_mutation_budget(alphas_a, alphas_b)
        if self.use_tree_mutation:
            new_alphas = self._apply_tree_mutation(new_alphas, num_mute)
        else:
            new_alphas = self._apply_mutation(new_alphas, num_mute)
        return self._repair_genotype(new_alphas)

    @staticmethod
    def _tree_normal_slots_for_subtree(subtree_name):
        if subtree_name == 'stage':
            return [2, 3, 4, 5, 6]
        if subtree_name == 'trunk_0':
            return [2, 4]
        if subtree_name == 'trunk_1':
            return [3, 5]
        if subtree_name == 'node2_to_node3':
            return [6]
        raise ValueError(f'Unknown subtree name: {subtree_name}')

    def _tree_mutation_context(self):
        layer = random.randint(0, self.num_layers - 1)
        subtree_name = random.choice(('stage', 'trunk_0', 'trunk_1', 'node2_to_node3'))
        return layer, subtree_name, self._tree_normal_slots_for_subtree(subtree_name)

    @staticmethod
    def _tree_group_for_slot(slot):
        if slot in (2, 3):
            return 2, 4
        if slot in (4, 5, 6):
            return 4, 7
        raise ValueError(f'Normal branch slot expected, got {slot}.')

    def _tree_can_delete_slot(self, genotype_G, layer, slot):
        start, end = self._tree_group_for_slot(slot)
        group = genotype_G[layer, start:end]
        return int(genotype_G[layer, slot]) != 0 and int(np.sum(group != 0)) > 1

    def _tree_replace_slot_nonzero(self, genotype_G, layer, slot):
        old_value = int(genotype_G[layer, slot])
        for _ in range(8):
            new_value = self._sample_guided_nonzero_op(layer, slot)
            if new_value != old_value:
                genotype_G[layer, slot] = new_value
                return genotype_G
        genotype_G[layer, slot] = self._sample_guided_nonzero_op(layer, slot)
        return genotype_G

    def _apply_tree_add_branch(self, genotype_G):
        genotype_G = np.asarray(genotype_G, dtype=np.int64).copy()
        layer, _subtree_name, slots = self._tree_mutation_context()
        zero_slots = [slot for slot in slots if int(genotype_G[layer, slot]) == 0]
        if len(zero_slots) == 0:
            return self._apply_tree_replace_branch(genotype_G)
        slot = random.choice(zero_slots)
        genotype_G[layer, slot] = self._sample_guided_nonzero_op(layer, slot)
        return self._repair_genotype(genotype_G)

    def _apply_tree_delete_branch(self, genotype_G):
        genotype_G = np.asarray(genotype_G, dtype=np.int64).copy()
        layer, _subtree_name, slots = self._tree_mutation_context()
        deletable = [
            slot for slot in slots
            if self._tree_can_delete_slot(genotype_G, layer, slot)
        ]
        if len(deletable) == 0:
            return self._apply_tree_replace_branch(genotype_G)
        genotype_G[layer, random.choice(deletable)] = 0
        return self._repair_genotype(genotype_G)

    def _apply_tree_replace_branch(self, genotype_G):
        genotype_G = np.asarray(genotype_G, dtype=np.int64).copy()
        layer, _subtree_name, slots = self._tree_mutation_context()
        slot = random.choice(slots)
        if int(genotype_G[layer, slot]) == 0:
            genotype_G[layer, slot] = self._sample_guided_nonzero_op(layer, slot)
        else:
            genotype_G = self._tree_replace_slot_nonzero(genotype_G, layer, slot)
        return self._repair_genotype(genotype_G)

    def _apply_tree_replace_subtree(self, genotype_G):
        genotype_G = np.asarray(genotype_G, dtype=np.int64).copy()
        layer, subtree_name, slots = self._tree_mutation_context()
        donor = np.asarray(self.gan_alg.search(remove=False), dtype=np.int64)
        if subtree_name == 'stage':
            genotype_G[layer] = donor[layer]
        else:
            for slot in slots:
                genotype_G[layer, slot] = donor[layer, slot]
            if subtree_name == 'trunk_0':
                genotype_G[layer, 0] = donor[layer, 0]
            elif subtree_name == 'trunk_1':
                genotype_G[layer, 1] = donor[layer, 1]
        return self._repair_genotype(genotype_G)

    def _apply_single_tree_mutation(self, genotype_G):
        probs = np.asarray([
            max(0.0, float(self.tree_mutation_add_prob)),
            max(0.0, float(self.tree_mutation_delete_prob)),
            max(0.0, float(self.tree_mutation_replace_prob)),
            max(0.0, float(self.tree_mutation_subtree_prob)),
        ], dtype=np.float64)
        if float(np.sum(probs)) <= 0.0:
            raise ValueError('Configured tree mutation probabilities must have positive total mass.')
        probs = probs / np.sum(probs)
        route = int(np.random.choice(4, p=probs))
        if route == 0:
            return self._apply_tree_add_branch(genotype_G)
        if route == 1:
            return self._apply_tree_delete_branch(genotype_G)
        if route == 2:
            return self._apply_tree_replace_branch(genotype_G)
        return self._apply_tree_replace_subtree(genotype_G)

    def _apply_tree_mutation(self, genotype_G, num_mute):
        genotype_G = np.asarray(genotype_G, dtype=np.int64).copy()
        steps = max(1, int(num_mute))
        for _ in range(steps):
            genotype_G = self._apply_single_tree_mutation(genotype_G)
        return self._repair_genotype(genotype_G)

    def crossover(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        while new_alphas[layer][index] == alphas_a[layer][index]:
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            new_alphas[layer][index] = alphas_b[layer][index]
            if index >= 2 and index < 4 and new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
            elif index >= 4 and new_alphas[layer][4] == 0 and new_alphas[layer][5] == 0 and new_alphas[layer][6] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
        return new_alphas

    def good_gene_crossover_mutation(self, alphas_a, alphas_b):
        """Good-gene crossover + link/operation mutation.

        Same positions in the two parents are treated as shared good genes and
        are kept directly. For different positions, the better cached/predicted
        parent is sampled with a higher probability, so crossover is no longer
        purely random. Mutation is then split into link-style mutation and
        operation mutation.
        """
        alphas_a = np.asarray(alphas_a, dtype=np.int64)
        alphas_b = np.asarray(alphas_b, dtype=np.int64)
        new_alphas = alphas_a.copy()

        quality_a = self._estimate_arch_quality(alphas_a)
        quality_b = self._estimate_arch_quality(alphas_b)
        if quality_a is None or quality_b is None:
            better_parent = None
        else:
            better_parent = alphas_a if quality_a >= quality_b else alphas_b

        for i in range(self.num_layers):
            for j in range(self.num_up_slots + self.num_normal_slots):
                if alphas_a[i][j] == alphas_b[i][j]:
                    # shared gene: keep it
                    new_alphas[i][j] = alphas_a[i][j]
                else:
                    if better_parent is not None and random.random() < self.good_gene_parent_bias:
                        new_alphas[i][j] = better_parent[i][j]
                    else:
                        new_alphas[i][j] = alphas_b[i][j] if random.random() < 0.5 else alphas_a[i][j]

        num_mute = self._adaptive_mutation_budget(alphas_a, alphas_b)
        new_alphas = self._apply_mutation(new_alphas, num_mute)
        return self._repair_genotype(new_alphas)

    def uniform_crossover_mutation(self, alphas_a, alphas_b):
        """Compatibility wrapper.

        The tree experiment changes only crossover representation/granularity.
        Existing good-gene and uniform implementations remain available as
        fallbacks, and the original mutation is used by every route.
        """
        if self.use_tree_subtree_crossover:
            return self.tree_subtree_crossover_mutation(alphas_a, alphas_b)

        if self.use_good_gene_crossover:
            return self.good_gene_crossover_mutation(alphas_a, alphas_b)

        new_alphas = alphas_a.copy()
        for i in range(3):
            for j in range(2):
                if random.random() >= 0.5:
                    new_alphas[i][j] = alphas_b[i][j]
            for k in range(5):
                if random.random() >= 0.5:
                    new_alphas[i][k + 2] = alphas_b[i][k + 2]

        num_mute = self._adaptive_mutation_budget(alphas_a, alphas_b)
        new_alphas = self._apply_mutation(new_alphas, num_mute)
        return self._repair_genotype(new_alphas)

    def single_point_crossover_mutation(self, alphas_a, alphas_b):
        """Single-point crossover + probability-guided mutation."""
        new_alphas = alphas_a.copy()
        a1 = random.randint(0, 2)
        a2 = random.randint(0, 7)
        for i in range(a1):
            for j in range(7):
                new_alphas[i][j] = alphas_b[i][j]
        for k in range(a2):
            new_alphas[a1][k] = alphas_b[a1][k]

        num_mute = self._adaptive_mutation_budget(alphas_a, alphas_b)
        new_alphas = self._apply_mutation(new_alphas, num_mute)
        return new_alphas

    def two_point_crossover_mutation(self, alphas_a, alphas_b):
        """Two-point crossover + probability-guided mutation."""
        alphas_a1 = alphas_a.ravel()
        alphas_b1 = alphas_b.ravel()
        new_alphas = alphas_a1.copy()
        a1 = random.randint(0, len(alphas_a1))
        a2 = random.randint(a1, len(alphas_a1))
        for i in range(a1, a2):
            new_alphas[i] = alphas_b1[i]
        new_alphas = new_alphas.reshape((3, 7))

        num_mute = self._adaptive_mutation_budget(alphas_a, alphas_b)
        new_alphas = self._apply_mutation(new_alphas, num_mute)
        return new_alphas

    def mutation(self, alphas_a, ratio=0.5):
        """Mutation for an individual using probability-guided sampling."""
        new_alphas = alphas_a.copy()
        num_mute = max(1, int(round(max(1.0, ratio)))) if ratio > 1 else 1
        new_alphas = self._apply_mutation(new_alphas, num_mute)
        return new_alphas

    # ---------------------------------------------------------------------
    # remaining original helper methods
    # ---------------------------------------------------------------------
    def select_best(self, epoch, fid_stat=None):
        if fid_stat is None:
            # Prefer already-evaluated exact architectures when available.
            best_idx = None
            best_is = -float('inf')
            for idx, genotype_G in enumerate(self.genotypes):
                cached = self.historical_population.get(self._encode_genotype(genotype_G), None)
                if cached is not None and cached.get('exact', False) and cached['is'] > best_is:
                    best_is = cached['is']
                    best_idx = idx
            if best_idx is not None:
                return self.genotypes[best_idx]
            raise ValueError('select_best requires fid_stat when no exact evaluation cache is available.')

        values = []
        for genotype_G in self.genotypes:
            is_value, _, _ = self.validate(genotype_G, fid_stat, epoch)
            values.append(is_value)
        max_index = int(np.argmax(values))
        return self.genotypes[max_index]

    def get_fitness(self, genotypes, fid_stat):
        fids = np.zeros(len(genotypes))
        for idx, genotype_G in enumerate(genotypes):
            gen_net = self.gen_net.eval()
            eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
            fakes = []
            with torch.no_grad():
                for iter_idx in tqdm(range(eval_iter), desc='sample images'):
                    z = torch.cuda.FloatTensor(np.random.normal(
                        0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
                    gen_imgs = (gen_net(z, genotype_G) + 1) / 2
                    fakes.append(gen_imgs.cpu())
            fakes = torch.cat(fakes, dim=0)
            fid_score = get_fid(fakes, fid_stat)
            fids[idx] = fid_score
        return fids

    def get_index(self, genotypes, fid_stat):
        fids_G = self.get_fitness(genotypes, fid_stat)
        index = fids_G.argsort()
        return index

    def get_operation_score(self, genotypes, index):
        scores_up = np.zeros((3, 2, 3), dtype=int)
        scores_normal = np.zeros((3, 5, 7), dtype=int)
        jj = 0
        num = len(genotypes) / 2
        for kk in index:
            for i in range(3):
                for j in range(2):
                    scores_up[i][j][genotypes[kk][i][j]] += 1
                for k in range(2):
                    scores_normal[i][k][genotypes[kk][i][k + 2]] += 1
                for k in range(2, 5):
                    scores_normal[i][k][genotypes[kk][i][k + 2]] += 1
            jj += 1
            if jj == num:
                break
        return scores_up, scores_normal

    def modify_fixed(self, Normal_G_fixed, Up_G_fixed, scores_up, scores_normal):
        index_up = scores_up.argsort()
        index_normal = scores_normal.argsort()
        for i in range(3):
            for j in range(2):
                Up_G_fixed[2 * i + j].remove(index_up[i][j][0])
        for i in range(3):
            for j in range(5):
                for kk in range(3):
                    Normal_G_fixed[5 * i + j].remove(index_normal[i][j][kk])
        return None

    def greedy_modify(self, Normal_G_fixed, Up_G_fixed, rank, genotypes):
        self.gan_alg.Normal_G_fixed = []
        for i in range(15):
            self.gan_alg.Normal_G_fixed.append([])
        self.gan_alg.Up_G_fixed = []
        for i in range(6):
            self.gan_alg.Up_G_fixed.append([])

        for r in rank:
            for i in range(3):
                for j in range(2):
                    if genotypes[r][i][j] not in self.gan_alg.Up_G_fixed[2 * i + j] and len(
                            self.gan_alg.Up_G_fixed[2 * i + j]) < 2:
                        self.gan_alg.Up_G_fixed[2 * i + j].append(genotypes[r][i][j])
            for i in range(3):
                for j in range(5):
                    if genotypes[r][i][j + 2] not in self.gan_alg.Normal_G_fixed[5 * i + j] and len(
                            self.gan_alg.Normal_G_fixed[5 * i + j]) < 4:
                        self.gan_alg.Normal_G_fixed[5 * i + j].append(genotypes[r][i][j + 2])
        return None

    def directly_modify_fixed(self, fid_stat):
        self.clear_bag()
        genotypes_to_eva = np.stack([self.gan_alg.search() for _ in range(42)], axis=0)
        rank = self.get_index(genotypes_to_eva, fid_stat)
        scores_up, scores_normal = self.get_operation_score(genotypes_to_eva, rank)
        self.modify_fixed(self.gan_alg.Normal_G_fixed, self.gan_alg.Up_G_fixed, scores_up, scores_normal)
        self.clear_bag()
        self.genotype_fixG = self.gan_alg.search(remove=False)

    def greedy_modify_fixed(self, num, fid_stat):
        self.clear_bag()
        genotypes_to_eva = np.stack([self.gan_alg.search() for _ in range(num)], axis=0)
        rank = self.get_index(genotypes_to_eva, fid_stat)
        self.greedy_modify(self.gan_alg.Normal_G_fixed, self.gan_alg.Up_G_fixed, rank, genotypes_to_eva)
        self.clear_bag()
        self.genotype_fixG = self.gan_alg.search(remove=False)

    def clear_bag(self):
        self.gan_alg.Normal_G = []
        for i in range(15):
            self.gan_alg.Normal_G.append([])
        self.gan_alg.Up_G = []
        for i in range(6):
            self.gan_alg.Up_G.append([])


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):
        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr
