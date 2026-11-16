import argparse
import math


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type')
    parser.add_argument('--img_size', type=int, default=32, help='image size, 32 for cifar10, 48 for stl10')
    parser.add_argument('--bottom_width', type=int, default=4, help='init resolution, 4 for cifar10, 6 for stl10')
    parser.add_argument('--channels', type=int, default=3, help='image channels')
    parser.add_argument('--num_op_g', type=int, default=1, help='The number of training operations over the network,/')
    parser.add_argument('--exp_name', type=str, default='arch_searchG_cifar10', help='experiment name')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')

    parser.add_argument('--gpu_ids', type=str, default='0', help='visible GPU ids')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')

    parser.add_argument('--arch', type=str, default='arch_cifar10', help='architecture name')
    parser.add_argument('--max_epoch_G', type=int, default=200, help='max number of epoch for training G')
    parser.add_argument('--max_iter_G', type=int, default=None, help='max number of iteration for training G')
    parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
    parser.add_argument('--gen_bs', type=int, default=256, help='batche size of G')
    parser.add_argument('--dis_bs', type=int, default=256, help='batche size of D')
    parser.add_argument('--gf_dim', type=int, default=256, help='base channel-dim of G')
    parser.add_argument('--df_dim', type=int, default=128, help='base channel-dim of D')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--lr_decay', action='store_true', help='learning rate decay or not')
    parser.add_argument('--beta1', type=float, default=0.0, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='init type')
    parser.add_argument('--d_spectral_norm', type=str2bool, default=True,
                        help='add spectral_norm on discriminator or not')
    parser.add_argument('--g_spectral_norm', type=str2bool, default=False,
                        help='add spectral_norm on generator or not')
    parser.add_argument('--latent_dim', type=int, default=120, help='dimensionality of the latent space')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help='path to resume checkpoint for generator/discriminator supernet')

    parser.add_argument('--evo_only', action='store_true',
                        help='skip supernet training and directly run evolutionary search')

    parser.add_argument('--final_evo_search', action='store_true',
                        help='run final evolutionary search after training')
    parser.add_argument('--print_freq', type=int, default=50, help='interval between each verbose')
    parser.add_argument('--val_freq', type=int, default=20, help='interval between each validation')
    parser.add_argument('--num_eval_imgs', type=int, default=5000)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--derived_start_epoch', type=int, default=0, help='')
    parser.add_argument('--derived_max_epoch', type=int, default=None, help='')
    parser.add_argument('--derived_epoch_interval', type=int, default=None, help='')
    parser.add_argument('--Total_evolutionary_algebra', type=int, default=30,
                        help='Algebra of Evolutionary Architecture')
    parser.add_argument('--tau_max', type=float, default=5, help='max tau for gumbel softmax')
    parser.add_argument('--tau_min', type=float, default=0.1, help='min tau for gumbel softmax')
    parser.add_argument('--gumbel_softmax', type=str2bool, default=False, help='use gumbel softmax or not')
    parser.add_argument('--amending_coefficient', type=float, default=0.1, help='')
    parser.add_argument('--derive_freq', type=int, default=1, help='')
    parser.add_argument('--derive_per_epoch', type=int, default=0, help='number of derive per epoch')
    parser.add_argument('--draw_arch', type=str2bool, default=True, help='visualize the searched architecture or not')
    parser.add_argument('--early_stop', type=str2bool, default=False, help='use early stop strategy or not')
    parser.add_argument('--genotypes_exp', type=str, help='ues genotypes of the experiment')
    parser.add_argument('--warmup', type=int, default=40, help='epochs before EA')
    parser.add_argument('--num_individual', type=int, default=32, help='numbers of  EA individual')
    parser.add_argument('--num_selected', type=int, default=12, help='numbers of  selected individual')
    parser.add_argument('--ga_interval', type=int, default=10, help='numbers of  EA interval')
    parser.add_argument('--start_search', type=int, default=200, help='Start searching arc')
    parser.add_argument('--epoch_generator', type=int, default=200, help='numbers of epoch for generator')
    parser.add_argument('--epoch_both', type=int, default=400, help='numbers of epoch for generator')
    parser.add_argument('--epoch_discriminator', type=int, default=200, help='numbers of epoch for discriminator')
    parser.add_argument('--mute_max_num', type=int, default=4, help='numbers of mute_max_num')
    parser.add_argument('--model_name', type=str, help='ues model of supernetG')
    parser.add_argument('--my_max_epoch_G', type=int, default=200, help='numbers of mute_max_num')
    parser.add_argument('--max_model_size', type=float, default=15, help='max model size')
    parser.add_argument('--tree_w_gene', type=float, default=0.40)
    parser.add_argument('--tree_w_active', type=float, default=0.35)
    parser.add_argument('--tree_w_subtree', type=float, default=0.25)
    # Experiment sampling settings. The trainer reads these values from args.
    parser.add_argument('--three_elite_fid_parent_weight', type=float, default=0.70)
    parser.add_argument('--three_elite_is_parent_weight', type=float, default=0.20)
    parser.add_argument('--three_elite_params_parent_weight', type=float, default=0.10)
    parser.add_argument('--tree_mutation_add_prob', type=float, default=0.35)
    parser.add_argument('--tree_mutation_delete_prob', type=float, default=0.25)
    parser.add_argument('--tree_mutation_replace_prob', type=float, default=0.30)
    parser.add_argument('--tree_mutation_subtree_prob', type=float, default=0.10)
    parser.add_argument('--good_gene_parent_bias', type=float, default=0.70)
    parser.add_argument(
        '--three_elite_quality_keep_ratio',
        type=float,
        default=0.60,
        help='Quality-retention ratio for objective-conditioned forests'
    )

    opt = parser.parse_args()

    opt = parser.parse_args()
    groups = (
        ('tree distance weights', (opt.tree_w_gene, opt.tree_w_active, opt.tree_w_subtree)),
        ('forest parent weights', (opt.three_elite_fid_parent_weight,
          opt.three_elite_is_parent_weight, opt.three_elite_params_parent_weight)),
        ('tree mutation probabilities', (opt.tree_mutation_add_prob,
          opt.tree_mutation_delete_prob, opt.tree_mutation_replace_prob,
          opt.tree_mutation_subtree_prob)),
    )
    for name, values in groups:
        if (not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
                or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9)):
            parser.error(name + ' must be finite probabilities summing to one')
    if not math.isfinite(opt.good_gene_parent_bias) or not 0.0 <= opt.good_gene_parent_bias <= 1.0:
        parser.error('--good_gene_parent_bias must be finite and between zero and one')
    return opt
