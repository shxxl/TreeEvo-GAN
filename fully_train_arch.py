from __future__ import absolute_import, division, print_function

import os
import json
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import archs
import cfg
import datasets
from archs.fully_super_network import Discriminator, Generator
from archs.network_64 import Discriminator64
from network import (LinearLrDecay, copy_params, load_params, my_validate,
                     save_dis_images, train)
from utils.flop_benchmark import print_FLOPs
from utils.utils import create_logger, count_parameters_in_MB, set_log_dir


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def _resolve_exps_path(relative_path):
    path = os.path.join('exps', relative_path)
    if not os.path.isfile(path):
        raise FileNotFoundError('Required experiment file does not exist: {}'.format(path))
    return path


def _atomic_torch_save(state, path):
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _empty_selected_metrics():
    return dict(selected_epoch=None, selected_fid=None,
                selected_is=None, selected_is_std=None)


def _update_selected_metrics(selected, epoch, is_value, is_std, fid_value, best_fid):
    """Select one EMA evaluation by FID; keep its IS/std together. Epoch is zero-based."""
    if not np.all(np.isfinite([is_value, is_std, fid_value])):
        raise ValueError('Cannot select a checkpoint using nonfinite EMA metrics.')
    if float(fid_value) >= float(best_fid):
        return False, dict(selected)
    return True, dict(selected_epoch=int(epoch), selected_fid=float(fid_value),
                      selected_is=float(is_value), selected_is_std=float(is_std))


def _restore_selected_metrics(checkpoint):
    selected = {key: checkpoint.get(key) for key in _empty_selected_metrics()}
    # Older best_is values were independently maximized. They cannot supply
    # the IS/std of the FID-selected checkpoint.
    if any(value is None for value in selected.values()):
        return _empty_selected_metrics()
    if not np.all(np.isfinite(list(selected.values()))):
        return _empty_selected_metrics()
    if not np.isclose(float(selected['selected_fid']), float(checkpoint['best_fid']),
                      rtol=1e-10, atol=1e-10):
        return _empty_selected_metrics()
    return selected


def _checkpoint_state(args, epoch, best_fid, gen_net, dis_net,
                      gen_optimizer, dis_optimizer, gen_avg_param, selected_metrics):
    avg_gen_net = deepcopy(gen_net)
    load_params(avg_gen_net, gen_avg_param)
    state = {
        'epoch': int(epoch),
        'model': args.arch,
        'genotype_of_G': args.genotype_of_G,
        'genotypes_exp': args.genotypes_exp,
        'random_seed': args.random_seed,
        'gen_state_dict': gen_net.state_dict(),
        'dis_state_dict': dis_net.state_dict(),
        'avg_gen_state_dict': avg_gen_net.state_dict(),
        'gen_optimizer': gen_optimizer.state_dict(),
        'dis_optimizer': dis_optimizer.state_dict(),
        'best_fid': float(best_fid),
        # Compatibility alias: IS paired with the selected FID, never max(IS).
        'best_is': selected_metrics['selected_is'],
        'selected_epoch_index_base': 0,
        'selected_checkpoint': 'checkpoint_best.pth',
        'selected_evaluation_weights': 'EMA',
        'path_helper': args.path_helper,
    }
    state.update(selected_metrics)
    del avg_gen_net
    return state


def main():
    args = cfg.parse_args()

    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    visible_ids = args.gpu_ids.split(',')
    args.gpu_ids = list(range(len(visible_ids)))
    if not args.gpu_ids:
        raise RuntimeError('At least one CUDA GPU is required.')

    genotype_g_path = _resolve_exps_path(args.genotype_of_G)
    genotype_G = np.load(genotype_g_path, allow_pickle=False)
    if genotype_G.shape != (3, 7):
        raise ValueError(
            'Generator genotype must have shape (3, 7), got {} from {}'.format(
                genotype_G.shape, genotype_g_path))

    basemodel_gen = Generator(args, genotype_G)
    gen_net = torch.nn.DataParallel(
        basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    if args.use_basemodel_D:
        basemodel_dis = Discriminator64(args)
    else:
        genotype_d_path = _resolve_exps_path(args.genotypes_exp)
        genotype_D = np.load(genotype_d_path, allow_pickle=False)
        basemodel_dis = Discriminator(args, genotype_D)
    dis_net = torch.nn.DataParallel(
        basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    def weights_init(module):
        classname = module.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(module.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight.data, 1.0)
            else:
                raise NotImplementedError(
                    '{} unknown init type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(
            args.max_iter_G * args.n_critic / len(train_loader))
    args.max_epoch_D = int(args.max_epoch_D)
    max_iter_D = args.max_epoch_D * len(train_loader)

    if args.optimizer.lower() == 'adam':
        gen_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, gen_net.parameters()),
            args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, dis_net.parameters()),
            args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer.lower() == 'rmsprop':
        gen_optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, gen_net.parameters()),
            args.g_lr, alpha=0.9)
        dis_optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, dis_net.parameters()),
            args.d_lr, alpha=0.9)
    elif args.optimizer.lower() == 'sgd':
        gen_optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, gen_net.parameters()),
            lr=args.g_lr, momentum=0.8)
        dis_optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, dis_net.parameters()),
            lr=args.d_lr, momentum=0.8)
    else:
        raise NotImplementedError('Unknown optimizer: {}'.format(args.optimizer))

    gen_scheduler = LinearLrDecay(
        gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(
        dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() == 'celeba':
        fid_stat = './fid_stat/celeba.3k.128.npz'
    else:
        raise NotImplementedError(
            'no fid stat for {}'.format(args.dataset.lower()))
    if not os.path.isfile(fid_stat):
        raise FileNotFoundError(fid_stat)

    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = float('inf')
    selected_metrics = _empty_selected_metrics()

    if args.checkpoint:
        checkpoint_dir = os.path.join('exps', args.checkpoint)
        checkpoint_file = os.path.join(
            checkpoint_dir, 'Model', 'checkpoint_best.pth')
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        start_epoch = int(checkpoint['epoch'])
        best_fid = float(checkpoint.get('best_fid', best_fid))
        selected_metrics = _restore_selected_metrics(checkpoint)
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net
        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info('=> loaded checkpoint %s (epoch %d)', checkpoint_file, start_epoch)
        if selected_metrics['selected_epoch'] is None:
            logger.warning(
                'Legacy checkpoint lacks paired selection metrics: historical best FID '
                'is retained; its IS/std stay unknown until a new FID best is found.')
    else:
        if not args.exp_name:
            raise ValueError('--exp_name is required for a new retraining run.')
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    logger.info('[FIXED-RETRAIN] generator=%s, discriminator=%s, seed=%d.',
                genotype_g_path,
                'Discriminator64' if args.use_basemodel_D else genotype_d_path,
                args.random_seed)
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)

    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': 0,
    }

    last_epoch = start_epoch
    for epoch in tqdm(
            range(start_epoch, args.max_epoch_D), desc='fixed architecture retraining'):
        last_epoch = epoch + 1
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer,
              gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)

        # Keep the original EWSGAN evaluation phase, plus a mandatory final eval.
        should_validate = (
            epoch == 210
            or (epoch > 400 and epoch % args.val_freq == 0)
            or epoch == args.max_epoch_D - 1
        )
        is_best_fid = False
        if should_validate:
            inception_score, std, fid_score = my_validate(args, gen_net, fid_stat)
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score_ema, std_ema, fid_score_ema = my_validate(
                args, gen_net, fid_stat)
            load_params(gen_net, backup_param)

            logger.info(
                'IS: %.3f, IS_std: %.3f, FID: %.3f, '
                'IS_ema: %.3f, IS_std_ema: %.3f, FID_ema: %.3f @ epoch %d.',
                inception_score, std, fid_score,
                inception_score_ema, std_ema, fid_score_ema, epoch)

            is_best_fid, selected_metrics = _update_selected_metrics(
                selected_metrics, epoch, inception_score_ema, std_ema,
                fid_score_ema, best_fid)
            if is_best_fid:
                best_fid = selected_metrics['selected_fid']
                logger.info(
                    '[SELECTED-EMA] epoch_index=%d IS=%.6f IS_std=%.6f FID=%.6f.',
                    selected_metrics['selected_epoch'], selected_metrics['selected_is'],
                    selected_metrics['selected_is_std'], selected_metrics['selected_fid'])

        # Overwrite latest instead of accumulating many large checkpoint files.
        if last_epoch % 25 == 0 or should_validate:
            state = _checkpoint_state(
                args, last_epoch, best_fid,
                gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, selected_metrics)
            _atomic_torch_save(
                state, os.path.join(args.path_helper['ckpt_path'], 'checkpoint_latest.pth'))
            if is_best_fid:
                _atomic_torch_save(
                    state, os.path.join(args.path_helper['ckpt_path'], 'checkpoint_best.pth'))
                selected_path = os.path.join(args.path_helper['ckpt_path'], 'selected_metrics.json')
                with open(selected_path + '.tmp', 'w', encoding='utf-8') as handle:
                    json.dump(dict(selected_metrics, selected_epoch_index_base=0,
                                   selected_checkpoint='checkpoint_best.pth', evaluation_weights='EMA'),
                              handle, indent=2)
                os.replace(selected_path + '.tmp', selected_path)
            del state

        if should_validate and best_fid >= 120.0:
            logger.info('[FIXED-RETRAIN] early stop: best FID %.3f >= 120.', best_fid)
            break

    final_state = _checkpoint_state(
        args, last_epoch, best_fid,
        gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, selected_metrics)
    _atomic_torch_save(
        final_state, os.path.join(args.path_helper['ckpt_path'], 'checkpoint_final.pth'))
    if not os.path.isfile(os.path.join(args.path_helper['ckpt_path'], 'checkpoint_best.pth')):
        _atomic_torch_save(
            final_state, os.path.join(args.path_helper['ckpt_path'], 'checkpoint_best.pth'))
    del final_state

    save_dis_images(args, gen_net, last_epoch, False)
    backup_param = copy_params(gen_net)
    load_params(gen_net, gen_avg_param)
    save_dis_images(args, gen_net, last_epoch, True)
    load_params(gen_net, backup_param)

    logger.info('[FIXED-RETRAIN] complete: epoch=%d, selected EMA evaluation=%s.',
                last_epoch, selected_metrics)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
