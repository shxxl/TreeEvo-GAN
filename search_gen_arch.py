from __future__ import absolute_import, division, print_function
import search_cfg as search_cfg
import archs
import datasets
from trainer.trainer_generator import GenTrainer
from trainer.trainer_utils import LinearLrDecay
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from archs.super_network import Generator, Discriminator
from archs.fully_super_network import simple_Discriminator
from algorithms.search_algs import GanAlgorithm
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
#from pytorch_gan_metrics import get_inception_score_and_fid
from pytorch_image_generation_metrics import get_inception_score_and_fid

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False



def main():
    args = search_cfg.parse_args()

    torch.cuda.manual_seed(args.random_seed)

    # ===================== GPU visible settings =====================
    # args.gpu_ids in parser usually string like "4" or "4,5"
    if len(args.gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # The first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for i in range(len(str_ids)):
        if i >= 0:
            args.gpu_ids.append(i)  # visible index, not physical id

    if len(args.gpu_ids) > 1:
        args.gpu_ids = args.gpu_ids[1:]  # leave visible GPU0 for evaluation
    else:
        args.gpu_ids = args.gpu_ids

    # ===================== init genotype algorithm =====================
    gan_alg = GanAlgorithm(args)

    # ===================== build network =====================
    basemodel_gen = Generator(args)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    basemodel_dis = simple_Discriminator()
    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # ===================== weight init OR resume =====================
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(f'{args.init_type} unknown init type')
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # If resuming from checkpoint, do NOT random init
    if not getattr(args, "resume_ckpt", ""):
        gen_net.apply(weights_init)
        dis_net.apply(weights_init)

    # ===================== data_loader =====================
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # epoch number for dis_net
    max_epoch_for_D = args.my_max_epoch_G * args.n_critic
    args.max_epoch_D = args.my_max_epoch_G * args.n_critic
    max_iter_D = max_epoch_for_D * len(train_loader)

    # ===================== fid stat =====================
    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # ===================== load checkpoint BEFORE optimizer =====================
    start_epoch = 0
    resume_ckpt = getattr(args, "resume_ckpt", "")
    need_modify_fixed = False  # mark only; actually run modify_fixed after trainer created

    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            print(f"==> Loading checkpoint from: {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location="cpu")

            # load weights
            if "weight_G" in ckpt:
                gen_net.load_state_dict(ckpt["weight_G"], strict=True)
            elif "G" in ckpt:
                gen_net.load_state_dict(ckpt["G"], strict=True)
            else:
                raise KeyError("Checkpoint missing generator weights: expected key 'weight_G' or 'G'")

            if "weight_D" in ckpt:
                dis_net.load_state_dict(ckpt["weight_D"], strict=True)
            elif "D" in ckpt:
                dis_net.load_state_dict(ckpt["D"], strict=True)

            # restore fixed info
            if "up_G_fixed" in ckpt:
                gan_alg.Up_G_fixed = ckpt["up_G_fixed"]
            if "normal_G_fixed" in ckpt:
                gan_alg.Normal_G_fixed = ckpt["normal_G_fixed"]

            # resume epoch
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1

            # infer prune state
            try:
                up_lens = [len(x) for x in getattr(gan_alg, "Up_G_fixed", [])]
                nm_lens = [len(x) for x in getattr(gan_alg, "Normal_G_fixed", [])]
                print(f"==> Up_G_fixed lens: {up_lens}")
                print(f"==> Normal_G_fixed lens (first 10): {nm_lens[:10]}")

                ckpt_name = os.path.basename(resume_ckpt)

                # rule1: filename contains before_prune => must prune once
                if "before_prune" in ckpt_name:
                    need_modify_fixed = True

                # rule2: still in original big space => also prune once
                if (len(up_lens) > 0 and max(up_lens) >= 3) and (len(nm_lens) > 0 and max(nm_lens) >= 7):
                    need_modify_fixed = True

            except Exception as e:
                print(f"[WARN] Cannot infer prune state from Up/Normal lens: {e}. "
                      f"Will NOT auto-run modify_fixed unless filename contains before_prune.")
                ckpt_name = os.path.basename(resume_ckpt)
                if "before_prune" in ckpt_name:
                    need_modify_fixed = True

            print(f"==> Loaded. start_epoch={start_epoch}, need_modify_fixed={need_modify_fixed}")
            del ckpt
        else:
            print(f"[WARN] resume_ckpt not found: {resume_ckpt}")

    # ===================== optimizer / scheduler =====================
    gen_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gen_net.parameters()),
        args.g_lr, (args.beta1, args.beta2)
    )
    dis_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dis_net.parameters()),
        args.d_lr, (args.beta1, args.beta2)
    )

    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # ===================== logger / writer =====================
    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)


    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    #logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))

    trainer_gen = GenTrainer(
        args, gen_net, dis_net, gen_optimizer, dis_optimizer,
        train_loader, gan_alg, None, None
    )

    # ===================== run modify_fixed ONCE if needed =====================
    # ===================== run modify_fixed ONCE if needed (RESUME ONLY) =====================
    # 注意：这里才可以调用 trainer_gen.directly_modify_fixed(fid_stat)
    # 因为 trainer_gen 已经构建完成，里面需要 dataloader / gan_alg 等依赖

    resume_ckpt = getattr(args, "resume_ckpt", "")

    if resume_ckpt:
        if need_modify_fixed:
            logger.info(
                "[RESUME] Detected unpruned supernet (or ckpt is before_prune). "
                "Run directly_modify_fixed() once..."
            )
            trainer_gen.directly_modify_fixed(fid_stat)

            # 打印剪枝后空间大小，确认真的变小了
            try:
                up_lens2 = [len(x) for x in getattr(gan_alg, "Up_G_fixed", [])]
                nm_lens2 = [len(x) for x in getattr(gan_alg, "Normal_G_fixed", [])]
                logger.info(f"[RESUME] After modify_fixed: Up_G_fixed lens: {up_lens2}")
                logger.info(f"[RESUME] After modify_fixed: Normal_G_fixed lens (first 10): {nm_lens2[:10]}")
            except Exception as e:
                logger.warning(f"[RESUME] After modify_fixed, cannot print lens: {e}")
        else:
            logger.info("[RESUME] Checkpoint looks already pruned. Skip directly_modify_fixed().")
    else:
        # 从头训练时不要打印 [RESUME]，避免误导
        # 真正的剪枝（directly_modify_fixed）应该在训练循环里按你的 warmup 触发点执行
        logger.info("[INIT] Start from scratch. Skip resume-only directly_modify_fixed().")

    # ===================== (optional) save current supernet snapshot (once) =====================
    def save_supernet_snapshot(tag: str):
        checkpoint_file = os.path.join(args.path_helper['ckpt_path'], f'{tag}.pt')
        ckpt = {
            'weight_G': gen_net.state_dict(),
            'weight_D': dis_net.state_dict(),
            'up_G_fixed': gan_alg.Up_G_fixed,
            'normal_G_fixed': gan_alg.Normal_G_fixed,
            'epoch': int(getattr(args, "current_epoch", -1)),
        }
        torch.save(ckpt, checkpoint_file)
        del ckpt

    save_supernet_snapshot("supernet_gen")  # initial snapshot after (optional) modify_fixed

    # ===================== train supernet OR evo-only =====================
    # During training, keep your original logic: no model size limit
    temp_model_size = args.max_model_size
    args.max_model_size = 999

    total_epochs = int(getattr(args, "my_max_epoch_G", 200))

    if not getattr(args, "evo_only", False):
        for epoch in tqdm(range(int(start_epoch), total_epochs), desc='training the supernet_G:'):
            args.current_epoch = epoch  # for snapshot metadata if needed
            lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
            trainer_gen.train(epoch, writer_dict, fid_stat, lr_schedulers)

            # keep your original discard/prune trigger if you still want it here
            # NOTE: if you already prune inside trainer_gen.train(), you can remove this block
            if epoch == args.warmup:  # Start discarding
                save_supernet_snapshot("gen_checkpoint_before_prune")
                trainer_gen.directly_modify_fixed(fid_stat)
                logger.info(
                    f'gan_alg.Normal_G_fixed: {gan_alg.Normal_G_fixed}, gan_alg.Up_G_fixed: {gan_alg.Up_G_fixed},@ epoch {epoch}.'
                )
    else:
        logger.info("EVO_ONLY enabled: skip supernet training loop.")
        ckpt_name = os.path.basename(resume_ckpt) if resume_ckpt else ""
        need_modify = ("before_prune" in ckpt_name)
        if need_modify:
            logger.info("[EVO_ONLY] resume from before_prune -> run directly_modify_fixed() once to align pipeline.")
            trainer_gen.directly_modify_fixed(fid_stat)
            logger.info(
                f'[EVO_ONLY] After modify_fixed: gan_alg.Normal_G_fixed={gan_alg.Normal_G_fixed}, '
                f'gan_alg.Up_G_fixed={gan_alg.Up_G_fixed}.'
            )
        else:
            logger.info("[EVO_ONLY] resume from supernet_gen (or other) -> skip directly_modify_fixed().")

    # ===================== save final supernet snapshot (once) =====================
    save_supernet_snapshot("supernet_gen_final")

    # ===================== restore model size constraint =====================
    args.max_model_size = temp_model_size

    # ===================== (IMPORTANT) final evo search: OPTIONAL =====================
    # Training-time "every 10 epoch search once" is inside trainer_gen.train(), so by default do NOT run offline final evo.
    run_final_evo = bool(getattr(args, "evo_only", False)) or bool(getattr(args, "final_evo_search", False))

    if run_final_evo:
        logger.info(
            f"[FINAL-EVO] run_final_evo={run_final_evo}, Total_evolutionary_algebra={args.Total_evolutionary_algebra}"
        )
        population = np.stack([gan_alg.search() for _ in range(args.num_individual)], axis=0)
        pop_selected = None

        for ii in tqdm(range(args.Total_evolutionary_algebra), desc='search generator using evo alg'):
            out = trainer_gen.my_search_evolv2(population, fid_stat, ii)
            if out is None:
                raise RuntimeError(f"my_search_evolv2 returned None at ii={ii}.")
            population, pop_selected, is_mean, fid_mean, _, _ = out

        if pop_selected is not None:
            for index, geno in enumerate(pop_selected):
                file_path = os.path.join(args.path_helper['ckpt_path'], f"best_gen_{index}.npy")
                np.save(file_path, geno)
    else:
        # Non evo_only: save training-time pool if any
        pool_to_save = None
        if hasattr(trainer_gen, "train_pool_genotypes") and trainer_gen.train_pool_genotypes is not None and len(
                trainer_gen.train_pool_genotypes) > 0:
            pool_to_save = trainer_gen.train_pool_genotypes
        elif hasattr(trainer_gen, "elite_genotypes") and trainer_gen.elite_genotypes is not None and len(
                trainer_gen.elite_genotypes) > 0:
            pool_to_save = trainer_gen.elite_genotypes

        if pool_to_save is not None:
            logger.info(f"[POOL-SAVE] saving pool (size={len(pool_to_save)})")
            for index, geno in enumerate(pool_to_save):
                file_path = os.path.join(args.path_helper['ckpt_path'], f"pool_gen_{index}.npy")
                np.save(file_path, geno)
        else:
            logger.info("[POOL-SAVE] pool is empty; skip saving pool architectures.")


if __name__ == '__main__':
    main()
