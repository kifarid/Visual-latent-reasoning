import argparse
import datetime
import glob
import os
import sys
import logging
import torch
import warnings

from omegaconf import OmegaConf
from functools import partial


from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy

from util import *


torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

#os.environ["TK_WORK_DIR"] = "/work/dlclarge2/faridk-diff_force/orbis/logs_tk"
#os.environ["WM_WORK_DIR"] = "/work/dlclarge2/faridk-diff_force/orbis/logs_wm"

def get_callbacks(opt, logdir, ckptdir, config, lightning_config, now):
    
    overrides = {
        "logdir": logdir,
        "ckptdir": ckptdir,
        "cfgdir": os.path.join(logdir, "configs"),
        "now": now,
        "lightning_config": lightning_config,
        "config": config,
        "resume": opt.resume,
        "log_img_frequency": opt.log_img_frequency,
        "log_val_img_frequency": opt.log_val_img_frequency,
        "periodic_checkpoint": opt.periodic_checkpoint,
        "log_ckpt_frequency": opt.log_ckpt_frequency,
        "increase_log_steps": opt.increase_log_steps,
        "tsne_epoch_frequency": opt.tsne_epoch_frequency,
        "log_img_epoch_frequency": opt.log_img_epoch_frequency,
        "save_on_train_epoch_end": opt.save_on_train_epoch_end,
        "bar_refresh_rate": 100 if os.environ.get("SLURM_JOB_ID") else 1,
    }
    
    # if config contains the callbacks field, use it
    if "callbacks" in config:
        callbacks_config = config.callbacks
    else:
        callbacks_config = OmegaConf.load("callbacks/configs/base.yaml").callbacks

    callbacks = []
    for cb in callbacks_config:
        # resolve interpolations in the config
        callbacks.append(instantiate_from_config(OmegaConf.merge(overrides, cb)))

    return callbacks



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-g",
        "--n_gpus",
        type=int,
        default=None,
        help="number of gpus per node",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=None,
        help="number of nodes",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="check validation every n epochs",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=None,
        help="Override validation check interval in epochs or steps (Lightning's val_check_interval)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="number of epochs",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "--profiler",
        type=str2bool,
        nargs="?",
        default=True,
    )
    parser.add_argument(
        "--tsne_epoch_frequency", 
        type=int, 
        default=0,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--periodic_checkpoint", 
        type=int, 
        default=5,
    )
    parser.add_argument(
        "--save_on_train_epoch_end",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, ModelCheckpoint saves at the end of every train epoch.",
    )
    parser.add_argument(
        "--enable_codebook_usage_logger", 
        action="store_true", 
        help="Enable CodebookUsageLogger callback")
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Enable Fully‐Sharded Data Parallel (FSDP) strategy")
    parser.add_argument(
        "--increase_log_steps",
        action="store_true",
        help="Increase log steps exponentially for ImageLogger",)
    parser.add_argument(
        "--log_img_frequency",
        type=int,
        default=2000,
        help="log image frequency",
    )
    parser.add_argument(
        "--log_val_img_frequency",
        type=int,
        default=400,
        help="log validation image frequency",
    )
    parser.add_argument(
        "--log_img_epoch_frequency",
        type=int,
        default=1,
        help="log images every n epochs (1 = every epoch)",
    )
    parser.add_argument(
        "--log_ckpt_frequency",
        type=int,
        default=2000,
        help="checkpointing frequency",
    )
    parser.add_argument(
        "--logger_type",
        type=str,
        choices=["tensorboard", "wandb"],
        default="tensorboard",
        help="Logging backend for pytorch lightning",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="orbis_cg",
        help="Weights & Biases project name (required if logger_type=wandb)",
    )
    return parser



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

        
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    lightning_config.trainer = trainer_config
    

    if opt.n_gpus is None:
        # use slurm env vars if available otherwise use all CUDA_AVAILABLE gpus
        opt.n_gpus = int(os.environ.get("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
    if opt.n_nodes is None:
        # use slurm env vars if available otherwise set to 1
        opt.n_nodes = int(os.environ.get("SLURM_NNODES", 1))

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
        
    if "first_stage" in config.model.target and not "second_stage" in config.model.target:
        workdir_env_name = "TK_WORK_DIR"
    else:
        workdir_env_name = "WM_WORK_DIR"
        
    workdir_env_name = os.environ.get(workdir_env_name)
        
    if workdir_env_name is None:
        default_path = os.getcwd()+'/logs'
        workdir_env_name = default_path
        warnings.warn(
            f"Environment variables 'TK_WORK_DIR' or 'WM_WORK_DIR' are not set. "
            f"Using default work directory: {default_path}"
        )
    
        
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            # determine logdir from checkpoint
            paths = opt.resume.split("/")
            log_name = workdir_env_name.split("/")[-1]
            idx = len(paths)-paths[::-1].index(log_name)+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
            logger.info("Resuming from checkpoint %s, logdir: %s", opt.resume, logdir)
        else:
            # determine checkpoint from logdir
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
            logger.info("Resuming from folder %s, logdir: %s", opt.resume, logdir)
        opt.resume_from_checkpoint = ckpt

        # determine configs from logdir
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base

        # determine name from logdir
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix+"_"+get_jobid()

        logdir = os.path.join(workdir_env_name, nowname)
    logger.info(">>> Logging to %s", logdir)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    # model
    model = instantiate_from_config(config.model)
    # Wire gradient clipping values onto the model for manual optimization paths.
    try:
        setattr(model, "grad_clip", config.model.get("grad_clip", 0.5))
        setattr(model, "gradient_clip_algorithm", config.model.get("gradient_clip_algorithm", "norm"))
    except Exception:
        pass

    # trainer and callbacks
    trainer_kwargs = dict()
    if opt.logger_type == "tensorboard":
        trainer_kwargs["logger"] = TensorBoardLogger(logdir, name="tb")
    else:
        if opt.wandb_project is None:
            raise ValueError("--wandb_project must be provided when using Wandb logger")
        os.makedirs(logdir, exist_ok=True)
        trainer_kwargs["logger"] = WandbLogger(
            name=nowname,
            project=opt.wandb_project,
            log_model=True,
            save_dir=logdir,
            config=OmegaConf.to_container(config, resolve=True),
        )
    logger.info("Logging to %s", logdir)
    trainer_kwargs["devices"] = opt.n_gpus
    trainer_kwargs["num_nodes"] = opt.n_nodes
    trainer_kwargs["check_val_every_n_epoch"] = opt.check_val_every_n_epoch
    if opt.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = None #opt.val_check_interval
    trainer_kwargs["max_epochs"] = opt.num_epochs
    trainer_kwargs["precision"] = config.model.get("precision", "bf16-mixed")
    trainer_kwargs["sync_batchnorm"] = config.model.get("sync_batchnorm", True)
    trainer_kwargs["gradient_clip_val"] = config.model.get("grad_clip", None)
    trainer_kwargs["gradient_clip_algorithm"] = config.model.get("gradient_clip_algorithm", None)
    # Respect manual optimization: Lightning does not support automatic gradient accumulation
    # together with manual optimization. When the model opts into manual optimization
    # (`self.automatic_optimization = False`), we let the model handle accumulation
    # internally via its own `grad_acc_steps` and disable Trainer accumulation.
    _acc_steps_cfg = config.model.get("grad_acc_steps", 1)
    _is_manual_opt = getattr(model, "automatic_optimization", True) is False
    if _is_manual_opt and _acc_steps_cfg != 1:
        logger.warning(
            "Manual optimization detected; ignoring accumulate_grad_batches=%s and using model-managed accumulation.",
            _acc_steps_cfg,
        )
        trainer_kwargs["accumulate_grad_batches"] = 1
    else:
        trainer_kwargs["accumulate_grad_batches"] = _acc_steps_cfg
    if opt.use_fsdp:
        tlc = get_obj_from_str(config.model.params.generator_config.target)
        my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={tlc})
        trainer_kwargs["strategy"] = FSDPStrategy(
            sharding_strategy=ShardingStrategy.FULL_SHARD, #NO_SHARD, #FULL_SHARD, # SHARD_GRAD_OP ,# HYBRID_SHARD
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, buffer_dtype=torch.float16, reduce_dtype=torch.float16),
            # limit_all_gathers=True,
        )
    else:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=config.model.get("find_unused_parameters", False), static_graph=config.model.get("static_graph", False)) # static_graph=True,   
    
    # profiler        
    if opt.profiler:
        profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler(logdir),
        schedule=torch.profiler.schedule(skip_first=5 ,wait=1, warmup=1, active=3, repeat=2) 
        )
        trainer_kwargs["profiler"] = profiler

    trainer_kwargs["callbacks"] = get_callbacks(opt, logdir, ckptdir, config, lightning_config, now)
    # trainer_kwargs["detect_anomaly"] = True
    trainer = Trainer(**trainer_kwargs)
    
    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    # configure learning rate
    bs, base_lr, adjust_learning_rate = config.data.params.batch_size, config.model.base_learning_rate, config.model.adjust_learning_rate
    grad_acc_steps = max(config.model.params.get("grad_acc_steps", 1), config.model.get("grad_acc_steps", 1))

    ngpu = opt.n_gpus


    model.num_iters_per_epoch = len(data.datasets["train"]) // (config.data.params.batch_size * opt.n_gpus * grad_acc_steps)
    # Make grad_acc_steps available to manual-optimization models
    try:
        setattr(model, "grad_acc_steps", int(grad_acc_steps))
    except Exception:
        logger.warning("Could not set grad_acc_steps attribute on model.")
        pass
    logger.info("Num iters per epoch: %s", model.num_iters_per_epoch)

    bs_acc_factor = max(grad_acc_steps//8, 1)
    if adjust_learning_rate:
        model.learning_rate = base_lr * bs * ngpu * grad_acc_steps * opt.n_nodes
        logger.info("Setting learning rate to %s = %s (num_gpus) * %s (batchsize) * %s (base_lr) * %s (grad accumulation factor)",
        model.learning_rate, ngpu, bs, base_lr, grad_acc_steps)
    else: 
        model.learning_rate = base_lr
        logger.info("Setting learning rate to %s", model.learning_rate)


    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            logger.warning("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb; pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    try:
        if opt.resume:
            trainer.fit(model, data, ckpt_path = opt.resume_from_checkpoint)
        else:
            trainer.fit(model, data)
    except Exception:
        melk()
        raise
