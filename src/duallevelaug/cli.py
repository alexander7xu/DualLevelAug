import argparse
import json
import sys
from pathlib import Path
from .config import load_config
from .utils import write_json


def parser():
    p=argparse.ArgumentParser(prog="dla",description="DualLevelAug reproduction: data preparation, augmentation, training, evaluation, and experiment orchestration")
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("prepare","stats","pretrain-graph","generate","train","evaluate","features","report"):
        command=sub.add_parser(name)
        command.add_argument("--config",required=True,help="JSON configuration path")
        if name in ("train","evaluate","features"):
            command.add_argument("--dry-run",action="store_true",help="Print process argv without launching it")
    experiment=sub.add_parser("experiments",aliases=["experiment"])
    exp=experiment.add_subparsers(dest="operation",required=True)
    for name in ("list","plan","run"):
        action=exp.add_parser(name)
        action.add_argument("selectors",nargs="*",default=["all"],help="Table/figure/condition ID or wildcard")
        action.add_argument("--root",default=".",help="Data/artifact root")
        action.add_argument("--overrides",help="JSON overrides, e.g. backend paths/Python interpreters")
        if name=="plan":
            action.add_argument("--output",help="Optionally save the expanded JSON plan")
        if name=="run":
            action.add_argument("--no-resume",action="store_true")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command in ("experiments","experiment"):
            from . import experiments
            from .utils import read_json
            overrides=read_json(args.overrides) if args.overrides else None
            if args.operation=="list":
                result=experiments.select(args.selectors)
            elif args.operation=="plan":
                result=experiments.plan(args.selectors,root=args.root,overrides=overrides)
                if args.output:
                    write_json(args.output,result)
            else:
                result=experiments.run(args.selectors,root=args.root,overrides=overrides,resume=not args.no_resume)
        else:
            cfg=load_config(args.config)
            if args.command=="prepare":
                from .data import prepare
                options={k:v for k,v in cfg.items() if k not in ("dataset","root","raw_root","output","protocol","seed")}
                result=prepare(cfg["dataset"],cfg.get("raw_root",cfg.get("root")),cfg["output"],protocol=cfg.get("protocol","default"),seed=cfg.get("seed",0),**options)
            elif args.command=="stats":
                from .statistics import collect
                result=collect(cfg)
            elif args.command=="pretrain-graph":
                from .graph import pretrain
                result=pretrain(cfg)
            elif args.command=="generate":
                from .generation import generate
                result=generate(cfg)
            elif args.command in ("train","evaluate","features"):
                from .backends import build_command,run
                result=build_command(args.command,args.config,cfg) if args.dry_run else run(args.command,args.config,cfg)
            else:
                from .reporting import report
                result=report(cfg)
        if result is not None:
            print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    except (ValueError,FileNotFoundError,KeyError,RuntimeError) as exc:
        print("dla: %s"%exc,file=sys.stderr)
        return 2
    return 0


if __name__=="__main__":
    sys.exit(main())
