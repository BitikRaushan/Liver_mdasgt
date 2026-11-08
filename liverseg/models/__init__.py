from .mda_sgt import MDASGT


def build_model(cfg: dict) -> MDASGT:
    """cfg: dict (e.g. vars(args)) - extra keys are ignored."""
    return MDASGT(
        in_ch=len(str(cfg["phases"]).split(",")),
        base=cfg.get("base", 32),
        img_size=cfg.get("img_size", 256),
        encoder=cfg.get("encoder", "cres2net"),
        use_transformer=not cfg.get("no_transformer", False),
        use_gsa=not cfg.get("no_gsa", False),
        skip=cfg.get("skip", "dense"),
        direction_guidance=not cfg.get("no_direction", False),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("depth", 4),
        heads=cfg.get("heads", 8),
        lambda_init=cfg.get("lambda_init", 0.0),
    )
