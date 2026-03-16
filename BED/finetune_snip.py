"""
Finetune SNIP (Symbolic-Numeric Integrated Pretraining) on expressions generated from a
SRToolkit SymbolLibrary.

SNIP learns a shared latent space between symbolic (equation) and numeric (data points)
representations via a CLIP-style contrastive objective. This script finetunes a pretrained
SNIP checkpoint so that the model specialises to the operator vocabulary defined by a given
SymbolLibrary.

Usage:

python BED/finetune_snip.py --checkpoint Multimodal-Math-Pretraining-main/weights/snip-10dmax.pth --output_dir ./finetuned_snip --num_variables 2 --symbols "+,-,*,/,sin,cos,exp,C" --n_expressions 50000 --n_epochs 20 --allow_ood

Notes on choosing a pretrained checkpoint:
    - snip-1d-normalized.pth  : 1-variable expressions, normalised y
    - snip-10dmax.pth         : 1-10 variable expressions (use for num_variables > 1)
    The number of variables must fit within the checkpoint's max_input_dimension.

Supported SRToolkit → SNIP operator mapping:
    +  → add       -   → sub       *   → mul       /   → div
    ^  → pow       sqrt→ sqrt      sin → sin       cos → cos
    exp→ exp       tan → tan       arcsin/arccos/arctan (unchanged)
    ln → log       ^-1 → inv       ^2  → pow2      ^3  → pow3
    pi → pi        e   → e         C   → CONSTANT
    X_i→ x_i

Unsupported (no SNIP equivalent): u-, sinh, cosh, tanh, floor, ceil, log (log10), ^4, ^5.
Expressions containing unsupported symbols are silently skipped.
"""

import sys
import os
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# ---------------------------------------------------------------------------
# Add the SNIP package to the Python path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
SNIP_DIR = _REPO_ROOT / "Multimodal-Math-Pretraining-main"
sys.path.insert(0, str(SNIP_DIR))

from SRToolkit.utils import (
    SymbolLibrary,
    tokens_to_tree,
    expr_to_executable_function,
    generate_n_expressions,
)

# ---------------------------------------------------------------------------
# numpy.compat.py3k was removed in NumPy 2.0; generators.py imports it but
# never uses it, so we provide a stub to keep SNIP importable.
# ---------------------------------------------------------------------------
import types
try:
    import numpy.compat.py3k  # noqa: F401
except (ModuleNotFoundError, ImportError):
    _stub = types.ModuleType("numpy.compat.py3k")
    _stub.npy_load_module = None
    sys.modules.setdefault("numpy.compat", types.ModuleType("numpy.compat"))
    sys.modules["numpy.compat.py3k"] = _stub

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol mapping: SRToolkit name → SNIP prefix-notation name
# None means the symbol has no direct SNIP equivalent.
# ---------------------------------------------------------------------------
_SRTOOLKIT_TO_SNIP: dict[str, str | None] = {
    # Binary operators
    "+":      "add",
    "-":      "sub",
    "*":      "mul",
    "/":      "div",
    "^":      "pow",
    # Unary functions
    "u-":     None,       # unary minus — not in SNIP
    "sqrt":   "sqrt",
    "sin":    "sin",
    "cos":    "cos",
    "exp":    "exp",
    "tan":    "tan",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "sinh":   None,
    "cosh":   None,
    "tanh":   None,
    "floor":  None,
    "ceil":   None,
    "ln":     "log",      # SNIP's "log" is the natural logarithm
    "log":    "log",      # SRToolkit log10; reuses SNIP's log token — numeric data disambiguates during finetuning
    "^-1":    "inv",
    "^2":     "pow2",
    "^3":     "pow3",
    "^4":     None,
    "^5":     None,
    # Constants / literals
    "pi":     "pi",
    "e":      "e",
    "C":      "CONSTANT", # free (unknown) constant placeholder
    # Numeric literals introduced by equivalence transformations
    "0":      "CONSTANT", # no zero leaf in SNIP; treat as opaque constant
    "0.5":    "CONSTANT", # float — absent from SNIP vocab
    "1":      "1",
    "2":      "2",
    "-1":     "-1",
    "10":     "10",
}


def _get_snip_token(
    symbol: str,
    num_variables: int,
    sl: SymbolLibrary | None = None,
    allow_ood: bool = False,
) -> str | None:
    """
    Return the SNIP vocabulary token for a SRToolkit symbol.

    If the symbol has no direct SNIP equivalent and ``allow_ood`` is True, the
    function falls back to SNIP's OOD tokens:
        - ``OOD_unary_op``  for symbols of type "fn"  (unary functions)
        - ``OOD_binary_op`` for symbols of type "op"  (binary operators)

    Returns None when the symbol is genuinely unresolvable (unknown variable
    index, unknown type, or allow_ood is False).
    """
    if symbol in _SRTOOLKIT_TO_SNIP:
        mapped = _SRTOOLKIT_TO_SNIP[symbol]
        if mapped is not None:
            return mapped
        # mapped is None → unsupported; fall through to OOD logic below
    elif symbol.startswith("X_"):
        try:
            idx = int(symbol[2:])
        except ValueError:
            return None
        if idx < num_variables:
            return f"x_{idx}"
        return None
    else:
        # Completely unknown symbol; fall through to OOD logic below
        pass

    if not allow_ood or sl is None:
        return None

    sym_type = sl.get_type(symbol)
    if sym_type == "fn":
        return "OOD_unary_op"
    if sym_type == "op":
        return "OOD_binary_op"
    return None


def _tree_to_snip_prefix(
    node,
    num_variables: int,
    sl: SymbolLibrary | None = None,
    allow_ood: bool = False,
) -> list[str] | None:
    """
    Walk a SRToolkit expression tree and produce a SNIP-compatible prefix token list.

    Returns None if the tree contains any symbol that cannot be mapped (either
    because it is unsupported and allow_ood is False, or because its arity is
    unknown even in OOD mode).
    """
    if node is None:
        return []

    snip_tok = _get_snip_token(node.symbol, num_variables, sl=sl, allow_ood=allow_ood)
    if snip_tok is None:
        return None

    left_tokens  = _tree_to_snip_prefix(node.left,  num_variables, sl=sl, allow_ood=allow_ood)
    right_tokens = _tree_to_snip_prefix(node.right, num_variables, sl=sl, allow_ood=allow_ood)

    if left_tokens is None or right_tokens is None:
        return None

    # SNIP prefix: operator first, then left subtree, then right subtree
    # (matches the SRToolkit Node.to_list(notation="prefix") convention)
    return [snip_tok] + left_tokens + right_tokens


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _exponent_ok(v: float, max_exponent: int, float_precision: int) -> bool:
    """
    Return True iff v can be encoded by SNIP's float encoder without producing
    an out-of-vocabulary exponent token (E-max_exponent … E+max_exponent).
    """
    if not np.isfinite(v) or v == 0.0:
        return False
    _, exp_str = (f"%.{float_precision}e" % abs(float(v))).split("e")
    expon = int(exp_str) - float_precision   # matches FloatSequences.encode: expon = int(e) - precision
    return abs(expon) <= max_exponent


def build_dataset(
    expressions: list[list[str]],
    symbol_library: SymbolLibrary,
    n_points: int,
    x_range: tuple[float, float],
    const_range: tuple[float, float],
    seed: int = 42,
    allow_ood: bool = False,
    max_exponent: int = 100,
    float_precision: int = 3,
) -> list[dict]:
    """
    Validate SRToolkit expressions and return lightweight training records.

    Each record stores only the SNIP prefix tokens, the compiled function, and
    the constant count — **no xy_pairs are stored in memory**.  Fresh (x, y)
    points are sampled per batch during training, keeping RAM usage proportional
    to one batch rather than the entire dataset.

    A small probe evaluation is run here to reject expressions that never
    produce any valid encoder-safe outputs on the given x_range.

    Each valid sample dict:
        prefix_tokens : list[str]   — SNIP prefix token sequence
        fn            : callable    — compiled numpy evaluator
        n_consts      : int         — number of CONSTANT placeholders
    """
    rng = np.random.default_rng(seed)
    num_variables = symbol_library.num_variables
    dataset: list[dict] = []

    for expr_tokens in expressions:
        try:
            tree = tokens_to_tree(expr_tokens, symbol_library)
            if tree is None:
                continue

            prefix = _tree_to_snip_prefix(
                tree, num_variables, sl=symbol_library, allow_ood=allow_ood
            )
            if not prefix:
                continue  # empty or unsupported

            n_consts = prefix.count("CONSTANT")

            fn = expr_to_executable_function(expr_tokens, symbol_library)
            if fn is None:
                continue

            # Probe: verify the expression produces ≥4 encoder-safe points
            X_probe = rng.uniform(x_range[0], x_range[1], size=(n_points, num_variables))
            C_probe = (
                rng.uniform(const_range[0], const_range[1], size=n_consts).tolist()
                if n_consts > 0 else []
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                y_probe = fn(X_probe, C_probe)

            valid = np.array([
                _exponent_ok(yi, max_exponent, float_precision) for yi in y_probe
            ])
            if valid.sum() < 4:
                continue

            dataset.append({"prefix_tokens": prefix, "fn": fn, "n_consts": n_consts})

        except Exception:
            continue

    return dataset


# ---------------------------------------------------------------------------
# Batch collation  (samples xy fresh per batch — no large dataset in RAM)
# ---------------------------------------------------------------------------

def _collate_batch(
    samples: list[dict],
    env,
    device: torch.device,
    rng: np.random.Generator,
    n_points: int,
    x_range: tuple[float, float],
    const_range: tuple[float, float],
    num_variables: int,
    max_exponent: int,
    float_precision: int,
) -> tuple:
    """
    Sample fresh (x, y) points for each expression and pack into tensors.

    Returns:
        x2, len2   : symbolic encoder input (equation token indices)
        xy_batch   : numeric encoder input (list of point-pair sequences)
    """
    all_token_seqs = [s["prefix_tokens"] for s in samples]

    # Sample xy_pairs first; track which samples yield ≥1 valid point.
    # A fresh C draw can produce no valid points even when the probe succeeded.
    xy_batch: list = []
    keep: list[int] = []
    for idx, s in enumerate(samples):
        X = rng.uniform(x_range[0], x_range[1], size=(n_points, num_variables))
        C_vals = (
            rng.uniform(const_range[0], const_range[1], size=s["n_consts"]).tolist()
            if s["n_consts"] > 0 else []
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = s["fn"](X, C_vals)
        valid = np.array([_exponent_ok(yi, max_exponent, float_precision) for yi in y])
        X, y = X[valid], y[valid]
        if len(X) == 0:
            continue  # skip — no encoder-safe points with this C draw
        xy_batch.append([[X[i], np.array([y[i]])] for i in range(len(X))])
        keep.append(idx)

    if len(keep) < 2:
        return None  # caller will skip this batch (CLIP needs ≥2 samples)

    token_seqs = [all_token_seqs[i] for i in keep]

    # Validate vocabulary on the surviving token sequences
    for seq in token_seqs:
        for tok in seq:
            if tok not in env.equation_word2id:
                raise KeyError(
                    f"Token '{tok}' is not in the SNIP vocabulary. "
                    "Check num_variables matches the pretrained checkpoint."
                )

    indexed = env.word_to_idx(token_seqs, float_input=False)
    x2, len2 = env.batch_equations(indexed)
    x2   = x2.to(device)
    len2 = len2.to(device)

    return x2, len2, xy_batch


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_modules(checkpoint_path: str, params, env) -> dict:
    """Build SNIP modules and load pretrained weights from a checkpoint."""
    from snip.model import build_modules

    modules = build_modules(env, params)

    logger.info(f"Loading pretrained weights from {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for name in ("embedder", "encoder_y", "encoder_f"):
        if name not in modules or name not in data:
            continue
        weights = data[name]
        # Strip "module." prefix added by DistributedDataParallel if present
        if all(k.startswith("module.") for k in weights):
            weights = {k[7:]: v for k, v in weights.items()}
        try:
            modules[name].load_state_dict(weights)
            logger.info(f"  Loaded {name}")
        except RuntimeError as exc:
            logger.warning(f"  Could not load {name}: {exc}")

    return modules


def _build_params(args):
    """
    Build a SNIP parameter namespace.

    Architecture params (latent_dim, max_input_dimension, enc_emb_dim, …) are
    read from the checkpoint's saved ``params`` dict so the rebuilt environment
    and modules have identical shapes to the pretrained weights.  Training
    params (lr, batch_size, …) come from ``args``.
    """
    from parsers import get_parser

    params = get_parser().parse_args([])  # all-defaults baseline

    # ------------------------------------------------------------------
    # Override architecture params from the checkpoint so that the
    # environment and model shapes match the pretrained weights exactly.
    # ------------------------------------------------------------------
    ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "params" in ckpt_data:
        arch_keys = [
            "latent_dim",
            "enc_emb_dim", "dec_emb_dim",
            "n_enc_layers", "n_dec_layers",
            "n_enc_heads", "n_dec_heads",
            "n_enc_hidden_layers", "n_dec_hidden_layers",
            "emb_expansion_factor",
            "enc_positional_embeddings", "dec_positional_embeddings",
            "max_input_dimension", "min_input_dimension", "max_output_dimension",
            "float_precision", "mantissa_len", "max_exponent", "max_int",
            "embedder_type",
            "normalize_y",
        ]
        saved = ckpt_data["params"]
        for key in arch_keys:
            if key in saved:
                setattr(params, key, saved[key])
        logger.info(
            f"Loaded architecture params from checkpoint "
            f"(latent_dim={params.latent_dim}, "
            f"max_input_dimension={params.max_input_dimension})"
        )
    else:
        logger.warning(
            "Checkpoint contains no saved 'params' dict — using parser defaults. "
            "If you see size-mismatch warnings, set latent_dim / max_input_dimension manually."
        )

    # Task
    params.tasks = "functions"
    params.loss_type = "CLIP"
    params.is_proppred = False
    params.use_skeleton = False
    params.export_data = False

    # Paths / identifiers
    params.dump_path = args.output_dir
    params.exp_name = "finetune_snip"
    params.exp_id = "run0"
    params.reload_model = ""
    params.reload_data = ""
    params.reload_checkpoint = ""

    # Distributed / hardware (disabled for standalone finetuning)
    params.is_master = True
    params.local_rank = -1
    params.multi_gpu = False
    params.fp16 = False
    params.amp = -1
    params.nvidia_apex = False
    params.accumulate_gradients = 1

    # Optimisation
    params.lr = args.lr
    params.optimizer = f"adam,lr={args.lr}"
    params.clip_grad_norm = 0.5
    params.clip_temperature = args.clip_temperature

    # Logging / checkpointing
    params.print_freq = 100
    params.save_periodic = args.save_every
    params.n_steps_per_epoch = args.steps_per_epoch
    params.debug = False
    params.debug_train_statistics = False
    params.eval_only = False
    params.stopping_criterion = ""
    params.validation_metrics = ""

    params.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    return params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(args, symbol_library: SymbolLibrary) -> None:
    """Run the full finetuning pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Warn about unsupported symbols
    # ------------------------------------------------------------------
    nv = symbol_library.num_variables
    unsupported = [
        sym for sym in symbol_library.symbols
        if _get_snip_token(sym, nv) is None
    ]
    if unsupported and not args.allow_ood:
        logger.warning(
            f"The following symbols have no direct SNIP equivalent: {unsupported}. "
            f"Expressions containing them will be skipped. "
            f"Pass --allow_ood to map them to SNIP's OOD tokens instead."
        )
    elif unsupported and args.allow_ood:
        ood_unary  = [s for s in unsupported if symbol_library.get_type(s) == "fn"]
        ood_binary = [s for s in unsupported if symbol_library.get_type(s) == "op"]
        if ood_unary:
            logger.info(f"OOD unary  (→ OOD_unary_op):  {ood_unary}")
        if ood_binary:
            logger.info(f"OOD binary (→ OOD_binary_op): {ood_binary}")

    # ------------------------------------------------------------------
    # Build SNIP environment and model
    # ------------------------------------------------------------------
    params = _build_params(args)

    from snip.envs.environment import FunctionEnvironment
    env = FunctionEnvironment(params)

    modules = _load_modules(args.checkpoint, params, env)
    device = params.device
    for m in modules.values():
        m.to(device)

    embedder  = modules["embedder"]
    encoder_y = modules["encoder_y"]
    encoder_f = modules["encoder_f"]

    all_params = [p for m in modules.values() for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    # ------------------------------------------------------------------
    # Generate expressions and evaluate them to build the dataset
    # ------------------------------------------------------------------
    logger.info(f"Generating {args.n_expressions} expressions …")
    raw_expressions = generate_n_expressions(
        symbol_library,
        args.n_expressions,
        unique=True,
        max_expression_length=args.max_expr_length,
        verbose=True,
    )
    logger.info(f"Generated {len(raw_expressions)} raw expressions.")

    logger.info("Evaluating expressions to build training dataset …")
    dataset = build_dataset(
        raw_expressions,
        symbol_library,
        n_points=args.n_points,
        x_range=tuple(args.x_range),
        const_range=tuple(args.const_range),
        seed=args.seed,
        allow_ood=args.allow_ood,
        max_exponent=params.max_exponent,
        float_precision=params.float_precision,
    )
    logger.info(f"Valid training samples: {len(dataset)}")

    if len(dataset) < args.batch_size:
        raise RuntimeError(
            f"Only {len(dataset)} valid samples, which is fewer than the batch size "
            f"({args.batch_size}). Increase --n_expressions or broaden --x_range / "
            f"--const_range, or use a symbol library with only supported operators."
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    accum_steps = args.accumulate_gradients
    use_cuda = device.type == "cuda"

    # Collation kwargs forwarded every batch
    collate_kw = dict(
        env=env,
        device=device,
        rng=rng,
        n_points=args.n_points,
        x_range=tuple(args.x_range),
        const_range=tuple(args.const_range),
        num_variables=nv,
        max_exponent=params.max_exponent,
        float_precision=params.float_precision,
    )

    for epoch in range(args.n_epochs):
        for m in modules.values():
            m.train()

        indices = rng.permutation(len(dataset))
        epoch_losses: list[float] = []

        n_batches = (len(dataset) - args.batch_size) // args.batch_size + 1
        optimizer.zero_grad()

        for step in range(n_batches):
            batch_idx = indices[step * args.batch_size: (step + 1) * args.batch_size]
            if len(batch_idx) < 2:
                continue  # CLIP needs at least 2 samples
            batch = [dataset[i] for i in batch_idx]

            try:
                result = _collate_batch(batch, **collate_kw)
            except KeyError as exc:
                logger.error(str(exc))
                raise
            if result is None:
                continue  # all samples in this batch had no valid points
            x2, len2, xy_batch = result

            # Numeric encoder input
            x1, len1 = embedder(xy_batch)
            x1   = x1.to(device)
            len1 = len1.to(device)

            # Forward pass
            encoded_y = encoder_y("fwd", x=x1, lengths=len1, causal=False)
            encoded_f = encoder_f("fwd", x=x2, lengths=len2, causal=False)

            # CLIP contrastive loss (scaled for gradient accumulation)
            B = encoded_f.shape[0]
            logits_per_f = (encoded_f @ encoded_y.T) / args.clip_temperature
            logits_per_y = (encoded_y @ encoded_f.T) / args.clip_temperature
            labels = torch.arange(B, device=device)
            loss = (
                F.cross_entropy(logits_per_f, labels) +
                F.cross_entropy(logits_per_y, labels)
            ) / 2

            (loss / accum_steps).backward()

            epoch_losses.append(loss.item())

            # Optimiser step every accum_steps micro-batches
            if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
                clip_grad_norm_(all_params, 0.5)
                optimizer.step()
                optimizer.zero_grad()
                if use_cuda:
                    torch.cuda.empty_cache()

            if step % 50 == 0:
                logger.info(
                    f"  epoch {epoch + 1}/{args.n_epochs}  "
                    f"step {step + 1}/{n_batches}  "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        logger.info(f"Epoch {epoch + 1}/{args.n_epochs}  avg_loss={avg_loss:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({"params": vars(params), **{k: v.state_dict() for k, v in modules.items()}}, ckpt)
            logger.info(f"  Saved checkpoint: {ckpt}")

        # Best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(args.output_dir, "best_checkpoint.pth")
            torch.save({"params": vars(params), **{k: v.state_dict() for k, v in modules.items()}}, best_ckpt)

    logger.info(f"Finetuning complete. Best avg loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved in: {args.output_dir}")


# ---------------------------------------------------------------------------
# Inference encoder
# ---------------------------------------------------------------------------

class SNIPEncoder:
    """
    Wraps the symbolic encoder (encoder_f) from a (fine-tuned) SNIP checkpoint
    for embedding SRToolkit expressions into a latent vector space.

    Usage::

        encoder = load_snip_encoder(
            "path/to/best_checkpoint.pth",
            symbol_library,
        )
        embeddings = encoder.encode(list_of_token_lists)
        dist = torch.cdist(embeddings, embeddings, p=2).numpy()
    """

    def __init__(self, encoder_f, env, num_variables: int, device: torch.device):
        self.encoder_f    = encoder_f
        self.env          = env
        self.num_variables = num_variables
        self.device       = device

    @torch.no_grad()
    def encode(
        self,
        expressions: list[list[str]],
        symbol_library: SymbolLibrary,
        allow_ood: bool = False,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Encode a list of SRToolkit token-list expressions.

        Args:
            expressions:    List of token lists (infix SRToolkit notation).
            symbol_library: The SymbolLibrary used to parse the expressions.
            allow_ood:      Map unsupported symbols to OOD tokens instead of
                            raising (default False).
            batch_size:     Number of expressions per forward pass.

        Returns:
            Tensor of shape (N, latent_dim) on CPU.
        """
        self.encoder_f.eval()

        # Convert every expression to SNIP prefix tokens
        prefix_seqs: list[list[str]] = []
        for expr_tokens in expressions:
            try:
                tree   = tokens_to_tree(expr_tokens, symbol_library)
                prefix = _tree_to_snip_prefix(
                    tree, self.num_variables, sl=symbol_library, allow_ood=allow_ood
                )
                if not prefix:
                    raise ValueError("empty prefix")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not convert expression {''.join(expr_tokens)!r} to SNIP prefix: {exc}"
                ) from exc
            prefix_seqs.append(prefix)

        # Encode in mini-batches
        all_embeddings: list[torch.Tensor] = []
        for i in range(0, len(prefix_seqs), batch_size):
            batch = prefix_seqs[i : i + batch_size]
            indexed    = self.env.word_to_idx(batch, float_input=False)
            x2, len2   = self.env.batch_equations(indexed)
            x2         = x2.to(self.device)
            len2       = len2.to(self.device)
            encoded    = self.encoder_f("fwd", x=x2, lengths=len2, causal=False)
            all_embeddings.append(encoded.cpu())

        return torch.cat(all_embeddings, dim=0)


def load_snip_encoder(
    checkpoint: str,
    symbol_library: SymbolLibrary,
    cpu: bool = False,
    base_checkpoint: str | None = None,
) -> SNIPEncoder:
    """
    Load the symbolic encoder from a SNIP checkpoint and return a
    :class:`SNIPEncoder` ready for inference.

    Args:
        checkpoint:      Path to a ``.pth`` checkpoint (pretrained or finetuned).
                         Checkpoints produced by :func:`finetune` include saved
                         architecture params and are fully self-contained.
        symbol_library:  The :class:`SymbolLibrary` whose expressions you will
                         encode.  Used only to determine the number of variables
                         for the SNIP token mapping.
        cpu:             Force CPU inference even when CUDA is available.
        base_checkpoint: Optional path to the *original* pretrained SNIP checkpoint
                         whose ``params`` dict will be used for architecture config.
                         Needed only for finetuned checkpoints that were saved
                         without embedded params (i.e. trained before this fix).

    Returns:
        A :class:`SNIPEncoder` instance.
    """
    # Minimal args namespace — only the fields _build_params actually reads.
    # If base_checkpoint is given, read arch params from there instead of
    # from the (possibly param-less) finetuned checkpoint.
    params_source = base_checkpoint if base_checkpoint is not None else checkpoint
    _args = argparse.Namespace(
        checkpoint       = params_source,
        output_dir       = "",
        lr               = 1e-5,
        clip_temperature = 1.0,
        save_every       = 1,
        steps_per_epoch  = 1,
        cpu              = cpu,
        accumulate_gradients = 1,
    )
    params = _build_params(_args)

    from snip.envs.environment import FunctionEnvironment
    env = FunctionEnvironment(params)

    modules = _load_modules(checkpoint, params, env)
    encoder_f = modules["encoder_f"]
    encoder_f.to(params.device)
    encoder_f.eval()

    return SNIPEncoder(
        encoder_f    = encoder_f,
        env          = env,
        num_variables = symbol_library.num_variables,
        device       = params.device,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune SNIP on a user-specified SRToolkit SymbolLibrary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a pretrained SNIP .pth checkpoint "
             "(e.g. Multimodal-Math-Pretraining-main/weights/snip-1d-normalized.pth).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where finetuned checkpoints will be written.",
    )

    # Symbol library
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated list of SRToolkit operator/function names "
             "(e.g. '+,-,*,/,sin,cos,C'). "
             "Omit to use the full default SymbolLibrary.",
    )
    parser.add_argument(
        "--num_variables", type=int, default=1,
        help="Number of input variables (X_0 … X_{n-1}). "
             "Must not exceed the pretrained checkpoint's max_input_dimension "
             "(1 for snip-1d-normalized, 10 for snip-10dmax).",
    )

    # Data generation
    parser.add_argument(
        "--n_expressions", type=int, default=50000,
        help="Number of expressions to generate for finetuning.",
    )
    parser.add_argument(
        "--n_points", type=int, default=50,
        help="Number of (x, y) evaluation points per expression.",
    )
    parser.add_argument(
        "--max_expr_length", type=int, default=30,
        help="Maximum token length of generated expressions.",
    )
    parser.add_argument(
        "--x_range", type=float, nargs=2, default=[-5.0, 5.0],
        metavar=("X_MIN", "X_MAX"),
        help="Uniform sampling range for input variable values.",
    )
    parser.add_argument(
        "--const_range", type=float, nargs=2, default=[0.1, 5.0],
        metavar=("C_MIN", "C_MAX"),
        help="Uniform sampling range for unknown constant (C) values.",
    )

    # Training
    parser.add_argument("--n_epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",        type=int,   default=64)
    parser.add_argument("--lr",                type=float, default=1e-5,
                        help="Adam learning rate.")
    parser.add_argument("--clip_temperature",  type=float, default=1.0,
                        help="Temperature for the CLIP contrastive loss.")
    parser.add_argument("--steps_per_epoch",   type=int,   default=1000,
                        help="Used for SNIP internal epoch tracking only.")
    parser.add_argument("--save_every",        type=int,   default=10,
                        help="Save a periodic checkpoint every N epochs.")
    parser.add_argument(
        "--accumulate_gradients", type=int, default=1,
        help="Accumulate gradients over this many micro-batches before each "
             "optimiser step (simulates a larger effective batch size without "
             "extra GPU memory). E.g. batch_size=16 + accumulate_gradients=4 "
             "≈ effective batch size 64.",
    )
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--cpu",               action="store_true",
                        help="Force CPU even when CUDA is available.")
    parser.add_argument(
        "--allow_ood", action="store_true",
        help="Map unsupported unary/binary symbols to SNIP's OOD tokens "
             "(OOD_unary_op / OOD_binary_op) instead of skipping those expressions. "
             "Useful for symbols like sinh, tanh, floor, ceil, log10, ^4, ^5, u-.",
    )

    return parser.parse_args()


def finetune_for_equivalence_classes(
    checkpoint: str,
    output_dir: str,
    n_expressions: int = 50000,
    n_epochs: int = 1,
    batch_size: int = 16,
    accumulate_gradients: int = 4,
    lr: float = 1e-5,
    seed: int = 42,
    cpu: bool = False,
) -> None:
    """
    Convenience wrapper that finetunes SNIP using the same symbol library and
    sampling setup as ``equivalence_classes.py``.

    The symbol library mirrors the one built in ``train_hvae_model()``:
    SymbolLibrary.default_symbols(2) plus the numeric literals 0, 0.5, 1, 2,
    -1, 10 that the equivalence transformations introduce as expression leaves.

    The x sampling range [1, 5] matches the range used when evaluating
    expressions in ``equivalence_classes.py``.

    Args:
        checkpoint:           Path to the pretrained SNIP checkpoint.
                              Use snip-10dmax.pth (trained with up to 10 variables).
        output_dir:           Directory where finetuned checkpoints are written.
        n_expressions:        Number of training expressions to generate (default 50000).
        n_epochs:             Number of finetuning epochs (default 50).
        batch_size:           Micro-batch size per step (default 16, GPU-memory friendly).
        accumulate_gradients: Accumulate this many micro-batches per optimiser step
                              (default 4, so effective batch size = 16×4 = 64).
        lr:                   Adam learning rate (default 1e-5).
        seed:                 Random seed (default 42).
        cpu:                  Force CPU even when CUDA is available (default False).
    """
    import argparse

    sl = SymbolLibrary.default_symbols(num_variables=2)
    sl.add_symbol("0",   "lit", 5, "np.full(X.shape[0], 0)",   "0")
    sl.add_symbol("0.5", "lit", 5, "np.full(X.shape[0], 0.5)", "0.5")
    sl.add_symbol("1",   "lit", 5, "np.full(X.shape[0], 1)",   "1")
    sl.add_symbol("2",   "lit", 5, "np.full(X.shape[0], 2)",   "2")
    sl.add_symbol("-1",  "lit", 5, "np.full(X.shape[0], -1)",  "-1")
    sl.add_symbol("10",  "lit", 5, "np.full(X.shape[0], 10)",  "10")

    args = argparse.Namespace(
        checkpoint           = checkpoint,
        output_dir           = output_dir,
        num_variables        = 2,
        n_expressions        = n_expressions,
        n_points             = 20,
        max_expr_length      = 30,
        x_range              = [1.0, 5.0],  # matches equivalence_classes.py evaluation range
        const_range          = [0.1, 5.0],
        n_epochs             = n_epochs,
        batch_size           = batch_size,
        accumulate_gradients = accumulate_gradients,
        lr                   = lr,
        clip_temperature     = 1.0,
        steps_per_epoch      = 1000,
        save_every           = 10,
        seed                 = seed,
        cpu                  = cpu,
        allow_ood            = True,
    )

    finetune(args, sl)


def finetune_for_smoothness(
    checkpoint: str,
    output_dir: str,
    num_variables: int = 9,
    n_expressions: int = 500,
    n_epochs: int = 1,
    batch_size: int = 16,
    accumulate_gradients: int = 4,
    lr: float = 1e-5,
    seed: int = 42,
    cpu: bool = False,
) -> None:
    """
    Convenience wrapper that finetunes SNIP using the symbol library and
    variable count that match the experiments in ``smoothness_to_true.py``.

    Symbol set (mirrors the grammar in ``smoothness_to_true.py``):
        +  -  *  /  u-  sqrt  sin  cos  exp  arcsin  tanh  ln  ^2  ^3  pi  C
        X_0 … X_{num_variables-1}

    ``u-`` (unary minus) and ``tanh`` have no direct SNIP equivalent and are
    mapped to SNIP's ``OOD_unary_op`` token (``allow_ood=True``).

    x is sampled from [0.5, 5.0] — a positive range that keeps sqrt, ln, and
    arcsin (after composition) mostly finite.  Adjust ``x_range`` via the
    returned args if you need a different range.

    Args:
        checkpoint:           Path to the pretrained SNIP checkpoint.
                              Use snip-10dmax.pth (supports up to 10 variables).
        output_dir:           Directory where finetuned checkpoints are written.
        num_variables:        Number of input variables (default 9, the maximum
                              used across the Feynman datasets).  Must be ≤ the
                              checkpoint's max_input_dimension (10 for snip-10dmax).
        n_expressions:        Number of training expressions to generate (default 50000).
        n_epochs:             Number of finetuning epochs (default 50).
        batch_size:           Micro-batch size per step (default 16).
        accumulate_gradients: Micro-batches per optimiser step (default 4,
                              giving effective batch size 64).
        lr:                   Adam learning rate (default 1e-5).
        seed:                 Random seed (default 42).
        cpu:                  Force CPU even when CUDA is available (default False).
    """
    _SMOOTHNESS_SYMBOLS = [
        "+", "-", "*", "/", "u-",
        "sqrt", "sin", "cos", "exp", "arcsin", "tanh",
        "ln", "^2", "^3",
        "pi", "C",
    ]
    sl = SymbolLibrary.from_symbol_list(_SMOOTHNESS_SYMBOLS, num_variables=num_variables)

    args = argparse.Namespace(
        checkpoint           = checkpoint,
        output_dir           = output_dir,
        num_variables        = num_variables,
        n_expressions        = n_expressions,
        n_points             = 20,
        max_expr_length      = 40,     # matches smoothness_to_true.py max_expression_length
        x_range              = [0.5, 5.0],  # positive — avoids domain errors for sqrt/ln
        const_range          = [0.1, 5.0],
        n_epochs             = n_epochs,
        batch_size           = batch_size,
        accumulate_gradients = accumulate_gradients,
        lr                   = lr,
        clip_temperature     = 1.0,
        steps_per_epoch      = 1000,
        save_every           = 10,
        seed                 = seed,
        cpu                  = cpu,
        allow_ood            = True,   # u- and tanh → OOD_unary_op
    )

    finetune(args, sl)


def main() -> None:
    args = parse_args()

    if args.symbols is not None:
        symbol_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
        symbol_library = SymbolLibrary.from_symbol_list(
            symbol_list, num_variables=args.num_variables
        )
    else:
        symbol_library = SymbolLibrary.default_symbols(num_variables=args.num_variables)

    logger.info(f"Symbol library ({len(symbol_library)} symbols): {symbol_library}")

    finetune(args, symbol_library)


if __name__ == "__main__":
    # main()
    # finetune_for_equivalence_classes("../Multimodal-Math-Pretraining-main/weights/snip-10dmax.pth",
    #                                  "../Multimodal-Math-Pretraining-main/weights/snip-10dmax-finetuned-equivalence-classes")
    finetune_for_smoothness("../Multimodal-Math-Pretraining-main/weights/snip-10dmax.pth",
                                     "../Multimodal-Math-Pretraining-main/weights/snip-10dmax-finetuned-smoothness")