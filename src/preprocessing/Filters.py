import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import SEED
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

from .constants import TARGET_BIN, TARGET_15C, VALID_TIME_COL

# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------

class ValidTimeFilter(BaseEstimator, TransformerMixin):
    """
    Drop rows where `col` == 0.
    Expects and returns a DataFrame (X contains all columns including targets).
    """

    def __init__(self, col: str = VALID_TIME_COL):
        self.col = col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.col not in X.columns:
            print(f"  WARNING: column '{self.col}' not found – skipping filter.")
            return X
        n_before = len(X)
        X = X[X[self.col] != 0].reset_index(drop=True)
        print(f"  [{self.__class__.__name__}] Dropped {n_before - len(X):,} rows  →  {X.shape}")
        return X


class CategoricalDropper(BaseEstimator, TransformerMixin):
    """
    Drop all columns with dtype object or category, except the target columns.
    """

    def __init__(self, keep_cols: list = None):
        self.keep_cols = keep_cols or [TARGET_BIN, TARGET_15C]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        cat_cols = [
            c for c in X.select_dtypes(include=["object", "category", "str"]).columns
            if c not in self.keep_cols
        ]
        X = X.drop(columns=cat_cols)
        print(
            f"  [{self.__class__.__name__}] "
            f"Dropped {len(cat_cols)} categorical columns: {cat_cols}  →  {X.shape}"
        )
        return X
    

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalStatisticsExtractor(BaseEstimator, TransformerMixin):
    """
    Featurise text columns with generic character-level statistics and attack-token counts.
    """
    # ------------------------------------------------------------------
    # "Missing" detection
    # ------------------------------------------------------------------
    # Values that mean "this field does not apply to this packet". Compared
    # case-insensitively after stripping whitespace.
    _PLACEHOLDER_VALUES = frozenset({
        "", "0", "0.0", "0.0.0.0",
        "nan", "none", "null", "na", "<na>",
    })

    # ------------------------------------------------------------------
    # Attack-token taxonomies (lowercase; matched as substrings)
    # ------------------------------------------------------------------
    # SQLi — covers raw and URL-encoded variants and DBMS-specific functions
    # observed in this dataset's payloads.
    _SQLI_TOKENS = (
        # Classic punctuation / keywords
        "'", '"', "--", ";", "/*", "*/", "=",
        "union", "select", "insert", "drop", "update", "delete", "into",
        "or ", " and ", " where ", "having ", "group by", "order by",
        # Bypass / fingerprinting helpers
        "0x", "char(", "chr(", "concat(", "cast(", "convert(",
        "benchmark(", "sleep(", "waitfor", "xp_", "information_schema",
        # DBMS-specific exploits seen in this dataset
        "extractvalue", "updatexml", "pg_sleep", "dbms_pipe", "elt(", "if(",
        # URL-encoded equivalents (very frequent in http.request.uri.query)
        "%27", "%22", "%20and%20", "%20or%20", "%3d", "%3b", "%29", "%28",
        "%2c", "%7c", "%23",
    )

    # XSS — script/event-handler injection
    _XSS_TOKENS = (
        "<", ">", "<script", "</script", "script>", "onerror", "onload",
        "onmouseover", "onfocus", "onclick", "javascript:", "vbscript:",
        "alert(", "prompt(", "confirm(", "expression(",
        "eval(", "fromcharcode", "document.cookie",
        "src=", "iframe", "<svg", "<img",
        "%3cscript", "%3c", "%3e", "%22",
    )

    # Path traversal / file-system probes
    _PATH_TOKENS = (
        "../", "..\\", "%2e%2e", "%252e",
        "/etc/", "/etc/passwd", "/etc/shadow", "/proc/", "/var/",
        "boot.ini", "win.ini", "system32",
    )

    # Command injection / RCE — includes Shellshock signatures
    _CMD_TOKENS = (
        # Generic shell metacharacters
        "&&", "||", "$(", "${", "`",
        # Common payload binaries / utilities
        "/bin/sh", "/bin/bash", "cmd.exe", "powershell",
        "nc ", "wget ", "curl ", "chmod ", "chown ", "id;", "uname",
        "system(", "passthru(", "shell_exec", "popen(", "exec(",
        # Shellshock (CVE-2014-6278) — appears verbatim in http.referer
        "() {", "_;}", "_; }", ">_[$(", "cve-2014",
    )

    # ------------------------------------------------------------------
    # Structural regexes
    # ------------------------------------------------------------------
    # IPv4 dotted-quad
    _IP_PAT = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    # Private RFC1918 + loopback
    _PRIVATE_IP_PAT = re.compile(
        r"^(?:10\.|127\.|192\.168\.|172\.(?:1[6-9]|2\d|3[0-1])\.)"
    )
    # Multicast (224.0.0.0/4)
    _MULTICAST_PAT = re.compile(r"^(?:22[4-9]|23\d)\.")
    # Hex-blob: ≥8 hex chars, even length (typical of TCP options / MQTT msg)
    _HEX_PAT = re.compile(r"^[0-9a-fA-F]{8,}$")
    # URL-encoded triple %XX
    _URL_ENC_PAT = re.compile(r"%[0-9a-fA-F]{2}")

    def __init__(self, text_cols: list = None, keep_cols: list = None,
                 drop_original: bool = True,
                 extra_placeholders=None,
                 skip_low_cardinality: int = 0):
        self.text_cols = text_cols
        self.keep_cols = keep_cols or [TARGET_BIN, TARGET_15C]
        self.drop_original = drop_original
        self.extra_placeholders = extra_placeholders
        self.skip_low_cardinality = skip_low_cardinality

    def fit(self, X: pd.DataFrame, y=None):
        # Resolve which columns to featurise (stateless, but recorded for transform).
        if self.text_cols is not None:
            self.text_cols_ = [c for c in self.text_cols if c in X.columns]
        else:
            self.text_cols_ = [
                c for c in X.select_dtypes(include=["object", "category", "str"]).columns
                if c not in self.keep_cols
            ]

        # Build the placeholder set once.
        ph = set(self._PLACEHOLDER_VALUES)
        if self.extra_placeholders:
            ph.update(str(p).strip().lower() for p in self.extra_placeholders)
        self._placeholders_ = frozenset(ph)
        return self

    @staticmethod
    def _char_entropy(s: str) -> float:
        """Shannon entropy over the character distribution of a string."""
        if not s:
            return 0.0
        counts = np.array(list(__import__("collections").Counter(s).values()), dtype=float)
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum())

    def _is_placeholder_mask(self, s: pd.Series) -> pd.Series:
        """Return a boolean mask of rows whose value is a 'missing' placeholder."""
        norm = s.fillna("").astype(str).str.strip().str.lower()
        return norm.isin(self._placeholders_)

    def _features_for_series(self, col: pd.Series) -> pd.DataFrame:
        raw = col.fillna("").astype(str)

        # 1 Missingness flag
        is_missing = self._is_placeholder_mask(col).astype(float)
        s = raw.where(is_missing == 0, "")
        lower = s.str.lower()
        length = s.str.len().astype(float)
        safe_len = length.replace(0, np.nan)

        # 2 Skip the expensive stats block on truly low-cardinality columns.
        if self.skip_low_cardinality:
            n_unique = s[s != ""].nunique()
            if n_unique < self.skip_low_cardinality:
                out = pd.DataFrame(
                    {"is_missing": is_missing},
                    index=col.index,
                )
                out.columns = [f"{col.name}__{c}" for c in out.columns]
                return out

        # 3 Character class counters
        n_special = s.str.count(r"[^A-Za-z0-9]").astype(float)
        n_digits = s.str.count(r"[0-9]").astype(float)
        n_alpha = s.str.count(r"[A-Za-z]").astype(float)
        n_pct = s.str.count(r"%").astype(float)
        n_spaces = s.str.count(r"\s").astype(float)
        n_hex_chars = s.str.count(r"[0-9a-fA-F]").astype(float)
        n_non_ascii = s.str.count(r"[^\x00-\x7f]").astype(float)

        # 4 Structural counters useful on URIs / payloads
        n_dot = s.str.count(r"\.").astype(float)
        n_slash = s.str.count(r"/").astype(float)
        n_eq = s.str.count(r"=").astype(float)
        n_amp = s.str.count(r"&").astype(float)
        n_semi = s.str.count(r";").astype(float)
        n_urlenc = s.str.count(self._URL_ENC_PAT).astype(float)

        # 5 Structural flags (IP / hex blob)
        is_ip = s.str.match(self._IP_PAT).fillna(False).astype(float)
        is_priv_ip = (
            s.str.match(self._PRIVATE_IP_PAT).fillna(False).astype(float) * is_ip
        )
        is_mcast_ip = (
            s.str.match(self._MULTICAST_PAT).fillna(False).astype(float) * is_ip
        )
        is_hex = s.str.match(self._HEX_PAT).fillna(False).astype(float)

        # 6 Numeric coercion
        as_float = pd.to_numeric(s, errors="coerce").astype(float)

        feats = {
            # Missingness
            "is_missing":       is_missing,
            "is_empty":         (length == 0).astype(float),
            # Basic shape
            "len":              length,
            "entropy":          s.map(self._char_entropy).astype(float),
            "special_ratio":    (n_special / safe_len).fillna(0.0),
            "digit_ratio":      (n_digits / safe_len).fillna(0.0),
            "alpha_ratio":      (n_alpha / safe_len).fillna(0.0),
            "pct_encode_ratio": (n_pct / safe_len).fillna(0.0),
            "space_ratio":      (n_spaces / safe_len).fillna(0.0),
            "hex_ratio":        (n_hex_chars / safe_len).fillna(0.0),
            "non_ascii_ratio":  (n_non_ascii / safe_len).fillna(0.0),
            "url_encoded_count": n_urlenc,
            # URI structure
            "n_dots":           n_dot,
            "n_slashes":        n_slash,
            "n_equals":         n_eq,
            "n_ampersands":     n_amp,
            "n_semicolons":     n_semi,
            # Structural flags
            "is_ip":            is_ip,
            "is_private_ip":    is_priv_ip,
            "is_multicast_ip":  is_mcast_ip,
            "is_hex_blob":      is_hex,
            # Numeric escape hatch
            "as_float":         as_float.fillna(0.0),
            "as_float_isnan":   as_float.isna().astype(float),
            # Attack-token counts
            "sqli_hits":  sum(lower.str.count(re.escape(t)) for t in self._SQLI_TOKENS).astype(float),
            "xss_hits":   sum(lower.str.count(re.escape(t)) for t in self._XSS_TOKENS).astype(float),
            "path_hits":  sum(lower.str.count(re.escape(t)) for t in self._PATH_TOKENS).astype(float),
            "cmd_hits":   sum(lower.str.count(re.escape(t)) for t in self._CMD_TOKENS).astype(float),
        }
        out = pd.DataFrame(feats, index=col.index)
        out.columns = [f"{col.name}__{c}" for c in out.columns]
        return out

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        blocks = [self._features_for_series(X[c]) for c in self.text_cols_]
        if self.drop_original:
            X = X.drop(columns=self.text_cols_)
        if blocks:
            X = pd.concat([X, *blocks], axis=1)
        n_feats = sum(b.shape[1] for b in blocks)
        print(
            f"  [{self.__class__.__name__}] "
            f"Featurised {len(self.text_cols_)} text columns → {n_feats} numeric "
            f"features {self.text_cols_}  →  {X.shape}"
        )
        return X


class CategoricalEmbedder(BaseEstimator, TransformerMixin):
    """
    Semantic embeddings for text columns via a frozen pretrained model.
    """

    def __init__(self, text_cols: list = None, keep_cols: list = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 n_components: int = 16, normal_label=0,
                 batch_size: int = 256, drop_original: bool = True):
        self.text_cols = text_cols
        self.keep_cols = keep_cols or [TARGET_BIN, TARGET_15C]
        self.model_name = model_name
        self.n_components = n_components
        self.normal_label = normal_label
        self.batch_size = batch_size
        self.drop_original = drop_original

    def _get_encoder(self):
        # Lazy import / load so the module doesn't hard-depend on sentence-transformers.
        if getattr(self, "_encoder", None) is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def _encode(self, col: pd.Series) -> np.ndarray:
        texts = col.fillna("").astype(str).tolist()
        return self._get_encoder().encode(
            texts, batch_size=self.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )

    def fit(self, X: pd.DataFrame, y=None):

        if self.text_cols is not None:
            self.text_cols_ = [c for c in self.text_cols if c in X.columns]
        else:
            self.text_cols_ = [
                c for c in X.select_dtypes(include=["object", "category", "str"]).columns
                if c not in self.keep_cols
            ]

        # Fit SVD on NORMAL rows only (leak-safe one-class reduction).
        if TARGET_BIN in X.columns:
            normal_mask = (X[TARGET_BIN] == self.normal_label).to_numpy()
        else:
            normal_mask = np.ones(len(X), dtype=bool)

        self.svd_ = {}
        self.embed_dim_ = {}
        for c in self.text_cols_:
            emb = self._encode(X[c])
            self.embed_dim_[c] = emb.shape[1]
            if self.n_components and self.n_components < emb.shape[1]:
                svd = TruncatedSVD(n_components=self.n_components, random_state=SEED)
                svd.fit(emb[normal_mask])
                self.svd_[c] = svd
            else:
                self.svd_[c] = None
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        blocks = []
        for c in self.text_cols_:
            emb = self._encode(X[c])
            svd = self.svd_.get(c)
            if svd is not None:
                emb = svd.transform(emb)
            cols = [f"{c}__emb{i}" for i in range(emb.shape[1])]
            blocks.append(pd.DataFrame(emb, index=X.index, columns=cols))

        if self.drop_original:
            X = X.drop(columns=self.text_cols_)
        if blocks:
            X = pd.concat([X, *blocks], axis=1)
        n_feats = sum(b.shape[1] for b in blocks)
        print(
            f"  [{self.__class__.__name__}] "
            f"Embedded {len(self.text_cols_)} text columns → {n_feats} numeric "
            f"features (model={self.model_name})  →  {X.shape}"
        )
        return X