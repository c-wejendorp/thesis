from pathlib import Path
from typing import Optional
from torchaudio.datasets import SPEECHCOMMANDS
import torch
from torch.nn.functional import pad

import random
from collections import defaultdict, Counter
from torch import Tensor

from .noise_functions import load_random_noise_chunk, add_noise_at_snr
from .config import NOISE_FOLDER, SAMPLE_RATE, TARGET_LENGTH, CANONICAL

class PadOrTrim():
    def __init__(self, max_len: int = TARGET_LENGTH, pad_value: float = 0.0,):
        self.max_len = max_len
        self.pad_value = pad_value
    
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return pad(w[:, :self.max_len], (0, max(0, self.max_len - w.shape[1])), value=self.pad_value)


class SpeechCommandsGoogle(SPEECHCOMMANDS):
    """
    Extended Speech Commands dataset.

    Options:
        - keyword_list: list of target keywords (default: CANONICAL).
        - use_unknown: add an 'unknown' class from non-keyword commands.
        - use_silence: add a 'silence' class from background noise chunks.
        - upsample:
            * False:
                - keep ALL keyword samples
                - make unknown and silence have as many samples
                  as the LARGEST keyword class.
            * True:
                - upsample EVERY active class (keywords, unknown, silence)
                  so they all have the same size as the LARGEST keyword class.

        - transform: applied to waveform (e.g. PadOrTrim()).
        - snr: float or (low, high) to add white noise at given SNR (dB).
    """

    def __init__(
        self,
        *args,
        subset: Optional[str] = None,
        keyword_list: list[str] | None = None,
        use_unknown: bool = True,
        use_silence: bool = True,
        upsample: bool = False,
        transform=None,
        snr: float | tuple[float, float] | None = None,  # in dB
        **kwargs,
    ):
        super().__init__(*args, subset=subset, **kwargs)

        self.subset = subset
        self.keywords = keyword_list if keyword_list is not None else CANONICAL
        self.keywords_set = set(self.keywords)

        self.use_unknown = use_unknown
        self.use_silence = use_silence
        self.upsample = upsample

        # Noise files for silence samples (clips without speech)
        self.noise_paths: list[str] = []
        if self.use_silence:
            self.noise_paths = sorted(
                str(p) for p in Path(self._path, NOISE_FOLDER).glob("*.wav")
            )

        self.transform = transform if transform is not None else PadOrTrim()
        self._set_snr(snr)

        # If nothing special is requested, fall back to "all labels" mode
        if (
            keyword_list is None
            and not use_unknown
            and not use_silence
            and not upsample
        ):
            self._use_custom_view = False
            self._init_all_labels()
        else:
            self._use_custom_view = True
            self._init_custom_labels()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _set_snr(self, snr: float | tuple[float, float] | None) -> None:
        """Validate and store SNR configuration."""
        if isinstance(snr, tuple):
            if len(snr) != 2:
                raise ValueError("snr must be a float or a (low, high) tuple.")
            low, high = snr
            if low > high:
                raise ValueError("snr range must be ordered (low <= high).")
            self.snr = (float(low), float(high))
        elif isinstance(snr, (int, float)) or snr is None:
            self.snr = float(snr) if snr is not None else None
        else:
            raise TypeError("snr must be float, (float, float), or None.")

    def _init_all_labels(self) -> None:
        """Plain parent behaviour, with label maps and transforms."""
        labels = sorted({Path(p).parent.name for p in self._walker})
        self.labels = labels
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        self.label_counts = Counter(Path(p).parent.name for p in self._walker)
        self.samples: list[dict] = []

    def _init_custom_labels(self) -> None:
        """
        Build a custom view with:
          - selected keywords (self.keywords)
          - optional 'unknown' from non-keyword commands
          - optional 'silence' from noise files

        upsample=False:
            - keep ALL keyword samples
            - size unknown and silence to match the LARGEST keyword class.

        upsample=True:
            - for each active class (keywords, unknown, silence),
              sample/oversample so they ALL have size = LARGEST keyword class.
        """
        rng = random.Random(0)  # deterministic selection

        # Map walker indices into keywords / unknown
        per_keyword: dict[str, list[int]] = defaultdict(list)
        unknown_idxs: list[int] = []

        for idx, path in enumerate(self._walker):
            label = Path(path).parent.name
            if label in self.keywords_set:
                per_keyword[label].append(idx)
            else:
                unknown_idxs.append(idx)

        # Sanity: all requested keywords must exist
        missing = [kw for kw in self.keywords_set if kw not in per_keyword]
        if missing:
            raise RuntimeError(f"Requested keywords not found in dataset: {missing}")

        kw_counts = [len(v) for v in per_keyword.values()]
        max_kw_count = max(kw_counts)

        samples: list[dict] = []

        # ---------- 1) Keyword samples ----------
        if self.upsample:
            # Upsample each keyword class to max_kw_count
            for kw in sorted(self.keywords_set):
                idxs = per_keyword[kw]
                if len(idxs) >= max_kw_count:
                    chosen = rng.sample(idxs, max_kw_count)
                else:
                    chosen = rng.choices(idxs, k=max_kw_count)
                for idx in chosen:
                    samples.append(
                        {
                            "kind": "keyword",
                            "label": kw,
                            "walker_idx": idx,
                        }
                    )
        else:
            # No upsampling: keep all samples for each keyword
            for kw in sorted(self.keywords_set):
                idxs = per_keyword[kw]
                for idx in idxs:
                    samples.append(
                        {
                            "kind": "keyword",
                            "label": kw,
                            "walker_idx": idx,
                        }
                    )

        # ---------- 2) Unknown samples (optional) ----------
        if self.use_unknown:
            if len(unknown_idxs) == 0:
                raise RuntimeError(
                    "No non-keyword commands available to form the 'unknown' class."
                )

            # Unknown size is always max_kw_count, regardless of upsample flag:
            n_unknown = max_kw_count

            if len(unknown_idxs) >= n_unknown:
                unknown_chosen = rng.sample(unknown_idxs, n_unknown)
            else:
                unknown_chosen = rng.choices(unknown_idxs, k=n_unknown)

            for idx in unknown_chosen:
                samples.append(
                    {
                        "kind": "unknown",
                        "label": "UNKNOWN",
                        "walker_idx": idx,
                    }
                )

        # ---------- 3) Silence / noise samples (optional) ----------
        if self.use_silence:
            if not self.noise_paths:
                raise RuntimeError(
                    "use_silence=True but no noise files were found "
                    f"in folder: {NOISE_FOLDER}"
                )

            n_silence = max_kw_count

            for _ in range(n_silence):
                samples.append(
                    {
                        "kind": "silence",
                        "label": "SILENCE",
                        "walker_idx": None,  # handled specially
                    }
                )

        self.samples = samples

        # ---------- Label mapping ----------
        labels = list(sorted(self.keywords_set))
        if self.use_unknown:
            labels.append("UNKNOWN")
        if self.use_silence:
            labels.append("SILENCE")

        self.labels = labels
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

        self.label_counts = Counter(s["label"] for s in self.samples)

    # ------------------------------------------------------------------
    # Core Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._use_custom_view:
            return len(self.samples)
        else:
            return len(self._walker)

    def __getitem__(self, n: int) -> tuple[Tensor, int, dict]:
        if self._use_custom_view:
            waveform, label_idx, meta_data = self._getitem_custom(n)
        else:
            waveform, label_idx, meta_data = self._getitem_all(n)

        # Add noise (Nothing happens if self.snr is None)
        snr_value = self._sample_snr_value()
        if snr_value is not None:
            waveform = self._add_noise(waveform, snr_value)
        meta_data["snr_value"] = snr_value
        return waveform, label_idx, meta_data
    
    # ------------------------------------------------------------------
    # __getitem__ helpers
    # ------------------------------------------------------------------

    def _getitem_all(self, n: int) -> tuple[Tensor, int, dict]:
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(n)
        waveform = self._apply_transform(waveform)

        meta_data = {
            "sample_rate": sample_rate,
            "label": label,
            "speaker_id": speaker_id,
            "utterance": label,
            "utterance_number": utterance_number,
            "raw_signal": waveform.clone(),
            "snr_value": None,
        }
        label_idx = self.label_to_idx[label]
        return waveform, label_idx, meta_data

    def _getitem_custom(self, n: int) -> tuple[Tensor, int, dict]:
        sample_info = self.samples[n]
        kind = sample_info["kind"]
        label_str = sample_info["label"]

        if kind in ("keyword", "unknown"):
            walker_idx = sample_info["walker_idx"]
            waveform, sample_rate, utterance, speaker_id, utterance_number = super().__getitem__(walker_idx)
        elif kind == "silence":
            waveform = self._load_random_noise_chunk()
            sample_rate = SAMPLE_RATE
            utterance = None
            speaker_id = None
            utterance_number = None
        else:
            raise RuntimeError(f"Unknown sample kind: {kind}")

        waveform = self._apply_transform(waveform)

        meta_data = {
            "sample_rate": sample_rate,
            "label": label_str,
            "utterance": utterance,
            "utterance_number": utterance_number,
            "speaker_id": speaker_id,
            "raw_signal": waveform.clone(),
            "snr_value": None,
        }

        label_idx = self.label_to_idx[label_str]
        return waveform, label_idx, meta_data

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _apply_transform(self, waveform: Tensor) -> Tensor:
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform
    
    def _sample_snr_value(self) -> Optional[float]:
        if self.snr is None:
            return None
        if isinstance(self.snr, (int, float)):
            return float(self.snr)
        low, high = self.snr
        return float(torch.empty(1).uniform_(low, high).item())

    def _load_random_noise_chunk(self):
        return load_random_noise_chunk(
            noise_paths=self.noise_paths,
            target_length=TARGET_LENGTH,
            sample_rate=SAMPLE_RATE,
        )
    
    def _add_noise(self, waveform: Tensor, snr_value: float) -> Tensor:
        return add_noise_at_snr(waveform, snr_value )
    
