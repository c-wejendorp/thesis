from pathlib import Path
from typing import Optional
from torchaudio.datasets import SPEECHCOMMANDS
import torch
from torch.nn.functional import pad

import random
from collections import defaultdict, Counter
from torch import Tensor

import warnings

from .noise_functions import (
    load_random_noise_chunk,
    attach_deterministic_noise_to_samples,
    add_noise_at_snr
)
from .config import NOISE_FOLDER, NOISE_FILES, SAMPLE_RATE, TARGET_LENGTH, CANONICAL, ALL_KEYWORDS

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
        keyword_list: Optional[list[str]] = CANONICAL, #
        use_unknown: bool = True, # unknown is samples from the rest of the keywords not present in the key_word_list
        use_silence: bool = True, # silence is just noise without any utterance present
        upsample: bool = False,
        seed: int = 789, # used for upsampling and unknown selection
        transform=None,
        add_noise: bool = True,
        noise_prob: float = 0.9,
        background_noise_folder: str = NOISE_FOLDER,
        noise_files: list[str] = NOISE_FILES,
        snr: float | tuple[float, float] = (-5,15),  # in dB
        **kwargs,
    ):
        super().__init__(*args, subset=subset, **kwargs)

        # training, validation, testing subsets
        self.subset = subset
        self.evaluation = subset in ("validation", "testing")

        # Keywords and class configuration
        self.keywords_set = set(keyword_list) if keyword_list is not None else set(ALL_KEYWORDS)
        self.use_unknown = use_unknown
        self.use_silence = use_silence
        self.background_noise_folder = Path(self._path, background_noise_folder)
        if use_silence:
            self.silence_paths = self._load_noise_paths(
                self.background_noise_folder, noise_files, "use_silence"
            )
        self.upsample = upsample
        self.seed = seed
        # Tranforms
        self.transform = transform if transform is not None else PadOrTrim()

        # Noise configuration
        # Noise files for silence samples (clips without speech) or to sample noise to add to speech samples
        self.add_noise = add_noise
        if self.add_noise:
            self.noise_paths = self._load_noise_paths(
                self.background_noise_folder, noise_files, "add_noise"
            )
            self.noise_prob = noise_prob
            self._set_snr(snr)
        else: 
            self.noise_paths = []
            self.noise_prob = 0.0
            self.snr = float('inf')

        self._init_labels()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _set_snr(self, snr: float | tuple[float, float]) -> None:
        """Validate and store SNR configuration."""
        assert self.add_noise, "Cannot set SNR if add_noise is False."

        # Ranges not allowed in evaluation mode
        if self.evaluation and isinstance(snr, tuple):
            raise ValueError("In evaluation mode, snr must be a single float (or inf).")

        # Float case (includes float('inf'))
        if isinstance(snr, (int, float)):
            self.snr = float(snr)
            return
        
        # Tuple case
        if (
            isinstance(snr, tuple)
            and len(snr) == 2
            and all(isinstance(x, (int, float)) for x in snr)
        ):
            low, high = snr
            if low > high:
                raise ValueError("snr range must satisfy low <= high.")
            self.snr = (float(low), float(high))
            return

        # Anything else is invalid
        raise TypeError("snr must be a float or (float, float).")
        
    def _load_noise_paths(self, base_dir: Path, file_list: list[str], flag_name: str) -> list[str]:
        """
        Returns sorted noise paths filtered by file_list.
        Falls back to white_noise.wav and warns if none found.
        """
        paths = sorted(str(p) for p in base_dir.glob("*.wav") if p.name in file_list)

        if not paths:
            white = base_dir / "white_noise.wav"
            paths = [str(white)]
            warnings.warn(
                f"{flag_name}=True but none of the requested files were found. "
                f"Falling back to white_noise.wav."
            )

        return paths

    def _init_labels(self) -> None:
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
        rng_upsample = random.Random(self.seed)
        for kw in sorted(self.keywords_set):
            idxs = per_keyword[kw]

            if self.upsample:
                if len(idxs) >= max_kw_count:
                    chosen = rng_upsample.sample(idxs, max_kw_count)
                else:
                    chosen = rng_upsample.choices(idxs, k=max_kw_count)
            else:
                # No upsampling: keep all samples for each keyword
                chosen = idxs

            samples.extend(
                {
                    "kind": "keyword",
                    "label": kw,
                    "walker_idx": idx,
                }
                for idx in chosen
            )

        # ---------- 2) Unknown samples (optional) ----------
        if self.use_unknown:
            if len(unknown_idxs) == 0:
                raise RuntimeError(
                    "No non-keyword commands available to form the 'unknown' class."
                )
            rng_unknown = random.Random(self.seed)  # deterministic selection for unknowns
            # Unknown size is always max_kw_count, regardless of upsample flag:
            n_unknown = max_kw_count

            if len(unknown_idxs) >= n_unknown:
                unknown_chosen = rng_unknown.sample(unknown_idxs, n_unknown)
            else:
                unknown_chosen = rng_unknown.choices(unknown_idxs, k=n_unknown)
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
            if not self.silence_paths:
                raise RuntimeError(
                    "use_silence=True but no noise files were found "
                    f"in folder: {self.background_noise_folder}"
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
        if self.evaluation and self.add_noise:
            self.samples = attach_deterministic_noise_to_samples(
                samples=self.samples,
                noise_paths=self.noise_paths,
                target_length=TARGET_LENGTH,
                sample_rate=SAMPLE_RATE,
                seed=self.seed,
            )
        
        # ---------- Label mapping ----------
        labels = list(sorted(self.keywords_set))
        if self.use_unknown:
            labels.append("UNKNOWN")
        if self.use_silence:
            labels.append("SILENCE")

        self.labels = labels
        self.label_counts = Counter(s["label"] for s in self.samples)
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

    # ------------------------------------------------------------------
    # Core Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, n: int) -> tuple[Tensor, int, dict]:
        sample_info = self.samples[n]
        label_str = sample_info["label"]
        kind = sample_info["kind"]
        walker_idx = sample_info["walker_idx"]

        noise = sample_info.get("noise", None)
        noise_type = sample_info.get("noise_type", None)

        # handle the silence class first since it needs special handling regardless of we add noise or not. 
        if kind == "silence":
            if self.evaluation and self.add_noise:
                # we should never see that we are in eval and then we have noise none
                assert noise is not None, ("Silence samples in evaluation should already have deterministic noise attached")
                assert noise_type is not None, ("Silence samples in evaluation should already have deterministic noise_type attached")
                waveform = noise.clone()
                utterance = noise_type
            else:
                waveform, utterance = self._load_random_noise_chunk(self.silence_paths)
                
            sample_rate = SAMPLE_RATE
            utterance_number = -1
            speaker_id = "SILENCE"
            
        elif kind in ("keyword", "unknown"):
            waveform, sample_rate, utterance, speaker_id, utterance_number = super().__getitem__(walker_idx)
        
        else: 
            raise RuntimeError(f"Unknown sample kind: {kind}")
        
        waveform = self._apply_transform(waveform)
        waveform_post_transform = waveform.clone()
        
        snr_value = float('inf')  # default: no noise added
        if not self.add_noise:
            noise = torch.zeros_like(waveform)
            noise_type = None
        else:
            snr_value = self._sample_snr_value() # in eval mode this inforced to be a fixed value

            if self.evaluation:
                assert noise is not None, "In evaluation mode and adding noise, all samples should already have deterministic noise attached"
                assert noise_type is not None, "In evaluation mode and adding noise, all samples should already have a deterministic noise_type attached"

                if kind == "silence":
                    snr_value = float('inf')
            
                else:
                    waveform = add_noise_at_snr(waveform, noise, snr_value)
                    
            else:
                if kind == "silence":
                        noise = waveform.clone()
                        noise_type = utterance   

                else: # keyword or unknown
                    if random.random() < self.noise_prob:
                        noise, noise_type  = self._load_random_noise_chunk(self.noise_paths)
                        waveform = add_noise_at_snr(waveform, noise, snr_value)
                    else:
                        # noise is a zero tensor length of waveform and noisetype is None.
                        noise =torch.zeros_like(waveform) # to avoid torch collate problems this cant be None
                        # noise type should still be none here 
                        assert noise_type is None, ("Noise_type is not None when adding noise and training mode")
        
        meta_data = {
            "sample_rate": sample_rate,
            "label": label_str,
            "utterance": utterance,
            "utterance_number": utterance_number,
            "speaker_id": speaker_id,
            "waveform_post_transform": waveform_post_transform,
            "noise": noise,
            "noise_type": noise_type,
            "snr": snr_value
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
    
    def _sample_snr_value(self) -> float:
        if isinstance(self.snr, (int, float)):
            return float(self.snr)
        low, high = self.snr
        return float(torch.empty(1).uniform_(low, high).item())

    def _load_random_noise_chunk(self, noise_paths: list[str]) -> tuple[Tensor, str]:
        return load_random_noise_chunk(
            noise_paths=noise_paths,
            target_length=TARGET_LENGTH,
            sample_rate=SAMPLE_RATE,
        )