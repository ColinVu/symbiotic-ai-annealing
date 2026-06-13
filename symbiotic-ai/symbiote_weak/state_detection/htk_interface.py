"""Interface between Python and the HTK Hidden Markov Model toolkit.

Provides helpers for:
* writing / reading HTK binary feature files (``.mfc``),
* writing HTK label files (``.lab``) and master label files (``.mlf``),
* generating the HMM prototype and configuration files,
* running HTK training (``HCompV``, ``HERest``) and decoding (``HVite``)
  via ``subprocess``.
"""

import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_HTK_CONFIG, HTKConfig


# Default HTK state labels (emitting states only)
_DEFAULT_STATES = ["PICK", "CARRY_WITH", "PLACE", "CARRY_EMPTY"]


class HTKStateDetector:
    """HTK-based HMM state detector."""

    def __init__(
        self,
        model_dir: str,
        config: Optional[HTKConfig] = None,
        state_labels: Optional[List[str]] = None,
    ):
        self.model_dir = model_dir
        self.config = config or DEFAULT_HTK_CONFIG
        self.state_labels = list(state_labels) if state_labels else list(_DEFAULT_STATES)

    # ==================================================================
    # Training
    # ==================================================================

    def train(
        self,
        training_data: List[Tuple[np.ndarray, pd.DataFrame]],
        output_dir: str,
        verbose: bool = True,
    ) -> None:
        """Train HTK HMM from ``(features, annotations)`` pairs.

        Creates the full HTK directory structure under *output_dir* and runs
        ``HCompV`` followed by iterative ``HERest`` re-estimation.
        """
        output_dir = os.path.abspath(output_dir)
        features_dir = os.path.join(output_dir, "features")
        labels_dir = os.path.join(output_dir, "labels")
        models_dir = os.path.join(output_dir, "models")
        config_dir = os.path.join(output_dir, "config")

        for d in [features_dir, labels_dir, models_dir, config_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # 1. Write feature files and label files -----------------------
        scp_entries: List[str] = []
        mlf_lines = ["#!MLF!#\n"]
        for i, (features, annotations) in enumerate(training_data):
            mfc_path = os.path.join(features_dir, f"train_{i}.mfc")
            lab_path = os.path.join(labels_dir, f"train_{i}.lab")
            self._write_htk_features(features, mfc_path)
            self._write_htk_labels(annotations, lab_path)
            scp_entries.append(mfc_path)

            # Append to master label file
            mlf_lines.append(f'"*/train_{i}.lab"\n')
            for _, row in annotations.iterrows():
                s = int(row["timestamp_start"] * 1e7)
                e = int(row["timestamp_end"] * 1e7)
                mlf_lines.append(f"{s} {e} {row['state']}\n")
            mlf_lines.append(".\n")

        # Write .scp (script) file
        scp_path = os.path.join(config_dir, "train.scp")
        with open(scp_path, "w") as f:
            f.write("\n".join(scp_entries) + "\n")

        # Write master label file
        mlf_path = os.path.join(labels_dir, "labels.mlf")
        with open(mlf_path, "w") as f:
            f.writelines(mlf_lines)

        # 2. Write configuration and prototype -------------------------
        htk_cfg_path = self._write_htk_config(config_dir)
        proto_path = self._write_proto(config_dir)
        wordlist_path = self._write_wordlist(config_dir, self.state_labels)
        grammar_path = self._write_grammar(config_dir)

        # 3. HCompV: compute global mean / variance --------------------
        hmm0_dir = os.path.join(models_dir, "hmm0")
        Path(hmm0_dir).mkdir(exist_ok=True)

        cmd_hcompv = [
            "HCompV", "-T", "1",
            "-C", htk_cfg_path,
            "-S", scp_path,
            "-M", hmm0_dir,
            "-f", "0.01",
            proto_path,
        ]
        if verbose:
            print(f"[HTK] Running HCompV ...")
        result = subprocess.run(cmd_hcompv, capture_output=True, text=True)
        if verbose and result.stdout:
            print(result.stdout[-500:])
        if result.returncode != 0:
            raise RuntimeError(
                f"HCompV failed (exit {result.returncode}):\n{result.stderr}"
            )

        # Build the macro / hmmdefs from HCompV output
        self._build_hmm0(hmm0_dir, proto_path)

        # 4. HERest: Baum-Welch re-estimation --------------------------
        prev_dir = hmm0_dir
        for iteration in range(1, self.config.num_training_iterations + 1):
            iter_dir = os.path.join(models_dir, f"hmm{iteration}")
            Path(iter_dir).mkdir(exist_ok=True)

            cmd_herest = [
                "HERest", "-T", "1",
                "-C", htk_cfg_path,
                "-S", scp_path,
                "-I", mlf_path,
                "-M", iter_dir,
                "-H", os.path.join(prev_dir, "macros"),
                "-H", os.path.join(prev_dir, "hmmdefs"),
                wordlist_path,
            ]

            if verbose:
                print(f"[HTK] HERest iteration {iteration} ...")
            result = subprocess.run(cmd_herest, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"HERest iter {iteration} failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )
            prev_dir = iter_dir

        # 5. Copy final model to canonical location --------------------
        final_dir = os.path.join(models_dir, "hmm_final")
        Path(final_dir).mkdir(exist_ok=True)
        
        # Copy macros as-is (don't strip anything - HVite needs STREAMINFO for networks)
        src = os.path.join(prev_dir, "macros")
        dst = os.path.join(final_dir, "macros")
        if os.path.exists(src):
            with open(src, "rb") as f_in, open(dst, "wb") as f_out:
                f_out.write(f_in.read())
        
        # Copy hmmdefs - strip any ~o header and ensure blank lines between HMMs
        src = os.path.join(prev_dir, "hmmdefs")
        dst = os.path.join(final_dir, "hmmdefs")
        if os.path.exists(src):
            with open(src, "r") as f_in:
                lines = f_in.readlines()
            with open(dst, "w") as f_out:
                skip_header = True
                prev_was_endhmm = False
                for line in lines:
                    stripped = line.strip()
                    # Skip ~o header and related lines at the start
                    if skip_header:
                        if stripped.startswith("~h"):
                            skip_header = False
                            f_out.write(line)
                            prev_was_endhmm = False
                        elif not (stripped.startswith("~o") or 
                                  stripped.startswith("<STREAMINFO>") or
                                  stripped.startswith("<VECSIZE>") or
                                  stripped == ""):
                            skip_header = False
                            f_out.write(line)
                            prev_was_endhmm = False
                    else:
                        # Add blank line between HMM definitions
                        if stripped.startswith("~h") and prev_was_endhmm:
                            f_out.write("\n")
                        f_out.write(line)
                        prev_was_endhmm = (stripped == "<ENDHMM>")

        # Copy grammar and wordlist for decoding
        for fn in [grammar_path, wordlist_path]:
            dst = os.path.join(final_dir, os.path.basename(fn))
            with open(fn, "rb") as f_in, open(dst, "wb") as f_out:
                f_out.write(f_in.read())

        if verbose:
            print(f"[HTK] Training complete. Final model: {final_dir}")

    # ==================================================================
    # Decoding
    # ==================================================================

    def decode(
        self,
        features: np.ndarray,
        fps: float,
        frame_numbers: Optional[List[int]] = None,
        verbose: bool = True,
        word_penalty: float = 0.0,
        grammar_scale: float = 5.0,
        strict_cycle: bool = True,
        expected_sequence: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Decode state sequence using Viterbi (HVite).

        Returns DataFrame with ``[timestamp_start, timestamp_end, state]``.
        """
        final_dir = os.path.join(self.model_dir, "models", "hmm_final")
        if not os.path.isdir(final_dir):
            final_dir = self.model_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            mfc_path = os.path.join(tmpdir, "input.mfc")
            self._write_htk_features(features, mfc_path)

            scp_path = os.path.join(tmpdir, "test.scp")
            with open(scp_path, "w") as f:
                f.write(mfc_path + "\n")

            output_mlf = os.path.join(tmpdir, "output.mlf")

            wordlist_path = os.path.join(final_dir, "wordlist")
            grammar_path = os.path.join(final_dir, "grammar")

            htk_cfg_path = self._write_htk_config(tmpdir)

            grammar_path = os.path.join(tmpdir, "grammar")
            with open(grammar_path, "w") as f:
                if expected_sequence:
                    f.write(self._expected_sequence_grammar(expected_sequence))
                elif strict_cycle:
                    # Enforce cyclic order while allowing decode to start at any phase.
                    f.write(self._strict_cycle_grammar())
                else:
                    # Unconstrained loop over state symbols.
                    states_list = " | ".join(self.state_labels)
                    f.write(f"$word = {states_list};\n")
                    f.write("( < $word > )\n")
            
            # Parse grammar into network
            net_path = os.path.join(tmpdir, "net.slf")
            hp_result = subprocess.run(
                ["HParse", grammar_path, net_path],
                capture_output=True, text=True
            )
            if hp_result.returncode != 0:
                raise RuntimeError(f"HParse failed: {hp_result.stderr}")

            # Create a dictionary that maps each word to itself
            dict_path = os.path.join(tmpdir, "dict")
            with open(dict_path, "w") as f:
                for state in self.state_labels:
                    f.write(f"{state} {state}\n")

            cmd = [
                "HVite", "-T", "1",
                "-C", htk_cfg_path,
                "-H", os.path.join(final_dir, "macros"),
                "-H", os.path.join(final_dir, "hmmdefs"),
                "-S", scp_path,
                "-i", output_mlf,
                "-w", net_path,
                "-p", str(word_penalty),  # Word insertion penalty
                "-s", str(grammar_scale),  # Grammar scale factor
                dict_path,
                wordlist_path,
            ]
            if verbose:
                print("[HTK] Running HVite ...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Debug: print stderr even on success
            if verbose and result.stderr:
                print(f"[HTK] HVite stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                # Also print the full command for debugging
                print(f"[HTK] Failed command: {' '.join(cmd)}")
                raise RuntimeError(
                    f"HVite failed (exit {result.returncode}):\n{result.stderr}"
                )

            segments = self._parse_mlf(
                output_mlf,
                fps=fps,
                n_frames=features.shape[0],
                frame_numbers=frame_numbers,
                sample_period=self.config.sample_period,
                state_labels=self.state_labels,
            )

        return segments

    # ==================================================================
    # HTK file I/O helpers
    # ==================================================================

    def _write_htk_features(
        self, features: np.ndarray, output_path: str
    ) -> str:
        """Write features to HTK binary ``.mfc`` format."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n_frames, n_features = features.shape
        sample_period = self.config.sample_period
        samp_size = n_features * 4  # 4 bytes per float32

        header = struct.pack(">IIHH", n_frames, sample_period, samp_size, 9)
        with open(output_path, "wb") as f:
            f.write(header)
            for row in features.astype(np.float32):
                f.write(struct.pack(f">{n_features}f", *row))
        return output_path

    @staticmethod
    def _read_htk_features(path: str) -> np.ndarray:
        """Read an HTK ``.mfc`` file back into a numpy array."""
        with open(path, "rb") as f:
            n_frames, sample_period, samp_size, parm_kind = struct.unpack(
                ">IIHH", f.read(12)
            )
            n_features = samp_size // 4
            data = []
            for _ in range(n_frames):
                row = struct.unpack(f">{n_features}f", f.read(samp_size))
                data.append(row)
        return np.array(data, dtype=np.float32)

    @staticmethod
    def _write_htk_labels(
        annotations: pd.DataFrame, output_path: str
    ) -> None:
        """Write annotations to an HTK ``.lab`` file."""
        with open(output_path, "w") as f:
            for _, row in annotations.iterrows():
                s = int(row["timestamp_start"] * 1e7)
                e = int(row["timestamp_end"] * 1e7)
                f.write(f"{s} {e} {row['state']}\n")

    # ------------------------------------------------------------------
    # Config / prototype generation
    # ------------------------------------------------------------------

    def _write_htk_config(self, output_dir: str) -> str:
        """Write minimal HTK configuration file."""
        path = os.path.join(output_dir, "htk.cfg")
        with open(path, "w") as f:
            f.write("TARGETKIND = USER\n")
            f.write(f"TARGETRATE = {self.config.sample_period}.0\n")
        return path

    def _write_proto(self, output_dir: str) -> str:
        """Write HMM prototype file with simple 3-state topology.

        Uses 3 states (entry + 1 emitting + exit) which is standard
        for HTK and should work with networks.
        """
        dim = self.config.feature_dim
        mean_str = " ".join(["0.0"] * dim)
        var_str = " ".join(["1.0"] * dim)
        path = os.path.join(output_dir, "proto")

        with open(path, "w") as f:
            f.write(f"~o <VECSIZE> {dim} <USER>\n")
            f.write(f'~h "proto"\n')
            f.write("<BEGINHMM>\n")
            f.write("<NUMSTATES> 3\n")  # entry + 1 emitting + exit
            f.write("<STATE> 2\n")
            f.write(f"<MEAN> {dim}\n  {mean_str}\n")
            f.write(f"<VARIANCE> {dim}\n  {var_str}\n")
            f.write("<TRANSP> 3\n")
            f.write("  0.0 1.0 0.0\n")
            f.write("  0.0 0.6 0.4\n")
            f.write("  0.0 0.0 0.0\n")
            f.write("<ENDHMM>\n")
        return path

    @staticmethod
    def _write_wordlist(output_dir: str, state_labels: List[str]) -> str:
        """Write HTK word list (one state label per line)."""
        path = os.path.join(output_dir, "wordlist")
        with open(path, "w") as f:
            for state in state_labels:
                f.write(state + "\n")
        return path

    def _write_grammar(self, output_dir: str) -> str:
        """Write a strict cyclic HTK grammar for state sequence recognition."""
        path = os.path.join(output_dir, "grammar")
        with open(path, "w") as f:
            f.write(self._strict_cycle_grammar())
        return path

    def _strict_cycle_grammar(self) -> str:
        states = self.state_labels
        if not states:
            raise ValueError("state_labels cannot be empty")
        lines: List[str] = []
        for i in range(len(states)):
            rotated = states[i:] + states[:i]
            lines.append(f"$cy{i} = {' '.join(rotated)};")
        cyc = " | ".join(f"$cy{i}" for i in range(len(states)))
        lines.append(f"( < {cyc} > )")
        return "\n".join(lines) + "\n"

    def _expected_sequence_grammar(self, expected_sequence: List[str]) -> str:
        seq = [s for s in expected_sequence if s in set(self.state_labels)]
        if not seq:
            return self._strict_cycle_grammar()
        return "( " + " ".join(seq) + " )\n"

    # ------------------------------------------------------------------
    # Post-HCompV model building
    # ------------------------------------------------------------------

    def _build_hmm0(self, hmm0_dir: str, proto_path: str) -> None:
        """Create per-state HMM definitions from the prototype.

        After ``HCompV`` runs on the prototype, we replicate it for
        each state label so HTK can train them independently.

        Uses simple 3-state HMMs.
        """
        dim = self.config.feature_dim

        # Parse mean and variance from HCompV output
        proto_out = os.path.join(hmm0_dir, "proto")
        if not os.path.exists(proto_out):
            proto_out = proto_path

        mean_vec: str = " ".join(["0.0"] * dim)
        var_vec: str  = " ".join(["1.0"] * dim)

        try:
            with open(proto_out, "r") as f:
                plines = f.readlines()
            i = 0
            while i < len(plines):
                stripped = plines[i].strip()
                if stripped.startswith("<MEAN>"):
                    if i + 1 < len(plines):
                        mean_vec = plines[i + 1].strip()
                    i += 2
                elif stripped.startswith("<VARIANCE>") and not stripped.startswith("<VARIANCE_FLOOR>"):
                    if i + 1 < len(plines):
                        var_vec = plines[i + 1].strip()
                    i += 2
                else:
                    i += 1
        except OSError:
            pass

        # Build variance floor
        vfloors_path = os.path.join(hmm0_dir, "vFloors")
        if os.path.exists(vfloors_path):
            with open(vfloors_path, "r") as f:
                vfloors_content = f.read().strip()
        else:
            floor_str = " ".join(["0.01"] * dim)
            vfloors_content = f'~v "varFloor1"\n<VARIANCE> {dim}\n  {floor_str}'

        # Write macros
        macros_path = os.path.join(hmm0_dir, "macros")
        with open(macros_path, "w") as f:
            f.write(f"~o <VECSIZE> {dim} <USER>\n")
            f.write(vfloors_content + "\n")

        # Write hmmdefs with 3-state HMMs
        transp = "  0.0 1.0 0.0\n  0.0 0.6 0.4\n  0.0 0.0 0.0"
        hmmdefs_path = os.path.join(hmm0_dir, "hmmdefs")
        with open(hmmdefs_path, "w") as f:
            for idx, state_name in enumerate(self.state_labels):
                if idx > 0:
                    f.write("\n")  # Blank line between HMMs
                f.write(f'~h "{state_name}"\n')
                f.write("<BEGINHMM>\n")
                f.write("<NUMSTATES> 3\n")
                f.write("<STATE> 2\n")
                f.write(f"<MEAN> {dim}\n")
                f.write(f"  {mean_vec}\n")
                f.write(f"<VARIANCE> {dim}\n")
                f.write(f"  {var_vec}\n")
                f.write("<TRANSP> 3\n")
                f.write(f"{transp}\n")
                f.write("<ENDHMM>\n")

    # ------------------------------------------------------------------
    # MLF parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mlf(
        mlf_path: str,
        fps: float,
        n_frames: int,
        frame_numbers: Optional[List[int]] = None,
        sample_period: int = 100000,
        state_labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Parse HTK Master Label File output into a DataFrame.

        HTK emits segment times in 100ns units based on the feature sample
        period used in the .mfc header (``sample_period``). If we have the
        original sampled frame indices from extraction, map HTK frame indices
        back to real video timestamps so output aligns with the source video.
        """
        segments: List[Dict] = []
        valid_labels = set(state_labels or _DEFAULT_STATES)

        has_frame_map = (
            frame_numbers is not None
            and len(frame_numbers) == n_frames
            and fps > 0
            and sample_period > 0
        )

        def _map_time(start_100ns: int, end_100ns: int) -> Tuple[float, float]:
            if not has_frame_map:
                return start_100ns / 1e7, end_100ns / 1e7

            # Convert HTK times to frame indices in the extracted feature stream.
            # HTK segment boundaries are aligned to frame periods.
            start_idx = int(round(start_100ns / sample_period))
            end_idx = int(round(end_100ns / sample_period))

            # Clamp to valid extracted-frame index range.
            start_idx = max(0, min(start_idx, n_frames - 1))
            end_idx = max(0, min(end_idx, n_frames - 1))
            if end_idx < start_idx:
                end_idx = start_idx

            # Map extracted feature indices back to original video timestamps.
            start_t = frame_numbers[start_idx] / fps
            end_t = frame_numbers[end_idx] / fps
            if end_t < start_t:
                end_t = start_t
            return start_t, end_t

        with open(mlf_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith('"') or line == ".":
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    start_100ns = int(parts[0])
                    end_100ns = int(parts[1])
                    state = parts[2]
                    if state in valid_labels:
                        start_t, end_t = _map_time(start_100ns, end_100ns)
                        segments.append({
                            "timestamp_start": start_t,
                            "timestamp_end": end_t,
                            "state": state,
                        })

        if not segments:
            return pd.DataFrame(
                columns=["timestamp_start", "timestamp_end", "state"]
            )
        return pd.DataFrame(segments)


__all__ = ["HTKStateDetector"]
