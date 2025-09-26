from pathlib import Path
from typing import Iterable, List
import logging
from tqdm import tqdm
import taglib
import matchering as mg
from .llm import metadata_extract


def normalize_audio_extract_tags(
    audio_files: Iterable[Path],
    reference_path: Path,
    collection_path: Path,
    *,
    overwrite: bool = False,
) -> List[Path]:
    """
    Master each audio file using Matchering with a single reference track,
    write mastered .wav into `collection_path`, and copy/augment metadata tags.

    Returns:
        List of paths to mastered files that were written (or already existed if overwrite=False).
    """
    written: List[Path] = []

    for file in tqdm(audio_files, desc="Mastering Songs"):

        out_name = f"{file.stem}.wav"
        out_path = collection_path / out_name

        try:
            if out_path.exists() and not overwrite:
                logger.info("Already mastered (skip): %s", out_path.name)
                written.append(out_path)
                continue

            # Ensure parent exists (may differ per file if you choose a nested strategy later)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # ---- Mastering ----
            mg.process(
                target=str(file),
                reference=str(reference_path),
                results=[mg.pcm24(str(out_path))],  # 24-bit PCM WAV
            )
            logger.info("Normalized: %s -> %s", file.name, out_path.name)

            # ---- Tags ----
            src = None
            dst = None
            try:
                src = taglib.File(file.as_posix())
                dst = taglib.File(out_path.as_posix())

                # Copy existing tags
                # taglib expects {str: List[str]}
                dst.tags = {k: [str(v) for v in (vals if isinstance(vals, list) else [vals])]
                            for k, vals in (src.tags or {}).items()}

                # LLM-based extraction from filename stem
                extracted = metadata_extract(file.stem)  # returns dict[str, List[str]]
                # Merge/override with extracted fields
                for k, v in extracted.items():
                    # ensure list[str]
                    v_list = [str(x) for x in (v if isinstance(v, list) else [v])]
                    dst.tags[k] = v_list

                dst.save()

            finally:
                # close TagLib handles to release file descriptors
                try:
                    if src is not None:
                        src.close()
                except Exception:
                    pass
                try:
                    if dst is not None:
                        dst.close()
                except Exception:
                    pass

            written.append(out_path)

        except taglib.TagLibException as e:
            logger.warning("TagLib error on %s: %s", file.name, e)
        except Exception as e:
            logger.warning("Error normalizing %s: %s", file.name, e)

    return written