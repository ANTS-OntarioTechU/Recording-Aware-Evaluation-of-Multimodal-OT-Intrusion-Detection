"""Keep newly generated training files separate from the canonical paper artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import project_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingWorkspace:
    """All writable training files must stay here, never in canonical folders.

    The only allowed directory inside the checkout is ``training_workspace/``.
    An external scratch or Drive directory is also allowed, but cannot contain
    the checkout. Existing output files are never replaced.
    """

    root: Path
    repository: Path | None = None

    def __post_init__(self):
        repo = Path(self.repository or project_root()).expanduser().resolve()
        root = Path(self.root).expanduser().resolve()
        if root == repo or repo.is_relative_to(root):
            raise ValueError('Training workspace must not contain the repository')
        if root.is_relative_to(repo) and not root.is_relative_to(repo / 'training_workspace'):
            raise ValueError('Inside the repository, use training_workspace/ only')
        object.__setattr__(self, 'root', root)
        object.__setattr__(self, 'repository', repo)

    def output_path(self, relative: str | Path) -> Path:
        path = (self.root / relative).resolve()
        if path == self.root or not path.is_relative_to(self.root):
            raise ValueError('Training output must stay inside its workspace')
        # Resolve again on every write to detect symlinks created after setup.
        if path.is_relative_to(self.repository) and not path.is_relative_to(
                self.repository / 'training_workspace'):
            raise ValueError('Refusing to write a canonical repository path')
        return path

    def _write_new(self, relative, writer) -> str:
        destination = self.output_path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f'Refusing to replace {destination.name}')
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.partial')
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                writer(handle)
            digest = sha256_file(temporary)
            # Run directories are exclusive; a second writer still must not replace a file.
            with destination.open('xb') as handle, temporary.open('rb') as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    handle.write(chunk)
            if sha256_file(destination) != digest:
                raise ValueError(f'Output verification failed: {destination.name}')
            return digest
        finally:
            temporary.unlink(missing_ok=True)

    def write_json(self, relative, value) -> str:
        payload = (json.dumps(value, indent=2, default=str) + '\n').encode()
        return self._write_new(relative, lambda handle: handle.write(payload))

    def write_csv(self, relative, frame: pd.DataFrame) -> str:
        return self._write_new(relative, lambda handle: handle.write(frame.to_csv(index=False).encode()))

    def write_parquet(self, relative, frame: pd.DataFrame) -> str:
        return self._write_new(relative, lambda handle: frame.to_parquet(handle, index=False))

    def write_npz(self, relative, **arrays) -> str:
        # File handle prevents numpy from silently appending an extra .npz suffix.
        return self._write_new(relative, lambda handle: np.savez_compressed(handle, **arrays))

    def write_model(self, relative, bundle) -> str:
        """Save only models you fitted yourself; never load untrusted joblib files."""
        import joblib
        return self._write_new(relative, lambda handle: joblib.dump(bundle, handle))

    def new_run(self) -> Path:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        relative = Path('runs') / run_id
        self.output_path(relative).mkdir(parents=True, exist_ok=False)
        return relative
