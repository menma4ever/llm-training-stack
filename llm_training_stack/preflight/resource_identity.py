"""Typed resource identity classification for model, tokenizer, and dataset paths.

Distinguishes Hugging Face Hub identifiers (e.g. 'HuggingFaceTB/SmolLM-135M', 'gpt2')
from local filesystem paths and remote URI schemes, preventing path jail false exemptions
and directory traversal vulnerabilities.
"""

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
from typing import Optional, List, Tuple


class ResourceKind(str, Enum):
    """Categorization of a model, tokenizer, or dataset target identifier."""
    HUB_ID = "hub_id"
    LOCAL_PATH = "local_path"
    REMOTE_URI = "remote_uri"
    INVALID = "invalid"


@dataclass(frozen=True)
class ResourceIdentity:
    """Immutable representation of a resolved resource identifier."""
    raw_identifier: str
    kind: ResourceKind
    is_hub: bool
    is_local: bool
    is_remote: bool
    resolved_path: Optional[Path] = None
    repo_id: Optional[str] = None
    subfolder: Optional[str] = None
    exists_locally: bool = False
    validation_error: Optional[str] = None


class ResourceClassifier:
    """Classifies string identifiers into typed resource identities."""

    # Matches valid Hugging Face repo IDs: username/repo_name or single repo_name
    # Allows alphanumeric characters, dots, underscores, and dashes
    _HUB_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)?$")

    @classmethod
    def classify_output_path(cls, identifier: Optional[str]) -> ResourceIdentity:
        """Classifies an output directory target.
        
        Output paths are ALWAYS local filesystem paths, never Hub IDs,
        regardless of whether they exist yet or whether they are single-component ('runs', 'output')
        or multi-component ('runs/exp1').
        """
        return cls.classify(identifier, allow_nonexistent_local=True, is_output_path=True)

    @classmethod
    def classify(
        cls,
        identifier: Optional[str],
        allow_nonexistent_local: bool = False,
        is_output_path: bool = False,
    ) -> ResourceIdentity:
        """Classifies an identifier into a typed ResourceIdentity.

        Args:
            identifier: The input string (path, Hub ID, or URI).
            allow_nonexistent_local: If True, identifiers intended as filesystem paths
                (e.g. output directories or new paths) are classified as LOCAL_PATH even
                if they do not yet exist on disk.
            is_output_path: If True, indicates that the target is an output destination.
                Output targets are unconditionally local filesystem paths and never Hub IDs.
        """
        if not identifier or not identifier.strip():
            return ResourceIdentity(
                raw_identifier=identifier or "",
                kind=ResourceKind.INVALID,
                is_hub=False,
                is_local=False,
                is_remote=False,
                validation_error="Identifier is empty or whitespace.",
            )

        ident = identifier.strip()

        # 1. Remote URI scheme detection (e.g. 'http://', 's3://', 'hf://')
        if "://" in ident:
            return ResourceIdentity(
                raw_identifier=ident,
                kind=ResourceKind.REMOTE_URI,
                is_hub=False,
                is_local=False,
                is_remote=True,
                validation_error="Remote URI protocol detected.",
            )

        p = Path(ident)

        # 2. Output directory or non-existent local target check:
        # Output paths are ALWAYS local filesystem paths, never Hugging Face Hub IDs,
        # whether single-component ('runs', 'checkpoints') or multi-component ('runs/experiment_01').
        if is_output_path or allow_nonexistent_local:
            try:
                resolved = p.expanduser().resolve()
                return ResourceIdentity(
                    raw_identifier=ident,
                    kind=ResourceKind.LOCAL_PATH,
                    is_hub=False,
                    is_local=True,
                    is_remote=False,
                    resolved_path=resolved,
                    exists_locally=resolved.exists(),
                )
            except Exception as e:
                return ResourceIdentity(
                    raw_identifier=ident,
                    kind=ResourceKind.INVALID,
                    is_hub=False,
                    is_local=True,
                    is_remote=False,
                    validation_error=f"Path resolution failed: {e}",
                )

        # 3. Explicit filesystem path indicators:
        # - Starts with '.', '~', '/', or Windows drive letter (e.g. 'C:\' or 'C:/')
        # - Contains Windows backslash '\\'
        # - Trailing slash
        has_local_path_indicators = (
            ident.startswith((".", "~", "/", "\\"))
            or "\\" in ident
            or ident.endswith(("/", "\\"))
            or (len(ident) >= 2 and ident[1] == ":" and ident[0].isalpha())
            or p.is_absolute()
        )

        if has_local_path_indicators:
            try:
                resolved = p.expanduser().resolve()
                exists = resolved.exists()
                return ResourceIdentity(
                    raw_identifier=ident,
                    kind=ResourceKind.LOCAL_PATH,
                    is_hub=False,
                    is_local=True,
                    is_remote=False,
                    resolved_path=resolved,
                    exists_locally=exists,
                )
            except Exception as e:
                return ResourceIdentity(
                    raw_identifier=ident,
                    kind=ResourceKind.INVALID,
                    is_hub=False,
                    is_local=True,
                    is_remote=False,
                    validation_error=f"Path resolution failed: {e}",
                )

        # 4. Check if an unqualified relative path actually exists on the local filesystem
        # (e.g. 'data.jsonl', 'artifacts', 'tests')
        if p.exists():
            return ResourceIdentity(
                raw_identifier=ident,
                kind=ResourceKind.LOCAL_PATH,
                is_hub=False,
                is_local=True,
                is_remote=False,
                resolved_path=p.resolve(),
                exists_locally=True,
            )

        # 5. Check if it matches valid Hugging Face Hub ID format
        # Hub IDs have no backslashes, at most one forward slash, and match alphanumeric rules
        if cls._HUB_PATTERN.match(ident):
            return ResourceIdentity(
                raw_identifier=ident,
                kind=ResourceKind.HUB_ID,
                is_hub=True,
                is_local=False,
                is_remote=False,
                repo_id=ident,
                exists_locally=False,
            )

        # 6. Fallback: treat unrecognized pattern as a potential local path
        return ResourceIdentity(
            raw_identifier=ident,
            kind=ResourceKind.LOCAL_PATH,
            is_hub=False,
            is_local=True,
            is_remote=False,
            resolved_path=p.resolve(),
            exists_locally=False,
        )

    @classmethod
    def validate_in_roots(
        cls,
        identity: ResourceIdentity,
        allowed_roots: List[Path],
    ) -> Tuple[bool, Optional[str]]:
        """Verifies that a local path resides within authorized filesystem roots.

        Hub IDs automatically pass local path jail checks.
        """
        if identity.is_hub:
            return True, None

        if identity.is_remote:
            return False, f"Remote URI protocol is forbidden: '{identity.raw_identifier}'."

        if not identity.resolved_path:
            return False, f"Unresolved path: '{identity.raw_identifier}'."

        resolved = identity.resolved_path
        for root in allowed_roots:
            canonical_root = root.resolve()
            if resolved == canonical_root or canonical_root in resolved.parents:
                return True, None

        return False, (
            f"Path '{identity.raw_identifier}' resolves to '{resolved}', "
            f"which is outside authorized filesystem roots."
        )
