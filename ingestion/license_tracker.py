"""License & Provenance Tracker.

`normalizer.normalize` already attaches a baseline `LicenseMeta` and
`IngestTrace` to every record (source, url, raw file/offset, parser).
This module applies the *policy* on top of that provenance: given a
source's declared license, decide whether the record requires legal
review and whether it's allowed to ship to the public template store.

Per spec section 7: pipeline must never silently mark unknown or
non-permissive licenses as publishable.
"""
from __future__ import annotations

from .models import CanonicalRecord

# Licenses considered safe to redistribute in a public template store.
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc0",
    "cc-by-4.0",
    "public-domain",
}

# Licenses explicitly known to be internal-only (educational scrapes,
# research dumps with no redistribution rights, etc.) — not "unknown",
# just not publishable.
INTERNAL_ONLY_LICENSES = {
    "educational",
    "research-only",
    "proprietary",
}


def classify_license(license_name: str) -> tuple[bool, bool, bool]:
    """Returns (requires_legal_review, public_allowed, internal_use_only)."""
    key = (license_name or "unknown").strip().lower()

    if key in PERMISSIVE_LICENSES:
        return False, True, False
    if key in INTERNAL_ONLY_LICENSES:
        return False, False, True
    if key == "unknown":
        return True, False, True
    # Any other declared-but-unrecognized license: flag for legal review
    # rather than guessing either way.
    return True, False, True


def apply_license_policy(record: CanonicalRecord) -> CanonicalRecord:
    if record.license_meta is None:
        return record

    requires_legal_review, public_allowed, internal_use_only = classify_license(
        record.license_meta.license
    )
    record.license_meta.requires_legal_review = requires_legal_review
    record.license_meta.public_allowed = public_allowed
    record.license_meta.internal_use_only = internal_use_only

    if requires_legal_review:
        record.validation.requires_legal_review = True
        record.validation.needs_review = True
        reason = f"license '{record.license_meta.license}' requires legal review"
        if reason not in record.validation.reasons:
            record.validation.reasons.append(reason)

    record.touch("license_policy_applied", detail=f"license={record.license_meta.license}")
    return record
