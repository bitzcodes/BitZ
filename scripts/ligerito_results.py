"""Reject late-OOD, unlabelled and inconsistent Ligerito result identities."""
VERSION = "bitz/ligerito-policy/early-ood/v1"


def validate_ligerito(report, target=None):
    if not isinstance(report, dict) or report.get("protocol_version") != VERSION:
        raise ValueError("historical or missing Ligerito protocol identity")
    for key in ("requested_profile", "resolved_profile"):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ValueError("missing Ligerito profile")
    fingerprint = report.get("configuration_fingerprint", "")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("missing Ligerito configuration fingerprint")
    config = report.get("configuration", {})
    levels = config.get("levels", [])
    johnson = report.get("regime") == "johnson"
    if report.get("regime") not in ("johnson", "udr") or not levels or config.get("hash") != "blake3":
        raise ValueError("invalid Ligerito regime/hash/configuration")
    if report.get("outer_ood") is not johnson:
        raise ValueError("outer OOD presence disagrees with Ligerito regime")
    if report.get("target_bits") != config.get("target_security_bits") or (target is not None and report.get("target_bits") != target):
        raise ValueError("Ligerito target disagrees with enclosing protocol")
    expected = "johnson_ood" if johnson else "udr"
    if any(level.get("regime") != expected for level in levels):
        raise ValueError("inconsistent Ligerito decoding regimes")
    counts = [level.get("ood_samples", 0) for level in levels]
    if counts[0] != 0 or any((n > 0) != johnson for n in counts[1:]) or report.get("recursive_ood") != counts:
        raise ValueError("inconsistent recursive OOD accounting")
    return report


def decode_identity(encoded):
    import json
    try:
        report = json.loads(bytes.fromhex(encoded))
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("invalid encoded Ligerito identity") from error
    return validate_ligerito(report)


def validate_result_fields(fields):
    if fields.get("schema") not in ("bitz/2", "bitz-cli/2", "bitz-cli-mul/2"):
        raise ValueError("historical or unsupported BitZ result schema")
    return decode_identity(fields.get("ligerito_hex"))
