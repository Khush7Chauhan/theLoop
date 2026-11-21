session_approvals: set[str] = set()


def reset_session_approvals() -> None:
    session_approvals.clear()


def is_approved(rule_hint: str | None) -> bool:
    return bool(rule_hint) and rule_hint in session_approvals


def request_approval(policy: dict, *, input_fn=input, output_fn=print) -> bool:
    """Ask the human to approve an ``ask`` policy decision.

    Returns True for "once" and "session" approvals; a "session" approval is
    remembered by its rule hint so the same path is not asked twice.
    """
    rule_hint = policy.get("rule_hint")
    if is_approved(rule_hint):
        return True

    output_fn("\nApproval needed")
    output_fn(policy["message"])
    output_fn(f"Risk: {policy.get('risk', 'unknown')}")
    output_fn("\nAllow?")
    output_fn("  y  yes, once")
    output_fn("  s  yes, for this session")
    output_fn("  n  no")

    choice = input_fn("> ").strip().lower()
    if choice == "s" and rule_hint:
        session_approvals.add(rule_hint)
        return True
    return choice == "y"
