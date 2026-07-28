"""Written policy definitions for the Phase 4.2 policy-conditioned classifier.

The fixed-label model in Phase 3 learns each label as an opaque column index: label 3
means whatever the annotators happened to put in column 3. Nothing about the *meaning*
of "threat" is available to it, which is why adding a seventh policy would require
retraining with new labeled data.

A policy-conditioned classifier instead takes the policy text as **input**:

    "does this comment violate policy X, where X is defined as <definition>?"

The definition is part of the input string, so the model learns to read a rule and
apply it, rather than memorizing six column positions. That is what makes the
held-out-policy experiment in ``src/policy.py`` possible — the model can be asked
about a policy whose definition it has never seen during training, which a fixed
six-head classifier structurally cannot do.

These definitions are written in the style of real content-moderation policy: a scope
statement, explicit inclusions, and explicit exclusions. The exclusions matter most —
they are where a human reviewer's judgment actually lives, and where a classifier
trained only on label columns has no guidance at all.

Kept deliberately compact (~60-80 tokens each) because the definition and the comment
share a 256-token window; a verbose definition would truncate the text being judged.
"""

from __future__ import annotations

# Policies with written definitions. The keys must be valid label names from config.LABELS.
POLICIES: dict[str, dict[str, str]] = {
    "toxic": {
        "title": "Toxic or abusive language",
        "definition": (
            "Policy: Toxic content. A comment violates this policy if it is rude, "
            "disrespectful, hostile, or abusive toward a person or group, such that a "
            "reasonable reader would be discouraged from taking part in the conversation. "
            "This includes name-calling, hostile mockery, profanity aimed at someone, and "
            "aggressive contempt. It does not include strong disagreement stated civilly, "
            "criticism of ideas or edits rather than people, blunt or terse tone, or "
            "profanity used as emphasis without a target."
        ),
    },
    "threat": {
        "title": "Threats of violence or harm",
        "definition": (
            "Policy: Threats. A comment violates this policy if it expresses intent to "
            "inflict physical harm, death, or violence on a person or group, or wishes such "
            "harm on them, or intimidates by warning of harm to come. Conditional and "
            "indirect phrasing still counts. It does not include threats of non-violent "
            "action such as reporting, banning, or legal steps; violence described in the "
            "third person or quoted from elsewhere; fictional or historical narration; or "
            "hyperbolic figures of speech carrying no intent of harm."
        ),
    },
    "identity_hate": {
        "title": "Hate directed at identity",
        "definition": (
            "Policy: Identity-based hate. A comment violates this policy if it attacks, "
            "demeans, or dehumanizes a person or group because of a protected characteristic "
            "such as race, ethnicity, national origin, religion, gender, sexual orientation, "
            "or disability. This includes slurs targeting those characteristics and claims of "
            "inherent inferiority. It does not include neutral or factual mention of an "
            "identity group, criticism of a religion's or nation's practices without "
            "demeaning its people, or reclaimed in-group usage with no hostile target."
        ),
    },
}

# Policy held out of training in the zero-shot experiment. `threat` is chosen because it
# is the rarest label (478 positives, 0.30%), which makes it the hardest and most
# informative generalization test — and the one where labeled data is most expensive to
# obtain in practice, so zero-shot transfer would be worth the most.
HELD_OUT_POLICY = "threat"


def policy_names() -> list[str]:
    return list(POLICIES)


def definition(label: str) -> str:
    if label not in POLICIES:
        raise KeyError(f"no written policy for '{label}'. Defined: {policy_names()}")
    return POLICIES[label]["definition"]


def training_policies() -> list[str]:
    """Policies the zero-shot variant is allowed to train on."""
    return [p for p in POLICIES if p != HELD_OUT_POLICY]
