# HITL Flowchart

```mermaid
flowchart TD
    A[User banking request] --> B{Prompt injection or anomaly?}
    B -->|Yes| C[Block and alert security reviewer]
    B -->|No| D{High-risk action?}
    D -->|Transfer, close account, identity change| E[Human-in-the-loop approval]
    D -->|General question| F[Generate grounded draft]
    F --> G{Confidence and judge scores}
    G -->|High confidence, all scores pass| H[Auto-send]
    G -->|Medium confidence or criteria disagree| I[Human-as-tiebreaker review queue]
    G -->|Low confidence or safety failure| J[Immediate escalation]
    E --> K{Reviewer decision}
    I --> K
    J --> K
    C --> L[Audit and incident response]
    K -->|Approve or edit| M[Send reviewed response]
    K -->|Reject| N[Safe refusal]
    H --> O[Audit and monitoring]
    M --> O
    N --> O
```

The three human decision points are high-risk action approval, ambiguous-response review, and
security-incident escalation. Their triggers and reviewer context are defined in `src/hitl/hitl.py`.
