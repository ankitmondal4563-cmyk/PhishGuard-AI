"""Build a downloadable Markdown risk-assessment report for one scan."""
from datetime import datetime

DISCLAIMER = ("This report is an automated estimate produced by a machine-learning model. "
              "It is not a guarantee. When in doubt, do not click links or share personal "
              "information - contact the organisation through its official website or app.")


def build_report(scan_type: str, result: dict, model_info: dict | None = None) -> str:
    lines = [
        "# PhishGuard AI - Risk Assessment Report",
        f"_Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}_",
        "",
        f"**Scan type:** {scan_type}",
        f"**Verdict:** {result['label']}",
        f"**Risk level:** {result['risk_level']} ({result['risk_score']}/100)",
        "",
        "## Scanned content",
        "```",
        result["input"][:2000],
        "```",
        "",
        "## Red flags found",
    ]
    lines += [f"- {i}" for i in result["indicators"]] or ["- None of the rule-based red flags triggered."]

    drivers = result.get("model_drivers")
    if isinstance(drivers, list) and drivers:
        lines += ["", "## What influenced the model most", ""]
        lines += [f"- {d['feature']} (value {d['value']}): "
                  f"{'raises' if d['contribution'] > 0 else 'lowers'} risk by {abs(d['contribution']):.2f}"
                  for d in drivers]
    elif isinstance(drivers, dict) and drivers.get("towards_phishing"):
        lines += ["", "## Words that raised the risk", ""]
        lines += [f"- {d['term']} (+{d['contribution']:.2f})" for d in drivers["towards_phishing"]]

    if result.get("links"):
        lines += ["", "## Links found"]
        lines += [f"- {l['url']} -> {l['risk_level']} ({l['probability']:.0%})" for l in result["links"]]

    if model_info:
        lines += ["", "## Model", f"- {model_info['name']} (trained {model_info['trained_at']})"]
        if model_info.get("is_synthetic"):
            lines += ["- **Warning: this model was trained on synthetic demo data and must not be "
                      "relied on for real security decisions.**"]
    lines += ["", "---", DISCLAIMER]
    return "\n".join(lines)
