"""
demo.py — Interactive CLI Demo for Finance Fraud Detection SLM

Allows interactive testing of the model from the command line,
with pre-loaded example transactions and the option to enter custom ones.
"""

import os
import sys

# ── Try rich for pretty output, fall back to plain print ────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None


def cprint(msg="", style="", markup=True):
    if RICH:
        console.print(msg, style=style)
    else:
        print(msg)


BANNER = """
╔═══════════════════════════════════════════════════════════╗
║       💳  Finance Fraud Detection SLM — Demo CLI          ║
║       ~1M Parameter Transformer Encoder                   ║
╚═══════════════════════════════════════════════════════════╝
"""

PRELOADED_EXAMPLES = [
    {
        "name": "🛒 Normal grocery purchase",
        "txn": {
            "amount": 47.50,
            "merchant_cat": "GROCERY",
            "country": "US",
            "is_domestic": True,
            "hour": 14,
            "day_of_week": 1,
            "channel": "POS_CHIP",
            "currency": "USD",
            "velocity": "NORMAL",
            "flags": ["VERIFIED"],
        },
    },
    {
        "name": "☕ Coffee shop — weekday morning",
        "txn": {
            "amount": 5.75,
            "merchant_cat": "RESTAURANT",
            "country": "US",
            "is_domestic": True,
            "hour": 8,
            "day_of_week": 2,
            "channel": "POS_TAP",
            "currency": "USD",
            "velocity": "NORMAL",
            "flags": ["VERIFIED", "3DS_PASSED"],
        },
    },
    {
        "name": "💻 Online electronics (suspicious)",
        "txn": {
            "amount": 1200.00,
            "merchant_cat": "ELECTRONICS",
            "country": "RO",
            "is_domestic": False,
            "hour": 23,
            "day_of_week": 6,
            "channel": "ONLINE",
            "currency": "EUR",
            "velocity": "HIGH",
            "flags": ["BILLING_MISMATCH", "FOREIGN_IP"],
        },
    },
    {
        "name": "🚨 Crypto purchase + Tor + extreme velocity",
        "txn": {
            "amount": 9999.99,
            "merchant_cat": "CRYPTO",
            "country": "NG",
            "is_domestic": False,
            "hour": 3,
            "day_of_week": 6,
            "channel": "ONLINE",
            "currency": "CRYPTO_BTC",
            "velocity": "EXTREME",
            "flags": ["NEW_DEVICE", "TOR_VPN", "FOREIGN_IP"],
        },
    },
    {
        "name": "💵 ATM withdrawal — geo-impossible",
        "txn": {
            "amount": 500.0,
            "merchant_cat": "ATM",
            "country": "GH",
            "is_domestic": False,
            "hour": 22,
            "day_of_week": 3,
            "channel": "ATM",
            "currency": "GHS",
            "velocity": "RAPID_SUCCESSION",
            "flags": ["GEO_IMPOSSIBLE", "RECENTLY_COMPROMISED"],
        },
    },
    {
        "name": "🧾 Card testing micro-transaction",
        "txn": {
            "amount": 0.99,
            "merchant_cat": "DIGITAL_GOODS",
            "country": "PH",
            "is_domestic": False,
            "hour": 2,
            "day_of_week": 5,
            "channel": "ONLINE",
            "currency": "USD",
            "velocity": "RAPID_SUCCESSION",
            "flags": ["NEW_DEVICE", "NO_HISTORY"],
        },
    },
]

MERCHANT_CATS = [
    "GROCERY", "GAS", "RESTAURANT", "RETAIL", "PHARMACY",
    "UTILITIES", "TELECOM", "SUBSCRIPTION", "ENTERTAINMENT",
    "ELECTRONICS", "ECOMMERCE", "LUXURY", "JEWELRY",
    "CRYPTO", "MONEY_TRANSFER", "GAMBLING", "DIGITAL_GOODS",
    "HEALTHCARE", "EDUCATION", "ATM",
]

CHANNELS = [
    "POS_CHIP", "POS_SWIPE", "POS_TAP", "ATM", "ONLINE",
    "PHONE", "RECURRING", "MOBILE_APP",
]

VELOCITIES = [
    "NORMAL", "ELEVATED", "HIGH", "VERY_HIGH",
    "EXTREME", "RAPID_SUCCESSION", "DORMANT_REUSE", "GEO_IMPOSSIBLE",
]

AVAILABLE_FLAGS = [
    "NEW_DEVICE", "FOREIGN_IP", "TOR_VPN", "BILLING_MISMATCH",
    "CVV_FAIL", "RECENTLY_COMPROMISED", "GEO_IMPOSSIBLE",
    "HIGH_DECLINE_RATE", "NO_HISTORY", "VERIFIED", "3DS_PASSED",
]


def print_result(txn_name: str, result: dict):
    """Print formatted prediction result."""
    prob  = result["fraud_probability"]
    label = result["risk_label"]
    action = result["action"]
    conf  = result["confidence"]
    factors = result.get("risk_factors", [])

    bar_len = 40
    filled = int(prob * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    if RICH:
        color = (
            "bold red"    if prob >= 0.75 else
            "bold orange3" if prob >= 0.50 else
            "bold yellow"  if prob >= 0.25 else
            "bold green"
        )
        panel_content = (
            f"[bold]{label}[/bold]\n"
            f"Action: {action}\n"
            f"Fraud probability: [{color}]{prob:.2%}[/{color}]  (confidence: {conf})\n"
            f"[{color}]{bar}[/{color}]\n"
        )
        if factors:
            panel_content += f"Risk factors: [italic yellow]{', '.join(factors[:5])}[/italic yellow]\n"
        panel_content += f"\nTokens: [dim]{' '.join(result['tokens'][:10])}{'...' if len(result['tokens'])>10 else ''}[/dim]"
        console.print(Panel(panel_content, title=f"[cyan]{txn_name}[/cyan]", expand=False))
    else:
        print(f"\n{'─'*60}")
        print(f"  {txn_name}")
        print(f"  {label} — {action}")
        print(f"  Fraud probability: {prob:.2%}  (confidence: {conf})")
        print(f"  [{bar}]")
        if factors:
            print(f"  Risk factors: {', '.join(factors[:5])}")
        print(f"{'─'*60}")


def prompt_custom_transaction() -> dict:
    """Interactively prompt user for a custom transaction."""
    cprint("\n[bold cyan]Enter transaction details[/bold cyan] (press Enter to use default):")

    def prompt(msg, default, cast=str):
        raw = input(f"  {msg} [{default}]: ").strip()
        return cast(raw) if raw else cast(default)

    amount = prompt("Amount (USD)", "100.00", float)

    cprint(f"\n  Merchant categories: {', '.join(MERCHANT_CATS[:10])}...")
    merchant = prompt("Merchant category", "RETAIL").upper()

    country = prompt("Country code (e.g. US, GB, NG)", "US").upper()
    is_dom_raw = prompt("Domestic transaction? (y/n)", "y")
    is_domestic = is_dom_raw.lower() in ("y", "yes", "1", "true")

    hour = prompt("Hour of day (0-23)", "14", int)
    dow = prompt("Day of week (0=Mon, 6=Sun)", "1", int)

    cprint(f"\n  Channels: {', '.join(CHANNELS)}")
    channel = prompt("Channel", "POS_CHIP").upper()

    currency = prompt("Currency", "USD").upper()

    cprint(f"\n  Velocities: {', '.join(VELOCITIES)}")
    velocity = prompt("Velocity", "NORMAL").upper()

    cprint(f"\n  Flags (comma-separated, or empty): {', '.join(AVAILABLE_FLAGS[:6])}...")
    flags_raw = input("  Flags []: ").strip()
    flags = [f.strip().upper() for f in flags_raw.split(",") if f.strip()]

    return {
        "amount":       amount,
        "merchant_cat": merchant,
        "country":      country,
        "is_domestic":  is_domestic,
        "hour":         hour,
        "day_of_week":  dow,
        "channel":      channel,
        "currency":     currency,
        "velocity":     velocity,
        "flags":        flags,
    }


def main():
    cprint(BANNER, style="bold cyan")

    # Load model
    checkpoint = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint):
        cprint(
            "[bold red]❌ No trained model found![/bold red]\n"
            "   Run [bold]python train.py[/bold] first to train the model.\n"
            "   Quick start: [bold]python train.py --epochs 5[/bold]",
        )
        sys.exit(1)

    cprint("⏳ Loading model from checkpoint...", style="dim")
    from inference import FraudDetector
    detector = FraudDetector.from_checkpoint(checkpoint)
    cprint("✅ Model loaded.\n", style="bold green")

    while True:
        cprint("[bold]Choose an option:[/bold]")
        cprint("  [1-6] Run a pre-loaded example transaction")
        cprint("  [c]   Enter a custom transaction")
        cprint("  [a]   Run all pre-loaded examples")
        cprint("  [q]   Quit")

        choice = input("\n> ").strip().lower()

        if choice == "q":
            cprint("\nGoodbye! 💳", style="bold")
            break

        elif choice == "a":
            cprint("\n[bold]Running all example transactions...[/bold]\n")
            for ex in PRELOADED_EXAMPLES:
                result = detector.predict(ex["txn"])
                print_result(ex["name"], result)

        elif choice == "c":
            txn = prompt_custom_transaction()
            result = detector.predict(txn)
            print_result("Custom Transaction", result)

        elif choice.isdigit() and 1 <= int(choice) <= len(PRELOADED_EXAMPLES):
            ex = PRELOADED_EXAMPLES[int(choice) - 1]
            result = detector.predict(ex["txn"])
            print_result(ex["name"], result)

        else:
            cprint("  Invalid choice. Please enter 1-6, c, a, or q.", style="dim")


if __name__ == "__main__":
    main()
