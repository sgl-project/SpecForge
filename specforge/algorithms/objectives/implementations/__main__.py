"""Print the registered loss-to-implementation mapping as Markdown."""

from .registry import implementation_registrations


def main():
    """Generate the inventory from registrations, without a parallel mapping."""
    print("| Implementation | Category | Mathematical losses |")
    print("| --- | --- | --- |")
    for name, entry in sorted(implementation_registrations().items()):
        losses = ", ".join(entry.losses) or "registered reference formulas"
        print(f"| {name} | {entry.category} | {losses} |")


if __name__ == "__main__":
    main()
