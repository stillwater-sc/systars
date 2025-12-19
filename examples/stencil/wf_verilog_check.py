from amaranth.back import verilog

from systars.stencil import StencilConfig, WindowFormer

config = StencilConfig(
    max_width=8,
    max_height=8,
    max_kernel_h=3,
    max_kernel_w=3,
    parallel_channels=4,
)

wf = WindowFormer(config)
output = verilog.convert(wf, name="WindowFormer")

# Look for fill_count logic
lines = output.split("\n")
in_block = False
for i, line in enumerate(lines):
    if "fill_count" in line:
        # Print surrounding context
        start = max(0, i - 2)
        end = min(len(lines), i + 3)
        print(f"=== Lines {start + 1}-{end} ===")
        for j in range(start, end):
            prefix = ">>> " if j == i else "    "
            print(f"{prefix}{lines[j]}")
        print()


# Find definition of $8
for i, line in enumerate(output.split("\n")):
    if "\\$8 " in line and "assign" in line:
        print(f"{i + 1}: {line}")
