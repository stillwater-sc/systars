# Files

  | File               | Components                                        | Description                                                                                        |
  |--------------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------|
  | config.py          | StencilConfig, StencilActivation, StencilDataflow | Configuration dataclass with hardware parameters, predefined configs (DEFAULT, SMALL, LARGE, EDGE) |
  | line_buffer.py     | LineBufferBank, LineBufferUnit                    | SRAM banks for row storage, circular buffer management, K_h parallel outputs                       |
  | window_former.py   | WindowFormer                                      | Shift register array (K_h × K_w), sliding window extraction, stride support                        |
  | mac_array.py       | MACBank, ChannelParallelMAC                       | K×K multipliers + adder tree, P_c parallel banks with broadcast                                    |
  | stencil_machine.py | StencilMachine, StencilState                      | Top-level integration, controller FSM, activation functions (ReLU, ReLU6)                          |

Tests created: tests/unit/test_stencil.py with 43 comprehensive tests covering:

- Configuration validation
- Line buffer read/write operations
- Window forming and validity
- MAC bank dot product computation
- Parallel channel processing
- FSM state transitions
- Verilog generation

Bug fix: Fixed address bit width calculation that was causing value overflow. Changed from (value-1).bit_length() to value.bit_length() for configuration signals.

Test results: All 271 unit tests pass (43 new stencil tests + 228 existing tests).
